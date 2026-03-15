import os
import sqlite3
import numpy as np
from fastapi import FastAPI, HTTPException, Header, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

from backend.core.ai_interpreter import AIInterpreter
from backend.core.knowledge_rag import RAGRetriever
from backend.core.calibrator import calibrator
from backend.core.algorithm_visualizer import AlgorithmVisualizer

app = FastAPI(title="CNTA ML Project API")

# 允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = r'd:\CNTDATA\CNTA_ML_Project\database\cnta_experiments.sqlite'
KB_DB_PATH = r'd:\CNTDATA\CNTA_ML_Project\database\cnta_knowledge_base.sqlite'
IMAGE_ROOT = r'd:\CNTDATA'

# 初始化 RAG 检索器（自动建表）
rag_retriever = RAGRetriever(DB_PATH, knowledge_db_path=KB_DB_PATH)

# 挂载图片目录，让前端能访问
if os.path.exists(IMAGE_ROOT):
    app.mount("/images", StaticFiles(directory=IMAGE_ROOT), name="images")

# 根路由：返回前端页面
@app.get("/")
async def read_root():
    """返回前端主页"""
    index_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "index.html")
    return FileResponse(index_path)

from fastapi.responses import Response
from PIL import Image
import io

# 图像缩略图缓存目录
CACHE_DIR = r'd:\CNTDATA\CNTA_ML_Project\backend\data\temp'
os.makedirs(CACHE_DIR, exist_ok=True)

@app.get("/api/view/tif")
async def view_tif(path: str):
    """
    将 TIFF 实时转码为 JPEG 供浏览器预览
    """
    full_path = os.path.join(IMAGE_ROOT, path.lstrip('/'))
    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail="Image not found")
        
    try:
        with Image.open(full_path) as img:
            # 转换为 RGB (有些 TIFF 是 16位或灰度)
            img = img.convert("RGB")
            # 缩小尺寸以加快预览速度
            img.thumbnail((800, 800))
            
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            return Response(content=buf.getvalue(), media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from src.analysis.feature_extractor import FeatureExtractor
import cv2

@app.post("/api/images/{image_id}/analyze")
async def analyze_image_v2(image_id: int):
    """
    手动触发 AI 重新分析（混合方案：快速响应 + 高精度可视化）
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM images WHERE id = ?", (image_id,))
    row = cursor.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Image not found")

    img_path = row['file_path']
    mag = row['magnification']

    if not os.path.exists(img_path):
        raise HTTPException(status_code=404, detail="Physical file missing")

    try:
        img = cv2.imread(img_path, 0)

        # 快速特征提取（保证响应速度）
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(img)
        smoothed = cv2.GaussianBlur(enhanced, (3, 3), 0)
        _, thresh = cv2.threshold(smoothed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 距离变换（不进行闭运算）
        dist = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
        diameter_px = np.median(dist[dist > 1.5]) * 2

        # 密度计算
        density = np.count_nonzero(thresh) / thresh.size * 100

        # 对齐计算（结构张量）
        Ix = cv2.Scharr(enhanced, cv2.CV_64F, 1, 0)
        Iy = cv2.Scharr(enhanced, cv2.CV_64F, 0, 1)
        Jxx = cv2.GaussianBlur(Ix * Ix, (3, 3), 0)
        Jyy = cv2.GaussianBlur(Iy * Iy, (3, 3), 0)
        trace = Jxx + Jyy + 1e-10
        coherence = np.abs(Jxx - Jyy) / trace
        hof = float(np.clip(np.mean(coherence), 0, 1))
        mean_phi_deg = float(np.degrees(np.arccos(np.sqrt(np.clip(np.mean(coherence), 0, 1)))))

        # 波浪系数计算（替代曲率）
        # 计算沿水平方向（垂直生长CNT）的强度波动
        row_intensities = smoothed.mean(axis=1)
        wave_coefficient = float(np.std(row_intensities) / np.mean(row_intensities) if np.mean(row_intensities) > 0 else 0.0)

        # 转换直径为nm
        px_per_um = 50.0 if mag and mag == 50000 else 100.0
        diameter_nm = diameter_px / px_per_um * 1000.0

        results = {
            'diameter': float(diameter_nm),
            'density': float(density),
            'alignment': float(hof),
            'curvature': float(wave_coefficient),  # 使用波浪系数
            'mean_phi_deg': float(mean_phi_deg),
            'px_per_um': float(px_per_um)
        }

        # 更新数据库（修改curvature说明为平均曲率）
        cursor.execute("""
            UPDATE images
            SET diameter = ?, density = ?, alignment = ?, curvature = ?, processed = 1
            WHERE id = ?
        """, (results['diameter'], results['density'], results['alignment'], results['curvature'], image_id))

        conn.commit()
        return {"status": "success", "results": results}
    except Exception as e:
        print(f"Error in analyze: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

class ImageDetail(BaseModel):
    file_path: Optional[str] = None
    source: Optional[str] = None
    sample_id: Optional[str] = None
    membrane_id: Optional[int] = None
    growth_temp: Optional[float] = None
    growth_time: Optional[float] = None
    ar_flow: Optional[float] = None
    h2_flow: Optional[float] = None
    c2h4_flow: Optional[float] = None
    al2o3_power: Optional[float] = None
    al2o3_thickness: Optional[float] = None
    fe_power: Optional[float] = None
    fe_thickness: Optional[float] = None
    anneal_temp: Optional[float] = None
    anneal_time: Optional[float] = None
    position_label: Optional[str] = None
    magnification: Optional[int] = None
    horizontal_pos: Optional[str] = None
    vertical_pos: Optional[int] = None
    repeat_id: Optional[int] = None
    catalyst_weight: Optional[float] = None
    actual_temp: Optional[float] = None
    membrane_pos_cm: Optional[float] = None
    diameter: Optional[float] = None
    density: Optional[float] = None
    alignment: Optional[float] = None
    curvature: Optional[Any] = None
    tortuosity: Optional[float] = None
    processed: Optional[int] = 0

@app.post("/api/images")
async def create_image(image: ImageDetail):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    data = calibrator.calibrate(image.dict())
    
    fields = []
    values = []
    placeholders = []
    
    for k, v in data.items():
        if v is not None:
            fields.append(k)
            values.append(v)
            placeholders.append("?")
            
    if not fields:
        raise HTTPException(status_code=400, detail="No data provided")
        
    query = f"INSERT INTO images ({', '.join(fields)}) VALUES ({', '.join(placeholders)})"
    cursor.execute(query, values)
    new_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return {"status": "success", "id": new_id}

@app.put("/api/images/{image_id}")
async def update_image(image_id: int, image: ImageDetail):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    data = calibrator.calibrate(image.dict())
    
    updates = []
    values = []
    
    for k, v in data.items():
        if v is not None:
            updates.append(f"{k} = ?")
            values.append(v)
            
    if not updates:
        raise HTTPException(status_code=400, detail="No data provided")
        
    values.append(image_id)
    query = f"UPDATE images SET {', '.join(updates)} WHERE id = ?"
    cursor.execute(query, values)
    conn.commit()
    conn.close()
    return {"status": "success"}

@app.delete("/api/images/{image_id}")
async def delete_image(image_id: int):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM images WHERE id = ?", (image_id,))
    conn.commit()
    conn.close()
    return {"status": "success"}

@app.post("/api/images/{image_id}/features")
async def update_features(image_id: int, features: Dict[str, Any]):
    # 保持兼容性，允许手动修正数值
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE images 
        SET diameter = ?, density = ?, alignment = ?, curvature = ?, tortuosity = ?, processed = 1
        WHERE id = ?
    """, (features.get('diameter'), features.get('density'), features.get('alignment'), features.get('curvature'), features.get('tortuosity'), image_id))
    conn.commit()
    conn.close()
    return {"status": "success"}

@app.get("/api/summary")
async def get_summary():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT source, COUNT(*) FROM images GROUP BY source")
    source_counts = dict(cursor.fetchall())
    
    cursor.execute("SELECT COUNT(*) FROM images WHERE processed = 1")
    processed_count = cursor.fetchone()[0]
    
    conn.close()
    return {
        "total_images": sum(source_counts.values()),
        "source_breakdown": source_counts,
        "processed_images": processed_count
    }

from backend.core.database_helpers import resolve_sort

@app.get("/api/images")
async def get_image_list(
    source: Optional[str] = None, 
    min_temp: Optional[float] = None,
    max_temp: Optional[float] = None,
    processed: Optional[int] = None,
    limit: int = 15, 
    offset: int = 0,
    sort_by: str = "id",
    order: str = "desc"
):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # 动态构建 WHERE 子句
    where_clauses = []
    params = []
    
    if source:
        where_clauses.append("source = ?")
        params.append(source)
    if min_temp is not None:
        where_clauses.append("growth_temp >= ?")
        params.append(min_temp)
    if max_temp is not None:
        where_clauses.append("growth_temp <= ?")
        params.append(max_temp)
    if processed is not None:
        where_clauses.append("processed = ?")
        params.append(processed)
        
    where_sql = " WHERE " + " AND ".join(where_clauses) if where_clauses else ""
    
    # 解析排序
    _, sort_col, sort_order = resolve_sort(sort_by, order)
    
    # 执行主查询
    query = f"SELECT * FROM images {where_sql} ORDER BY {sort_col} {sort_order} LIMIT ? OFFSET ?"
    cursor.execute(query, params + [limit, offset])
    rows = cursor.fetchall()
    
    # 获取总数用于前端分页计算
    count_query = f"SELECT COUNT(*) FROM images {where_sql}"
    cursor.execute(count_query, params)
    total_count = cursor.fetchone()[0]
    
    results = []
    for row in rows:
        d = dict(row)
        rel_path = os.path.relpath(d['file_path'], IMAGE_ROOT).replace("\\", "/")
        d['url'] = f"/images/{rel_path}"
        results.append(d)
        
    conn.close()
    return {
        "items": results,
        "total": total_count,
        "limit": limit,
        "offset": offset
    }

# ─────────────────────────────────────────────────────────────────── #
#  AI 解释 & 对话 Endpoints
# ─────────────────────────────────────────────────────────────────── #

class ChatRequest(BaseModel):
    message: str
    image_id: Optional[int] = None
    history: Optional[List[Dict[str, str]]] = []


def _get_interpreter(x_provider: str, x_api_key: str, x_model: Optional[str] = None) -> AIInterpreter:
    """从请求头构建 AIInterpreter，支持指定模型"""
    if not x_api_key:
        raise HTTPException(status_code=401, detail="缺少 X-Api-Key 请求头")
    provider = (x_provider or "glm").lower()
    try:
        return AIInterpreter(provider=provider, api_key=x_api_key, model=x_model)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/images/{image_id}/interpret")
async def interpret_image(
    image_id: int,
    x_provider: str = Header(default="glm"),
    x_api_key: str = Header(default=""),
    x_model: Optional[str] = Header(default=None),
    x_temperature: Optional[str] = Header(default="0.5"),
):
    """
    流式返回特征可解释性分析报告（SSE）。
    前端通过 fetch + ReadableStream 接收。
    """
    # 读取图像记录
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM images WHERE id = ?", (image_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="Image not found")

    row_dict = dict(row)
    features = {
        "density": row_dict.get("density"),
        "alignment": row_dict.get("alignment"),
        "diameter": row_dict.get("diameter"),
        "curvature": row_dict.get("curvature"),
        "tortuosity": row_dict.get("tortuosity"),
    }
    params = {k: row_dict.get(k) for k in [
        "sample_id", "source", "growth_temp", "actual_temp", "growth_time",
        "anneal_temp", "anneal_time", "ar_flow", "h2_flow", "c2h4_flow",
        "fe_thickness", "fe_power", "al2o3_thickness", "al2o3_power",
        "position_label", "membrane_pos_cm", "magnification",
    ]}
    params["current_id"] = image_id

    # RAG 检索
    rag_results = rag_retriever.retrieve_all(
        features, params,
        query=f"CNT density {features.get('density')} alignment {features.get('alignment')} diameter {features.get('diameter')}"
    )

    interpreter = _get_interpreter(x_provider, x_api_key, x_model)
    temperature = float(x_temperature or 0.5)

    def event_stream():
        yield from interpreter.interpret_stream(
            features=features,
            params=params,
            similar_exps=rag_results["similar_experiments"],
            pdf_passages=rag_results["pdf_passages"],
            temperature=temperature,
        )

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/api/chat")
async def chat_with_ai(
    req: ChatRequest,
    x_provider: str = Header(default="glm"),
    x_api_key: str = Header(default=""),
    x_model: Optional[str] = Header(default=None),
    x_temperature: Optional[str] = Header(default="0.5"),
):
    """
    流式对话接口（SSE）。
    可传入 image_id 以携带当前图像上下文。
    """
    context = None
    if req.image_id is not None:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM images WHERE id = ?", (req.image_id,)).fetchone()
        conn.close()
        if row:
            row_dict = dict(row)
            features = {
                "density": row_dict.get("density"),
                "alignment": row_dict.get("alignment"),
                "diameter": row_dict.get("diameter"),
                "curvature": row_dict.get("curvature"),
                "tortuosity": row_dict.get("tortuosity"),
            }
            params = {k: row_dict.get(k) for k in [
                "sample_id", "source", "growth_temp", "fe_thickness",
                "al2o3_thickness", "c2h4_flow", "h2_flow", "ar_flow",
            ]}
            params["current_id"] = req.image_id
            similar = rag_retriever.retrieve_from_db(features, params, top_k=3)
            context = {
                "features": features,
                "params": params,
                "similar_experiments": similar,
            }

    interpreter = _get_interpreter(x_provider, x_api_key, x_model)
    temperature = float(x_temperature or 0.5)

    def event_stream():
        yield from interpreter.chat_stream(
            history=req.history or [],
            user_message=req.message,
            context=context,
            temperature=temperature,
        )

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ─────────────────────────────────────────────────────────────────── #
#  RAG 文献管理 Endpoints
# ─────────────────────────────────────────────────────────────────── #

@app.post("/api/rag/upload")
async def upload_rag_document(file: UploadFile = File(...)):
    """上传 PDF 文献，提取文本并分块存入数据库"""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="仅支持 PDF 格式")
    content = await file.read()
    try:
        result = rag_retriever.add_pdf(content, file.filename)
        return {"status": "success", **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF处理失败: {e}")


@app.get("/api/rag/documents")
async def list_rag_documents():
    """列出所有已上传的文献"""
    return rag_retriever.list_documents()


@app.delete("/api/rag/documents/{doc_id}")
async def delete_rag_document(doc_id: int):
    """删除指定文献及其所有分块"""
    try:
        rag_retriever.delete_document(doc_id)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class RAGSearchRequest(BaseModel):
    query: str
    task_name: Optional[str] = None
    top_k: int = 5


@app.get("/api/rag/stats")
async def get_rag_stats():
    return rag_retriever.get_stats()


@app.post("/api/rag/search")
async def search_rag_documents(req: RAGSearchRequest):
    return {
        "items": rag_retriever.retrieve_from_pdf(
            query=req.query,
            top_k=req.top_k,
            task_name=req.task_name,
        )
    }


# ========================================================================
# 算法可视化API
# ========================================================================

class VisualizationResponse(BaseModel):
    steps: List[Dict[str, Any]]
    total_steps: int


@app.get("/api/images/{image_id}/visualize", response_model=VisualizationResponse)
async def visualize_image_analysis(image_id: int):
    """获取图像分析的可视化步骤"""
    # 查询图像路径
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT file_path, magnification FROM images WHERE id = ?", (image_id,))
    result = cursor.fetchone()
    conn.close()

    if not result:
        raise HTTPException(status_code=404, detail="Image not found")

    filepath = result['file_path']
    mag = result['magnification'] if result['magnification'] else 50000

    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Image file not found")

    # 读取图像
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise HTTPException(status_code=400, detail="Failed to read image")

    # 生成可视化步骤
    visualizer = AlgorithmVisualizer(magnification=mag)
    visualizer.visualize_extraction(img)

    return {
        "steps": visualizer.get_steps(),
        "total_steps": len(visualizer.get_steps())
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
