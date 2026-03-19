import os
import json
import re
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
    return FileResponse(
        index_path,
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )

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

        # 快速特征提取
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(img)
        smoothed = cv2.GaussianBlur(enhanced, (3, 3), 0)
        _, thresh = cv2.threshold(smoothed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 距离变换
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

        # 骨架追踪曲率计算
        from skimage.morphology import skeletonize
        from skimage.measure import label
        from scipy.ndimage import convolve

        # 完整骨架
        skel_full = skeletonize(thresh > 0)

        # 提取最长骨架
        longest_skel = _extract_longest_skeleton(skel_full).astype(np.uint8) * 255
        labeled = label(longest_skel, connectivity=2)

        px_per_nm = 0.05 if mag and mag == 50000 else 0.1  # 像素转nm系数
        all_curvatures = []

        n_regions = labeled.max()
        if n_regions > 0:
            for rid in range(1, min(n_regions + 1, 15)):
                region_mask = (labeled == rid).astype(np.uint8)
                coords = np.argwhere(region_mask).astype(float)

                if len(coords) < 10:
                    continue

                # 找端点
                kernel = np.ones((3, 3), dtype=np.uint8)
                kernel[1, 1] = 0
                neighbor_count = convolve(region_mask, kernel, mode='constant', cval=0)
                endpoints = np.argwhere((region_mask > 0) & (neighbor_count == 1))

                if len(endpoints) >= 2:
                    # 追踪路径并计算曲率
                    path = _trace_skeleton(region_mask, endpoints[0], endpoints[1])
                    if len(path) > 5:
                        for i in range(2, min(len(path) - 2, 50)):
                            p0, p1, p2, p3, p4 = path[i-2], path[i-1], path[i], path[i+1], path[i+2]
                            v1 = p2 - p0
                            v2 = p4 - p2
                            angle1 = np.arctan2(v1[0], v1[1])
                            angle2 = np.arctan2(v2[0], v2[1])
                            d_theta = abs(angle2 - angle1)
                            ds = np.linalg.norm(p4 - p0)
                            if ds > 0:
                                curvature_px = d_theta / ds
                                curvature_nm = curvature_px / px_per_nm
                                if curvature_nm < 2.0:
                                    all_curvatures.append(curvature_nm)

        avg_curvature_nm = float(np.median(all_curvatures)) if all_curvatures else 0.0

        # 转换直径为nm
        px_per_um = 50.0 if mag and mag == 50000 else 100.0
        diameter_nm = diameter_px / px_per_um * 1000.0

        results = {
            'diameter': float(diameter_nm),
            'density': float(density),
            'alignment': float(hof),
            'curvature': float(avg_curvature_nm),  # 骨架追踪曲率
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

# 标记图像为已删除
@app.put("/api/images/{image_id}/delete")
async def soft_delete_image(image_id: int):
    """
    标记图像为已删除（逻辑删除）
    - is_deleted = 1: 已删除
    - is_deleted = 0: 正常
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("UPDATE images SET is_deleted = 1 WHERE id = ?", (image_id,))
    if cursor.rowcount == 0:
        conn.close()
        raise HTTPException(status_code=404, detail="Image not found")

    conn.commit()
    conn.close()

    return {"status": "success", "deleted_id": image_id}


# 恢复已删除的图像
@app.put("/api/images/{image_id}/restore")
async def restore_deleted_image(image_id: int):
    """
    恢复已删除的图像
    - is_deleted = 0: 恢复正常状态
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("UPDATE images SET is_deleted = 0 WHERE id = ?", (image_id,))
    if cursor.rowcount == 0:
        conn.close()
        raise HTTPException(status_code=404, detail="Image not found")

    conn.commit()
    conn.close()

    return {"status": "success", "restored_id": image_id}


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


_XR_SAMPLE_PATTERN = re.compile(r"([A-Za-z])\s*(\d+)\s*-?\s*([A-Za-z])\s*(\d+)")
_XR_TARGET_KEYS = ("diameter", "density", "alignment", "curvature")
_XR_FEATURE_KEYS = ("actual_temp", "flow_rate", "catalyst_concentration")


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _sse_payload(payload: Dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _stream_with_error_boundary(stream_factory):
    try:
        yield from stream_factory()
    except Exception as e:
        detail = str(e) or "unknown stream error"
        yield _sse_payload({"type": "error", "detail": detail})
        yield _sse_payload({"type": "done"})


def _parse_xr_label(sample_id: Optional[str], file_path: Optional[str]) -> Optional[Dict[str, Any]]:
    candidates: List[str] = []
    if sample_id:
        candidates.append(sample_id)
    if file_path:
        stem = os.path.splitext(os.path.basename(file_path))[0]
        candidates.append(stem)

    for text in candidates:
        match = _XR_SAMPLE_PATTERN.search(text)
        if not match:
            continue
        return {
            "orientation": match.group(1).upper(),
            "sample_no": int(match.group(2)),
            "position": match.group(3).upper(),
            "shot_no": int(match.group(4)),
        }
    return None


def _fit_linear_model(rows: List[Dict[str, Any]], target_field: str) -> Dict[str, Any]:
    x_rows = []
    y_vals = []
    for row in rows:
        x = [row.get("actual_temp"), row.get("flow_rate"), row.get("catalyst_concentration")]
        y = row.get(target_field)
        if any(v is None for v in x) or y is None:
            continue
        x_rows.append([1.0] + [float(v) for v in x])
        y_vals.append(float(y))

    n_train = len(y_vals)
    if n_train == 0:
        return {
            "method": "none",
            "n_train": 0,
            "coef": [0.0, 0.0, 0.0, 0.0],
            "mean": None,
        }

    y_arr = np.asarray(y_vals, dtype=float)
    y_mean = float(np.mean(y_arr))

    if n_train < 4:
        return {
            "method": "mean",
            "n_train": n_train,
            "coef": [y_mean, 0.0, 0.0, 0.0],
            "mean": y_mean,
        }

    x_arr = np.asarray(x_rows, dtype=float)
    coef, _, _, _ = np.linalg.lstsq(x_arr, y_arr, rcond=None)
    return {
        "method": "ols",
        "n_train": n_train,
        "coef": [float(v) for v in coef.tolist()],
        "mean": y_mean,
    }


def _predict_linear_model(model: Dict[str, Any], row: Dict[str, Any]) -> Optional[float]:
    x = [row.get("actual_temp"), row.get("flow_rate"), row.get("catalyst_concentration")]
    if any(v is None for v in x):
        return None
    coef = model.get("coef") or [0.0, 0.0, 0.0, 0.0]
    x_full = [1.0] + [float(v) for v in x]
    pred = sum(c * v for c, v in zip(coef, x_full))
    return float(pred)


def _pearson(xs: List[float], ys: List[float]) -> Optional[float]:
    if len(xs) < 2 or len(ys) < 2:
        return None
    x_arr = np.asarray(xs, dtype=float)
    y_arr = np.asarray(ys, dtype=float)
    if float(np.std(x_arr)) == 0.0 or float(np.std(y_arr)) == 0.0:
        return None
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


@app.get("/api/ml/xr/simple-model")
async def get_xr_simple_model_data(limit: int = 2000):
    """
    XR 简易建模数据接口：
    - 仅保留首字母 C 的样本
    - 样本定义：run + 第一个数字
    - 输出字段：样本号/拍摄部位/拍摄编号 + 工艺变量 + 4个形貌特征(实际/预测)
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT
            id, file_path, sample_id, actual_temp, growth_temp,
            ar_flow, catalyst_weight,
            diameter, density, alignment, curvature
        FROM images
        WHERE source = 'XR'
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    )
    db_rows = cursor.fetchall()
    conn.close()

    rows: List[Dict[str, Any]] = []
    for db_row in db_rows:
        file_path = db_row["file_path"]
        parsed = _parse_xr_label(db_row["sample_id"], file_path)
        if not parsed or parsed["orientation"] != "C":
            continue

        run_name = os.path.basename(os.path.dirname(file_path or ""))
        actual_temp = _safe_float(db_row["actual_temp"])
        if actual_temp is None:
            actual_temp = _safe_float(db_row["growth_temp"])

        row = {
            "id": int(db_row["id"]),
            "run_name": run_name,
            "sample_id": db_row["sample_id"],
            "sample_no": parsed["sample_no"],
            "position": parsed["position"],
            "shot_no": parsed["shot_no"],
            "actual_temp": actual_temp,
            "flow_rate": _safe_float(db_row["ar_flow"]),
            "catalyst_concentration": _safe_float(db_row["catalyst_weight"]),
            "diameter_actual": _safe_float(db_row["diameter"]),
            "density_actual": _safe_float(db_row["density"]),
            "alignment_actual": _safe_float(db_row["alignment"]),
            "curvature_actual": _safe_float(db_row["curvature"]),
        }
        rows.append(row)

    if not rows:
        return {
            "summary": {
                "total_rows": 0,
                "n_runs": 0,
                "n_samples": 0,
                "label_counts": {key: 0 for key in _XR_TARGET_KEYS},
            },
            "coefficients": {},
            "correlations": {},
            "rows": [],
        }

    # build simple model per target
    coefficients: Dict[str, Dict[str, Any]] = {}
    label_counts: Dict[str, int] = {}
    correlations: Dict[str, Dict[str, Optional[float]]] = {}

    for target in _XR_TARGET_KEYS:
        target_field = f"{target}_actual"
        model = _fit_linear_model(rows, target_field)
        coefficients[target] = {
            "method": model["method"],
            "n_train": model["n_train"],
            "intercept": model["coef"][0],
            "coef_temp": model["coef"][1],
            "coef_flow": model["coef"][2],
            "coef_catalyst": model["coef"][3],
        }
        label_counts[target] = model["n_train"]

        for row in rows:
            row[f"{target}_pred"] = _predict_linear_model(model, row)

        # feature-target Pearson correlations (labeled subset)
        corr_result: Dict[str, Optional[float]] = {}
        for feature in _XR_FEATURE_KEYS:
            xs: List[float] = []
            ys: List[float] = []
            for row in rows:
                x_val = row.get(feature)
                y_val = row.get(target_field)
                if x_val is None or y_val is None:
                    continue
                xs.append(float(x_val))
                ys.append(float(y_val))
            corr_result[feature] = _pearson(xs, ys)
        correlations[target] = corr_result

    # row-level label status
    for row in rows:
        labeled_num = sum(1 for target in _XR_TARGET_KEYS if row.get(f"{target}_actual") is not None)
        row["label_status"] = "labeled" if labeled_num == len(_XR_TARGET_KEYS) else ("partial" if labeled_num > 0 else "unlabeled")

    run_keys = set(row["run_name"] for row in rows)
    sample_keys = set((row["run_name"], row["sample_no"]) for row in rows)

    return {
        "summary": {
            "total_rows": len(rows),
            "n_runs": len(run_keys),
            "n_samples": len(sample_keys),
            "label_counts": label_counts,
        },
        "coefficients": coefficients,
        "correlations": correlations,
        "rows": rows,
    }


def _has_xr_schema(cursor: sqlite3.Cursor) -> bool:
    rows = cursor.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table' AND name IN ('xr_runs', 'xr_images', 'xr_targets')
        """
    ).fetchall()
    return len(rows) == 3


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

    if source == "XR" and _has_xr_schema(cursor):
        _, sort_col, sort_order = resolve_sort(sort_by, order)

        where_clauses = ["i.source = 'XR'"]
        params = []
        if min_temp is not None:
            where_clauses.append("COALESCE(r.set_temp_c, i.growth_temp) >= ?")
            params.append(min_temp)
        if max_temp is not None:
            where_clauses.append("COALESCE(r.set_temp_c, i.growth_temp) <= ?")
            params.append(max_temp)
        if processed is not None:
            where_clauses.append("COALESCE(xi.processed, i.processed, 0) = ?")
            params.append(processed)
        where_sql = " WHERE " + " AND ".join(where_clauses)

        xr_sort_map = {
            "id": "id",
            "sample_id": "sample_id",
            "growth_temp": "growth_temp",
            "growth_time": "growth_time",
            "ar_flow": "ar_flow",
            "catalyst_weight": "catalyst_weight",
            "magnification": "magnification",
            "diameter": "diameter",
            "density": "density",
            "alignment": "alignment",
            "curvature": "curvature",
            "source": "source",
        }
        xr_sort_col = xr_sort_map.get(sort_col, "id")

        query = f"""
            SELECT
                i.id AS id,
                i.file_path AS file_path,
                'XR' AS source,
                COALESCE(xi.sample_id, i.sample_id) AS sample_id,
                COALESCE(r.set_temp_c, i.growth_temp) AS growth_temp,
                COALESCE(r.growth_time_h, i.growth_time) AS growth_time,
                COALESCE(r.ar_flow, i.ar_flow) AS ar_flow,
                COALESCE(r.catalyst_concentration, i.catalyst_weight) AS catalyst_weight,
                COALESCE(xi.position_label, i.position_label) AS position_label,
                COALESCE(xi.horizontal_pos, i.horizontal_pos) AS horizontal_pos,
                COALESCE(xi.vertical_pos, i.vertical_pos) AS vertical_pos,
                COALESCE(xi.magnification, i.magnification) AS magnification,
                COALESCE(xi.actual_temp_c, i.actual_temp) AS actual_temp,
                COALESCE(xi.membrane_pos_cm, i.membrane_pos_cm) AS membrane_pos_cm,
                t.diameter AS diameter,
                t.density AS density,
                t.alignment AS alignment,
                t.curvature AS curvature,
                t.tortuosity AS tortuosity,
                COALESCE(xi.processed, i.processed, 0) AS processed
            FROM images i
            LEFT JOIN xr_images xi ON xi.file_path = i.file_path
            LEFT JOIN xr_runs r ON r.run_id = xi.run_id
            LEFT JOIN xr_targets t ON t.target_id = (
                SELECT MAX(t2.target_id)
                FROM xr_targets t2
                WHERE t2.image_id = xi.image_id
            )
            {where_sql}
            ORDER BY {xr_sort_col} {sort_order}
            LIMIT ? OFFSET ?
        """
        cursor.execute(query, params + [limit, offset])
        rows = cursor.fetchall()

        count_query = f"""
            SELECT COUNT(*)
            FROM images i
            LEFT JOIN xr_images xi ON xi.file_path = i.file_path
            LEFT JOIN xr_runs r ON r.run_id = xi.run_id
            {where_sql}
        """
        cursor.execute(count_query, params)
        total_count = cursor.fetchone()[0]

        results = []
        for row in rows:
            d = dict(row)
            rel_path = os.path.relpath(d["file_path"], IMAGE_ROOT).replace("\\", "/")
            d["url"] = f"/images/{rel_path}"
            results.append(d)

        conn.close()
        return {
            "items": results,
            "total": total_count,
            "limit": limit,
            "offset": offset,
        }
    
    # 动态构建 WHERE 子句
    where_clauses = []
    params = []

    # 始终排除已删除的数据
    where_clauses.append("COALESCE(is_deleted, 0) = 0")

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
        yield from _stream_with_error_boundary(
            lambda: interpreter.interpret_stream(
                features=features,
                params=params,
                similar_exps=rag_results["similar_experiments"],
                pdf_passages=rag_results["pdf_passages"],
                knowledge_links=rag_results.get("knowledge_links", []),
                temperature=temperature,
            )
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
        yield from _stream_with_error_boundary(
            lambda: interpreter.chat_stream(
                history=req.history or [],
                user_message=req.message,
                context=context,
                temperature=temperature,
            )
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


@app.post("/api/rag/links")
async def search_rag_links(req: RAGSearchRequest):
    links = rag_retriever.knowledge_base.search_links(query=req.query, top_k=req.top_k)
    return {
        "items": links,
        "chain": rag_retriever.knowledge_base.get_relation_chain_summary(
            query=req.query,
            top_k=max(req.top_k, 20),
        ),
    }


# ========================================================================
# 算法可视化API
# ========================================================================

class VisualizationResponse(BaseModel):
    steps: List[Dict[str, Any]]
    total_steps: int


def prepare_visualization_image(img_gray: np.ndarray, max_side: int = 1280) -> np.ndarray:
    """Downscale only for visualization pipeline to keep endpoint latency bounded."""
    if img_gray is None:
        return img_gray
    h, w = img_gray.shape[:2]
    side = max(h, w)
    if side <= max_side:
        return img_gray
    scale = max_side / float(side)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(img_gray, (new_w, new_h), interpolation=cv2.INTER_AREA)


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
    img = prepare_visualization_image(img, max_side=1280)

    # 生成可视化步骤
    visualizer = AlgorithmVisualizer(magnification=mag)
    visualizer.visualize_extraction(img)

    return {
        "steps": visualizer.get_steps(),
        "total_steps": len(visualizer.get_steps())
    }


def _trace_skeleton(mask, start, end):
    """追踪骨架路径（辅助函数）"""
    path = []
    current = start.copy()
    visited = set()
    visited.add(tuple(current))

    max_steps = 500
    step = 0

    while step < max_steps:
        y, x = int(current[0]), int(current[1])
        path.append(current)

        if np.linalg.norm(current - end) < 3:
            break

        next_point = None
        min_dist = float('inf')

        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if (ny, nx) not in visited and 0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1]:
                    if mask[ny, nx] > 0:
                        dist = np.linalg.norm([ny, nx] - end)
                        if dist < min_dist:
                            min_dist = dist
                            next_point = np.array([ny, nx])

        if next_point is None:
            break

        visited.add(tuple(next_point))
        current = next_point
        step += 1

    return np.array(path) if len(path) > 0 else np.array([])


def _extract_longest_skeleton(skeleton):
    """在完整骨架中提取最长的一条连续路径"""
    from skimage.measure import label
    from scipy.ndimage import convolve

    skel = skeleton.copy().astype(np.uint8)

    labeled = label(skel, connectivity=2)
    n_regions = labeled.max()

    pruned = np.zeros_like(skel)

    if n_regions > 0:
        for rid in range(1, min(n_regions + 1, 20)):
            region_mask = (labeled == rid).astype(np.uint8)
            coords = np.argwhere(region_mask)

            if len(coords) < 10:
                continue

            kernel = np.ones((3, 3), dtype=np.uint8)
            kernel[1, 1] = 0
            neighbor_count = convolve(region_mask, kernel, mode='constant', cval=0)
            endpoints = np.argwhere((region_mask > 0) & (neighbor_count == 1))

            if len(endpoints) < 2:
                pruned[region_mask > 0] = 255
                continue

            longest_path = []
            longest_length = 0

            max_pairs = min(len(endpoints), 10)
            for i in range(max_pairs):
                for j in range(i + 1, max_pairs):
                    path = _trace_skeleton(region_mask, endpoints[i], endpoints[j])
                    if len(path) > longest_length:
                        longest_path = path
                        longest_length = len(path)

            if len(longest_path) > 5:
                for point in longest_path:
                    pruned[int(point[0]), int(point[1])] = 255

    return pruned


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
