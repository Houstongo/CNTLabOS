import os
import json
import re
import sqlite3
import numpy as np
from datetime import datetime
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
from backend.core.batch_processor import _extract_image_features, DEFAULT_CNTSEGNET_CHECKPOINT
from backend.core.knowledge_driven_predictor import KnowledgeDrivenPredictor
from backend.core.tccer_retriever import TCCERRetriever
from backend.core.msfu_extractor import MSFUExtractor, MSFUMetadata, store_msfus_in_db, get_msfu_stats
from backend.core.qa_service import QAService
from backend.core.segmentation_backend import (
    BOTH,
    CANONICAL_WCNTSEGNET,
    CNTSEGNET,
    LEGACY_THRESHOLD,
    normalize_segmentation_backend,
)

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

# 初始化 TCCER 检索器
tccer_retriever = TCCERRetriever(kb_db_path=KB_DB_PATH)

# 初始化知识驱动预测器
knowledge_predictor = KnowledgeDrivenPredictor(DB_PATH, rag_retriever)

# 初始化QA服务
qa_service = QAService(kb_db_path=KB_DB_PATH)

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


def _read_grayscale_image(image_path: str):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        return img

    try:
        data = np.fromfile(image_path, dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    except Exception:
        return None


class BatchImageActionRequest(BaseModel):
    image_ids: List[int]
    backend: Optional[str] = CANONICAL_WCNTSEGNET
    device: Optional[str] = "cpu"
    checkpoint: Optional[str] = None
    tile_size: Optional[int] = 512
    overlap: Optional[int] = 64
    seg_threshold: Optional[float] = 0.5


def _validate_batch_ids(image_ids: List[int]) -> List[int]:
    normalized = []
    seen = set()
    for image_id in image_ids or []:
        try:
            value = int(image_id)
        except (TypeError, ValueError):
            continue
        if value not in seen:
            seen.add(value)
            normalized.append(value)
    if not normalized:
        raise HTTPException(status_code=400, detail="image_ids must contain at least one valid id")
    return normalized


def _analyze_image_with_cursor(
    cursor: sqlite3.Cursor,
    image_id: int,
    backend: str = CANONICAL_WCNTSEGNET,
    device: str = "cpu",
    checkpoint: Optional[str] = None,
    tile_size: int = 512,
    overlap: int = 64,
    seg_threshold: float = 0.5,
) -> Dict[str, Any]:
    active_clause = _active_images_clause(cursor)
    cursor.execute(f"SELECT * FROM images WHERE id = ? AND {active_clause}", (image_id,))
    row = cursor.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Image not found")

    img_path = row["file_path"]
    mag = row["magnification"]

    if not os.path.exists(img_path):
        raise HTTPException(status_code=404, detail="Physical file missing")

    try:
        normalized_backend = normalize_segmentation_backend(backend, allow_both=False)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    results = _extract_image_features(
        file_path=img_path,
        magnification=mag,
        diameter_method="enhanced",
        segmentation_backend=normalized_backend,
        device=device or "cpu",
        checkpoint_path=checkpoint or DEFAULT_CNTSEGNET_CHECKPOINT,
        tile_size=tile_size or 512,
        overlap=overlap or 64,
        seg_threshold=seg_threshold if seg_threshold is not None else 0.5,
    )

    update_values = {
        "diameter": results.get("diameter"),
        "density": results.get("density"),
        "alignment": results.get("alignment"),
        "curvature": results.get("curvature_nm"), # 存储数值型曲率 (nm^-1) 用于机器学习
        "processed": 1,
    }
    for column in (
        "tortuosity",
        "waviness_ratio",
        "waviness_height_nm",
        "waviness_wavelength_nm",
        "waviness_branches",
    ):
        if _images_has_column(cursor, column):
            update_values[column] = results.get(column)

    assignments = ", ".join(f"{column} = ?" for column in update_values)
    cursor.execute(
        f"UPDATE images SET {assignments} WHERE id = ?",
        tuple(update_values.values()) + (image_id,),
    )
    return results


@app.post("/api/images/batch/analyze")
async def batch_analyze_images(req: BatchImageActionRequest):
    image_ids = _validate_batch_ids(req.image_ids)
    backend = normalize_segmentation_backend(req.backend, allow_both=False)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    items = []
    success_count = 0
    failed_count = 0
    skipped_count = 0

    try:
        for image_id in image_ids:
            cursor.execute(
                "SELECT id, COALESCE(is_deleted, 0) AS is_deleted FROM images WHERE id = ?",
                (image_id,),
            )
            row = cursor.fetchone()
            if not row:
                failed_count += 1
                items.append({"image_id": image_id, "status": "failed", "detail": "Image not found"})
                continue

            if int(row["is_deleted"] or 0) == 1:
                skipped_count += 1
                failed_count += 1
                items.append({"image_id": image_id, "status": "skipped", "detail": "Image is logically deleted"})
                continue

            try:
                results = _analyze_image_with_cursor(
                    cursor,
                    image_id,
                    backend=backend,
                    device=req.device or "cpu",
                    checkpoint=req.checkpoint,
                    tile_size=req.tile_size or 512,
                    overlap=req.overlap or 64,
                    seg_threshold=req.seg_threshold if req.seg_threshold is not None else 0.5,
                )
                success_count += 1
                items.append({"image_id": image_id, "status": "success", "results": results})
            except HTTPException as exc:
                failed_count += 1
                items.append({"image_id": image_id, "status": "failed", "detail": exc.detail})
            except Exception as exc:
                failed_count += 1
                items.append({"image_id": image_id, "status": "failed", "detail": str(exc)})

        conn.commit()
        return {
            "status": "success",
            "summary": {
                "requested_count": len(image_ids),
                "success_count": success_count,
                "failed_count": failed_count,
                "skipped_count": skipped_count,
            },
            "items": items,
        }
    finally:
        conn.close()


@app.put("/api/images/batch/delete")
async def batch_soft_delete_images(req: BatchImageActionRequest):
    image_ids = _validate_batch_ids(req.image_ids)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    if not _images_has_column(cursor, "is_deleted"):
        conn.close()
        raise HTTPException(status_code=400, detail="is_deleted column is not available")

    items = []
    success_count = 0
    failed_count = 0
    skipped_count = 0

    try:
        for image_id in image_ids:
            cursor.execute(
                "SELECT id, COALESCE(is_deleted, 0) AS is_deleted FROM images WHERE id = ?",
                (image_id,),
            )
            row = cursor.fetchone()
            if not row:
                failed_count += 1
                items.append({"image_id": image_id, "status": "failed", "detail": "Image not found"})
                continue

            if int(row["is_deleted"] or 0) == 1:
                skipped_count += 1
                failed_count += 1
                items.append({"image_id": image_id, "status": "skipped", "detail": "Image already in trash"})
                continue

            cursor.execute("UPDATE images SET is_deleted = 1 WHERE id = ?", (image_id,))
            success_count += 1
            items.append({"image_id": image_id, "status": "success"})

        conn.commit()
        return {
            "status": "success",
            "summary": {
                "requested_count": len(image_ids),
                "success_count": success_count,
                "failed_count": failed_count,
                "skipped_count": skipped_count,
            },
            "items": items,
        }
    finally:
        conn.close()

@app.post("/api/images/{image_id}/analyze")
async def analyze_image_v2(
    image_id: int,
    backend: str = CANONICAL_WCNTSEGNET,
    device: str = "cpu",
    checkpoint: str = None,
    tile_size: int = 512,
    overlap: int = 64,
    seg_threshold: float = 0.5,
):
    """
    ?????? AI ??????????????????????+ ???????????
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    try:
        results = _analyze_image_with_cursor(
            cursor,
            image_id,
            backend=backend,
            device=device,
            checkpoint=checkpoint,
            tile_size=tile_size,
            overlap=overlap,
            seg_threshold=seg_threshold,
        )
        conn.commit()
        return {"status": "success", "results": results}
    except HTTPException:
        raise
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
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    if _images_has_column(cursor, "is_deleted"):
        cursor.execute("SELECT id, COALESCE(is_deleted, 0) AS is_deleted FROM images WHERE id = ?", (image_id,))
        row = cursor.fetchone()
        if not row:
            conn.close()
            raise HTTPException(status_code=404, detail="Image not found")
        if int(row["is_deleted"] or 0) != 1:
            conn.close()
            raise HTTPException(status_code=409, detail="Image must be moved to trash before permanent deletion")

    cursor.execute("DELETE FROM images WHERE id = ?", (image_id,))
    if cursor.rowcount == 0:
        conn.close()
        raise HTTPException(status_code=404, detail="Image not found")

    conn.commit()
    conn.close()
    return {"status": "success", "deleted_id": image_id, "mode": "hard"}

@app.post("/api/images/{image_id}/features")
async def update_features(image_id: int, features: Dict[str, Any]):
    # 保持兼容性，允许手动修正数值
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # 优先获取数值型曲率，如果前端传来的是旧版标签字符串，尝试解析或设为 None
    curv = features.get('curvature_nm') or features.get('curvature')
    try:
        curv = float(curv)
    except (TypeError, ValueError):
        curv = None

    cursor.execute("""
        UPDATE images 
        SET diameter = ?, density = ?, alignment = ?, curvature = ?, tortuosity = ?, processed = 1
        WHERE id = ?
    """, (features.get('diameter'), features.get('density'), features.get('alignment'), curv, features.get('tortuosity'), image_id))
    conn.commit()
    conn.close()
    return {"status": "success"}

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


@app.get("/api/summary")
async def get_summary():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    active_clause = _active_images_clause(cursor)
    cursor.execute(f"SELECT source, COUNT(*) FROM images WHERE {active_clause} GROUP BY source")
    source_counts = dict(cursor.fetchall())
    
    cursor.execute(f"SELECT COUNT(*) FROM images WHERE {active_clause} AND processed = 1")
    processed_count = cursor.fetchone()[0]
    
    conn.close()
    return {
        "total_images": sum(source_counts.values()),
        "source_breakdown": source_counts,
        "processed_images": processed_count
    }

from backend.core.database_helpers import resolve_sort


_XR_SAMPLE_PATTERN = re.compile(r"([A-Za-z])\s*(\d+)\s*-?\s*([A-Za-z])\s*(\d+)")
_XR_TARGET_KEYS = ("diameter", "density", "alignment", "curvature", "tortuosity")
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


def _images_has_column(cursor: sqlite3.Cursor, column_name: str) -> bool:
    rows = cursor.execute("PRAGMA table_info(images)").fetchall()
    return any(row[1] == column_name for row in rows)


def _active_images_clause(cursor: sqlite3.Cursor, alias: Optional[str] = None) -> str:
    if not _images_has_column(cursor, "is_deleted"):
        return "1=1"
    prefix = f"{alias}." if alias else ""
    return f"COALESCE({prefix}is_deleted, 0) = 0"


def _deleted_images_clause(cursor: sqlite3.Cursor, alias: Optional[str] = None) -> str:
    if not _images_has_column(cursor, "is_deleted"):
        return "1=0"
    prefix = f"{alias}." if alias else ""
    return f"COALESCE({prefix}is_deleted, 0) = 1"


def _image_visibility_clause(
    cursor: sqlite3.Cursor,
    deletion_view: str = "active",
    alias: Optional[str] = None,
) -> str:
    normalized = (deletion_view or "active").strip().lower()
    if normalized == "deleted":
        return _deleted_images_clause(cursor, alias)
    if normalized == "all":
        return "1=1"
    return _active_images_clause(cursor, alias)


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
    - 输出字段：样本号/拍摄部位/拍摄编号 + 工艺变量 + 5个形貌特征(实际/预测)
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    active_clause = _active_images_clause(cursor)
    cursor.execute(
        """
        SELECT
            id, file_path, sample_id, actual_temp, growth_temp,
            ar_flow, catalyst_weight,
            diameter, density, alignment, curvature, tortuosity
        FROM images
        WHERE source = 'XR' AND """ + active_clause + """
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
            "tortuosity_actual": _safe_float(db_row["tortuosity"]),
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
    is_deleted: Optional[int] = None,
    deletion_view: str = "active",
    limit: int = 15, 
    offset: int = 0,
    sort_by: str = "id",
    order: str = "desc"
):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    effective_view = deletion_view
    if is_deleted is not None:
        effective_view = "deleted" if int(is_deleted) == 1 else "active"

    if source == "XR" and _has_xr_schema(cursor):
        _, sort_col, sort_order = resolve_sort(sort_by, order)

        where_clauses = ["i.source = 'XR'", _image_visibility_clause(cursor, effective_view, "i")]
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
                COALESCE(xi.processed, i.processed, 0) AS processed,
                COALESCE(i.is_deleted, 0) AS is_deleted
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
    where_clauses.append(_image_visibility_clause(cursor, effective_view))

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
    """?????? AIInterpreter????????"""
    if not x_api_key:
        raise HTTPException(status_code=401, detail="?? X-Api-Key ???")
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
    active_clause = _active_images_clause(cursor)
    cursor.execute(f"SELECT * FROM images WHERE id = ? AND {active_clause}", (image_id,))
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
        active_clause = _active_images_clause(conn.cursor())
        row = conn.execute(f"SELECT * FROM images WHERE id = ? AND {active_clause}", (req.image_id,)).fetchone()
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
#  QA Chat Assistant Endpoints
# ─────────────────────────────────────────────────────────────────── #

class QAChatRequest(BaseModel):
    """QA对话请求"""
    message: str
    session_id: Optional[str] = None
    rag_enabled: bool = True
    top_k: int = 5
    task_name: Optional[str] = "科研问答"


@app.post("/api/qa/chat")
async def qa_chat(
    req: QAChatRequest,
    x_provider: str = Header(default="glm"),
    x_api_key: str = Header(default=""),
    x_model: Optional[str] = Header(default=None),
    x_temperature: Optional[str] = Header(default="0.5"),
):
    """
    QA专用对话接口，自动检索RAG知识库并保存对话历史

    SSE响应格式：
    - type: "content" - 回答内容
    - type: "sources" - 引用来源
    - type: "done" - 对话完成
    """
    if not x_api_key:
        raise HTTPException(status_code=401, detail="请提供 X-Api-Key")

    # 生成或获取会话ID
    session_id = req.session_id or f"qa_{int(datetime.now().timestamp())}"

    # 保存用户消息
    qa_service.save_conversation(session_id, "user", req.message)

    # 检索RAG知识
    rag_context = None
    sources = []

    if req.rag_enabled:
        retrieval_result = rag_retriever.retrieve_for_qa(
            query=req.message,
            task_name=req.task_name,
            top_k=req.top_k,
        )
        rag_context = {
            "context_summary": retrieval_result["context_summary"],
            "retrieval_stats": retrieval_result["retrieval_stats"],
        }
        sources = retrieval_result["pdf_passages"] + retrieval_result["knowledge_links"]

    # 获取对话历史
    history = qa_service.get_conversations(session_id, limit=20)
    history_list = [
        {"role": conv["role"], "content": conv["content"]}
        for conv in history
    ]

    # 构建LLM上下文
    llm_context = None
    if rag_context:
        llm_context = {
            "rag_enabled": True,
            "context_summary": rag_context["context_summary"],
        }

    # 调用LLM
    interpreter = _get_interpreter(x_provider, x_api_key, x_model)
    temperature = float(x_temperature or 0.5)

    def event_stream():
        assistant_response = []

        def stream_with_sources():
            # 流式输出内容
            for chunk in interpreter.chat_stream(
                history=history_list,
                user_message=req.message,
                context=llm_context,
                temperature=temperature,
            ):
                data = json.loads(chunk.replace("data: ", "").replace("\n\n", ""))
                if data.get("type") == "content":
                    assistant_response.append(data.get("text", ""))
                    yield chunk
                elif data.get("type") == "done":
                    # 发送完成信号
                    yield chunk
                    break

        yield from _stream_with_error_boundary(stream_with_sources)

        # 发送引用来源
        if sources:
            sources_data = json.dumps(
                {"type": "sources", "sources": sources},
                ensure_ascii=False,
            )
            yield f"data: {sources_data}\n\n"

        # 保存助手响应
        assistant_text = "".join(assistant_response)
        if assistant_text:
            qa_service.save_conversation(
                session_id,
                "assistant",
                assistant_text,
                rag_context=rag_context,
                sources=sources,
            )

        # 返回会话ID（新会话时需要）
        yield f"data: {json.dumps({'type': 'done', 'session_id': session_id}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/api/qa/history")
async def get_qa_history(session_id: str, limit: int = 50):
    """获取对话历史"""
    conversations = qa_service.get_conversations(session_id, limit)
    return {"session_id": session_id, "conversations": conversations}


@app.delete("/api/qa/session/{session_id}")
async def clear_qa_session(session_id: str):
    """清空会话"""
    deleted_count = qa_service.clear_session(session_id)
    return {"status": "success", "deleted_count": deleted_count}


@app.get("/api/qa/templates")
async def get_qa_templates():
    """获取预设问题模板"""
    templates = qa_service.get_templates()
    return {"templates": templates}


@app.get("/api/qa/sessions")
async def list_qa_sessions():
    """获取所有会话列表"""
    sessions = qa_service.get_session_list()
    return {"sessions": sessions}


@app.post("/api/qa/export")
async def export_qa_conversation(
    session_id: str,
    format: str = "markdown",
):
    """导出对话历史"""
    try:
        content = qa_service.export_conversation(session_id, format)
        return {
            "status": "success",
            "content": content,
            "format": format,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


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


# ------------------------------------------------------------------ #
#  知识驱动预测 API
# ------------------------------------------------------------------ #

class PredictionRequest(BaseModel):
    """预测请求"""
    source: str = "ZZY"
    growth_temp: Optional[float] = None
    growth_time: Optional[float] = None
    actual_temp: Optional[float] = None
    membrane_pos_cm: Optional[float] = None
    fe_thickness: Optional[float] = None
    al2o3_thickness: Optional[float] = None
    ar_flow: Optional[float] = None
    h2_flow: Optional[float] = None
    c2h4_flow: Optional[float] = None
    anneal_temp: Optional[float] = None
    anneal_time: Optional[float] = None
    target: str = "diameter"  # diameter, density, alignment, curvature
    query: Optional[str] = None  # 用于RAG检索


@app.post("/api/predict")
async def predict_features(req: PredictionRequest):
    """
    知识驱动预测接口
    结合RAG文献、专家知识和相似实验进行预测
    """
    try:
        # 构造参数字典
        params = {
            'source': req.source,
            'growth_temp': req.growth_temp,
            'growth_time': req.growth_time,
            'actual_temp': req.actual_temp,
            'membrane_pos_cm': req.membrane_pos_cm,
            'fe_thickness': req.fe_thickness,
            'al2o3_thickness': req.al2o3_thickness,
            'ar_flow': req.ar_flow,
            'h2_flow': req.h2_flow,
            'c2h4_flow': req.c2h4_flow,
            'anneal_temp': req.anneal_temp,
            'anneal_time': req.anneal_time,
        }

        # 执行预测
        result = knowledge_predictor.predict(
            params=params,
            target=req.target,
            query=req.query,
            use_knowledge=True
        )

        return {
            "status": "success",
            "prediction": {
                "target": req.target,
                "predicted_value": result.predicted_value,
                "confidence": result.confidence,
                "knowledge_baseline": result.knowledge_baseline,
                "ml_residual": result.ml_residual,
            },
            "evidence": {
                "similar_experiments": result.similar_experiments,
                "rag_evidence": result.rag_evidence,
                "physical_constraints": result.physical_constraints,
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"预测失败: {e}")


class BatchPredictionRequest(BaseModel):
    """批量预测请求"""
    params_list: List[Dict[str, Any]]
    target: str = "diameter"
    query: Optional[str] = None


@app.post("/api/predict/batch")
async def batch_predict_features(req: BatchPredictionRequest):
    """
    批量预测接口
    """
    try:
        results = knowledge_predictor.batch_predict(
            params_list=req.params_list,
            target=req.target,
            query=req.query
        )

        return {
            "status": "success",
            "predictions": [
                {
                    "predicted_value": r.predicted_value,
                    "confidence": r.confidence,
                    "knowledge_baseline": r.knowledge_baseline,
                    "ml_residual": r.ml_residual,
                }
                for r in results
            ],
            "count": len(results)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"批量预测失败: {e}")


@app.post("/api/ml/train")
async def train_ml_models(source: Optional[str] = None):
    """
    训练机器学习模型
    """
    try:
        knowledge_predictor.train_models(source=source)
        return {"status": "success", "message": "模型训练完成"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"训练失败: {e}")


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
        "subgraph": rag_retriever.knowledge_base.retrieve_local_relation_subgraph(
            query=req.query,
            top_k=max(req.top_k, 20),
            max_hops=1,
            max_expanded_edges=60,
        ),
        "constrained_chain": rag_retriever.knowledge_base.generate_constrained_evidence_chain(
            query=req.query,
            top_k=max(3, min(req.top_k, 8)),
            min_confidence=0.45,
        ),
        "theme_aggregation": rag_retriever.knowledge_base.get_theme_level_aggregation(
            query=req.query,
            top_k=max(req.top_k, 30),
        ),
    }


# ========================================================================
# TCCER (Task-Constrained Chain Evidence Retrieval) API
# ========================================================================

class TCCERQueryRequest(BaseModel):
    query: str
    task_name: Optional[str] = None
    max_hops: int = 3
    min_confidence: float = 0.4


@app.post("/api/tccer/query")
async def tccer_query(req: TCCERQueryRequest):
    """
    TCCER约束链式证据检索

    返回主链、辅助链、冲突链等结构化证据
    """
    try:
        result = tccer_retriever.retrieve(
            query=req.query,
            task_name=req.task_name
        )
        return result.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TCCER检索失败: {str(e)}")


# ========================================================================
# MSFU (Minimal Semantic Fact Unit) API
# ========================================================================

class MSFUExtractRequest(BaseModel):
    text: str
    doc_title: str = ""
    use_llm_refinement: bool = False
    provider: Optional[str] = None
    api_key: Optional[str] = None


@app.post("/api/msfu/extract")
async def msfu_extract(req: MSFUExtractRequest):
    """
    手动触发MSFU提取

    从给定文本中提取语义事实单元
    """
    try:
        metadata = MSFUMetadata(
            doc_id="manual",
            chunk_id="manual",
            doc_title=req.doc_title or "Manual Extraction",
            doc_type="manual"
        )

        # 设置LLM客户端
        llm_client = None
        if req.use_llm_refinement and req.provider and req.api_key:
            llm_client = AIInterpreter(provider=req.provider, api_key=req.api_key)

        extractor = MSFUExtractor(
            llm_client=llm_client,
            use_llm_refinement=req.use_llm_refinement
        )

        msfus = extractor.extract(req.text, metadata, req.doc_title)

        return {
            "status": "success",
            "count": len(msfus),
            "msfus": [m.to_dict() for m in msfus]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MSFU提取失败: {str(e)}")


@app.get("/api/msfu/stats")
async def msfu_stats():
    """
    获取MSFU统计信息

    包括总数、按关系类型/方向/提取方法的分布、平均置信度等
    """
    try:
        stats = get_msfu_stats(KB_DB_PATH)
        return {"status": "success", "stats": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取MSFU统计失败: {str(e)}")


class MSFUDocExtractRequest(BaseModel):
    doc_id: int
    use_llm_refinement: bool = False
    provider: Optional[str] = None
    api_key: Optional[str] = None


@app.post("/api/msfu/reextract-doc")
async def msfu_reextract_doc(req: MSFUDocExtractRequest):
    """
    对已有文档重新提取MSFU

    清除该文档的MSFU后重新提取
    """
    try:
        conn = sqlite3.connect(KB_DB_PATH)
        cursor = conn.cursor()

        # 获取文档信息
        cursor.execute(
            "SELECT id, title, source_type FROM kb_documents WHERE id = ?",
            (req.doc_id,)
        )
        doc_row = cursor.fetchone()
        if not doc_row:
            conn.close()
            raise HTTPException(status_code=404, detail="文档不存在")

        doc_id, title, source_type = doc_row

        # 获取chunks
        cursor.execute(
            "SELECT id, text FROM kb_chunks WHERE doc_id = ? ORDER BY chunk_index",
            (doc_id,)
        )
        chunks = cursor.fetchall()

        # 删除旧MSFU
        cursor.execute("DELETE FROM kb_msfu WHERE doc_id = ?", (doc_id,))

        # 设置LLM客户端
        llm_client = None
        if req.use_llm_refinement and req.provider and req.api_key:
            llm_client = AIInterpreter(provider=req.provider, api_key=req.api_key)

        extractor = MSFUExtractor(
            llm_client=llm_client,
            use_llm_refinement=req.use_llm_refinement
        )

        # 提取新MSFU
        total_count = 0
        for chunk_id, chunk_text in chunks:
            metadata = MSFUMetadata(
                doc_id=str(doc_id),
                chunk_id=str(chunk_id),
                doc_title=title,
                doc_type=source_type
            )
            msfus = extractor.extract(chunk_text, metadata, title)
            stored_ids = store_msfus_in_db(msfus, KB_DB_PATH, doc_id, chunk_id)
            total_count += len(stored_ids)

        conn.commit()
        conn.close()

        return {
            "status": "success",
            "doc_id": doc_id,
            "doc_title": title,
            "chunks_processed": len(chunks),
            "msfu_count": total_count
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"重新提取MSFU失败: {str(e)}")


# ========================================================================
# 算法可视化API
# ========================================================================

class VisualizationResponse(BaseModel):
    backend: Optional[str] = None
    steps: Optional[List[Dict[str, Any]]] = None
    total_steps: Optional[int] = None
    phases: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None
    comparison: Optional[Dict[str, Any]] = None


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
async def visualize_image_analysis(
    image_id: int,
    backend: str = CANONICAL_WCNTSEGNET,
    device: str = "cpu",
    checkpoint: str = None,
    tile_size: int = 512,
    overlap: int = 64,
    seg_threshold: float = 0.5
):
    """获取图像分析的可视化步骤

    Args:
        backend: 分割后端，可选值 "wcntsegnet"(主传统算法) | "threshold"(兼容别名) | "cntsegnet"(深度学习) | "both"(对比查看)
        device: CNTSegNet设备，"cpu" 或 "cuda"
        checkpoint: CNTSegNet模型权重路径
        tile_size: 分块大小
        overlap: 分块重叠像素
        seg_threshold: 分割阈值
    """
    # 查询图像路径
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    active_clause = _active_images_clause(cursor)
    cursor.execute(
        f"SELECT file_path, magnification FROM images WHERE id = ? AND {active_clause}",
        (image_id,),
    )
    result = cursor.fetchone()
    conn.close()

    if not result:
        raise HTTPException(status_code=404, detail="Image not found")

    filepath = result['file_path']
    mag = result['magnification'] if result['magnification'] else 50000

    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Image file not found")

    # 读取图像
    img = _read_grayscale_image(filepath)
    if img is None:
        raise HTTPException(status_code=400, detail="Failed to read image")
    img = prepare_visualization_image(img, max_side=1280)

    # 定义阶段配置
    threshold_phases = [
        {"name": "图像预处理", "steps": [0, 1, 2, 3]},
        {"name": "基础特征", "steps": [4, 5, 6, 7]},
        {"name": "高级特征", "steps": [8, 9, 10]}
    ]

    cntsegnet_phases = [
        {"name": "模型初始化", "steps": [0, 1, 2]},
        {"name": "分块推理", "steps": [3, 4, 5]},
        {"name": "特征计算", "steps": [6, 7, 8, 9, 10]}
    ]

    try:
        normalized_backend = normalize_segmentation_backend(backend, allow_both=True)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # 根据backend参数返回不同结果
    if normalized_backend == CANONICAL_WCNTSEGNET:
        # WCNTSegNET 传统分割
        visualizer = AlgorithmVisualizer(magnification=mag)
        visualizer.visualize_extraction(img)
        return {
            "backend": CANONICAL_WCNTSEGNET,
            "steps": visualizer.get_steps(),
            "total_steps": len(visualizer.get_steps()),
            "phases": threshold_phases
        }

    elif normalized_backend == CNTSEGNET:
        # CNTSegNet深度学习分割
        try:
            from backend.core.cntsegnet_visualizer import CNTSegNetVisualizer
            visualizer = CNTSegNetVisualizer(
                magnification=mag,
                device=device,
                checkpoint_path=checkpoint,
                tile_size=tile_size,
                overlap=overlap,
                seg_threshold=seg_threshold
            )
            visualizer.visualize_extraction(img)
            return {
                "backend": "cntsegnet",
                "steps": visualizer.get_steps(),
                "total_steps": len(visualizer.get_steps()),
                "phases": cntsegnet_phases,
                "metadata": {
                    "model_info": visualizer.get_model_info(),
                    "inference_time": visualizer.get_inference_time(),
                    "tile_config": {"tile_size": tile_size, "overlap": overlap}
                }
            }
        except Exception as e:
            # CNTSegNet失败，降级到WCNTSegNET
            print(f"CNTSegNet failed, falling back to WCNTSegNET: {e}")
            visualizer = AlgorithmVisualizer(magnification=mag)
            visualizer.visualize_extraction(img)
            return {
                "backend": CANONICAL_WCNTSEGNET,
                "steps": visualizer.get_steps(),
                "total_steps": len(visualizer.get_steps()),
                "phases": threshold_phases,
                "metadata": {"fallback_reason": f"CNTSegNet failed: {str(e)}"}
            }

    elif normalized_backend == BOTH:
        # 对比查看模式
        threshold_viz = AlgorithmVisualizer(magnification=mag)
        threshold_viz.visualize_extraction(img)

        try:
            from backend.core.cntsegnet_visualizer import CNTSegNetVisualizer
            cntsegnet_viz = CNTSegNetVisualizer(
                magnification=mag,
                device=device,
                checkpoint_path=checkpoint,
                tile_size=tile_size,
                overlap=overlap,
                seg_threshold=seg_threshold
            )
            cntsegnet_viz.visualize_extraction(img)

            return {
                "backend": BOTH,
                "comparison": {
                    CANONICAL_WCNTSEGNET: {
                        "steps": threshold_viz.get_steps(),
                        "total_steps": len(threshold_viz.get_steps()),
                        "phases": threshold_phases
                    },
                    LEGACY_THRESHOLD: {
                        "steps": threshold_viz.get_steps(),
                        "total_steps": len(threshold_viz.get_steps()),
                        "phases": threshold_phases
                    },
                    CNTSEGNET: {
                        "steps": cntsegnet_viz.get_steps(),
                        "total_steps": len(cntsegnet_viz.get_steps()),
                        "phases": cntsegnet_phases
                    },
                    "metadata": {
                        "model_info": cntsegnet_viz.get_model_info(),
                        "inference_time": cntsegnet_viz.get_inference_time(),
                        "tile_config": {"tile_size": tile_size, "overlap": overlap}
                    }
                }
            }
        except Exception as e:
            # CNTSegNet失败，只返回阈值分割
            print(f"CNTSegNet failed in both mode: {e}")
            return {
                "backend": BOTH,
                "comparison": {
                    CANONICAL_WCNTSEGNET: {
                        "steps": threshold_viz.get_steps(),
                        "total_steps": len(threshold_viz.get_steps()),
                        "phases": threshold_phases
                    },
                    LEGACY_THRESHOLD: {
                        "steps": threshold_viz.get_steps(),
                        "total_steps": len(threshold_viz.get_steps()),
                        "phases": threshold_phases
                    },
                    CNTSEGNET: None,
                    "metadata": {"fallback_reason": f"CNTSegNet failed: {str(e)}"}
                }
            }

    else:
        raise HTTPException(status_code=400, detail=f"Invalid backend parameter: {backend}")


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
