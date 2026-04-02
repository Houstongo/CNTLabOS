"""
批量特征提取脚本  v2.0
========================
调用 src/analysis/feature_extractor.py（v2.0）对数据库中未处理的图像
逐一提取四特征，并将结果写回数据库。

运行方式：
    cd d:\\CNTDATA\\CNTA_ML_Project
    python backend/core/batch_processor.py

可选参数：
    --reprocess   重新处理已处理过的图像（processed=1 的也会被重跑）
    --limit N     只处理前 N 张（用于测试）
    --source ZZY  只处理指定来源
"""

import os
import sys
import sqlite3
import argparse
import cv2
import time
import traceback
import multiprocessing as mp
import queue
import json
import tempfile
import numpy as np

# 确保可以 import src.analysis.feature_extractor
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.analysis.feature_extractor import FeatureExtractor
from experiments.cnt_paper_repro.config import load_config
from experiments.cnt_paper_repro.model import ResNet34UNet
from experiments.cnt_paper_repro.patching import extract_patch, extract_patch_specs

DB_PATH = r'd:\CNTDATA\CNTA_ML_Project\database\cnta_experiments.sqlite'

CLDICE_CONFIG_PATH = os.path.join(
    PROJECT_ROOT,
    'experiments', 'cnt_paper_repro', 'runs',
    'cnt_paper_repro_100000x_center768_cldice_seed42',
    'config_snapshot.yaml',
)
CLDICE_CHECKPOINT_PATH = os.path.join(
    PROJECT_ROOT,
    'experiments', 'cnt_paper_repro', 'runs',
    'cnt_paper_repro_100000x_center768_cldice_seed42',
    'best_model.pth',
)

_CLDICE_RUNTIME = {
    "segmenter": None,
    "config": None,
}


class _CLDiceSegmenter:
    """CNTSegNet-clDice (ResNet34UNet) 分割器 — 1ch 灰度输入。"""

    def __init__(self, config_path: str, checkpoint_path: str, device: str):
        try:
            import torch
        except Exception as exc:
            raise RuntimeError(f"Failed to import Torch: {exc}") from exc

        self._torch = torch
        self.device = torch.device(device)

        if self.device.type.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA device but torch.cuda.is_available() is False")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"clDice checkpoint not found: {checkpoint_path}")

        cfg = load_config(config_path)
        self._normalize_mean = float(cfg["data"].get("normalize_mean", 0.5))
        self._normalize_std = float(cfg["data"].get("normalize_std", 0.5))
        self._patch_size = int(cfg["data"].get("patch_size", 768))
        self._threshold = float(cfg.get("inference", {}).get("threshold", 0.7))
        # stride = patch_size // 2 以保证重叠覆盖
        self._stride = self._patch_size // 2

        model = ResNet34UNet(
            in_channels=int(cfg["model"].get("in_channels", 1)),
            num_classes=int(cfg["model"].get("num_classes", 1)),
            encoder_weights=None,
        ).to(self.device)

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        model.eval()
        self.model = model

    def predict_mask(self, roi_gray: np.ndarray) -> np.ndarray:
        if roi_gray is None or roi_gray.size == 0:
            raise ValueError("Empty ROI for clDice segmentation")

        torch = self._torch
        h, w = roi_gray.shape[:2]
        specs = extract_patch_specs(roi_gray, patch_size=self._patch_size, mode="grid", stride=self._stride)
        accum = np.zeros((h, w), dtype=np.float32)
        counts = np.zeros((h, w), dtype=np.float32)
        batch_size = 8 if self.device.type == "cuda" else 2

        with torch.no_grad():
            for start in range(0, len(specs), batch_size):
                batch_specs = specs[start:start + batch_size]
                batch_patches = []
                for spec in batch_specs:
                    patch = extract_patch(roi_gray, spec).astype(np.float32) / 255.0
                    batch_patches.append(
                        ((patch - self._normalize_mean) / max(self._normalize_std, 1e-6)).astype(np.float32)
                    )

                tensor = torch.from_numpy(np.stack(batch_patches, axis=0)).unsqueeze(1).to(self.device)
                probs = torch.sigmoid(self.model(tensor)).detach().cpu().numpy()[:, 0]

                for spec, prob in zip(batch_specs, probs):
                    valid_prob = prob[:spec.height, :spec.width]
                    accum[spec.top:spec.top + spec.height, spec.left:spec.left + spec.width] += valid_prob
                    counts[spec.top:spec.top + spec.height, spec.left:spec.left + spec.width] += 1.0

        prob_map = accum / np.maximum(counts, 1.0)
        return (prob_map >= self._threshold).astype(np.uint8)


def _get_cldice_segmenter(device: str):
    config = (str(device),)
    if _CLDICE_RUNTIME["segmenter"] is None or _CLDICE_RUNTIME["config"] != config:
        _CLDICE_RUNTIME["segmenter"] = _CLDiceSegmenter(
            config_path=CLDICE_CONFIG_PATH,
            checkpoint_path=CLDICE_CHECKPOINT_PATH,
            device=device,
        )
        _CLDICE_RUNTIME["config"] = config
    return _CLDICE_RUNTIME["segmenter"]


def _images_has_column(cursor: sqlite3.Cursor, column_name: str) -> bool:
    rows = cursor.execute("PRAGMA table_info(images)").fetchall()
    return any(row[1] == column_name for row in rows)


def _make_feature_extractor(magnification=None, diameter_method: str = "standard"):
    try:
        return FeatureExtractor(magnification=magnification, diameter_method=diameter_method)
    except TypeError:
        # Keep compatibility with older test doubles / helper scripts.
        return FeatureExtractor(magnification=magnification)


def _resolve_diameter_method(source: str, magnification, requested_method: str):
    method = requested_method or "standard"
    mag_value = int(magnification) if magnification else None

    if method == "enhanced" and source == "XR" and mag_value is not None and mag_value <= 20_000:
        return "standard", "xr_low_mag_guard"

    return method, None


def _extract_image_features(
    file_path: str,
    magnification,
    diameter_method: str = "standard",
    progress_callback=None,
    device: str = "cpu",
):
    # 使用 np.fromfile + cv2.imdecode 支持中文路径
    try:
        img_array = np.fromfile(file_path, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
    except Exception:
        img = None
    if img is None:
        raise ValueError(f"imread 失败: {os.path.basename(file_path)}")

    extractor = _make_feature_extractor(
        magnification=int(magnification) if magnification else None,
        diameter_method=diameter_method,
    )

    roi = extractor.extract_roi(img)
    segmenter = _get_cldice_segmenter(device=device)
    mask = segmenter.predict_mask(roi)
    if progress_callback is not None:
        progress_callback(
            "segmentation",
            0.0,
            {
                "backend": "cldice",
                "device": device,
                "fg_ratio": round(float(mask.mean()), 4),
            },
        )
    try:
        return extractor.extract_all(
            img,
            progress_callback=progress_callback,
            external_binary_mask=mask,
        )
    except TypeError:
        return extractor.extract_all(img, progress_callback=progress_callback)


def _extract_worker(
    file_path: str,
    magnification,
    diameter_method: str,
    message_queue,
    device: str,
):
    def emit_progress(step_name: str, elapsed_s: float, payload: dict):
        message_queue.put({
            "type": "progress",
            "step": step_name,
            "elapsed_s": elapsed_s,
            "payload": payload or {},
        })

    try:
        result = _extract_image_features(
            file_path=file_path,
            magnification=magnification,
            diameter_method=diameter_method,
            progress_callback=emit_progress,
            device=device,
        )
        message_queue.put({
            "type": "result",
            "result": result,
        })
    except Exception as exc:
        message_queue.put({
            "type": "error",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        })


def _extract_worker_fileipc(
    file_path: str,
    magnification,
    diameter_method: str,
    result_file: str,
    device: str,
):
    payload = None
    try:
        result = _extract_image_features(
            file_path=file_path,
            magnification=magnification,
            diameter_method=diameter_method,
            progress_callback=None,
            device=device,
        )
        payload = {
            "type": "result",
            "result": result,
        }
    except Exception as exc:
        payload = {
            "type": "error",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }

    try:
        with open(result_file, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, ensure_ascii=False)
    except Exception:
        pass


def _format_step_payload(payload: dict) -> str:
    if not payload:
        return ""

    parts = []
    for key, value in payload.items():
        if value is None:
            continue
        parts.append(f"{key}={value}")
    return " ".join(parts)


def _run_with_timeout(
    file_path: str,
    magnification,
    diameter_method: str,
    timeout_s: float,
    prefix: str,
    log_steps: bool,
    device: str,
):
    ctx = mp.get_context("spawn")
    started_at = time.perf_counter()
    last_step = "spawn"
    final_message = None

    try:
        result_queue = ctx.Queue()
        process = ctx.Process(
            target=_extract_worker,
            args=(
                file_path,
                magnification,
                diameter_method,
                result_queue,
                device,
            ),
        )
        process.start()
        use_queue = True
    except PermissionError:
        use_queue = False
        tmp_dir = tempfile.mkdtemp(prefix="cnta_timeout_")
        result_file = os.path.join(tmp_dir, "result.json")
        process = ctx.Process(
            target=_extract_worker_fileipc,
            args=(
                file_path,
                magnification,
                diameter_method,
                result_file,
                device,
            ),
        )
        process.start()

    if use_queue:
        while True:
            while True:
                try:
                    message = result_queue.get_nowait()
                except queue.Empty:
                    break

                if message.get("type") == "progress":
                    last_step = message.get("step") or last_step
                    if log_steps:
                        payload_str = _format_step_payload(message.get("payload") or {})
                        payload_suffix = f" {payload_str}" if payload_str else ""
                        print(
                            f"{prefix} STEP {last_step:<10} t={message.get('elapsed_s', 0):7.2f}s{payload_suffix}",
                            flush=True,
                        )
                else:
                    final_message = message

            if final_message is not None:
                process.join(timeout=1)
                final_message["elapsed_s"] = round(time.perf_counter() - started_at, 3)
                final_message["last_step"] = last_step
                return final_message

            process.join(timeout=0.2)
            elapsed_s = time.perf_counter() - started_at

            if timeout_s is not None and elapsed_s > timeout_s:
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)

                return {
                    "type": "timeout",
                    "elapsed_s": round(elapsed_s, 3),
                    "last_step": last_step,
                }

            if not process.is_alive():
                while True:
                    try:
                        message = result_queue.get_nowait()
                    except queue.Empty:
                        break

                    if message.get("type") == "progress":
                        last_step = message.get("step") or last_step
                        if log_steps:
                            payload_str = _format_step_payload(message.get("payload") or {})
                            payload_suffix = f" {payload_str}" if payload_str else ""
                            print(
                                f"{prefix} STEP {last_step:<10} t={message.get('elapsed_s', 0):7.2f}s{payload_suffix}",
                                flush=True,
                            )
                    else:
                        final_message = message

                if final_message is not None:
                    final_message["elapsed_s"] = round(time.perf_counter() - started_at, 3)
                    final_message["last_step"] = last_step
                    return final_message

                return {
                    "type": "error",
                    "error": f"worker exited unexpectedly with code {process.exitcode}",
                    "elapsed_s": round(elapsed_s, 3),
                    "last_step": last_step,
                }
    else:
        while True:
            process.join(timeout=0.2)
            elapsed_s = time.perf_counter() - started_at

            if timeout_s is not None and elapsed_s > timeout_s:
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)
                return {
                    "type": "timeout",
                    "elapsed_s": round(elapsed_s, 3),
                    "last_step": "worker_fileipc",
                }

            if process.is_alive():
                continue

            # Worker finished (or crashed): collect final payload from file.
            while True:
                if os.path.exists(result_file):
                    break
                # A tiny wait for potential late file flush.
                if time.perf_counter() - started_at > elapsed_s + 0.5:
                    break
                time.sleep(0.01)

            if os.path.exists(result_file):
                try:
                    with open(result_file, "r", encoding="utf-8") as fp:
                        final_message = json.load(fp)
                except Exception as exc:
                    final_message = {
                        "type": "error",
                        "error": f"failed to parse worker result file: {exc}",
                    }
            else:
                final_message = {
                    "type": "error",
                    "error": f"worker exited with code {process.exitcode} and no result file",
                }

            final_message["elapsed_s"] = round(elapsed_s, 3)
            final_message["last_step"] = "worker_fileipc"
            return final_message


def batch_process(
    reprocess: bool = False,
    limit: int = None,
    source: str = None,
    diameter_method: str = "standard",
    per_image_timeout: float = None,
    log_steps: bool = False,
    device: str = "cpu",
):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    optional_columns = {
        name for name in (
            "tortuosity",
            "waviness_ratio",
            "waviness_height_nm",
            "waviness_wavelength_nm",
            "waviness_branches",
        )
        if _images_has_column(cursor, name)
    }

    # 构建查询条件
    where_clauses = []
    params = []

    if _images_has_column(cursor, "is_deleted"):
        where_clauses.append("COALESCE(is_deleted, 0) = 0")

    # --only-incomplete: 只处理未完成的数据
    # 未完成定义: processed=0 或 curvature为空/字符串 或 waviness_ratio为空
    has_waviness = "waviness_ratio" in optional_columns
    if reprocess and has_waviness:
        # 选择需要重处理的不完整数据
        where_clauses.append(
            "(COALESCE(processed, 0) = 0 "
            "OR curvature IS NULL "
            "OR typeof(curvature) = 'text' "
            f"OR waviness_ratio IS NULL)"
        )
    elif not reprocess:
        where_clauses.append("COALESCE(processed, 0) = 0")

    if source:
        where_clauses.append("source = ?")
        params.append(source)

    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
    limit_sql = f"LIMIT {limit}" if limit else ""

    cursor.execute(
        f"SELECT id, file_path, source, magnification FROM images {where_sql} {limit_sql}",
        params
    )
    rows = cursor.fetchall()
    total = len(rows)
    print(f"待处理图像: {total} 张")

    success = 0
    skipped = 0
    errors = 0
    timeouts = 0

    print(
        f"批处理配置: source={source or 'ALL'} "
        f"diameter_method={diameter_method} "
        f"segmentation_backend=cldice "
        f"device={device} "
        f"per_image_timeout={per_image_timeout if per_image_timeout is not None else 'OFF'} "
        f"log_steps={log_steps}",
        flush=True,
    )

    for i, row in enumerate(rows, 1):
        img_id   = row['id']
        path     = row['file_path']
        mag      = row['magnification']

        prefix = f"[{i:4d}/{total}]"
        print(
            f"{prefix} START id={img_id} mag={mag or 'N/A'} file={os.path.basename(path)}",
            flush=True,
        )

        if not os.path.exists(path):
            print(f"{prefix} SKIP (文件不存在): {os.path.basename(path)}")
            skipped += 1
            continue

        try:
            effective_diameter_method, fallback_reason = _resolve_diameter_method(
                source=row["source"],
                magnification=mag,
                requested_method=diameter_method,
            )
            if fallback_reason:
                print(
                    f"{prefix} FALLBACK diameter_method={effective_diameter_method} "
                    f"reason={fallback_reason}",
                    flush=True,
                )

            if per_image_timeout is not None:
                outcome = _run_with_timeout(
                    file_path=path,
                    magnification=mag,
                    diameter_method=effective_diameter_method,
                    timeout_s=per_image_timeout,
                    prefix=prefix,
                    log_steps=log_steps,
                    device=device,
                )
                if outcome["type"] == "timeout":
                    timeouts += 1
                    print(
                        f"{prefix} TIMEOUT after {outcome['elapsed_s']:.2f}s "
                        f"last_step={outcome.get('last_step', 'unknown')} "
                        f"| {os.path.basename(path)}",
                        flush=True,
                    )
                    continue
                if outcome["type"] == "error":
                    errors += 1
                    print(
                        f"{prefix} ERROR after {outcome.get('elapsed_s', 0):.2f}s "
                        f"last_step={outcome.get('last_step', 'unknown')} "
                        f"| {os.path.basename(path)} -> {outcome.get('error')}",
                        flush=True,
                    )
                    if outcome.get("traceback"):
                        print(outcome["traceback"], flush=True)
                    continue
                res = outcome["result"]
                elapsed_s = outcome["elapsed_s"]
            else:
                def emit_progress(step_name: str, elapsed_s: float, payload: dict):
                    if not log_steps:
                        return
                    payload_str = _format_step_payload(payload or {})
                    payload_suffix = f" {payload_str}" if payload_str else ""
                    print(
                        f"{prefix} STEP {step_name:<10} t={elapsed_s:7.2f}s{payload_suffix}",
                        flush=True,
                    )

                started_at = time.perf_counter()
                res = _extract_image_features(
                    file_path=path,
                    magnification=mag,
                    diameter_method=effective_diameter_method,
                    progress_callback=emit_progress if log_steps else None,
                    device=device,
                )
                elapsed_s = round(time.perf_counter() - started_at, 3)

            curvature_um = None
            if res.get("curvature_nm") is not None:
                curvature_um = float(res.get("curvature_nm")) * 1000.0

            update_values = {
                "diameter": res["diameter"],
                "density": res["density"],
                "alignment": res["alignment"],
                "curvature": curvature_um,  # 使用数值型曲率（um^-1）
                "processed": 1,
            }
            for column in optional_columns:
                update_values[column] = res.get(column)

            assignments = ", ".join(f"{column}=?" for column in update_values)
            cursor.execute(
                f"UPDATE images SET {assignments} WHERE id=?",
                tuple(update_values.values()) + (img_id,),
            )
            conn.commit()

            diam_str = f"{res['diameter']:.1f}nm" if res['diameter'] is not None else "N/A"
            print(
                f"{prefix} OK  density={res['density']:.1f}%  "
                f"S={res['alignment']:.3f}  diameter={diam_str}  "
                f"curv={res['curvature']}  waviness={res.get('waviness_ratio')}  "
                f"t={elapsed_s:.2f}s  | {os.path.basename(path)}",
                flush=True,
            )
            success += 1

        except Exception as e:
            print(f"{prefix} ERROR: {os.path.basename(path)} → {e}", flush=True)
            errors += 1

    conn.close()
    print(
        f"\n完成：成功 {success}，跳过 {skipped}，超时 {timeouts}，错误 {errors}",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量提取 CNT 图像特征")
    parser.add_argument("--reprocess", action="store_true", help="重新处理已处理过的图像")
    parser.add_argument("--limit",     type=int, default=None, help="只处理前 N 张")
    parser.add_argument("--source",    type=str, default=None, help="只处理指定来源 (ZZY/XR)")
    parser.add_argument(
        "--diameter-method",
        type=str,
        choices=["standard", "enhanced"],
        default="standard",
        help="管径计算方法",
    )
    parser.add_argument(
        "--per-image-timeout",
        type=float,
        default=None,
        help="单张图像最大处理秒数；超时则终止该图并继续下一张",
    )
    parser.add_argument(
        "--step-timings",
        action="store_true",
        help="打印每张图的分步骤耗时日志",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="推理设备: cpu 或 cuda",
    )
    args = parser.parse_args()

    batch_process(
        reprocess=args.reprocess,
        limit=args.limit,
        source=args.source,
        diameter_method=args.diameter_method,
        per_image_timeout=args.per_image_timeout,
        log_steps=args.step_timings,
        device=args.device,
    )
