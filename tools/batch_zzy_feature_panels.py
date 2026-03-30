"""ZZY 批量特征提取 + 可视化面板生成

完整管线（与 batch_processor / 网站 API 一致）：
  1. CLDice 深度学习分割 → binary mask
  2. mask 传入 extract_all → 骨架化
  3. 分支剪枝 / 拓扑清理 (_clean_branch_skeleton)
  4. V3 特征提取（alignment, diameter, curvature V3 多统计量, waviness）

从数据库读取未被逻辑删除的 ZZY 图像，生成「左图右数据」面板。

输出:
  reports/zzy_feature_panels_{timestamp}/
    ├── {stem}__panel.png          可视化面板
    ├── {stem}__features.json      完整特征 JSON
    ├── summary.csv                汇总表
    └── summary.json               汇总 JSON
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.feature_extractor import FeatureExtractor  # noqa: E402
from backend.core.batch_processor import (  # noqa: E402
    _CLDiceSegmenter,
    _get_cldice_segmenter,
    _make_feature_extractor,
)

DB_PATH = PROJECT_ROOT / "database" / "cnta_experiments.sqlite"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "reports"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ZZY 批量特征可视化面板（clDice 分割 + V3 特征）")
    p.add_argument("--speed-profile", default="accurate", choices=["accurate", "fast"])
    p.add_argument("--diameter-method", default="enhanced", choices=["standard", "enhanced"])
    p.add_argument("--device", default="cuda", help="clDice 推理设备: cuda / cpu")
    p.add_argument("--limit", type=int, default=0, help="限制处理数量, 0=全部")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--resume", action="store_true", help="跳过已存在的面板")
    p.add_argument("--only-unprocessed", action="store_true", help="只处理未提取特征的图像")
    p.add_argument("--min-mag", type=int, default=0, help="最低倍率过滤（如 50000）")
    return p.parse_args()


# ── 数据库 ──────────────────────────────────────────────────────────────────

def _images_has_column(cursor, col: str) -> bool:
    cursor.execute("PRAGMA table_info(images)")
    return any(row[1] == col for row in cursor.fetchall())


def fetch_zzy_images(cursor, only_unprocessed: bool = False,
                     limit: int = 0, min_mag: int = 0) -> List[dict]:
    where_parts = ["source = 'ZZY'"]
    params: list = []
    if _images_has_column(cursor, "is_deleted"):
        where_parts.append("COALESCE(is_deleted, 0) = 0")
    if only_unprocessed:
        where_parts.append("COALESCE(processed, 0) = 0")
    if min_mag > 0:
        where_parts.append("COALESCE(magnification, 0) >= ?")
        params.append(min_mag)
    where_sql = "WHERE " + " AND ".join(where_parts)
    limit_sql = f"LIMIT {limit}" if limit > 0 else ""
    cursor.execute(
        f"SELECT id, file_path, magnification FROM images {where_sql} {limit_sql}",
        params,
    )
    return [dict(row) for row in cursor.fetchall()]


# ── 图像读写 ────────────────────────────────────────────────────────────────

def read_gray(path: Path) -> Optional[np.ndarray]:
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)


# ── 面板渲染 ────────────────────────────────────────────────────────────────

def fmt(v: Any, d: int = 4) -> str:
    if v is None:
        return "N/A"
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.{d}f}"
    return str(v)


def render_panel(image_gray: np.ndarray, features: dict,
                 file_name: str, output_path: Path) -> None:
    """左图右数据面板，V3 曲率多统计量"""
    h, w = image_gray.shape
    panel_w = max(1200, w + 700)
    panel_h = max(h + 140, 1050)
    panel = np.full((panel_h, panel_w, 3), 248, dtype=np.uint8)

    # 标题
    cv2.putText(panel, "ZZY Feature Panel  |  clDice + V3 Curvature",
                (28, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(panel, file_name[:130], (28, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (80, 80, 80), 1, cv2.LINE_AA)

    # 左侧原图
    img_bgr = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
    y0 = 100
    panel[y0:y0 + h, 28:28 + w] = img_bgr
    cv2.rectangle(panel, (27, y0 - 1), (28 + w, y0 + h), (170, 170, 170), 1)

    # 右侧特征文本
    x0 = 28 + w + 30
    y_start = y0 + 20
    gap = 33

    lines: list[str] = []

    lines.append("=== Basic Features ===")
    lines.append(f"density (%)          {fmt(features.get('density'), 2)}")
    lines.append(f"diameter (nm)        {fmt(features.get('diameter'), 2)}")
    lines.append(f"alignment (HOF)      {fmt(features.get('alignment'), 4)}")
    lines.append(f"mean_phi_deg         {fmt(features.get('mean_phi_deg'), 2)}")
    lines.append(f"px_per_um            {fmt(features.get('px_per_um'), 2)}")
    lines.append(f"n_branches           {features.get('n_branches', 'N/A')}")
    lines.append(f"hof_method           {features.get('hof_method', 'N/A')}")
    lines.append(f"diameter_method      {features.get('diameter_method', 'N/A')}")
    lines.append("")

    lines.append("=== V3 Curvature ===")
    lines.append(f"label                {features.get('curvature_v3', 'N/A')}")
    lines.append(f"curvature_nm_v3      {fmt(features.get('curvature_nm_v3'), 6)}")
    lines.append(f"branch_count         {features.get('curvature_v3_branch_count', 'N/A')}")
    lines.append("")

    lines.append("=== V3 Curvature Statistics ===")
    lines.append(f"p50_length           {fmt(features.get('curvature_nm_v3_p50_length'), 6)}")
    lines.append(f"p50_sqrt_length      {fmt(features.get('curvature_nm_v3_p50_sqrt_length'), 6)}")
    lines.append(f"p75_length           {fmt(features.get('curvature_nm_v3_p75_length'), 6)}")
    lines.append(f"p75_sqrt_length      {fmt(features.get('curvature_nm_v3_p75_sqrt_length'), 6)}")
    lines.append(f"mean_length          {fmt(features.get('curvature_nm_v3_mean_length'), 6)}")
    lines.append(f"mean_sqrt_length     {fmt(features.get('curvature_nm_v3_mean_sqrt_length'), 6)}")
    lines.append(f"trimmed_mean_length  {fmt(features.get('curvature_nm_v3_trimmed_mean_length'), 6)}")
    lines.append(f"trimmed_mean_sqrt    {fmt(features.get('curvature_nm_v3_trimmed_mean_sqrt_length'), 6)}")
    lines.append("")

    lines.append("=== Waviness / Tortuosity ===")
    lines.append(f"waviness_ratio_v2    {fmt(features.get('waviness_ratio_v2'), 4)}")
    lines.append(f"height_nm (v2)       {fmt(features.get('waviness_height_nm_v2'), 2)}")
    lines.append(f"wavelength_nm (v2)   {fmt(features.get('waviness_wavelength_nm_v2'), 2)}")
    lines.append(f"waviness_branches    {features.get('waviness_branches_v2', 'N/A')}")
    lines.append(f"tortuosity_v2        {fmt(features.get('tortuosity_v2'), 3)}")

    # 骨架清理信息
    lines.append("")
    lines.append("=== Branch Cleanup ===")
    lines.append(f"removed_components   {features.get('removed_short_component_count', 'N/A')}")
    lines.append(f"removed_spurs        {features.get('removed_spur_count', 'N/A')}")

    for i, line in enumerate(lines):
        cv2.putText(panel, line, (x0, y_start + i * gap),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.58, (28, 28, 28), 1, cv2.LINE_AA)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    enc = cv2.imencode(".png", panel)[1]
    output_path.write_bytes(enc.tobytes())


# ── 主流程 ──────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    if args.output_dir:
        out_dir = args.output_dir
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = DEFAULT_OUTPUT_ROOT / f"zzy_feature_panels_cldice_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 数据库
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    rows = fetch_zzy_images(cursor, only_unprocessed=args.only_unprocessed,
                            limit=args.limit, min_mag=args.min_mag)
    conn.close()

    if not rows:
        print("没有找到符合条件的 ZZY 图像")
        return

    print(f"待处理: {len(rows)} 张图像 (min_mag={args.min_mag}, device={args.device})")

    # 加载 CLDice 分割模型
    print("加载 CLDice 分割模型...")
    segmenter = _get_cldice_segmenter(device=args.device)
    print("CLDice 模型加载完成")

    summary_rows: List[dict] = []
    t0 = time.perf_counter()
    success = 0
    skip = 0
    error = 0

    for idx, row in enumerate(rows, 1):
        file_path = Path(row["file_path"])
        file_name = file_path.name
        mag = row["magnification"]
        stem = file_path.stem.replace(" ", "_")

        panel_path = out_dir / f"{stem}__panel.png"
        json_path = out_dir / f"{stem}__features.json"

        print(f"[{idx}/{len(rows)}] {file_name}", end="", flush=True)

        # resume
        if args.resume and panel_path.exists() and json_path.exists():
            features = json.loads(json_path.read_text(encoding="utf-8"))
            print("  [SKIP-resume]")
            summary_rows.append({"file_name": file_name, "status": "resume", **features})
            skip += 1
            continue

        if not file_path.exists():
            print(f"  [SKIP-file missing]")
            skip += 1
            continue

        img_gray = read_gray(file_path)
        if img_gray is None:
            print(f"  [ERROR-read]")
            error += 1
            continue

        # ── 完整管线：clDice → 骨架剪枝 → 特征提取 ──
        try:
            extractor = _make_feature_extractor(
                magnification=int(mag) if mag else None,
                diameter_method=args.diameter_method,
            )
            roi = extractor.extract_roi(img_gray)
            mask = segmenter.predict_mask(roi)
            features = extractor.extract_all(img_gray, external_binary_mask=mask)
        except Exception as e:
            print(f"  [ERROR-{e}]")
            error += 1
            continue

        # 渲染 + 保存
        try:
            render_panel(img_gray, features, file_name, panel_path)
            json_path.write_text(
                json.dumps(features, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
        except Exception as e:
            print(f"  [ERROR-render-{e}]")
            error += 1
            continue

        summary_rows.append({"file_name": file_name, "status": "ok", **features})
        elapsed = time.perf_counter() - t0
        avg = elapsed / idx
        eta = avg * (len(rows) - idx)
        print(f"  [OK] {elapsed:.1f}s  ETA {eta:.0f}s")
        success += 1

    # ── 汇总 ──
    total_elapsed = time.perf_counter() - t0
    print(f"\n完成: {success} 成功, {skip} 跳过, {error} 错误, 耗时 {total_elapsed:.1f}s")
    print(f"输出目录: {out_dir}")

    if summary_rows:
        fieldnames = list(summary_rows[0].keys())
        csv_path = out_dir / "summary.csv"
        with csv_path.open("w", encoding="utf-8-sig", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)

    summary_payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "pipeline": "clDice_segmentation → skeleton_branch_cleanup → feature_extraction",
        "speed_profile": args.speed_profile,
        "diameter_method": args.diameter_method,
        "device": args.device,
        "total": len(rows),
        "success": success,
        "skip": skip,
        "error": error,
        "elapsed_s": round(total_elapsed, 1),
        "rows": summary_rows,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
