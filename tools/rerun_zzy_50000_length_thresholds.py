from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.core.batch_processor import _get_cldice_segmenter  # noqa: E402
from src.analysis.feature_extractor import FeatureExtractor  # noqa: E402
from tools.generate_xr_slice_standard_batch import (  # noqa: E402
    aggregate_cached_waviness,
    cache_branch_measurements,
    cache_branch_waviness_metrics,
    classify_v3_curvature,
    compute_junction_metrics,
    sample_branch_diameters_nm,
    sample_branch_curvatures_nm,
    summarize_values,
)


DB_PATH = PROJECT_ROOT / "database" / "cnta_experiments.sqlite"
OUTPUT_ROOT = PROJECT_ROOT / "reports"
THRESHOLDS = [("L0", 1.0), ("L1", 3.0), ("L2", 5.0), ("L3", 7.0), ("L4", 9.0)]
CURVATURE_STATS = ("p50", "p70", "p75", "mean", "trimmed_mean")
WEIGHT_MODES = ("uniform", "sqrt_length", "length")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rerun ZZY 50000x active rows with L0-L4 length thresholds.")
    parser.add_argument("--device", default="cpu", help="cpu/cuda")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_gray(path: Path) -> np.ndarray:
    arr = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    return img


def infer_sample_prefix(file_path: str, sample_id: str | None) -> str:
    stem = Path(file_path).stem
    match = re.search(r"(No\d+)", stem)
    if match:
        return match.group(1)
    if sample_id:
        match = re.search(r"(No\d+)", sample_id)
        if match:
            return match.group(1)
    return "UNKNOWN"


def infer_group_key(file_path: str) -> str:
    stem = Path(file_path).stem.strip()
    stem = re.sub(r"\s+", " ", stem)
    stem = re.sub(r"( 50000(?: \d+)?)\s*-\d+$", r"\1", stem)
    stem = re.sub(r"([+-]?\d+(?:\.\d+)?)-\d+$", r"\1", stem)
    stem = re.sub(r"\b(top|mid|bottom)(\d+)$", r"\1", stem, flags=re.IGNORECASE)
    return stem


def load_active_rows(limit: int = 0) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    query = """
    SELECT
        id AS image_id,
        sample_id,
        file_path,
        magnification,
        al2o3_power,
        al2o3_thickness,
        fe_power,
        fe_thickness,
        ar_flow,
        h2_flow,
        c2h4_flow,
        anneal_temp,
        growth_temp,
        anneal_time,
        growth_time
    FROM images
    WHERE source='ZZY'
      AND COALESCE(is_deleted, 0)=0
      AND magnification=50000
      AND al2o3_power = 200
      AND al2o3_thickness = 5.0
      AND ar_flow = 600
      AND h2_flow = 300
      AND c2h4_flow = 150
      AND anneal_temp = 600
      AND growth_temp = 750
      AND growth_time = 3.0
      AND fe_power IS NOT NULL
      AND fe_thickness IS NOT NULL
      AND anneal_time IS NOT NULL
    ORDER BY id
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    if limit > 0:
        df = df.head(int(limit)).copy()

    numeric_cols = [
        "magnification",
        "al2o3_power",
        "al2o3_thickness",
        "fe_power",
        "fe_thickness",
        "ar_flow",
        "h2_flow",
        "c2h4_flow",
        "anneal_temp",
        "growth_temp",
        "anneal_time",
        "growth_time",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["sample_prefix"] = [
        infer_sample_prefix(path, sample_id)
        for path, sample_id in zip(df["file_path"], df["sample_id"])
    ]
    df["group_key"] = [infer_group_key(path) for path in df["file_path"]]
    df["fe_deposition_index"] = df["fe_power"] * df["fe_thickness"]
    return df


def aggregate_branch_curvature_nm(
    extractor: FeatureExtractor,
    branches: List[Dict[str, Any]],
    branch_stat: str,
    weight_mode: str,
) -> float:
    branch_curvatures = []
    weights = []
    for branch in branches:
        stats = branch.get("_curvature_stats_nm") or {}
        key = "median" if branch_stat == "p50" else branch_stat
        branch_curvature_nm = float(stats.get(key, 0.0))
        if branch_curvature_nm <= 0:
            continue
        branch_curvatures.append(branch_curvature_nm)
        path_length_px = float(branch.get("path_length_px", 0.0))
        if weight_mode == "uniform":
            weights.append(1.0)
        elif weight_mode == "sqrt_length":
            weights.append(np.sqrt(max(path_length_px, 1.0)))
        elif weight_mode == "length":
            weights.append(max(path_length_px, 1.0))
        else:
            raise ValueError(f"Unsupported weight_mode: {weight_mode}")
    if not branch_curvatures:
        return 0.0
    return float(np.average(np.asarray(branch_curvatures, dtype=float), weights=np.asarray(weights, dtype=float)))


def analyze_thresholds(mask: np.ndarray, magnification: int) -> Dict[str, Any]:
    extractor = FeatureExtractor(magnification=int(magnification), speed_profile="accurate")
    extractor._calibrate(mask.shape[1])
    density = float(np.count_nonzero(mask) / max(mask.size, 1) * 100.0)
    diameter_nm, skeleton = extractor.calculate_diameter(mask)
    distance_map = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    base_components = extractor._collect_components(skeleton)
    processed = extractor.preprocess(mask * 255)
    alignment_metrics = extractor.calculate_hof_skeleton_adaptive(
        skeleton,
        processed=processed,
        base_components=base_components,
    )
    junction_metrics = compute_junction_metrics(skeleton, extractor.px_per_um)

    base_min_points = max(extractor.V3_MIN_BRANCH_POINTS, int(round(extractor.expected_tube_px * 1.5)))
    relaxed_branches = extractor._collect_ordered_branches_v2(
        skeleton,
        min_points=base_min_points,
        min_length_factor=min(factor for _, factor in THRESHOLDS),
    )
    cache_branch_measurements(extractor, relaxed_branches, distance_map)
    cache_branch_waviness_metrics(extractor, relaxed_branches)

    global_diameters_nm = sample_branch_diameters_nm(extractor, distance_map, relaxed_branches)
    diameter_stats_nm = summarize_values(global_diameters_nm)

    thresholds: Dict[str, Any] = {}
    for label, factor in THRESHOLDS:
        branches = extractor._filter_ordered_branches(
            relaxed_branches,
            min_points=base_min_points,
            min_length_factor=factor,
        )
        curvature_distribution_nm = sample_branch_curvatures_nm(extractor, branches)
        diameter_distribution_nm = sample_branch_diameters_nm(extractor, distance_map, branches)
        waviness_v2 = aggregate_cached_waviness(branches)
        record = {
            "min_length_factor": factor,
            "branch_count": len(branches),
            "curvature_label": classify_v3_curvature(
                aggregate_branch_curvature_nm(extractor, branches, "p75", "sqrt_length")
            )
            if branches
            else "Unknown",
            "waviness_ratio_v2": float(waviness_v2["waviness_ratio_v2"]) if waviness_v2["waviness_ratio_v2"] is not None else None,
            "tortuosity_v2": float(waviness_v2["tortuosity_v2"]),
            "curvature_point_count": int(curvature_distribution_nm.size),
            "diameter_point_count": int(diameter_distribution_nm.size),
            "diameter_stats_nm": summarize_values(diameter_distribution_nm),
        }
        for stat in CURVATURE_STATS:
            for weight in WEIGHT_MODES:
                key = f"curvature_{stat}_{weight}_nm"
                record[key] = aggregate_branch_curvature_nm(extractor, branches, stat, weight)
        thresholds[label] = record

    return {
        "density": density,
        "diameter_nm": float(diameter_nm) if diameter_nm is not None else None,
        "diameter_stats_nm": diameter_stats_nm,
        "alignment": float(alignment_metrics["alignment"]),
        "alignment_raw": float(alignment_metrics["alignment_raw"]),
        "mean_phi_deg": float(alignment_metrics["mean_phi_deg"]),
        "mean_phi_raw_deg": float(alignment_metrics["mean_phi_raw_deg"]),
        "hof_method": alignment_metrics["hof_method"],
        "junction_count": float(junction_metrics["junction_count"]),
        "junction_ratio": float(junction_metrics["junction_ratio"]),
        "skeleton_length_px": float(junction_metrics["skeleton_length_px"]),
        "skeleton_length_um": float(junction_metrics["skeleton_length_um"]),
        "thresholds": thresholds,
    }


def flatten_record(base_row: Dict[str, Any], analysis: Dict[str, Any], elapsed_s: float) -> Dict[str, Any]:
    row = {
        "image_id": int(base_row["image_id"]),
        "sample_id": base_row["sample_id"],
        "sample_prefix": base_row["sample_prefix"],
        "group_key": base_row["group_key"],
        "file_path": base_row["file_path"],
        "al2o3_power": base_row["al2o3_power"],
        "al2o3_thickness": base_row["al2o3_thickness"],
        "fe_power": base_row["fe_power"],
        "fe_thickness": base_row["fe_thickness"],
        "fe_deposition_index": base_row["fe_deposition_index"],
        "ar_flow": base_row["ar_flow"],
        "h2_flow": base_row["h2_flow"],
        "c2h4_flow": base_row["c2h4_flow"],
        "anneal_temp": base_row["anneal_temp"],
        "growth_temp": base_row["growth_temp"],
        "anneal_time": base_row["anneal_time"],
        "growth_time": base_row["growth_time"],
        "density": analysis["density"],
        "alignment": analysis["alignment"],
        "alignment_raw": analysis["alignment_raw"],
        "mean_phi_deg": analysis["mean_phi_deg"],
        "mean_phi_raw_deg": analysis["mean_phi_raw_deg"],
        "hof_method": analysis["hof_method"],
        "diameter_nm": analysis["diameter_nm"],
        "diameter_mean_nm": analysis["diameter_stats_nm"]["mean"],
        "diameter_std_nm": analysis["diameter_stats_nm"]["std"],
        "diameter_min_nm": analysis["diameter_stats_nm"]["min"],
        "diameter_p25_nm": analysis["diameter_stats_nm"]["p25"],
        "diameter_p30_nm": analysis["diameter_stats_nm"]["p30"],
        "diameter_p50_nm": analysis["diameter_stats_nm"]["p50"],
        "diameter_p75_nm": analysis["diameter_stats_nm"]["p75"],
        "diameter_max_nm": analysis["diameter_stats_nm"]["max"],
        "junction_count": analysis["junction_count"],
        "junction_ratio": analysis["junction_ratio"],
        "skeleton_length_px": analysis["skeleton_length_px"],
        "skeleton_length_um": analysis["skeleton_length_um"],
        "elapsed_s": round(float(elapsed_s), 3),
    }
    for label, profile in analysis["thresholds"].items():
        prefix = label.lower()
        row[f"{prefix}_min_length_factor"] = profile["min_length_factor"]
        row[f"{prefix}_branch_count"] = profile["branch_count"]
        row[f"{prefix}_curvature_label"] = profile["curvature_label"]
        row[f"{prefix}_waviness_ratio_v2"] = profile["waviness_ratio_v2"]
        row[f"{prefix}_tortuosity_v2"] = profile["tortuosity_v2"]
        row[f"{prefix}_curvature_point_count"] = profile["curvature_point_count"]
        row[f"{prefix}_diameter_point_count"] = profile["diameter_point_count"]
        row[f"{prefix}_diameter_mean_nm"] = profile["diameter_stats_nm"]["mean"]
        row[f"{prefix}_diameter_p30_nm"] = profile["diameter_stats_nm"]["p30"]
        row[f"{prefix}_diameter_p50_nm"] = profile["diameter_stats_nm"]["p50"]
        row[f"{prefix}_diameter_p75_nm"] = profile["diameter_stats_nm"]["p75"]
        for stat in CURVATURE_STATS:
            for weight in WEIGHT_MODES:
                row[f"{prefix}_curvature_{stat}_{weight}_nm"] = profile[f"curvature_{stat}_{weight}_nm"]
    return row


def build_l_model_table(df: pd.DataFrame, label: str) -> pd.DataFrame:
    prefix = label.lower()
    out = df[
        [
            "image_id",
            "sample_id",
            "sample_prefix",
            "group_key",
            "file_path",
            "al2o3_power",
            "al2o3_thickness",
            "fe_power",
            "fe_thickness",
            "fe_deposition_index",
            "ar_flow",
            "h2_flow",
            "c2h4_flow",
            "anneal_temp",
            "growth_temp",
            "anneal_time",
            "growth_time",
            "density",
            "alignment",
            "diameter_nm",
            "diameter_mean_nm",
            "junction_count",
            "junction_ratio",
            f"{prefix}_branch_count",
            f"{prefix}_curvature_label",
            f"{prefix}_waviness_ratio_v2",
            f"{prefix}_tortuosity_v2",
            f"{prefix}_curvature_p50_uniform_nm",
            f"{prefix}_curvature_p50_sqrt_length_nm",
            f"{prefix}_curvature_p50_length_nm",
            f"{prefix}_curvature_p70_uniform_nm",
            f"{prefix}_curvature_p70_sqrt_length_nm",
            f"{prefix}_curvature_p70_length_nm",
            f"{prefix}_curvature_p75_uniform_nm",
            f"{prefix}_curvature_p75_sqrt_length_nm",
            f"{prefix}_curvature_p75_length_nm",
            f"{prefix}_curvature_mean_uniform_nm",
            f"{prefix}_curvature_mean_sqrt_length_nm",
            f"{prefix}_curvature_mean_length_nm",
            f"{prefix}_curvature_trimmed_mean_uniform_nm",
            f"{prefix}_curvature_trimmed_mean_sqrt_length_nm",
            f"{prefix}_curvature_trimmed_mean_length_nm",
        ]
    ].copy()
    rename_map = {col: col[len(prefix) + 1 :] for col in out.columns if col.startswith(f"{prefix}_")}
    out.rename(columns=rename_map, inplace=True)
    return out


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows_df = load_active_rows(limit=args.limit)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (OUTPUT_ROOT / f"zzy_50000_length_threshold_reextract_{timestamp}")
    ensure_dir(output_dir)

    segmenter = _get_cldice_segmenter(device=args.device)
    results: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    for idx, row in rows_df.iterrows():
        started = time.perf_counter()
        path = Path(str(row["file_path"]))
        print(f"[{idx + 1}/{len(rows_df)}] {path.name}", flush=True)
        try:
            image = read_gray(path)
            extractor = FeatureExtractor(magnification=int(row["magnification"]), speed_profile="accurate")
            roi = extractor.extract_roi(image)
            mask = segmenter.predict_mask(roi)
            analysis = analyze_thresholds(mask, int(row["magnification"]))
            results.append(flatten_record(row.to_dict(), analysis, time.perf_counter() - started))
        except Exception as exc:
            errors.append(
                {
                    "image_id": row["image_id"],
                    "file_path": str(path),
                    "error": str(exc),
                }
            )

    summary_csv = output_dir / "summary.csv"
    write_csv(summary_csv, results)
    summary_json = output_dir / "summary.json"
    summary_json.write_text(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "count": len(results),
                "errors": errors,
                "thresholds": [label for label, _ in THRESHOLDS],
                "curvature_stats": list(CURVATURE_STATS),
                "weight_modes": list(WEIGHT_MODES),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    result_df = pd.DataFrame(results)
    if not result_df.empty:
        for label, _ in THRESHOLDS:
            table = build_l_model_table(result_df, label)
            table.to_csv(output_dir / f"{label.lower()}_modeling_table.csv", index=False, encoding="utf-8-sig")

    if errors:
        pd.DataFrame(errors).to_csv(output_dir / "errors.csv", index=False, encoding="utf-8-sig")

    print(output_dir)


if __name__ == "__main__":
    main()
