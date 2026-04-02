from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.feature_extractor import FeatureExtractor
from tools.generate_xr_slice_standard_batch import (
    REFERENCE_THRESHOLD_LABEL,
    THRESHOLD_COLORS,
    analyze_threshold_profiles,
    draw_branch_overlay,
    ensure_dir,
    read_gray_image,
    render_panel,
    slugify,
    write_png,
)


DEFAULT_ANOMALY_CSV = PROJECT_ROOT / "reports" / "slice_standard_batch_20260331_005741" / "data_cleaning_review" / "modeling_prep" / "xr_modeling_anomaly_flags.csv"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports" / "slice_standard_batch_20260331_005741"
DEFAULT_OUTPUT_DIR = DEFAULT_REPORT_DIR / "data_cleaning_review" / "xr_anomaly_review_bundle"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a review bundle for flagged XR anomaly samples.")
    parser.add_argument("--anomaly-csv", type=Path, default=DEFAULT_ANOMALY_CSV)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def load_flagged_rows(anomaly_csv: Path, limit: int | None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with anomaly_csv.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if str(row.get("is_anomaly", "")).strip().lower() != "true":
                continue
            rows.append(row)
    rows.sort(key=lambda row: (-int(row.get("anomaly_flag_count", "0") or 0), row.get("sample_id", ""), row.get("image_id", "")))
    if limit is not None:
        rows = rows[: max(0, int(limit))]
    return rows


def find_features_path(report_dir: Path, image_id: str) -> Path:
    matches = sorted((report_dir / "items").glob(f"xr_{image_id}_*/features.json"))
    if not matches:
        raise FileNotFoundError(f"features.json not found for image_id={image_id}")
    return matches[0]


def read_mask(mask_path: Path) -> np.ndarray:
    mask = cv2.imdecode(np.fromfile(str(mask_path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Failed to read mask: {mask_path}")
    return mask


def build_review_record(source_record: dict[str, Any], item_dir: Path) -> dict[str, Any]:
    record = json.loads(json.dumps(source_record))
    record["panel_path"] = str(item_dir / "panel.png")
    record["mask_path"] = str(item_dir / "mask.png")
    record["original_path"] = str(item_dir / "original.png")
    record["l2_overlay_path"] = str(item_dir / "l2_overlay.png")
    return record


def process_one(row: dict[str, str], report_dir: Path, output_dir: Path, rank: int) -> dict[str, Any]:
    image_id = row["image_id"]
    features_path = find_features_path(report_dir, image_id)
    source_record = json.loads(features_path.read_text(encoding="utf-8"))

    sample_slug = slugify(source_record["sample_id"])
    item_slug = f"{rank:03d}_xr_{image_id}_{sample_slug}"
    item_dir = output_dir / "items" / item_slug
    ensure_dir(item_dir)

    image_gray = read_gray_image(Path(source_record["file_path"]))
    extractor = FeatureExtractor(magnification=int(source_record["magnification"]), speed_profile="accurate")
    roi = extractor.extract_roi(image_gray)
    write_png(item_dir / "original.png", roi.astype(np.uint8))

    source_mask_path = Path(source_record["mask_path"])
    mask = read_mask(source_mask_path)
    shutil.copy2(source_mask_path, item_dir / "mask.png")

    analysis = analyze_threshold_profiles(roi, (mask > 0).astype(np.uint8), int(source_record["magnification"]))
    l2_overlay = draw_branch_overlay(
        (mask > 0).astype(np.uint8),
        analysis["thresholds"][REFERENCE_THRESHOLD_LABEL]["branches"],
        THRESHOLD_COLORS[REFERENCE_THRESHOLD_LABEL],
    )
    write_png(item_dir / "l2_overlay.png", l2_overlay.astype(np.uint8))

    review_record = build_review_record(source_record, item_dir)
    render_panel(item_dir / "panel.png", roi, (mask > 0).astype(np.uint8), review_record, analysis["thresholds"])
    (item_dir / "features.json").write_text(json.dumps(review_record, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "rank": rank,
        "image_id": source_record["image_id"],
        "sample_id": source_record["sample_id"],
        "file_name": source_record["file_name"],
        "file_path": source_record["file_path"],
        "magnification": source_record["magnification"],
        "anomaly_flag_count": row.get("anomaly_flag_count"),
        "anomaly_reasons": row.get("anomaly_reasons"),
        "density": row.get("density"),
        "alignment": row.get("alignment"),
        "diameter_mean_nm": row.get("diameter_mean_nm"),
        "l2_curvature_trimmed_mean_sqrt_length_nm": row.get("l2_curvature_trimmed_mean_sqrt_length_nm"),
        "l2_waviness_ratio_v2": row.get("l2_waviness_ratio_v2"),
        "original_path": str(item_dir / "original.png"),
        "mask_path": str(item_dir / "mask.png"),
        "l2_overlay_path": str(item_dir / "l2_overlay.png"),
        "panel_path": str(item_dir / "panel.png"),
        "features_path": str(item_dir / "features.json"),
    }


def write_manifest(rows: list[dict[str, Any]], output_dir: Path) -> None:
    manifest_path = output_dir / "review_manifest.csv"
    if not rows:
        manifest_path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(rows: list[dict[str, Any]], output_dir: Path) -> None:
    summary = {
        "count": len(rows),
        "items": rows,
    }
    (output_dir / "review_manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    ensure_dir(args.output_dir / "items")

    flagged_rows = load_flagged_rows(args.anomaly_csv, args.limit)
    manifest_rows: list[dict[str, Any]] = []

    for idx, row in enumerate(flagged_rows, start=1):
        try:
            manifest_rows.append(process_one(row, args.report_dir, args.output_dir, idx))
            print(f"[{idx}/{len(flagged_rows)}] processed image_id={row['image_id']} sample_id={row['sample_id']}")
        except Exception as exc:
            print(f"[{idx}/{len(flagged_rows)}] failed image_id={row.get('image_id')} sample_id={row.get('sample_id')}: {exc}")

    write_manifest(manifest_rows, args.output_dir)
    write_summary(manifest_rows, args.output_dir)
    print(args.output_dir)


if __name__ == "__main__":
    main()
