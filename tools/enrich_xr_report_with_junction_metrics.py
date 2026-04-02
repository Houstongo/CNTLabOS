from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.feature_extractor import FeatureExtractor
from tools.generate_xr_slice_standard_batch import (
    compute_junction_metrics,
    load_completed_records,
    write_summary_files,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill junction features into an existing XR report directory.")
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def read_mask(mask_path: Path) -> np.ndarray:
    mask = cv2.imdecode(np.fromfile(str(mask_path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Failed to read mask: {mask_path}")
    return (mask > 0).astype(np.uint8) * 255


def update_record(features_path: Path) -> dict | None:
    record = json.loads(features_path.read_text(encoding="utf-8"))
    mask_path = Path(record["mask_path"])
    if not mask_path.exists():
        return None

    mask = read_mask(mask_path)
    extractor = FeatureExtractor(magnification=int(record["magnification"]), speed_profile="accurate")
    extractor._calibrate(mask.shape[1])
    _, skeleton = extractor.calculate_diameter(mask)
    metrics = compute_junction_metrics(skeleton, extractor.px_per_um)

    record["junction_count"] = int(round(metrics["junction_count"]))
    record["junction_ratio"] = round(metrics["junction_ratio"], 6)
    record["skeleton_length_px"] = round(metrics["skeleton_length_px"], 3)
    record["skeleton_length_um"] = round(metrics["skeleton_length_um"], 6)

    features_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record


def main() -> None:
    args = parse_args()
    feature_paths = sorted((args.report_dir / "items").glob("*/features.json"))
    if args.limit is not None:
        feature_paths = feature_paths[: max(0, int(args.limit))]

    updated = 0
    for idx, features_path in enumerate(feature_paths, start=1):
        result = update_record(features_path)
        if result is not None:
            updated += 1
            print(f"[{idx}/{len(feature_paths)}] updated image_id={result['image_id']} sample_id={result['sample_id']}")

    records = load_completed_records(args.report_dir)
    write_summary_files(args.report_dir, records)
    print(f"updated={updated}")
    print(args.report_dir)


if __name__ == "__main__":
    main()
