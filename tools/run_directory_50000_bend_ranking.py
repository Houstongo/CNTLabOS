from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.core.batch_processor import _get_cldice_segmenter, _make_feature_extractor  # noqa: E402
from tools.batch_zzy_feature_panels import augment_features_with_junction_metrics  # noqa: E402

VALID_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
KNOWN_MAGS = (10000, 20000, 50000, 100000)
CALIBRATION_CONSTANT = 269792.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process arbitrary directory images and rank 50000X bend metrics.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda", help="clDice device: cuda/cpu")
    parser.add_argument("--diameter-method", default="enhanced", choices=["standard", "enhanced"])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def read_gray(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return image


def load_fei_xml(path: Path) -> Optional[ET.Element]:
    try:
        info = Image.open(path).info
        xml_text = info.get("34683")
        if not xml_text:
            return None
        return ET.fromstring(xml_text)
    except Exception:
        return None


def infer_magnification(path: Path) -> Optional[int]:
    root = load_fei_xml(path)
    if root is None:
        return None

    pixel_width = None
    databar_label = ""
    for elem in root.iter():
        tag = elem.tag.split("}")[-1]
        text = (elem.text or "").strip()
        if tag == "pixelWidth" and text:
            try:
                pixel_width = float(text)
            except ValueError:
                pixel_width = None
        elif tag == "databarLabel" and text:
            databar_label = text

    if databar_label:
        for mag in KNOWN_MAGS:
            if str(mag) in databar_label:
                return mag

    if pixel_width and pixel_width > 0:
        estimated = CALIBRATION_CONSTANT / pixel_width
        return min(KNOWN_MAGS, key=lambda mag: abs(mag - estimated))
    return None


def build_bend_metrics(features: Dict[str, Any]) -> Dict[str, Any]:
    curvature_trimmed_length = features.get("curvature_nm_v3_trimmed_mean_length")
    curvature_mean_length = features.get("curvature_nm_v3_mean_length")
    curvature = features.get("curvature_nm_v3_trimmed_mean_sqrt_length")
    diameter = features.get("diameter")
    dk_bend_index = None
    if curvature is not None and diameter is not None:
        dk_bend_index = float(curvature) * float(diameter)

    return {
        "curvature_nm_v3_trimmed_mean_length": curvature_trimmed_length,
        "curvature_nm_v3_mean_length": curvature_mean_length,
        "curvature_nm_v3_trimmed_mean_sqrt_length": curvature,
        "tortuosity_v2": features.get("tortuosity_v2"),
        "waviness_ratio_v2": features.get("waviness_ratio_v2"),
        "alignment": features.get("alignment"),
        "diameter": diameter,
        "junction_ratio": features.get("junction_ratio"),
        "junctions_per_100um": features.get("junctions_per_100um"),
        "dk_bend_index": dk_bend_index,
    }


def folder_index_for_path(path: Path, input_dir: Path) -> str:
    try:
        relative = path.relative_to(input_dir)
        if relative.parts:
            return relative.parts[0]
    except Exception:
        pass
    return path.parent.name


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_predicted_tier_sequence(rows: List[Dict[str, Any]]) -> List[int]:
    counts_by_tier: Dict[int, int] = {}
    for row in rows:
        tier = int(row["source_path"])
        counts_by_tier[tier] = counts_by_tier.get(tier, 0) + 1

    predicted: List[int] = []
    for tier in sorted(counts_by_tier.keys(), reverse=True):
        predicted.extend([tier] * counts_by_tier[tier])
    return predicted


def ranking_rows(
    rows: List[Dict[str, Any]],
    metric: str,
    descending: bool = True,
    predicted_tiers: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    sortable = [row for row in rows if row.get(metric) is not None]
    sortable.sort(key=lambda row: float(row[metric]), reverse=descending)
    ranked: List[Dict[str, Any]] = []
    for idx, row in enumerate(sortable, start=1):
        predicted_tier = predicted_tiers[idx - 1] if predicted_tiers and idx - 1 < len(predicted_tiers) else None
        ranked.append(
            {
                "rank": idx,
                "metric": metric,
                "file_name": row["file_name"],
                "source_path": row["source_path"],
                "真实属于哪个档位": row["source_path"],
                "预测排在哪个档位": predicted_tier,
                "value": row[metric],
                "tortuosity_v2": row.get("tortuosity_v2"),
                "waviness_ratio_v2": row.get("waviness_ratio_v2"),
                "curvature_nm_v3_trimmed_mean_length": row.get("curvature_nm_v3_trimmed_mean_length"),
                "curvature_nm_v3_mean_length": row.get("curvature_nm_v3_mean_length"),
                "curvature_nm_v3_trimmed_mean_sqrt_length": row.get("curvature_nm_v3_trimmed_mean_sqrt_length"),
                "alignment": row.get("alignment"),
                "diameter": row.get("diameter"),
                "junction_ratio": row.get("junction_ratio"),
                "junctions_per_100um": row.get("junctions_per_100um"),
                "dk_bend_index": row.get("dk_bend_index"),
            }
        )
    return ranked


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir
    if not input_dir.exists():
        raise SystemExit(f"Input dir not found: {input_dir}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (PROJECT_ROOT / "reports" / f"arbitrary_50000_bend_ranking_{stamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_images = sorted(
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in VALID_IMAGE_SUFFIXES
    )
    candidates: List[Dict[str, Any]] = []
    for path in all_images:
        mag = infer_magnification(path)
        if mag == 50000:
            candidates.append({"image_path": path, "magnification": mag})

    if args.limit > 0:
        candidates = candidates[: args.limit]

    if not candidates:
        raise SystemExit("No 50000X images found.")

    segmenter = _get_cldice_segmenter(device=args.device)
    rows: List[Dict[str, Any]] = []
    sidecar_dir = output_dir / "features"
    sidecar_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    for idx, item in enumerate(candidates, start=1):
        path = item["image_path"]
        print(f"[{idx}/{len(candidates)}] {path.name}", flush=True)
        img_gray = read_gray(path)
        extractor = _make_feature_extractor(magnification=50000, diameter_method=args.diameter_method)
        roi = extractor.extract_roi(img_gray)
        mask = segmenter.predict_mask(roi)
        features = extractor.extract_all(img_gray, external_binary_mask=mask)
        features = augment_features_with_junction_metrics(extractor, mask, features)
        bend_metrics = build_bend_metrics(features)

        payload = {
            "file_name": path.name,
            "source_path": folder_index_for_path(path, input_dir),
            "真实属于哪个档位": folder_index_for_path(path, input_dir),
            "magnification": 50000,
            **bend_metrics,
        }
        rows.append(payload)

        sidecar_path = sidecar_dir / f"{path.stem}__features.json"
        sidecar_path.write_text(
            json.dumps(
                {
                    "file_name": path.name,
                    "source_path": str(path),
                    "source_folder_index": folder_index_for_path(path, input_dir),
                    "magnification": 50000,
                    "features": features,
                    "bend_metrics": bend_metrics,
                },
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

    write_csv(output_dir / "summary.csv", rows)

    all_rankings: List[Dict[str, Any]] = []
    metric_specs = [
        ("tortuosity_v2", True),
        ("waviness_ratio_v2", True),
        ("curvature_nm_v3_trimmed_mean_length", True),
        ("curvature_nm_v3_mean_length", True),
        ("curvature_nm_v3_trimmed_mean_sqrt_length", True),
        ("dk_bend_index", True),
        ("alignment", False),
        ("junction_ratio", True),
        ("junctions_per_100um", True),
        ("diameter", True),
    ]
    rankings_dir = output_dir / "rankings"
    rankings_dir.mkdir(parents=True, exist_ok=True)
    predicted_tiers = build_predicted_tier_sequence(rows)
    for metric, descending in metric_specs:
        ranked = ranking_rows(rows, metric, descending=descending, predicted_tiers=predicted_tiers)
        write_csv(rankings_dir / f"{metric}.csv", ranked)
        all_rankings.extend(ranked)

    write_csv(output_dir / "all_rankings.csv", all_rankings)

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "device": args.device,
        "diameter_method": args.diameter_method,
        "selection_rule": "FEI metadata inferred magnification == 50000",
        "candidate_count": len(candidates),
        "elapsed_s": round(time.perf_counter() - t0, 1),
        "metrics": [metric for metric, _ in metric_specs],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"OUTPUT_DIR={output_dir}")


if __name__ == "__main__":
    main()
