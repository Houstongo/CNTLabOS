import csv
import json
import sys
import argparse
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.feature_extractor import FeatureExtractor


ROOT = Path(r"D:\CNTDATA\coredata\selected_No28_No39_No41_No42\rough_curvature_buckets_visual")
OUTPUT_ROOT = Path(r"D:\CNTDATA\CNTA_ML_Project\reports")

SELECTION_PLAN = [
    ("50000x", "low_curvature_near_straight"),
    ("50000x", "high_curvature_curly_entangled"),
    ("100000x", "low_curvature_near_straight"),
    ("100000x", "medium_curvature_visible_waviness"),
    ("100000x", "high_curvature_curly_entangled"),
]


def pick_samples():
    samples = []
    for mag_dir, bucket_dir in SELECTION_PLAN:
        bucket = ROOT / mag_dir / bucket_dir
        candidates = sorted(
            path for path in bucket.iterdir()
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
        )
        if not candidates:
            continue
        samples.append({
            "magnification": int(mag_dir.replace("x", "")),
            "bucket": bucket_dir,
            "path": candidates[0],
        })
    return samples


def render_panel(image_gray, record, output_path: Path):
    fig = plt.figure(figsize=(12, 6.75), dpi=160, constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0])

    ax_img = fig.add_subplot(grid[0, 0])
    ax_img.imshow(image_gray, cmap="gray")
    ax_img.set_title("SEM ROI Preview", fontsize=14)
    ax_img.axis("off")

    ax_text = fig.add_subplot(grid[0, 1])
    ax_text.axis("off")

    lines = [
        f"file: {record['file_name']}",
        f"magnification: {record['magnification']}x",
        f"bucket: {record['bucket']}",
        "",
        f"density: {record['density']:.2f} %",
        f"diameter: {record['diameter']}",
        f"legacy curvature_nm: {record['curvature_nm']:.6f}",
        f"v2 curvature_nm: {record['curvature_nm_v2']:.6f}",
        f"v3 curvature_nm: {record['curvature_nm_v3']:.6f}",
        "",
        f"legacy label: {record['curvature']}",
        f"v2 label: {record['curvature_v2']}",
        f"v3 label: {record['curvature_v3']}",
        "",
        f"waviness_ratio: {record['waviness_ratio']}",
        f"waviness_ratio_v2: {record['waviness_ratio_v2']}",
        f"tortuosity: {record['tortuosity']}",
        f"tortuosity_v2: {record['tortuosity_v2']}",
        "",
        f"px_per_um: {record['px_per_um']}",
    ]
    ax_text.text(
        0.0,
        1.0,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=12,
        family="monospace",
    )
    fig.suptitle("Legacy vs V2 vs V3 Curvature Quick Compare", fontsize=16)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--speed-profile", default="accurate", choices=["accurate", "fast"])
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUT_ROOT / f"curvature_v3_quick_compare_{args.speed_profile}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for sample in pick_samples():
        image_gray = cv2.imread(str(sample["path"]), cv2.IMREAD_GRAYSCALE)
        if image_gray is None:
            continue

        extractor = FeatureExtractor(
            magnification=sample["magnification"],
            speed_profile=args.speed_profile,
        )
        features = extractor.extract_all(image_gray)

        record = {
            "file_name": sample["path"].name,
            "file_path": str(sample["path"]),
            "magnification": sample["magnification"],
            "bucket": sample["bucket"],
            "density": features["density"],
            "diameter": features["diameter"],
            "curvature": features["curvature"],
            "curvature_nm": features["curvature_nm"],
            "curvature_v2": features["curvature_v2"],
            "curvature_nm_v2": features["curvature_nm_v2"],
            "curvature_v3": features["curvature_v3"],
            "curvature_nm_v3": features["curvature_nm_v3"],
            "waviness_ratio": features["waviness_ratio"],
            "waviness_ratio_v2": features["waviness_ratio_v2"],
            "tortuosity": features["tortuosity"],
            "tortuosity_v2": features["tortuosity_v2"],
            "px_per_um": features["px_per_um"],
        }
        records.append(record)

        stem = sample["path"].stem.replace(" ", "_")
        render_panel(image_gray, record, out_dir / f"{stem}__compare.png")
        with (out_dir / f"{stem}__features.json").open("w", encoding="utf-8") as fh:
            json.dump(record, fh, ensure_ascii=False, indent=2)

    summary_csv = out_dir / "summary.csv"
    if records:
        with summary_csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)

    with (out_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "output_dir": str(out_dir),
                "speed_profile": args.speed_profile,
                "count": len(records),
                "records": records,
            },
            fh,
            ensure_ascii=False,
            indent=2,
        )

    print(out_dir)


if __name__ == "__main__":
    main()
