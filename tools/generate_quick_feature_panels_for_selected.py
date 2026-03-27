"""Quick first-pass feature extraction panels for selected CNT images."""

from __future__ import annotations

import csv
import json
import sys
import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.feature_extractor import FeatureExtractor  # noqa: E402


INPUT_ROOT = Path(r"D:\CNTDATA\coredata\selected_No28_No39_No41_No42\rough_curvature_buckets_visual")
OUTPUT_PARENT = INPUT_ROOT
VALID_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate quick feature panels for selected CNT images.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--resume", action="store_true", help="Skip panels that already exist in output-dir.")
    return parser.parse_args()


def read_gray_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return image


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def slugify(text: str) -> str:
    chars = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_"}:
            chars.append(ch)
        else:
            chars.append("_")
    return "".join(chars).strip("_") or "item"


def iter_input_images(root: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for mag_dir in sorted(root.glob("*x")):
        if not mag_dir.is_dir():
            continue
        mag_text = mag_dir.name.lower()
        try:
            magnification = int(mag_text.replace("x", ""))
        except ValueError:
            continue
        for bucket_dir in sorted([p for p in mag_dir.iterdir() if p.is_dir()]):
            for image_path in sorted([p for p in bucket_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_IMAGE_SUFFIXES]):
                items.append(
                    {
                        "magnification": magnification,
                        "magnification_dir": mag_dir.name,
                        "bucket": bucket_dir.name,
                        "image_path": image_path,
                    }
                )
    return items


def format_metric(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, (float, int, np.floating, np.integer)):
        return f"{float(value):.{digits}f}"
    return str(value)


def add_text_block(canvas: np.ndarray, lines: List[str], x: int, y: int, line_gap: int = 34) -> None:
    for idx, line in enumerate(lines):
        y_pos = y + idx * line_gap
        cv2.putText(canvas, line, (x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (28, 28, 28), 2, cv2.LINE_AA)


def build_panel(image_gray: np.ndarray, rel_info: Dict[str, Any], features: Dict[str, Any]) -> np.ndarray:
    h, w = image_gray.shape
    panel_w = max(950, w + 520)
    panel_h = max(h + 120, 900)
    panel = np.full((panel_h, panel_w, 3), 248, dtype=np.uint8)

    title = f"{rel_info['magnification_dir']} | {rel_info['bucket']}"
    name = rel_info["image_path"].name
    cv2.putText(panel, title, (28, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(panel, name[:110], (28, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 60), 1, cv2.LINE_AA)

    image_bgr = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
    panel[110 : 110 + h, 28 : 28 + w] = image_bgr
    cv2.rectangle(panel, (27, 109), (28 + w, 110 + h), (180, 180, 180), 1)

    x0 = 28 + w + 34
    text_lines = [
        "Quick Feature Pass (fast profile)",
        f"density (%)            {format_metric(features.get('density'), 2)}",
        f"curvature_nm           {format_metric(features.get('curvature_nm'), 4)}",
        f"curvature_nm_v2        {format_metric(features.get('curvature_nm_v2'), 6)}",
        f"waviness_ratio         {format_metric(features.get('waviness_ratio'), 4)}",
        f"waviness_ratio_v2      {format_metric(features.get('waviness_ratio_v2'), 4)}",
        f"tortuosity             {format_metric(features.get('tortuosity'), 3)}",
        f"tortuosity_v2          {format_metric(features.get('tortuosity_v2'), 3)}",
        f"alignment              {format_metric(features.get('alignment'), 4)}",
        f"diameter (nm)          {format_metric(features.get('diameter'), 2)}",
        f"curvature label        {features.get('curvature')}",
        f"curvature label v2     {features.get('curvature_v2')}",
        f"waviness branches      {features.get('waviness_branches')}",
        f"waviness branches v2   {features.get('waviness_branches_v2')}",
        f"px_per_um              {format_metric(features.get('px_per_um'), 2)}",
        f"n_branches             {features.get('n_branches')}",
        f"speed_profile          {features.get('speed_profile')}",
    ]
    add_text_block(panel, text_lines, x0, 150, line_gap=38)

    footer_lines = [
        "Note:",
        "This first version reuses FeatureExtractor.extract_all(img)",
        'with speed_profile="fast" for quick triage.',
        "Values are suitable for browsing and rough comparison,",
        "not yet a final calibrated release report.",
    ]
    add_text_block(panel, footer_lines, 28, panel_h - 170, line_gap=30)
    return panel


def main() -> None:
    args = parse_args()
    items = iter_input_images(INPUT_ROOT)
    if not items:
        raise SystemExit(f"No images found under: {INPUT_ROOT}")

    if args.output_dir is not None:
        output_root = args.output_dir
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_root = OUTPUT_PARENT / f"quick_feature_panels_v1_{stamp}"
    ensure_dir(output_root)

    summary_rows: List[Dict[str, Any]] = []

    for index, item in enumerate(items, start=1):
        image_path = item["image_path"]
        print(f"[{index}/{len(items)}] {item['magnification_dir']} | {item['bucket']} | {image_path.name}")
        img_gray = read_gray_image(image_path)
        extractor = FeatureExtractor(magnification=item["magnification"], speed_profile="fast")
        rel_dir = output_root / item["magnification_dir"] / item["bucket"]
        ensure_dir(rel_dir)
        stem = slugify(image_path.stem)
        panel_path = rel_dir / f"{stem}__feature_panel.jpg"

        sidecar_path = rel_dir / f"{stem}__features.json"
        if args.resume and panel_path.exists() and sidecar_path.exists():
            features = json.loads(sidecar_path.read_text(encoding="utf-8"))
        else:
            features = extractor.extract_all(img_gray)
            panel = build_panel(img_gray, item, features)
            cv2.imencode(".jpg", panel, [int(cv2.IMWRITE_JPEG_QUALITY), 92])[1].tofile(str(panel_path))
            sidecar_path.write_text(json.dumps(features, indent=2, ensure_ascii=False), encoding="utf-8")

        row = {
            "magnification_dir": item["magnification_dir"],
            "bucket": item["bucket"],
            "file_name": image_path.name,
            "source_path": str(image_path),
            "panel_path": str(panel_path),
            "feature_json_path": str(sidecar_path),
            "density": features.get("density"),
            "curvature_nm": features.get("curvature_nm"),
            "curvature_nm_v2": features.get("curvature_nm_v2"),
            "waviness_ratio": features.get("waviness_ratio"),
            "waviness_ratio_v2": features.get("waviness_ratio_v2"),
            "tortuosity": features.get("tortuosity"),
            "tortuosity_v2": features.get("tortuosity_v2"),
            "alignment": features.get("alignment"),
            "diameter": features.get("diameter"),
            "curvature_label": features.get("curvature"),
            "curvature_label_v2": features.get("curvature_v2"),
            "speed_profile": features.get("speed_profile"),
        }
        summary_rows.append(row)

    csv_path = output_root / "summary.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    summary_payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "input_root": str(INPUT_ROOT),
        "output_root": str(output_root),
        "count": len(summary_rows),
        "method": "FeatureExtractor.extract_all(img_gray) with speed_profile=fast",
        "items": summary_rows,
    }
    (output_root / "summary.json").write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"OUTPUT_DIR={output_root}")
    print(f"CSV_PATH={csv_path}")


if __name__ == "__main__":
    main()
