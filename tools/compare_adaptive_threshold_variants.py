"""Compare adaptive-threshold variants against the current weak masks."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.feature_extractor import FeatureExtractor


VARIANT_SETS = {
    "paper_like_v1": [
        {"name": "current_wcntsegnet", "method": "current", "preprocess": True, "adaptive_method": "gaussian", "block_size": None, "c": None},
        {"name": "gauss_pre_b35_c5", "method": "custom", "preprocess": True, "adaptive_method": "gaussian", "block_size": 35, "c": 5},
        {"name": "gauss_pre_b51_c10", "method": "custom", "preprocess": True, "adaptive_method": "gaussian", "block_size": 51, "c": 10},
        {"name": "gauss_pre_b71_c15", "method": "custom", "preprocess": True, "adaptive_method": "gaussian", "block_size": 71, "c": 15},
        {"name": "mean_pre_b51_c10", "method": "custom", "preprocess": True, "adaptive_method": "mean", "block_size": 51, "c": 10},
        {"name": "gauss_raw_b51_c10", "method": "custom", "preprocess": False, "adaptive_method": "gaussian", "block_size": 51, "c": 10},
    ],
    "low_block_scan_v1": [
        {"name": "current_wcntsegnet", "method": "current", "preprocess": True, "adaptive_method": "gaussian", "block_size": None, "c": None},
        {"name": "gauss_pre_b15_c2", "method": "custom", "preprocess": True, "adaptive_method": "gaussian", "block_size": 15, "c": 2},
        {"name": "gauss_pre_b21_c2", "method": "custom", "preprocess": True, "adaptive_method": "gaussian", "block_size": 21, "c": 2},
        {"name": "gauss_pre_b27_c2", "method": "custom", "preprocess": True, "adaptive_method": "gaussian", "block_size": 27, "c": 2},
        {"name": "gauss_pre_b31_c2", "method": "custom", "preprocess": True, "adaptive_method": "gaussian", "block_size": 31, "c": 2},
        {"name": "gauss_pre_b35_c2", "method": "custom", "preprocess": True, "adaptive_method": "gaussian", "block_size": 35, "c": 2},
        {"name": "gauss_pre_b27_c5", "method": "custom", "preprocess": True, "adaptive_method": "gaussian", "block_size": 27, "c": 5},
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare adaptive-threshold weak-label variants on CNT patches.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-items", type=int, default=50)
    parser.add_argument("--variant-set", default="paper_like_v1", choices=sorted(VARIANT_SETS.keys()))
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_gray(path: str | Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    ensure_dir(path.parent)
    encoded = cv2.imencode(path.suffix, image)[1]
    path.write_bytes(encoded.tobytes())


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def adaptive_method_flag(name: str) -> int:
    if name.lower() == "gaussian":
        return cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    return cv2.ADAPTIVE_THRESH_MEAN_C


def crop_patch_from_source(source_image: np.ndarray, top: int, left: int, size: int) -> np.ndarray:
    return source_image[top:top + size, left:left + size]


def generate_variant_mask(source_image: np.ndarray, row: Dict[str, str], variant: Dict[str, object]) -> np.ndarray:
    top = int(row["patch_top"])
    left = int(row["patch_left"])
    patch_size = int(row["patch_size"])
    patch = crop_patch_from_source(source_image, top=top, left=left, size=patch_size)

    if variant["method"] == "current":
        current_mask = read_gray(row["patch_mask_path"])
        return current_mask

    magnification = int(float(row["magnification"])) if row.get("magnification") else None
    extractor = FeatureExtractor(magnification=magnification)
    extractor._calibrate(source_image.shape[1])
    processed = extractor.preprocess(patch) if bool(variant["preprocess"]) else patch
    mask = cv2.adaptiveThreshold(
        processed,
        255,
        adaptive_method_flag(str(variant["adaptive_method"])),
        cv2.THRESH_BINARY,
        int(variant["block_size"]),
        float(variant["c"]),
    )
    return mask


def panelize(images: List[np.ndarray], labels: List[str]) -> np.ndarray:
    tiles = []
    for image, label in zip(images, labels):
        if image.ndim == 2:
            tile = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            tile = image.copy()
        h, w = tile.shape[:2]
        banner_h = 28
        canvas = np.zeros((h + banner_h, w, 3), dtype=np.uint8)
        canvas[:banner_h, :] = 245
        canvas[banner_h:, :] = tile
        cv2.putText(canvas, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (20, 20, 20), 1, cv2.LINE_AA)
        tiles.append(canvas)

    cols = 3
    rows = int(np.ceil(len(tiles) / cols))
    h, w = tiles[0].shape[:2]
    sheet = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for idx, tile in enumerate(tiles):
        r = idx // cols
        c = idx % cols
        sheet[r * h:(r + 1) * h, c * w:(c + 1) * w] = tile
    return sheet


def mask_stats(mask: np.ndarray) -> Dict[str, object]:
    num_labels, _ = cv2.connectedComponents((mask > 0).astype(np.uint8), connectivity=8)
    return {
        "foreground_ratio_pct": round(float(np.count_nonzero(mask) / max(mask.size, 1) * 100.0), 4),
        "connected_components": int(max(num_labels - 1, 0)),
    }


def write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_panel_filename(row: Dict[str, str]) -> str:
    image_id = str(row.get("image_id", "")).strip() or "unknown"
    patch_index = str(row.get("patch_index", "")).strip() or "patch"
    top = str(row.get("patch_top", "")).strip() or "r0"
    left = str(row.get("patch_left", "")).strip() or "c0"
    size = str(row.get("patch_size", "")).strip() or "s0"
    return f"{image_id}_{patch_index}_r{top}_c{left}_{size}_adaptive_compare.png"


def main() -> None:
    args = parse_args()
    rows = load_rows(args.manifest)[: args.max_items]
    output_root = args.output_root
    panel_dir = output_root / "comparison_panels"
    ensure_dir(panel_dir)
    variants = VARIANT_SETS[args.variant_set]

    stats_rows: List[Dict[str, object]] = []
    preview_paths: List[Path] = []

    for idx, row in enumerate(rows, 1):
        source_image = read_gray(row["image_path"])
        patch = read_gray(row["patch_image_path"])
        current_mask = read_gray(row["patch_mask_path"])

        images = [patch, current_mask]
        labels = ["Original patch", "Current weak"]

        base_stats: Dict[str, object] = {
            "image_id": row["image_id"],
            "patch_filename": row["patch_filename"],
        }
        cur_stats = mask_stats(current_mask)
        base_stats["current_fg_pct"] = cur_stats["foreground_ratio_pct"]
        base_stats["current_components"] = cur_stats["connected_components"]

        for variant in variants[1:]:
            mask = generate_variant_mask(source_image, row, variant)
            images.append(mask)
            labels.append(str(variant["name"]))
            stats = mask_stats(mask)
            base_stats[f"{variant['name']}_fg_pct"] = stats["foreground_ratio_pct"]
            base_stats[f"{variant['name']}_components"] = stats["connected_components"]

        panel = panelize(images, labels)
        out_path = panel_dir / build_panel_filename(row)
        write_image(out_path, panel)
        preview_paths.append(out_path)
        stats_rows.append(base_stats)
        print(f"[{idx}/{len(rows)}] {out_path.name}")

    write_csv(output_root / "adaptive_variant_stats.csv", stats_rows)
    summary = {
        "manifest": str(args.manifest),
        "count": len(rows),
        "variant_set": args.variant_set,
        "variants": [variant["name"] for variant in variants],
        "panel_dir": str(panel_dir),
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
