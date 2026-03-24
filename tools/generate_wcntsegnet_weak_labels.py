"""
Generate WCNTSegNET weak-label assets for a curated experiment dataset.

This script reads the fixed train/test/reserve manifests, generates one
full-image weak mask per image using the current WCNTSegNET pipeline, and
exports dataset-level manifest/stat tables plus preview overlays.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.feature_extractor import FeatureExtractor


DEFAULT_SPLITS = ("train", "test", "reserve")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate WCNTSegNET weak labels for a prepared dataset.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS))
    parser.add_argument("--preview-per-split", type=int, default=12)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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


def load_split_manifest(dataset_root: Path, split: str) -> List[Dict[str, str]]:
    manifest_path = dataset_root / "manifests" / f"{split}_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing split manifest: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def dataset_image_filename_from_row(row: Dict[str, str]) -> str:
    return f"{int(row['image_id']):05d}_{Path(row['file_path']).name}"


def dataset_mask_filename(image_filename: str) -> str:
    image_path = Path(image_filename)
    return f"{image_path.stem}_mask.png"


def derive_output_paths(dataset_root: Path, split: str, image_filename: str) -> Dict[str, Path]:
    mask_filename = dataset_mask_filename(image_filename)
    return {
        "image_path": dataset_root / split / "images" / image_filename,
        "mask_path": dataset_root / split / "masks_wcntsegnet" / mask_filename,
        "overlay_path": dataset_root / "previews_wcntsegnet" / split / f"{Path(image_filename).stem}_overlay.png",
    }


def read_gray_image(image_path: Path) -> np.ndarray:
    img_array = np.fromfile(str(image_path), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")
    return img


def write_image(path: Path, image: np.ndarray) -> None:
    ensure_dir(path.parent)
    ok = cv2.imencode(path.suffix, image)[1]
    path.write_bytes(ok.tobytes())


def overlay_mask_on_gray(img_gray: np.ndarray, full_mask: np.ndarray) -> np.ndarray:
    overlay = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    overlay[full_mask > 0] = (64, 200, 64)
    return overlay


def build_preview_contact_sheet(overlay_paths: Sequence[Path], output_path: Path, thumb_size: Tuple[int, int] = (220, 160), cols: int = 4) -> None:
    images: List[np.ndarray] = []
    for overlay_path in overlay_paths:
        image = cv2.imread(str(overlay_path))
        if image is None:
            continue
        thumb = cv2.resize(image, thumb_size, interpolation=cv2.INTER_AREA)
        label = overlay_path.stem[:32]
        cv2.rectangle(thumb, (0, thumb_size[1] - 22), (thumb_size[0], thumb_size[1]), (0, 0, 0), -1)
        cv2.putText(thumb, label, (6, thumb_size[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        images.append(thumb)

    if not images:
        return

    rows = int(np.ceil(len(images) / cols))
    sheet = np.full((rows * thumb_size[1], cols * thumb_size[0], 3), 255, dtype=np.uint8)
    for idx, thumb in enumerate(images):
        row = idx // cols
        col = idx % cols
        y0 = row * thumb_size[1]
        x0 = col * thumb_size[0]
        sheet[y0:y0 + thumb_size[1], x0:x0 + thumb_size[0]] = thumb

    write_image(output_path, sheet)


def generate_wcntsegnet_mask(img_gray: np.ndarray, magnification: int | None) -> Tuple[np.ndarray, np.ndarray, float]:
    extractor = FeatureExtractor(magnification=magnification)
    roi = extractor.extract_roi(img_gray)
    extractor._calibrate(roi.shape[1])
    processed = extractor.preprocess(roi)
    density_roi, roi_mask = extractor.calculate_density(processed)

    full_mask = np.zeros_like(img_gray, dtype=np.uint8)
    full_mask[: roi.shape[0], :] = roi_mask.astype(np.uint8)
    return roi_mask.astype(np.uint8), full_mask, float(density_roi)


def compute_mask_stats(roi_mask: np.ndarray, full_mask: np.ndarray) -> Dict[str, object]:
    num_labels, _labels = cv2.connectedComponents((roi_mask > 0).astype(np.uint8), connectivity=8)
    return {
        "image_height": int(full_mask.shape[0]),
        "image_width": int(full_mask.shape[1]),
        "roi_height": int(roi_mask.shape[0]),
        "roi_width": int(roi_mask.shape[1]),
        "roi_foreground_ratio_pct": round(float(np.count_nonzero(roi_mask) / max(roi_mask.size, 1) * 100.0), 4),
        "full_foreground_ratio_pct": round(float(np.count_nonzero(full_mask) / max(full_mask.size, 1) * 100.0), 4),
        "connected_components": int(max(num_labels - 1, 0)),
    }


def build_manifest_row(
    dataset_root: Path,
    split: str,
    manifest_row: Dict[str, str],
    image_filename: str,
    status: str,
    stats: Dict[str, object] | None,
    error: str | None = None,
) -> Dict[str, object]:
    output_paths = derive_output_paths(dataset_root, split, image_filename)
    row: Dict[str, object] = {
        "split": split,
        "status": status,
        "error": error,
        "image_id": manifest_row.get("image_id"),
        "image_filename": image_filename,
        "image_path": str(output_paths["image_path"]),
        "mask_filename": output_paths["mask_path"].name,
        "mask_path": str(output_paths["mask_path"]),
        "overlay_path": str(output_paths["overlay_path"]),
    }

    for key, value in manifest_row.items():
        if key not in row:
            row[key] = value

    if stats:
        row.update(stats)

    return row


def build_stats_row(manifest_row: Dict[str, object]) -> Dict[str, object]:
    keys = [
        "split",
        "status",
        "image_id",
        "image_filename",
        "sample_id",
        "magnification",
        "repeat_id",
        "roi_foreground_ratio_pct",
        "full_foreground_ratio_pct",
        "connected_components",
        "roi_height",
        "image_height",
        "image_width",
        "error",
    ]
    return {key: manifest_row.get(key) for key in keys}


def process_split(dataset_root: Path, split: str, preview_per_split: int) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Path]]:
    rows = load_split_manifest(dataset_root, split)
    image_dir = dataset_root / split / "images"
    if not image_dir.exists():
        raise FileNotFoundError(f"Missing split image directory: {image_dir}")

    exported_rows: List[Dict[str, object]] = []
    stat_rows: List[Dict[str, object]] = []
    preview_paths: List[Path] = []

    for index, row in enumerate(rows, 1):
        image_filename = dataset_image_filename_from_row(row)
        output_paths = derive_output_paths(dataset_root, split, image_filename)
        image_path = output_paths["image_path"]

        print(f"[{split}] {index}/{len(rows)} {image_filename}")
        try:
            img_gray = read_gray_image(image_path)
            roi_mask, full_mask, density_roi = generate_wcntsegnet_mask(
                img_gray=img_gray,
                magnification=int(row["magnification"]) if row.get("magnification") else None,
            )
            stats = compute_mask_stats(roi_mask=roi_mask, full_mask=full_mask)
            stats["density_roi_pct"] = round(density_roi, 4)

            write_image(output_paths["mask_path"], full_mask)

            if len(preview_paths) < preview_per_split:
                overlay = overlay_mask_on_gray(img_gray, full_mask)
                write_image(output_paths["overlay_path"], overlay)
                preview_paths.append(output_paths["overlay_path"])

            manifest_row = build_manifest_row(
                dataset_root=dataset_root,
                split=split,
                manifest_row=row,
                image_filename=image_filename,
                status="success",
                stats=stats,
            )
        except Exception as exc:
            manifest_row = build_manifest_row(
                dataset_root=dataset_root,
                split=split,
                manifest_row=row,
                image_filename=image_filename,
                status="failed",
                stats=None,
                error=str(exc),
            )

        exported_rows.append(manifest_row)
        stat_rows.append(build_stats_row(manifest_row))

    contact_sheet_path = dataset_root / "previews_wcntsegnet" / f"{split}_contact_sheet.jpg"
    build_preview_contact_sheet(preview_paths, contact_sheet_path)
    return exported_rows, stat_rows, preview_paths


def summarize(all_rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "total": len(all_rows),
        "success": sum(1 for row in all_rows if row["status"] == "success"),
        "failed": sum(1 for row in all_rows if row["status"] != "success"),
        "splits": {},
    }
    split_names = sorted({str(row["split"]) for row in all_rows})
    for split in split_names:
        split_rows = [row for row in all_rows if row["split"] == split]
        success_rows = [row for row in split_rows if row["status"] == "success"]
        summary["splits"][split] = {
            "count": len(split_rows),
            "success": len(success_rows),
            "failed": len(split_rows) - len(success_rows),
            "mean_roi_foreground_ratio_pct": round(
                float(np.mean([row["roi_foreground_ratio_pct"] for row in success_rows])) if success_rows else 0.0,
                4,
            ),
        }
    return summary


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root
    all_rows: List[Dict[str, object]] = []
    all_stats: List[Dict[str, object]] = []

    for split in args.splits:
        exported_rows, stat_rows, _preview_paths = process_split(
            dataset_root=dataset_root,
            split=split,
            preview_per_split=args.preview_per_split,
        )
        all_rows.extend(exported_rows)
        all_stats.extend(stat_rows)

    write_csv(dataset_root / "labels_manifest.csv", all_rows)
    write_csv(dataset_root / "label_stats.csv", all_stats)

    summary = summarize(all_rows)
    summary["dataset_root"] = str(dataset_root)
    summary["splits"] = summary.get("splits", {})
    (dataset_root / "labels_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"DATASET_ROOT={dataset_root}")
    print(f"LABELS_MANIFEST={dataset_root / 'labels_manifest.csv'}")
    print(f"LABEL_STATS={dataset_root / 'label_stats.csv'}")
    print(f"LABEL_SUMMARY={dataset_root / 'labels_summary.json'}")


if __name__ == "__main__":
    main()
