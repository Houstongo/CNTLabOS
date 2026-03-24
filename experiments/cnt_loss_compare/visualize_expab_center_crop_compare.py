"""Generate center-cropped Exp-A vs Exp-B threshold comparisons on the test set."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch

if __package__ is None or __package__ == "":
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from experiments.cnt_loss_compare.config import load_config
    from experiments.cnt_loss_compare.visualize_predictions import (
        DEFAULT_EXP_A_CHECKPOINT,
        DEFAULT_EXP_A_CONFIG,
        DEFAULT_EXP_B_CHECKPOINT,
        DEFAULT_EXP_B_CONFIG,
        build_model,
        load_manifest_rows,
        preprocess_roi_for_model,
        read_image,
        restore_probability_to_full_image,
        save_png,
        title_panel,
    )
    from experiments.cnt_loss_compare.visualize_expb_threshold_sweep import training_view_weak_mask
    from src.analysis.feature_extractor import FeatureExtractor
else:
    from .config import load_config
    from .visualize_predictions import (
        DEFAULT_EXP_A_CHECKPOINT,
        DEFAULT_EXP_A_CONFIG,
        DEFAULT_EXP_B_CHECKPOINT,
        DEFAULT_EXP_B_CONFIG,
        build_model,
        load_manifest_rows,
        preprocess_roi_for_model,
        read_image,
        restore_probability_to_full_image,
        save_png,
        title_panel,
    )
    from .visualize_expb_threshold_sweep import training_view_weak_mask
    from src.analysis.feature_extractor import FeatureExtractor


DEFAULT_OUTPUT_DIR = Path(
    r"D:\CNTDATA\CNTA_ML_Project\experiments\cnt_loss_compare\reports\testset_expAB_center_crop_500_thr60_70_step02_20260325"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ROI-centered 500x500 Exp-A vs Exp-B threshold comparisons.")
    parser.add_argument("--exp-a-config", type=Path, default=DEFAULT_EXP_A_CONFIG)
    parser.add_argument("--exp-a-checkpoint", type=Path, default=DEFAULT_EXP_A_CHECKPOINT)
    parser.add_argument("--exp-b-config", type=Path, default=DEFAULT_EXP_B_CONFIG)
    parser.add_argument("--exp-b-checkpoint", type=Path, default=DEFAULT_EXP_B_CHECKPOINT)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--crop-size", type=int, default=500)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.60, 0.62, 0.64, 0.66, 0.68, 0.70])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def mask_to_bgr(mask: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)


def get_center_crop_bounds(image_gray: np.ndarray, crop_size: int) -> Tuple[int, int, int, int, int, int]:
    roi = FeatureExtractor.extract_roi(image_gray)
    roi_h, roi_w = roi.shape[:2]
    crop_h = min(crop_size, roi_h)
    crop_w = min(crop_size, roi_w)
    top = max(0, (roi_h - crop_h) // 2)
    left = max(0, (roi_w - crop_w) // 2)
    bottom = top + crop_h
    right = left + crop_w
    return top, bottom, left, right, roi_h, roi_w


def crop_full_image(array_full: np.ndarray, top: int, bottom: int, left: int, right: int) -> np.ndarray:
    return array_full[top:bottom, left:right]


@torch.no_grad()
def predict_full_probability(model, image_gray: np.ndarray, image_size: int, device: torch.device) -> np.ndarray:
    image_tensor, roi_h, roi_w = preprocess_roi_for_model(image_gray, image_size)
    logits = model(image_tensor.to(device))
    probability_512 = torch.sigmoid(logits)[0, 0].detach().cpu().numpy().astype(np.float32)
    return restore_probability_to_full_image(probability_512, image_gray.shape, (roi_h, roi_w))


def ensure_dirs(output_dir: Path, thresholds: List[float]) -> Dict[str, Path]:
    paths: Dict[str, Path] = {
        "original": output_dir / "original_crops",
        "weak_full": output_dir / "weak_full_crops",
        "weak_train": output_dir / "weak_train_view_crops",
        "panels": output_dir / "comparison_panels",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    for threshold in thresholds:
        key = int(round(threshold * 100))
        exp_a_dir = output_dir / f"exp_a_thr_{key:02d}_crops"
        exp_b_dir = output_dir / f"exp_b_thr_{key:02d}_crops"
        exp_a_dir.mkdir(parents=True, exist_ok=True)
        exp_b_dir.mkdir(parents=True, exist_ok=True)
        paths[f"exp_a_{key:02d}"] = exp_a_dir
        paths[f"exp_b_{key:02d}"] = exp_b_dir
    return paths


def make_panel(
    original_crop: np.ndarray,
    weak_full_crop: np.ndarray,
    weak_train_crop: np.ndarray,
    exp_a_crop: np.ndarray,
    exp_b_crop: np.ndarray,
    threshold: float,
) -> np.ndarray:
    original_bgr = cv2.cvtColor(original_crop, cv2.COLOR_GRAY2BGR)
    panels = [
        title_panel(original_bgr, "Original"),
        title_panel(mask_to_bgr(weak_full_crop), "Weak Full"),
        title_panel(mask_to_bgr(weak_train_crop), "Weak Train-View"),
        title_panel(mask_to_bgr(exp_a_crop), f"Exp-A@{threshold:.2f}"),
        title_panel(mask_to_bgr(exp_b_crop), f"Exp-B@{threshold:.2f}"),
    ]
    spacer = np.full((panels[0].shape[0], 16, 3), 235, dtype=np.uint8)
    stitched = panels[0]
    for panel in panels[1:]:
        stitched = np.concatenate([stitched, spacer, panel], axis=1)
    return stitched


def threshold_mask(probability: np.ndarray, threshold: float) -> np.ndarray:
    return (probability >= threshold).astype(np.uint8) * 255


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest or Path(load_config(args.exp_b_config)["data"]["test_manifest"])
    rows = load_manifest_rows(manifest_path)
    if args.limit and args.limit > 0:
        rows = rows[:args.limit]

    thresholds = [float(value) for value in args.thresholds]
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    exp_a_model, exp_a_config = build_model(args.exp_a_config, args.exp_a_checkpoint, device)
    exp_b_model, _ = build_model(args.exp_b_config, args.exp_b_checkpoint, device)
    image_size = int(exp_a_config["data"]["image_size"])
    output_paths = ensure_dirs(args.output_dir, thresholds)
    summary_rows: List[Dict[str, str]] = []

    for index, row in enumerate(rows, start=1):
        image_path = Path(row["image_path"])
        weak_mask_path = Path(row["mask_path"])
        image_gray = read_image(image_path, cv2.IMREAD_GRAYSCALE)
        weak_full = read_image(weak_mask_path, cv2.IMREAD_GRAYSCALE)
        weak_train = training_view_weak_mask(image_gray, weak_full, image_size)

        top, bottom, left, right, roi_h, roi_w = get_center_crop_bounds(image_gray, args.crop_size)
        original_crop = crop_full_image(image_gray, top, bottom, left, right)
        weak_full_crop = crop_full_image(weak_full, top, bottom, left, right)
        weak_train_crop = crop_full_image(weak_train, top, bottom, left, right)

        exp_a_prob = predict_full_probability(exp_a_model, image_gray, image_size, device)
        exp_b_prob = predict_full_probability(exp_b_model, image_gray, image_size, device)

        stem = Path(row["image_filename"]).stem
        original_crop_path = output_paths["original"] / f"{stem}_original_crop.png"
        weak_full_crop_path = output_paths["weak_full"] / f"{stem}_weak_full_crop.png"
        weak_train_crop_path = output_paths["weak_train"] / f"{stem}_weak_train_view_crop.png"
        save_png(original_crop, original_crop_path)
        save_png(weak_full_crop, weak_full_crop_path)
        save_png(weak_train_crop, weak_train_crop_path)

        summary_row: Dict[str, str] = {
            "index": str(index),
            "image_id": row["image_id"],
            "sample_id": row.get("sample_id", ""),
            "image_filename": row["image_filename"],
            "crop_top": str(top),
            "crop_bottom": str(bottom),
            "crop_left": str(left),
            "crop_right": str(right),
            "roi_height": str(roi_h),
            "roi_width": str(roi_w),
            "original_crop_path": str(original_crop_path),
            "weak_full_crop_path": str(weak_full_crop_path),
            "weak_train_view_crop_path": str(weak_train_crop_path),
        }

        for threshold in thresholds:
            threshold_key = int(round(threshold * 100))
            exp_a_crop = crop_full_image(threshold_mask(exp_a_prob, threshold), top, bottom, left, right)
            exp_b_crop = crop_full_image(threshold_mask(exp_b_prob, threshold), top, bottom, left, right)
            exp_a_crop_path = output_paths[f"exp_a_{threshold_key:02d}"] / f"{stem}_exp_a_thr_{threshold_key:02d}_crop.png"
            exp_b_crop_path = output_paths[f"exp_b_{threshold_key:02d}"] / f"{stem}_exp_b_thr_{threshold_key:02d}_crop.png"
            panel_path = output_paths["panels"] / f"{stem}_thr_{threshold_key:02d}_compare.png"
            save_png(exp_a_crop, exp_a_crop_path)
            save_png(exp_b_crop, exp_b_crop_path)
            save_png(make_panel(original_crop, weak_full_crop, weak_train_crop, exp_a_crop, exp_b_crop, threshold), panel_path)
            summary_row[f"exp_a_thr_{threshold_key:02d}_crop_path"] = str(exp_a_crop_path)
            summary_row[f"exp_b_thr_{threshold_key:02d}_crop_path"] = str(exp_b_crop_path)
            summary_row[f"panel_thr_{threshold_key:02d}_path"] = str(panel_path)

        summary_rows.append(summary_row)
        print(f"[{index}/{len(rows)}] {row['image_filename']}")

    summary_csv = args.output_dir / "center_crop_compare_manifest.csv"
    with summary_csv.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Saved {len(summary_rows)} center-crop comparisons to {args.output_dir}")


if __name__ == "__main__":
    main()
