"""Generate threshold-sweep visualizations against original and weak labels."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List

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
        predict_full_mask,
        read_image,
        resize_and_pad,
        save_png,
        save_probability_map,
        title_panel,
    )
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
        predict_full_mask,
        read_image,
        resize_and_pad,
        save_png,
        save_probability_map,
        title_panel,
    )
    from src.analysis.feature_extractor import FeatureExtractor


DEFAULT_OUTPUT_DIR = Path(r"D:\CNTDATA\CNTA_ML_Project\experiments\cnt_loss_compare\reports\testset_expB_threshold_sweep_20260325")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate threshold sweep comparisons for the full test set.")
    parser.add_argument("--model-config", "--exp-b-config", type=Path, default=DEFAULT_EXP_B_CONFIG, dest="model_config")
    parser.add_argument("--model-checkpoint", "--exp-b-checkpoint", type=Path, default=DEFAULT_EXP_B_CHECKPOINT, dest="model_checkpoint")
    parser.add_argument("--model-label", type=str, default="Exp-B")
    parser.add_argument("--manifest", type=Path, default=None, help="Defaults to the selected model config test manifest.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--panel-size", type=int, default=512)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.4, 0.5, 0.6, 0.7])
    return parser.parse_args()


def mask_to_bgr(mask: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)


def training_view_weak_mask(image_gray: np.ndarray, weak_mask_full: np.ndarray, image_size: int) -> np.ndarray:
    roi = FeatureExtractor.extract_roi(image_gray)
    roi_h = roi.shape[0]
    weak_roi = weak_mask_full[:roi_h, :]
    weak_train_view = resize_and_pad(weak_roi, image_size, interpolation=cv2.INTER_NEAREST)

    h, w = weak_roi.shape[:2]
    scale = min(image_size / max(h, 1), image_size / max(w, 1))
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    top = (image_size - new_h) // 2
    left = (image_size - new_w) // 2

    cropped = weak_train_view[top:top + new_h, left:left + new_w]
    restored_roi = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_NEAREST)
    full_mask = np.zeros_like(weak_mask_full)
    full_mask[:roi_h, :] = restored_roi
    return full_mask


def ensure_dirs(output_dir: Path, thresholds: List[float]) -> Dict[str, Path]:
    paths: Dict[str, Path] = {
        "panels": output_dir / "comparison_panels",
        "prob_maps": output_dir / "prediction_prob_maps",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    for threshold in thresholds:
        key = f"thr_{int(round(threshold * 100)):02d}"
        path = output_dir / f"prediction_{key}_masks"
        path.mkdir(parents=True, exist_ok=True)
        paths[key] = path
    return paths


def make_panel(
    original_gray: np.ndarray,
    weak_full: np.ndarray,
    weak_train_view: np.ndarray,
    threshold_masks: Dict[float, np.ndarray],
    panel_size: int,
    model_label: str,
) -> np.ndarray:
    original_bgr = cv2.cvtColor(original_gray, cv2.COLOR_GRAY2BGR)
    panels = [
        title_panel(resize_and_pad(original_bgr, panel_size, cv2.INTER_LINEAR), "Original"),
        title_panel(resize_and_pad(mask_to_bgr(weak_full), panel_size, cv2.INTER_NEAREST), "Weak Full"),
        title_panel(resize_and_pad(mask_to_bgr(weak_train_view), panel_size, cv2.INTER_NEAREST), "Weak Train-View"),
    ]
    for threshold, mask in threshold_masks.items():
        panels.append(
            title_panel(
                resize_and_pad(mask_to_bgr(mask), panel_size, cv2.INTER_NEAREST),
                f"{model_label}@{threshold:.1f}",
            )
        )
    spacer = np.full((panels[0].shape[0], 16, 3), 235, dtype=np.uint8)
    stitched = panels[0]
    for panel in panels[1:]:
        stitched = np.concatenate([stitched, spacer, panel], axis=1)
    return stitched


def main() -> None:
    args = parse_args()
    model_config = load_config(args.model_config)
    manifest_path = args.manifest or Path(model_config["data"]["test_manifest"])
    rows = load_manifest_rows(manifest_path)
    if args.limit and args.limit > 0:
        rows = rows[:args.limit]

    thresholds = [float(value) for value in args.thresholds]
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    prediction_model, config = build_model(args.model_config, args.model_checkpoint, device)
    image_size = int(config["data"]["image_size"])
    output_paths = ensure_dirs(args.output_dir, thresholds)
    summary_rows: List[Dict[str, str]] = []

    for index, row in enumerate(rows, start=1):
        image_path = Path(row["image_path"])
        weak_mask_path = Path(row["mask_path"])
        image_gray = read_image(image_path, cv2.IMREAD_GRAYSCALE)
        weak_full = read_image(weak_mask_path, cv2.IMREAD_GRAYSCALE)
        weak_train = training_view_weak_mask(image_gray, weak_full, image_size)

        threshold_masks: Dict[float, np.ndarray] = {}
        prob_map = None
        input_mode = str(config["model"].get("input_mode", "rgb_replicated"))
        for threshold in thresholds:
            mask, prob = predict_full_mask(prediction_model, image_gray, image_size, input_mode, device, threshold)
            threshold_masks[threshold] = mask
            if prob_map is None:
                prob_map = prob

        stem = Path(row["image_filename"]).stem
        panel = make_panel(image_gray, weak_full, weak_train, threshold_masks, args.panel_size, args.model_label)
        panel_path = output_paths["panels"] / f"{stem}_threshold_sweep.png"
        save_png(panel, panel_path)

        prob_path = output_paths["prob_maps"] / f"{stem}_{args.model_label.lower().replace(' ', '_')}_prob.png"
        save_probability_map(prob_map, prob_path)

        summary_row = {
            "index": str(index),
            "image_id": row["image_id"],
            "sample_id": row.get("sample_id", ""),
            "image_filename": row["image_filename"],
            "model_label": args.model_label,
            "panel_path": str(panel_path),
            "weak_full_path": str(weak_mask_path),
            "weak_train_view_path": "",
            "prediction_prob_path": str(prob_path),
        }

        weak_train_path = args.output_dir / "weak_train_view_masks" / f"{stem}_weak_train_view.png"
        weak_train_path.parent.mkdir(parents=True, exist_ok=True)
        save_png(weak_train, weak_train_path)
        summary_row["weak_train_view_path"] = str(weak_train_path)

        for threshold, mask in threshold_masks.items():
            threshold_key = f"prediction_thr_{int(round(threshold * 100)):02d}_path"
            threshold_dir_key = f"thr_{int(round(threshold * 100)):02d}"
            threshold_path = output_paths[threshold_dir_key] / f"{stem}_{args.model_label.lower().replace(' ', '_')}_thr_{int(round(threshold * 100)):02d}.png"
            save_png(mask, threshold_path)
            summary_row[threshold_key] = str(threshold_path)

        summary_rows.append(summary_row)
        print(f"[{index}/{len(rows)}] {row['image_filename']}")

    summary_csv = args.output_dir / "threshold_sweep_manifest.csv"
    fieldnames = list(summary_rows[0].keys()) if summary_rows else []
    with summary_csv.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Saved {len(summary_rows)} threshold sweeps to {args.output_dir}")


if __name__ == "__main__":
    main()
