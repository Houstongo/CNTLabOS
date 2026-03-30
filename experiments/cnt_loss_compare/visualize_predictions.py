"""Generate test-set visual comparisons for weak labels and trained experiment checkpoints."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn

if __package__ is None or __package__ == "":
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from experiments.cnt_loss_compare.backbone import build_model_from_config
    from experiments.cnt_loss_compare.config import load_config
    from experiments.cnt_loss_compare.data import prepare_model_input_from_gray_roi
    from src.analysis.feature_extractor import FeatureExtractor
else:
    from .backbone import build_model_from_config
    from .config import load_config
    from .data import prepare_model_input_from_gray_roi
    from src.analysis.feature_extractor import FeatureExtractor


DEFAULT_EXP_A_CONFIG = Path(r"D:\CNTDATA\CNTA_ML_Project\experiments\cnt_loss_compare\configs\exp_cntsegnet.yaml")
DEFAULT_EXP_B_CONFIG = Path(r"D:\CNTDATA\CNTA_ML_Project\experiments\cnt_loss_compare\configs\exp_no_orientation.yaml")
DEFAULT_EXP_A_CHECKPOINT = Path(r"D:\CNTDATA\CNTA_ML_Project\experiments\cnt_loss_compare\runs\exp_cntsegnet_seed42\best_model.pth")
DEFAULT_EXP_B_CHECKPOINT = Path(r"D:\CNTDATA\CNTA_ML_Project\experiments\cnt_loss_compare\runs\exp_no_orientation_seed42\best_model.pth")
DEFAULT_OUTPUT_DIR = Path(r"D:\CNTDATA\CNTA_ML_Project\experiments\cnt_loss_compare\reports\testset_expA_vs_expB_visual_20260325")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate full test-set visual comparisons for Exp-A and Exp-B.")
    parser.add_argument("--exp-a-config", type=Path, default=DEFAULT_EXP_A_CONFIG)
    parser.add_argument("--exp-b-config", type=Path, default=DEFAULT_EXP_B_CONFIG)
    parser.add_argument("--exp-a-checkpoint", type=Path, default=DEFAULT_EXP_A_CHECKPOINT)
    parser.add_argument("--exp-b-checkpoint", type=Path, default=DEFAULT_EXP_B_CHECKPOINT)
    parser.add_argument("--manifest", type=Path, default=None, help="Defaults to Exp-B test manifest.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--panel-size", type=int, default=512)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def read_image(path: Path, flags: int) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), flags)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return image


def load_manifest_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def resize_geometry(height: int, width: int, target_size: int) -> Tuple[int, int, int, int]:
    scale = min(target_size / max(height, 1), target_size / max(width, 1))
    new_h = max(1, int(round(height * scale)))
    new_w = max(1, int(round(width * scale)))
    top = (target_size - new_h) // 2
    left = (target_size - new_w) // 2
    return new_h, new_w, top, left


def resize_and_pad(image: np.ndarray, target_size: int, interpolation: int) -> np.ndarray:
    height, width = image.shape[:2]
    new_h, new_w, top, left = resize_geometry(height, width, target_size)
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    if image.ndim == 2:
        canvas = np.zeros((target_size, target_size), dtype=resized.dtype)
    else:
        canvas = np.zeros((target_size, target_size, image.shape[2]), dtype=resized.dtype)
    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas


def preprocess_roi_for_model(image_gray: np.ndarray, image_size: int, input_mode: str) -> Tuple[torch.Tensor, int, int]:
    roi = FeatureExtractor.extract_roi(image_gray)
    image_tensor = prepare_model_input_from_gray_roi(roi, image_size, input_mode=input_mode)
    return image_tensor.unsqueeze(0), roi.shape[0], roi.shape[1]


def restore_probability_to_full_image(probability_512: np.ndarray, full_shape: Tuple[int, int], roi_shape: Tuple[int, int]) -> np.ndarray:
    roi_h, roi_w = roi_shape
    new_h, new_w, top, left = resize_geometry(roi_h, roi_w, probability_512.shape[0])
    cropped = probability_512[top:top + new_h, left:left + new_w]
    restored_roi = cv2.resize(cropped, (roi_w, roi_h), interpolation=cv2.INTER_LINEAR)
    full_prob = np.zeros(full_shape, dtype=np.float32)
    full_prob[:roi_h, :] = restored_roi
    return full_prob


def build_model(config_path: Path, checkpoint_path: Path, device: torch.device) -> Tuple[nn.Module, Dict[str, object]]:
    config = load_config(config_path)
    model = build_model_from_config(config["model"], num_classes=1).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config


@torch.no_grad()
def predict_full_mask(
    model: nn.Module,
    image_gray: np.ndarray,
    image_size: int,
    input_mode: str,
    device: torch.device,
    threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    image_tensor, roi_h, roi_w = preprocess_roi_for_model(image_gray, image_size, input_mode=input_mode)
    logits = model(image_tensor.to(device))
    probability_512 = torch.sigmoid(logits)[0, 0].detach().cpu().numpy().astype(np.float32)
    full_prob = restore_probability_to_full_image(probability_512, image_gray.shape, (roi_h, roi_w))
    full_mask = (full_prob >= threshold).astype(np.uint8) * 255
    return full_mask, full_prob


def title_panel(image_bgr: np.ndarray, title: str) -> np.ndarray:
    title_h = 42
    canvas = np.full((image_bgr.shape[0] + title_h, image_bgr.shape[1], 3), 255, dtype=np.uint8)
    canvas[title_h:, :] = image_bgr
    cv2.putText(canvas, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2, cv2.LINE_AA)
    return canvas


def mask_to_bgr(mask: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)


def make_panel(original_gray: np.ndarray, weak_mask: np.ndarray, exp_a_mask: np.ndarray, exp_b_mask: np.ndarray, panel_size: int) -> np.ndarray:
    original_bgr = cv2.cvtColor(original_gray, cv2.COLOR_GRAY2BGR)
    panels = [
        title_panel(resize_and_pad(original_bgr, panel_size, cv2.INTER_LINEAR), "Original"),
        title_panel(resize_and_pad(mask_to_bgr(weak_mask), panel_size, cv2.INTER_NEAREST), "WCNTSegNET Weak"),
        title_panel(resize_and_pad(mask_to_bgr(exp_a_mask), panel_size, cv2.INTER_NEAREST), "Exp-A"),
        title_panel(resize_and_pad(mask_to_bgr(exp_b_mask), panel_size, cv2.INTER_NEAREST), "Exp-B"),
    ]
    spacer = np.full((panels[0].shape[0], 16, 3), 235, dtype=np.uint8)
    return np.concatenate([panels[0], spacer, panels[1], spacer, panels[2], spacer, panels[3]], axis=1)


def ensure_dirs(output_dir: Path) -> Dict[str, Path]:
    paths = {
        "panels": output_dir / "comparison_panels",
        "exp_a_masks": output_dir / "exp_a_masks",
        "exp_b_masks": output_dir / "exp_b_masks",
        "exp_a_probs": output_dir / "exp_a_prob_maps",
        "exp_b_probs": output_dir / "exp_b_prob_maps",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def save_probability_map(probability: np.ndarray, output_path: Path) -> None:
    probability_u8 = np.clip(np.round(probability * 255.0), 0, 255).astype(np.uint8)
    cv2.imencode(".png", probability_u8)[1].tofile(str(output_path))


def save_png(image: np.ndarray, output_path: Path) -> None:
    cv2.imencode(".png", image)[1].tofile(str(output_path))


def main() -> None:
    args = parse_args()
    exp_b_config = load_config(args.exp_b_config)
    manifest_path = args.manifest or Path(exp_b_config["data"]["test_manifest"])
    rows = load_manifest_rows(manifest_path)
    if args.limit and args.limit > 0:
        rows = rows[:args.limit]

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    exp_a_model, exp_a_config = build_model(args.exp_a_config, args.exp_a_checkpoint, device)
    exp_b_model, exp_b_config = build_model(args.exp_b_config, args.exp_b_checkpoint, device)
    image_size = int(exp_a_config["data"]["image_size"])
    exp_a_input_mode = str(exp_a_config["model"].get("input_mode", "rgb_replicated"))
    exp_b_input_mode = str(exp_b_config["model"].get("input_mode", "rgb_replicated"))

    output_paths = ensure_dirs(args.output_dir)
    summary_rows: List[Dict[str, str]] = []

    for index, row in enumerate(rows, start=1):
        image_path = Path(row["image_path"])
        weak_mask_path = Path(row["mask_path"])
        image_gray = read_image(image_path, cv2.IMREAD_GRAYSCALE)
        weak_mask = read_image(weak_mask_path, cv2.IMREAD_GRAYSCALE)

        exp_a_mask, exp_a_prob = predict_full_mask(exp_a_model, image_gray, image_size, exp_a_input_mode, device, args.threshold)
        exp_b_mask, exp_b_prob = predict_full_mask(exp_b_model, image_gray, image_size, exp_b_input_mode, device, args.threshold)

        stem = Path(row["image_filename"]).stem
        panel = make_panel(image_gray, weak_mask, exp_a_mask, exp_b_mask, args.panel_size)

        panel_path = output_paths["panels"] / f"{stem}_compare.png"
        exp_a_mask_path = output_paths["exp_a_masks"] / f"{stem}_exp_a_mask.png"
        exp_b_mask_path = output_paths["exp_b_masks"] / f"{stem}_exp_b_mask.png"
        exp_a_prob_path = output_paths["exp_a_probs"] / f"{stem}_exp_a_prob.png"
        exp_b_prob_path = output_paths["exp_b_probs"] / f"{stem}_exp_b_prob.png"

        save_png(panel, panel_path)
        save_png(exp_a_mask, exp_a_mask_path)
        save_png(exp_b_mask, exp_b_mask_path)
        save_probability_map(exp_a_prob, exp_a_prob_path)
        save_probability_map(exp_b_prob, exp_b_prob_path)

        summary_rows.append(
            {
                "index": str(index),
                "image_id": row["image_id"],
                "sample_id": row.get("sample_id", ""),
                "image_filename": row["image_filename"],
                "panel_path": str(panel_path),
                "weak_mask_path": str(weak_mask_path),
                "exp_a_mask_path": str(exp_a_mask_path),
                "exp_b_mask_path": str(exp_b_mask_path),
                "exp_a_prob_path": str(exp_a_prob_path),
                "exp_b_prob_path": str(exp_b_prob_path),
            }
        )
        print(f"[{index}/{len(rows)}] {row['image_filename']}")

    summary_csv = args.output_dir / "comparison_manifest.csv"
    with summary_csv.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Saved {len(summary_rows)} comparisons to {args.output_dir}")


if __name__ == "__main__":
    main()
