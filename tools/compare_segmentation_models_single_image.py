from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.cnt_loss_compare.backbone import CNTSegNet
from experiments.cnt_loss_compare.config import load_config as load_loss_compare_config
from experiments.cnt_loss_compare.data import IMAGENET_MEAN, IMAGENET_STD
from experiments.cnt_loss_compare.visualize_predictions import (
    preprocess_roi_for_model,
    read_image,
    resize_and_pad,
    resize_geometry,
    restore_probability_to_full_image,
)
from experiments.cnt_paper_repro.config import load_config as load_paper_config
from experiments.cnt_paper_repro.model import ResNet34UNet
from experiments.cnt_paper_repro.patching import extract_patch, grid_patch_specs
from src.analysis.feature_extractor import FeatureExtractor


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    config_path: Path
    checkpoint_path: Path


DEFAULT_IMAGE = Path(
    r"C:\Users\clearlove\Desktop\text\No28 200w 15.0nm 50w 0.5nm 600 300 150 600 750 15min 180min bottom 50000-1.png"
)
DEFAULT_OUTPUT_ROOT = Path(r"D:\CNTDATA\CNTA_ML_Project\reports")
DEFAULT_MODELS: List[ModelSpec] = [
    ModelSpec(
        name="CNTSegNet Baseline",
        family="cntsegnet",
        config_path=Path(r"D:\CNTDATA\CNTA_ML_Project\experiments\cnt_loss_compare\configs\exp_cntsegnet.yaml"),
        checkpoint_path=Path(r"D:\CNTDATA\CNTA_ML_Project\experiments\cnt_loss_compare\runs\exp_cntsegnet_seed42\best_model.pth"),
    ),
    ModelSpec(
        name="CNTSegNet Smoke40",
        family="cntsegnet",
        config_path=Path(r"D:\CNTDATA\CNTA_ML_Project\experiments\cnt_loss_compare\configs\exp_cntsegnet_smoke40.yaml"),
        checkpoint_path=Path(r"D:\CNTDATA\CNTA_ML_Project\experiments\cnt_loss_compare\runs\exp_cntsegnet_smoke40_seed42\best_model.pth"),
    ),
    ModelSpec(
        name="Paper Repro Baseline",
        family="paper_repro",
        config_path=Path(r"D:\CNTDATA\CNTA_ML_Project\experiments\cnt_paper_repro\runs\cnt_paper_repro_100000x_center768_seed42\config_snapshot.yaml"),
        checkpoint_path=Path(r"D:\CNTDATA\CNTA_ML_Project\experiments\cnt_paper_repro\runs\cnt_paper_repro_100000x_center768_seed42\best_model.pth"),
    ),
    ModelSpec(
        name="Paper Repro clDice",
        family="paper_repro",
        config_path=Path(r"D:\CNTDATA\CNTA_ML_Project\experiments\cnt_paper_repro\runs\cnt_paper_repro_100000x_center768_cldice_seed42\config_snapshot.yaml"),
        checkpoint_path=Path(r"D:\CNTDATA\CNTA_ML_Project\experiments\cnt_paper_repro\runs\cnt_paper_repro_100000x_center768_cldice_seed42\best_model.pth"),
    ),
    ModelSpec(
        name="Paper Repro clDice+Ridge",
        family="paper_repro",
        config_path=Path(r"D:\CNTDATA\CNTA_ML_Project\experiments\cnt_paper_repro\runs\cnt_paper_repro_100000x_center768_cldice_ridge_seed42\config_snapshot.yaml"),
        checkpoint_path=Path(r"D:\CNTDATA\CNTA_ML_Project\experiments\cnt_paper_repro\runs\cnt_paper_repro_100000x_center768_cldice_ridge_seed42\best_model.pth"),
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare several trained CNT segmentation models on one image.")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--paper-stride", type=int, default=512)
    parser.add_argument("--panel-size", type=int, default=512)
    return parser.parse_args()


def save_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".png", image)[1].tofile(str(path))


def title_panel(image_bgr: np.ndarray, title: str) -> np.ndarray:
    title_h = 42
    canvas = np.full((image_bgr.shape[0] + title_h, image_bgr.shape[1], 3), 255, dtype=np.uint8)
    canvas[title_h:, :] = image_bgr
    cv2.putText(canvas, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2, cv2.LINE_AA)
    return canvas


def mask_to_bgr(mask: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)


def overlay_mask(gray_image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int]) -> np.ndarray:
    base = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    overlay = base.copy()
    overlay[mask > 0] = color
    return cv2.addWeighted(base, 0.60, overlay, 0.40, 0.0)


def build_cntsegnet_model(spec: ModelSpec, device: torch.device) -> tuple[CNTSegNet, dict]:
    config = load_loss_compare_config(spec.config_path)
    model = CNTSegNet(num_classes=1, encoder_weights=config["model"].get("encoder_weights")).to(device)
    checkpoint = torch.load(spec.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config


def predict_cntsegnet_full(spec: ModelSpec, image_gray: np.ndarray, device: torch.device) -> Dict[str, object]:
    model, config = build_cntsegnet_model(spec, device)
    image_size = int(config["data"]["image_size"])
    threshold = float(config.get("inference", {}).get("threshold", 0.5))
    image_tensor, roi_h, roi_w = preprocess_roi_for_model(image_gray, image_size)
    with torch.no_grad():
        logits = model(image_tensor.to(device))
        probability_512 = torch.sigmoid(logits)[0, 0].detach().cpu().numpy().astype(np.float32)
    full_prob = restore_probability_to_full_image(probability_512, image_gray.shape, (roi_h, roi_w))
    full_mask = (full_prob >= threshold).astype(np.uint8) * 255
    return {
        "probability": full_prob,
        "mask": full_mask,
        "threshold": threshold,
        "roi_shape": [int(roi_h), int(roi_w)],
        "image_size": image_size,
    }


def build_paper_model(spec: ModelSpec, device: torch.device) -> tuple[ResNet34UNet, dict]:
    config = load_paper_config(spec.config_path)
    model = ResNet34UNet(
        in_channels=int(config["model"].get("in_channels", 1)),
        num_classes=int(config["model"].get("num_classes", 1)),
        encoder_weights=None,
    ).to(device)
    checkpoint = torch.load(spec.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config


def prepare_paper_roi(image_gray: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    roi = FeatureExtractor.extract_roi(image_gray)
    roi_h, roi_w = roi.shape[:2]
    return roi.astype(np.float32) / 255.0, (roi_h, roi_w)


def stitch_probability_map(prob_sum: np.ndarray, weight_sum: np.ndarray, roi_shape: tuple[int, int], full_shape: tuple[int, int]) -> np.ndarray:
    valid = np.maximum(weight_sum, 1e-6)
    roi_prob = (prob_sum / valid)[: roi_shape[0], : roi_shape[1]]
    full_prob = np.zeros(full_shape, dtype=np.float32)
    full_prob[: roi_shape[0], :] = roi_prob
    return full_prob


def predict_paper_full(spec: ModelSpec, image_gray: np.ndarray, device: torch.device, stride: int) -> Dict[str, object]:
    model, config = build_paper_model(spec, device)
    threshold = float(config["inference"].get("threshold", 0.7))
    patch_size = int(config["data"]["patch_size"])
    roi_image, roi_shape = prepare_paper_roi(image_gray)
    specs = grid_patch_specs(roi_shape[0], roi_shape[1], patch_size=patch_size, stride=stride)
    prob_sum = np.zeros((patch_size if roi_shape[0] <= patch_size else roi_shape[0], patch_size if roi_shape[1] <= patch_size else roi_shape[1]), dtype=np.float32)
    weight_sum = np.zeros_like(prob_sum)

    with torch.no_grad():
        for patch_spec in specs:
            patch = extract_patch(roi_image, patch_spec)
            input_tensor = torch.from_numpy(((patch - 0.5) / 0.5).astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
            patch_prob = torch.sigmoid(model(input_tensor))[0, 0].detach().cpu().numpy().astype(np.float32)
            prob_sum[patch_spec.top : patch_spec.top + patch_spec.height, patch_spec.left : patch_spec.left + patch_spec.width] += patch_prob[: patch_spec.height, : patch_spec.width]
            weight_sum[patch_spec.top : patch_spec.top + patch_spec.height, patch_spec.left : patch_spec.left + patch_spec.width] += 1.0

    full_prob = stitch_probability_map(prob_sum, weight_sum, roi_shape, image_gray.shape)
    full_mask = (full_prob >= threshold).astype(np.uint8) * 255
    return {
        "probability": full_prob,
        "mask": full_mask,
        "threshold": threshold,
        "roi_shape": [int(roi_shape[0]), int(roi_shape[1])],
        "patch_size": patch_size,
        "stride": stride,
        "patch_count": len(specs),
    }


def probability_to_u8(probability: np.ndarray) -> np.ndarray:
    return np.clip(np.round(probability * 255.0), 0, 255).astype(np.uint8)


def make_panel(original_gray: np.ndarray, masks: List[tuple[str, np.ndarray]], panel_size: int) -> np.ndarray:
    panels = [title_panel(resize_and_pad(cv2.cvtColor(original_gray, cv2.COLOR_GRAY2BGR), panel_size, cv2.INTER_LINEAR), "Original")]
    colors = [
        (255, 200, 0),
        (0, 220, 255),
        (255, 128, 255),
        (120, 255, 120),
        (255, 170, 90),
    ]
    for idx, (name, mask) in enumerate(masks):
        overlay = overlay_mask(original_gray, mask, colors[idx % len(colors)])
        panels.append(title_panel(resize_and_pad(overlay, panel_size, cv2.INTER_LINEAR), name))
        panels.append(title_panel(resize_and_pad(mask_to_bgr(mask), panel_size, cv2.INTER_NEAREST), f"{name} Mask"))
    spacer = np.full((panels[0].shape[0], 16, 3), 235, dtype=np.uint8)
    return np.concatenate([item for panel in panels for item in (panel, spacer)][:-1], axis=1)


def run_inference(spec: ModelSpec, image_gray: np.ndarray, device: torch.device, paper_stride: int) -> Dict[str, object]:
    if spec.family == "cntsegnet":
        return predict_cntsegnet_full(spec, image_gray, device)
    if spec.family == "paper_repro":
        return predict_paper_full(spec, image_gray, device, stride=paper_stride)
    raise ValueError(f"Unsupported model family: {spec.family}")


def main() -> None:
    args = parse_args()
    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_root / f"segmentation_model_compare_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    image_gray = read_image(args.image, cv2.IMREAD_GRAYSCALE)
    save_png(out_dir / "original.png", image_gray)

    summary: Dict[str, object] = {
        "image_path": str(args.image),
        "output_dir": str(out_dir),
        "device": str(device),
        "paper_stride": int(args.paper_stride),
        "models": [],
    }

    mask_panels: List[tuple[str, np.ndarray]] = []
    for spec in DEFAULT_MODELS:
        print(f"Running {spec.name} ...")
        result = run_inference(spec, image_gray, device, paper_stride=args.paper_stride)
        mask = result["mask"]
        probability = result["probability"]
        save_png(out_dir / f"{spec.name.replace(' ', '_').lower()}_mask.png", mask)
        save_png(out_dir / f"{spec.name.replace(' ', '_').lower()}_prob.png", probability_to_u8(probability))
        mask_panels.append((spec.name, mask))
        summary["models"].append(
            {
                "name": spec.name,
                "family": spec.family,
                "config_path": str(spec.config_path),
                "checkpoint_path": str(spec.checkpoint_path),
                "mask_path": str(out_dir / f"{spec.name.replace(' ', '_').lower()}_mask.png"),
                "prob_path": str(out_dir / f"{spec.name.replace(' ', '_').lower()}_prob.png"),
                **{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in result.items() if k not in {"mask", "probability"}},
            }
        )

    panel = make_panel(image_gray, mask_panels, panel_size=args.panel_size)
    save_png(out_dir / "comparison_panel.png", panel)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
