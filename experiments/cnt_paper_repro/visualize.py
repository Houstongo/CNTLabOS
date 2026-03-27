"""Visualization helper for paper-reproduction checkpoints."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

if __package__ is None or __package__ == "":
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from experiments.cnt_paper_repro.config import load_config
    from experiments.cnt_paper_repro.model import ResNet34UNet
else:
    from .config import load_config
    from .model import ResNet34UNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize paper-reproduction predictions.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--max-items", type=int, default=12)
    parser.add_argument("--baseline-checkpoint", type=Path, default=None)
    parser.add_argument("--original-expc-checkpoint", type=Path, default=None)
    parser.add_argument("--resumed-expc-checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def read_gray(path: str | Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = cv2.imencode(path.suffix, image)[1]
    path.write_bytes(encoded.tobytes())


def load_model(config: dict, checkpoint_path: Path, device: torch.device) -> ResNet34UNet:
    model = ResNet34UNet(
        in_channels=int(config["model"].get("in_channels", 1)),
        num_classes=int(config["model"].get("num_classes", 1)),
        encoder_weights=None,
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def predict_binary(model: ResNet34UNet, image: np.ndarray, device: torch.device, threshold: float) -> np.ndarray:
    image_tensor = torch.from_numpy(((image - 0.5) / 0.5).astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        prob = torch.sigmoid(model(image_tensor))[0, 0].detach().cpu().numpy()
    return (prob >= threshold).astype(np.uint8) * 255


def _ensure_color(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def make_labeled_panel(items: list[tuple[str, np.ndarray]]) -> np.ndarray:
    header_height = 42
    tile_height = items[0][1].shape[0]
    tile_widths = [image.shape[1] for _, image in items]
    canvas = np.full((header_height + tile_height, sum(tile_widths), 3), 255, dtype=np.uint8)

    offset_x = 0
    for label, image in items:
        color_image = _ensure_color(image)
        width = color_image.shape[1]
        canvas[header_height : header_height + tile_height, offset_x : offset_x + width] = color_image
        cv2.putText(canvas, label, (offset_x + 12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (20, 20, 20), 2, cv2.LINE_AA)
        offset_x += width
    return canvas


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device_name = config["training"].get("device", "auto")
    device = torch.device(device_name if device_name != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    threshold = float(config["inference"].get("threshold", 0.7))

    rows = list(csv.DictReader(args.manifest.open("r", encoding="utf-8-sig")))
    comparison_mode = all(
        checkpoint_path is not None
        for checkpoint_path in (args.baseline_checkpoint, args.original_expc_checkpoint, args.resumed_expc_checkpoint)
    )
    if comparison_mode:
        baseline_model = load_model(config, args.baseline_checkpoint, device)
        original_expc_model = load_model(config, args.original_expc_checkpoint, device)
        resumed_expc_model = load_model(config, args.resumed_expc_checkpoint, device)
        out_dir = args.output_dir or (args.resumed_expc_checkpoint.parent / f"compare_{args.manifest.stem}")
    else:
        if args.checkpoint is None:
            raise ValueError("Single-checkpoint mode requires --checkpoint.")
        model = load_model(config, args.checkpoint, device)
        out_dir = args.output_dir or (args.checkpoint.parent / f"visuals_{args.manifest.stem}")
    out_dir.mkdir(parents=True, exist_ok=True)

    for row in rows[: args.max_items]:
        image = read_gray(row["patch_image_path"]).astype(np.float32) / 255.0
        mask = read_gray(row["patch_mask_path"])
        original = (image * 255.0).astype(np.uint8)
        stem = Path(row["patch_filename"]).stem

        if comparison_mode:
            baseline_pred = predict_binary(baseline_model, image, device, threshold)
            original_expc_pred = predict_binary(original_expc_model, image, device, threshold)
            resumed_expc_pred = predict_binary(resumed_expc_model, image, device, threshold)
            panel = make_labeled_panel(
                [
                    ("Original", original),
                    ("WCNTSegNet", mask),
                    ("Baseline", baseline_pred),
                    ("ExpC-Orig", original_expc_pred),
                    ("ExpC-Resume", resumed_expc_pred),
                ]
            )
            write_image(out_dir / f"{stem}_compare.png", panel)
        else:
            image_tensor = torch.from_numpy(((image - 0.5) / 0.5).astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                prob = torch.sigmoid(model(image_tensor))[0, 0].detach().cpu().numpy()
            pred = (prob >= threshold).astype(np.uint8) * 255
            prob_vis = np.clip(prob * 255.0, 0, 255).astype(np.uint8)
            panel = np.concatenate([original, mask, prob_vis, pred], axis=1)
            write_image(out_dir / f"{stem}_panel.png", panel)


if __name__ == "__main__":
    main()
