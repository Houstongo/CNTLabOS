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
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--max-items", type=int, default=12)
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


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device_name = config["training"].get("device", "auto")
    device = torch.device(device_name if device_name != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    threshold = float(config["inference"].get("threshold", 0.7))

    model = ResNet34UNet(
        in_channels=int(config["model"].get("in_channels", 1)),
        num_classes=int(config["model"].get("num_classes", 1)),
        encoder_weights=None,
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    rows = list(csv.DictReader(args.manifest.open("r", encoding="utf-8-sig")))
    out_dir = args.checkpoint.parent / f"visuals_{args.manifest.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for row in rows[: args.max_items]:
        image = read_gray(row["patch_image_path"]).astype(np.float32) / 255.0
        mask = read_gray(row["patch_mask_path"])
        image_tensor = torch.from_numpy(((image - 0.5) / 0.5).astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            prob = torch.sigmoid(model(image_tensor))[0, 0].detach().cpu().numpy()
        pred = (prob >= threshold).astype(np.uint8) * 255

        original = (image * 255.0).astype(np.uint8)
        prob_vis = np.clip(prob * 255.0, 0, 255).astype(np.uint8)
        panel = np.concatenate([original, mask, prob_vis, pred], axis=1)
        write_image(out_dir / f"{Path(row['patch_filename']).stem}_panel.png", panel)


if __name__ == "__main__":
    main()
