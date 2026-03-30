"""Standalone evaluation entry for CNT loss-comparison experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

if __package__ is None or __package__ == "":
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from experiments.cnt_loss_compare.backbone import build_model_from_config
    from experiments.cnt_loss_compare.config import load_config
    from experiments.cnt_loss_compare.data import CNTManifestDataset
    from experiments.cnt_loss_compare.losses import build_loss_from_config
    from experiments.cnt_loss_compare.train import evaluate
else:
    from .backbone import build_model_from_config
    from .config import load_config
    from .data import CNTManifestDataset
    from .losses import build_loss_from_config
    from .train import evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained CNT loss-comparison checkpoint.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    manifest = args.manifest or Path(config["data"]["test_manifest"])
    device = torch.device(config["training"]["device"] if config["training"]["device"] != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    dataset = CNTManifestDataset(
        manifest_path=manifest,
        image_size=int(config["data"]["image_size"]),
        augment=False,
        input_mode=str(config["model"].get("input_mode", "rgb_replicated")),
    )
    loader = DataLoader(dataset, batch_size=int(config["training"]["batch_size"]), shuffle=False, num_workers=int(config["training"]["num_workers"]))
    model = build_model_from_config(config["model"], num_classes=1).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    criterion = build_loss_from_config(config["loss"])
    metrics = evaluate(model, loader, criterion, device)
    print(json.dumps({"manifest": str(manifest), "metrics": metrics}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
