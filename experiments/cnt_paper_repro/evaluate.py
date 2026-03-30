"""Evaluation entrypoint for the paper-reproduction model."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

if __package__ is None or __package__ == "":
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from experiments.cnt_paper_repro.config import load_config
    from experiments.cnt_paper_repro.data import CNTPatchDataset
    from experiments.cnt_paper_repro.metrics import pixel_metrics_from_logits
    from experiments.cnt_paper_repro.model import build_model_from_config
else:
    from .config import load_config
    from .data import CNTPatchDataset
    from .metrics import pixel_metrics_from_logits
    from .model import build_model_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate paper-reproduction checkpoint.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    split_manifest = Path(config["data"][f"{args.split}_manifest"])
    run_root = args.checkpoint.parent
    out_dir = run_root / f"eval_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)

    device_name = config["training"].get("device", "auto")
    device = torch.device(device_name if device_name != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    threshold = float(config["inference"].get("threshold", 0.7))

    dataset = CNTPatchDataset(
        manifest_path=split_manifest,
        augment=False,
        normalize_mean=float(config["data"].get("normalize_mean", 0.5)),
        normalize_std=float(config["data"].get("normalize_std", 0.5)),
    )
    loader = DataLoader(dataset, batch_size=int(config["training"]["batch_size"]), shuffle=False, num_workers=int(config["training"].get("num_workers", 0)))

    model = build_model_from_config(
        {
            **config["model"],
            "encoder_weights": None,
        }
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    rows: List[Dict[str, object]] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"eval-{args.split}", leave=False):
            image = batch["image"].to(device)
            mask = batch["mask"].to(device)
            logits = model(image)
            batch_metrics = pixel_metrics_from_logits(logits, mask, threshold=threshold)
            for i in range(image.shape[0]):
                rows.append(
                    {
                        "image_id": int(batch["image_id"][i]),
                        "patch_index": int(batch["patch_index"][i]),
                        "patch_filename": batch["patch_filename"][i],
                        **batch_metrics,
                    }
                )

    if rows:
        with (out_dir / "metrics.csv").open("w", newline="", encoding="utf-8-sig") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    summary = {
        "split": args.split,
        "count": len(rows),
        "mean_dice": float(sum(row["dice"] for row in rows) / len(rows)) if rows else 0.0,
        "mean_iou": float(sum(row["iou"] for row in rows) / len(rows)) if rows else 0.0,
        "mean_precision": float(sum(row["precision"] for row in rows) / len(rows)) if rows else 0.0,
        "mean_recall": float(sum(row["recall"] for row in rows) / len(rows)) if rows else 0.0,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
