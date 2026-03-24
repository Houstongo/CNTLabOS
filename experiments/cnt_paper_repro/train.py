"""Staged paper-reproduction training entrypoint."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

if __package__ is None or __package__ == "":
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from experiments.cnt_paper_repro.config import load_config, save_config_snapshot, set_seed
    from experiments.cnt_paper_repro.data import CNTPatchDataset
    from experiments.cnt_paper_repro.losses import compute_phase_loss
    from experiments.cnt_paper_repro.metrics import pixel_metrics_from_logits
    from experiments.cnt_paper_repro.model import ResNet34UNet
else:
    from .config import load_config, save_config_snapshot, set_seed
    from .data import CNTPatchDataset
    from .losses import compute_phase_loss
    from .metrics import pixel_metrics_from_logits
    from .model import ResNet34UNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train paper-reproduction CNT segmentation experiment.")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def build_loader(manifest_path: Path, batch_size: int, shuffle: bool, augment: bool, num_workers: int, normalize_mean: float, normalize_std: float) -> DataLoader:
    dataset = CNTPatchDataset(
        manifest_path=manifest_path,
        augment=augment,
        normalize_mean=normalize_mean,
        normalize_std=normalize_std,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=torch.cuda.is_available())


def _to_device(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    return {
        **batch,
        "image": batch["image"].to(device),
        "gray": batch["gray"].to(device),
        "mask": batch["mask"].to(device),
    }


def save_checkpoint(path: Path, model, optimizer, epoch: int, best_metric: float, history: List[Dict[str, float]], phase_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "phase_name": phase_name,
            "best_metric": best_metric,
            "history": history,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def run_epoch(model, loader, optimizer, device, threshold: float, phase_cfg: Dict[str, object], is_train: bool) -> Dict[str, float]:
    model.train(is_train)
    summaries: List[Dict[str, float]] = []
    iterator = tqdm(loader, desc="train" if is_train else "val", leave=False)

    for batch in iterator:
        batch = _to_device(batch, device)
        with torch.set_grad_enabled(is_train):
            logits = model(batch["image"])
            loss, details = compute_phase_loss(phase_cfg, logits, batch["mask"], batch["gray"])
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        metrics = pixel_metrics_from_logits(logits, batch["mask"], threshold=threshold)
        metrics.update({f"loss_{key}": value for key, value in details.items()})
        summaries.append(metrics)
        iterator.set_postfix({"loss": f"{metrics['loss_total']:.4f}", "dice": f"{metrics['dice']:.4f}"})

    if not summaries:
        return {"loss_total": 0.0}
    keys = summaries[0].keys()
    return {key: float(sum(item[key] for item in summaries) / len(summaries)) for key in keys}


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config["seed"]))

    run_root = Path(config["output"]["run_root"]) / f"{config['experiment_name']}_seed{config['seed']}"
    run_root.mkdir(parents=True, exist_ok=True)
    save_config_snapshot(config, run_root / "config_snapshot.yaml")

    device_name = config["training"].get("device", "auto")
    device = torch.device(device_name if device_name != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    threshold = float(config["inference"].get("threshold", 0.7))
    normalize_mean = float(config["data"].get("normalize_mean", 0.5))
    normalize_std = float(config["data"].get("normalize_std", 0.5))

    train_loader = build_loader(
        Path(config["data"]["train_manifest"]),
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        augment=bool(config["data"].get("augment", False)),
        num_workers=int(config["training"].get("num_workers", 0)),
        normalize_mean=normalize_mean,
        normalize_std=normalize_std,
    )
    val_loader = build_loader(
        Path(config["data"]["val_manifest"]),
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        augment=False,
        num_workers=int(config["training"].get("num_workers", 0)),
        normalize_mean=normalize_mean,
        normalize_std=normalize_std,
    )
    test_loader = build_loader(
        Path(config["data"]["test_manifest"]),
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        augment=False,
        num_workers=int(config["training"].get("num_workers", 0)),
        normalize_mean=normalize_mean,
        normalize_std=normalize_std,
    )

    model = ResNet34UNet(
        in_channels=int(config["model"].get("in_channels", 1)),
        num_classes=int(config["model"].get("num_classes", 1)),
        encoder_weights=config["model"].get("encoder_weights"),
    ).to(device)

    optimizer = Adam(
        model.parameters(),
        lr=float(config["training"].get("learning_rate", 5e-4)),
        weight_decay=float(config["training"].get("weight_decay", 0.0)),
    )

    history: List[Dict[str, float]] = []
    best_metric = float("-inf")
    best_epoch = 0
    best_phase = ""
    epoch_global = 0
    started_at = time.time()

    for phase_cfg in config["training"]["phases"]:
        phase_name = str(phase_cfg["name"])
        for phase_epoch in range(1, int(phase_cfg["epochs"]) + 1):
            epoch_global += 1
            train_metrics = run_epoch(model, train_loader, optimizer, device, threshold, phase_cfg, is_train=True)
            val_metrics = run_epoch(model, val_loader, optimizer, device, threshold, phase_cfg, is_train=False)

            row = {
                "epoch": epoch_global,
                "phase": phase_name,
                "phase_epoch": phase_epoch,
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
            row.update({f"train_{key}": value for key, value in train_metrics.items()})
            row.update({f"val_{key}": value for key, value in val_metrics.items()})
            history.append(row)

            current_metric = float(val_metrics.get(config["training"].get("selection_metric", "dice"), 0.0))
            if current_metric > best_metric:
                best_metric = current_metric
                best_epoch = epoch_global
                best_phase = phase_name
                save_checkpoint(run_root / "best_model.pth", model, optimizer, epoch_global, best_metric, history, phase_name)

            save_checkpoint(run_root / "last_model.pth", model, optimizer, epoch_global, best_metric, history, phase_name)

    best_checkpoint = torch.load(run_root / "best_model.pth", map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    final_phase_cfg = config["training"]["phases"][-1]
    test_metrics = run_epoch(model, test_loader, optimizer=None, device=device, threshold=threshold, phase_cfg=final_phase_cfg, is_train=False)

    if history:
        with (run_root / "history.csv").open("w", newline="", encoding="utf-8-sig") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)

    summary = {
        "experiment_name": config["experiment_name"],
        "seed": config["seed"],
        "device": str(device),
        "best_epoch": best_epoch,
        "best_phase": best_phase,
        "best_metric_name": config["training"].get("selection_metric", "dice"),
        "best_metric": best_metric,
        "test_metrics": test_metrics,
        "elapsed_seconds": round(time.time() - started_at, 2),
    }
    (run_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
