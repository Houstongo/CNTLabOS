"""Unified training entry for CNT loss-comparison experiments."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

if __package__ is None or __package__ == "":
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from experiments.cnt_loss_compare.backbone import build_model_from_config
    from experiments.cnt_loss_compare.config import load_config, save_config_snapshot, set_seed
    from experiments.cnt_loss_compare.data import CNTManifestDataset
    from experiments.cnt_loss_compare.losses import build_loss_from_config
    from experiments.cnt_loss_compare.metrics import cldice_metric_from_logits, pixel_metrics_from_logits
else:
    from .backbone import build_model_from_config
    from .config import load_config, save_config_snapshot, set_seed
    from .data import CNTManifestDataset
    from .losses import build_loss_from_config
    from .metrics import cldice_metric_from_logits, pixel_metrics_from_logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CNT loss-comparison experiment.")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def _to_device(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    return {
        **batch,
        "image": batch["image"].to(device),
        "mask": batch["mask"].to(device),
        "gray": batch["gray"].to(device),
    }


def build_loader(
    manifest_path: Path,
    image_size: int,
    batch_size: int,
    shuffle: bool,
    augment: bool,
    num_workers: int,
    input_mode: str,
) -> DataLoader:
    dataset = CNTManifestDataset(
        manifest_path=manifest_path,
        image_size=image_size,
        augment=augment,
        input_mode=input_mode,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=torch.cuda.is_available())


def run_epoch(model, loader, criterion, optimizer, device, is_train: bool) -> Dict[str, float]:
    model.train(is_train)
    metrics_accum: List[Dict[str, float]] = []
    loss_values: List[float] = []
    iterator = tqdm(loader, desc="train" if is_train else "val", leave=False)
    for batch in iterator:
        batch = _to_device(batch, device)
        with torch.set_grad_enabled(is_train):
            logits = model(batch["image"])
            loss, loss_details = criterion(logits, batch["mask"], batch["gray"])
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        pixel = pixel_metrics_from_logits(logits, batch["mask"])
        pixel["cldice"] = cldice_metric_from_logits(logits, batch["mask"])
        pixel.update({f"loss_{k}": v for k, v in loss_details.items()})
        metrics_accum.append(pixel)
        loss_values.append(float(loss.detach().cpu()))
        iterator.set_postfix({"loss": f"{sum(loss_values)/len(loss_values):.4f}", "dice": f"{pixel['dice']:.4f}", "cldice": f"{pixel['cldice']:.4f}"})
    summary: Dict[str, float] = {}
    if not metrics_accum:
        return {"loss": 0.0}
    for key in metrics_accum[0].keys():
        summary[key] = float(sum(item[key] for item in metrics_accum) / len(metrics_accum))
    summary["loss"] = float(sum(loss_values) / len(loss_values))
    return summary


def evaluate(model, loader, criterion, device) -> Dict[str, float]:
    return run_epoch(model, loader, criterion, optimizer=None, device=device, is_train=False)


def save_checkpoint(path: Path, model, optimizer, scheduler, epoch: int, best_metric: float, history: List[Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_metric": best_metric,
            "history": history,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config["seed"]))
    run_root = Path(config["output"]["run_root"]) / f"{config['experiment_name']}_seed{config['seed']}"
    run_root.mkdir(parents=True, exist_ok=True)
    save_config_snapshot(config, run_root / "config_snapshot.yaml")

    device = torch.device(config["training"]["device"] if config["training"]["device"] != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    data_cfg = config["data"]
    model_cfg = config["model"]
    input_mode = str(model_cfg.get("input_mode", "rgb_replicated"))
    train_loader = build_loader(Path(data_cfg["train_manifest"]), int(data_cfg["image_size"]), int(config["training"]["batch_size"]), True, True, int(config["training"]["num_workers"]), input_mode)
    val_loader = build_loader(Path(data_cfg["val_manifest"]), int(data_cfg["image_size"]), int(config["training"]["batch_size"]), False, False, int(config["training"]["num_workers"]), input_mode)
    test_loader = build_loader(Path(data_cfg["test_manifest"]), int(data_cfg["image_size"]), int(config["training"]["batch_size"]), False, False, int(config["training"]["num_workers"]), input_mode)

    model = build_model_from_config(model_cfg, num_classes=1).to(device)
    criterion = build_loss_from_config(config["loss"])
    optimizer = AdamW(model.parameters(), lr=float(config["training"]["learning_rate"]), weight_decay=float(config["training"]["weight_decay"]))
    scheduler = CosineAnnealingLR(optimizer, T_max=int(config["training"]["epochs"]), eta_min=float(config["training"].get("min_learning_rate", 1e-6)))

    history: List[Dict[str, float]] = []
    best_metric = float("-inf")
    best_metric_name = config["training"].get("selection_metric", "cldice")
    best_epoch = 0
    started_at = time.time()

    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, is_train=True)
        val_metrics = run_epoch(model, val_loader, criterion, optimizer, device, is_train=False)
        scheduler.step()

        row = {"epoch": epoch}
        row.update({f"train_{k}": v for k, v in train_metrics.items()})
        row.update({f"val_{k}": v for k, v in val_metrics.items()})
        row["lr"] = float(optimizer.param_groups[0]["lr"])
        history.append(row)

        current_metric = val_metrics.get(best_metric_name, val_metrics.get("dice", 0.0))
        if current_metric > best_metric:
            best_metric = current_metric
            best_epoch = epoch
            save_checkpoint(run_root / "best_model.pth", model, optimizer, scheduler, epoch, best_metric, history)
        save_checkpoint(run_root / "last_model.pth", model, optimizer, scheduler, epoch, best_metric, history)

    best_checkpoint = torch.load(run_root / "best_model.pth", map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    test_metrics = evaluate(model, test_loader, criterion, device)

    with (run_root / "history.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    summary = {
        "experiment_name": config["experiment_name"],
        "seed": config["seed"],
        "best_epoch": best_epoch,
        "best_metric_name": best_metric_name,
        "best_metric": best_metric,
        "test_metrics": test_metrics,
        "elapsed_seconds": round(time.time() - started_at, 2),
    }
    (run_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
