"""Staged paper-reproduction training entrypoint."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
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
    from experiments.cnt_paper_repro.model import build_model_from_config
else:
    from .config import load_config, save_config_snapshot, set_seed
    from .data import CNTPatchDataset
    from .losses import compute_phase_loss
    from .metrics import pixel_metrics_from_logits
    from .model import build_model_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train paper-reproduction CNT segmentation experiment.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path, default=None, help="Resume from an existing last-model checkpoint.")
    parser.add_argument(
        "--extra-final-phase-epochs",
        type=int,
        default=0,
        help="Append this many epochs to the configured final phase when resuming.",
    )
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


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    epoch: int,
    best_metric: float,
    history: List[Dict[str, float]],
    phase_name: str,
    phase_epoch: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "phase_name": phase_name,
            "phase_epoch": phase_epoch,
            "best_metric": best_metric,
            "history": history,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def _best_state_from_history(history: List[Dict[str, float]], selection_metric: str) -> tuple[float, int, str]:
    if not history:
        return float("-inf"), 0, ""
    metric_key = f"val_{selection_metric}"
    best_row = max(history, key=lambda row: float(row.get(metric_key, 0.0)))
    return float(best_row.get(metric_key, 0.0)), int(best_row.get("epoch", 0)), str(best_row.get("phase", ""))


def _snapshot_pre_resume_artifacts(run_root: Path) -> None:
    snapshot_map = {
        "best_model.pth": "pre_resume_best_model.pth",
        "last_model.pth": "pre_resume_last_model.pth",
        "history.csv": "pre_resume_history.csv",
        "summary.json": "pre_resume_summary.json",
    }
    for source_name, snapshot_name in snapshot_map.items():
        source_path = run_root / source_name
        snapshot_path = run_root / snapshot_name
        if source_path.exists() and not snapshot_path.exists():
            shutil.copy2(source_path, snapshot_path)


def _write_history_csv(run_root: Path, history: List[Dict[str, float]]) -> None:
    if not history:
        return
    with (run_root / "history.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def _phase_plan_for_resume(
    phases: List[Dict[str, object]],
    history: List[Dict[str, float]],
    extra_final_phase_epochs: int,
) -> List[tuple[Dict[str, object], int, int]]:
    if not history:
        return []

    last_row = history[-1]
    checkpoint_phase_name = str(last_row.get("phase", ""))
    checkpoint_phase_epoch = int(last_row.get("phase_epoch", 0))
    final_phase_name = str(phases[-1]["name"])

    phase_names = [str(phase["name"]) for phase in phases]
    if checkpoint_phase_name not in phase_names:
        raise ValueError(f"Checkpoint phase '{checkpoint_phase_name}' not found in configured phases: {phase_names}")

    plan: List[tuple[Dict[str, object], int, int]] = []
    checkpoint_phase_index = phase_names.index(checkpoint_phase_name)

    for index, phase_cfg in enumerate(phases[checkpoint_phase_index:], start=checkpoint_phase_index):
        phase_name = str(phase_cfg["name"])
        total_epochs = int(phase_cfg["epochs"])
        if phase_name == final_phase_name:
            total_epochs += extra_final_phase_epochs

        start_epoch = 1
        if index == checkpoint_phase_index:
            start_epoch = checkpoint_phase_epoch + 1

        if start_epoch <= total_epochs:
            plan.append((dict(phase_cfg), start_epoch, total_epochs))

    return plan


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

    model = build_model_from_config(config["model"]).to(device)

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
    resumed_from = None
    resume_epoch = 0
    original_history_length = 0
    extra_final_phase_epochs = max(0, int(args.extra_final_phase_epochs))
    started_at = time.time()
    selection_metric = str(config["training"].get("selection_metric", "dice"))

    if args.resume is not None:
        _snapshot_pre_resume_artifacts(run_root)
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        history = list(checkpoint.get("history", []))
        original_history_length = len(history)
        epoch_global = int(checkpoint.get("epoch", original_history_length))
        resumed_from = str(args.resume)
        resume_epoch = epoch_global
        best_metric, best_epoch, best_phase = _best_state_from_history(history, selection_metric)
        best_metric = float(checkpoint.get("best_metric", best_metric))

    phase_plan: List[tuple[Dict[str, object], int, int]]
    if args.resume is not None:
        phase_plan = _phase_plan_for_resume(
            phases=list(config["training"]["phases"]),
            history=history,
            extra_final_phase_epochs=extra_final_phase_epochs,
        )
    else:
        phase_plan = [(dict(phase_cfg), 1, int(phase_cfg["epochs"])) for phase_cfg in config["training"]["phases"]]

    for phase_cfg, start_phase_epoch, total_phase_epochs in phase_plan:
        phase_name = str(phase_cfg["name"])
        for phase_epoch in range(start_phase_epoch, total_phase_epochs + 1):
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

            current_metric = float(val_metrics.get(selection_metric, 0.0))
            if current_metric > best_metric:
                best_metric = current_metric
                best_epoch = epoch_global
                best_phase = phase_name
                save_checkpoint(run_root / "best_model.pth", model, optimizer, epoch_global, best_metric, history, phase_name, phase_epoch)

            save_checkpoint(run_root / "last_model.pth", model, optimizer, epoch_global, best_metric, history, phase_name, phase_epoch)
            _write_history_csv(run_root, history)

    best_checkpoint = torch.load(run_root / "best_model.pth", map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    final_phase_cfg = config["training"]["phases"][-1]
    test_metrics = run_epoch(model, test_loader, optimizer=None, device=device, threshold=threshold, phase_cfg=final_phase_cfg, is_train=False)

    _write_history_csv(run_root, history)

    summary = {
        "experiment_name": config["experiment_name"],
        "seed": config["seed"],
        "device": str(device),
        "best_epoch": best_epoch,
        "best_phase": best_phase,
        "best_metric_name": selection_metric,
        "best_metric": best_metric,
        "test_metrics": test_metrics,
        "resumed_from": resumed_from,
        "resume_epoch": resume_epoch,
        "original_history_length": original_history_length,
        "extra_final_phase_epochs": extra_final_phase_epochs,
        "elapsed_seconds": round(time.time() - started_at, 2),
    }
    (run_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
