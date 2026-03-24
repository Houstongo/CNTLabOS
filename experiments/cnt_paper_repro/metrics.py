"""Metrics for paper-reproduction runs."""

from __future__ import annotations

from typing import Dict

import torch


def binarize_logits(logits: torch.Tensor, threshold: float = 0.7) -> torch.Tensor:
    return (torch.sigmoid(logits) >= threshold).float()


def pixel_metrics_from_logits(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.7, eps: float = 1e-8) -> Dict[str, float]:
    pred = binarize_logits(logits, threshold=threshold)
    target = (target >= 0.5).float()

    pred = pred.reshape(pred.shape[0], -1)
    target = target.reshape(target.shape[0], -1)
    tp = (pred * target).sum(dim=1)
    fp = (pred * (1.0 - target)).sum(dim=1)
    fn = ((1.0 - pred) * target).sum(dim=1)

    precision = ((tp + eps) / (tp + fp + eps)).mean()
    recall = ((tp + eps) / (tp + fn + eps)).mean()
    dice = ((2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)).mean()
    iou = ((tp + eps) / (tp + fp + fn + eps)).mean()

    return {
        "dice": float(dice.detach().cpu()),
        "iou": float(iou.detach().cpu()),
        "precision": float(precision.detach().cpu()),
        "recall": float(recall.detach().cpu()),
    }
