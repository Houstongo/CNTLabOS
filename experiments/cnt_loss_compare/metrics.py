"""Evaluation metrics for CNT segmentation experiments."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from skimage.morphology import skeletonize

from .losses import soft_skeletonize


def pixel_metrics_from_logits(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    pred = (torch.sigmoid(logits) > threshold).float()
    pred_flat = pred.view(pred.shape[0], -1)
    target_flat = target.view(target.shape[0], -1)
    tp = (pred_flat * target_flat).sum(dim=1)
    fp = (pred_flat * (1 - target_flat)).sum(dim=1)
    fn = ((1 - pred_flat) * target_flat).sum(dim=1)
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    dice = (2 * tp) / (2 * tp + fp + fn + 1e-7)
    iou = tp / (tp + fp + fn + 1e-7)
    return {
        "precision": float(precision.mean().item()),
        "recall": float(recall.mean().item()),
        "dice": float(dice.mean().item()),
        "iou": float(iou.mean().item()),
    }


def cldice_metric_from_logits(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    pred = (torch.sigmoid(logits) > threshold).float()
    pred_skeleton = soft_skeletonize(pred, iterations=20)
    target_skeleton = soft_skeletonize(target, iterations=20)
    tprec = (pred_skeleton * target).sum(dim=(1, 2, 3)) / (pred_skeleton.sum(dim=(1, 2, 3)) + 1.0)
    tsens = (target_skeleton * pred).sum(dim=(1, 2, 3)) / (target_skeleton.sum(dim=(1, 2, 3)) + 1.0)
    score = (2.0 * tprec * tsens) / (tprec + tsens + 1e-7)
    return float(score.mean().item())


def skeleton_precision_recall_from_arrays(pred_mask: np.ndarray, target_mask: np.ndarray) -> Dict[str, float]:
    pred_skeleton = skeletonize(pred_mask > 0)
    target_skeleton = skeletonize(target_mask > 0)
    tp_precision = np.logical_and(pred_skeleton, target_mask > 0).sum()
    tp_recall = np.logical_and(target_skeleton, pred_mask > 0).sum()
    precision = tp_precision / (pred_skeleton.sum() + 1e-7)
    recall = tp_recall / (target_skeleton.sum() + 1e-7)
    return {"skeleton_precision": float(precision), "skeleton_recall": float(recall)}

