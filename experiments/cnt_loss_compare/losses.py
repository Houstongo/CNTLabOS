"""Configurable loss blocks for CNT loss-comparison experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss_from_logits(logits: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    pred = torch.sigmoid(logits).contiguous().view(logits.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)
    intersection = (pred * target).sum(dim=1)
    dice = (2.0 * intersection + smooth) / (pred.sum(dim=1) + target.sum(dim=1) + smooth)
    return 1.0 - dice.mean()


def bce_loss_from_logits(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, target)


def _sobel_kernels(device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    kernel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=device, dtype=dtype)
    kernel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], device=device, dtype=dtype)
    return kernel_x.view(1, 1, 3, 3), kernel_y.view(1, 1, 3, 3)


def _orientation_histogram(x: torch.Tensor, bins: int = 18, sigma: float = 0.25, eps: float = 1e-6) -> torch.Tensor:
    kernel_x, kernel_y = _sobel_kernels(x.device, x.dtype)
    gx = F.conv2d(x, kernel_x, padding=1)
    gy = F.conv2d(x, kernel_y, padding=1)
    magnitude = torch.sqrt(gx.pow(2) + gy.pow(2) + 1e-8)
    # `atan2(0, 0)` has undefined backward behavior and can poison training with NaN gradients
    # on flat regions, so we offset both inputs by a tiny epsilon before computing angles.
    angle = torch.atan2(gy + eps, gx + eps) % math.pi
    centers = torch.linspace(0.0, math.pi, steps=bins + 1, device=x.device, dtype=x.dtype)[:-1].view(1, bins, 1, 1)
    angle = angle.expand(-1, bins, -1, -1)
    magnitude = magnitude.expand(-1, bins, -1, -1)
    diff = torch.abs(angle - centers)
    diff = torch.minimum(diff, math.pi - diff)
    weights = torch.exp(-(diff ** 2) / max(2.0 * sigma * sigma, 1e-6))
    hist = (weights * magnitude).sum(dim=(-1, -2))
    return hist / (hist.sum(dim=1, keepdim=True) + 1e-8)


def orientation_guided_loss(logits: torch.Tensor, gray: torch.Tensor, bins: int = 18, sigma: float = 0.25, eps: float = 1e-6) -> torch.Tensor:
    pred_hist = _orientation_histogram(torch.sigmoid(logits), bins=bins, sigma=sigma, eps=eps)
    gray_hist = _orientation_histogram(gray, bins=bins, sigma=sigma, eps=eps)
    return F.mse_loss(pred_hist, gray_hist)


def _soft_erode(img: torch.Tensor) -> torch.Tensor:
    return -F.max_pool2d(-img, kernel_size=3, stride=1, padding=1)


def _soft_dilate(img: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(img, kernel_size=3, stride=1, padding=1)


def _soft_open(img: torch.Tensor) -> torch.Tensor:
    return _soft_dilate(_soft_erode(img))


def soft_skeletonize(img: torch.Tensor, iterations: int = 20) -> torch.Tensor:
    img = img.clamp(0.0, 1.0)
    skeleton = F.relu(img - _soft_open(img))
    for _ in range(iterations - 1):
        img = _soft_erode(img)
        delta = F.relu(img - _soft_open(img))
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton


def cldice_loss(logits: torch.Tensor, target: torch.Tensor, iterations: int = 20, smooth: float = 1.0) -> torch.Tensor:
    pred = torch.sigmoid(logits)
    pred_skeleton = soft_skeletonize(pred, iterations=iterations)
    target_skeleton = soft_skeletonize(target, iterations=iterations)
    tprec = (pred_skeleton * target).sum(dim=(1, 2, 3)) / (pred_skeleton.sum(dim=(1, 2, 3)) + smooth)
    tsens = (target_skeleton * pred).sum(dim=(1, 2, 3)) / (target_skeleton.sum(dim=(1, 2, 3)) + smooth)
    cl_dice = (2.0 * tprec * tsens) / (tprec + tsens + 1e-8)
    return 1.0 - cl_dice.mean()


def boundary_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = torch.sigmoid(logits)
    kernel_x, kernel_y = _sobel_kernels(pred.device, pred.dtype)
    pred_edge = torch.sqrt(F.conv2d(pred, kernel_x, padding=1).pow(2) + F.conv2d(pred, kernel_y, padding=1).pow(2) + 1e-8)
    target_edge = torch.sqrt(F.conv2d(target, kernel_x, padding=1).pow(2) + F.conv2d(target, kernel_y, padding=1).pow(2) + 1e-8)
    return F.l1_loss(pred_edge, target_edge)


@dataclass
class LossTerm:
    name: str
    weight: float
    params: Dict[str, float]


class CombinedLoss(nn.Module):
    """Composable experiment loss driven by config."""

    def __init__(self, terms: List[LossTerm]):
        super().__init__()
        self.terms = terms

    def forward(self, logits: torch.Tensor, target: torch.Tensor, gray: torch.Tensor) -> tuple[torch.Tensor, Dict[str, float]]:
        total = logits.new_tensor(0.0)
        details: Dict[str, float] = {}
        for term in self.terms:
            if term.name == "dice":
                value = dice_loss_from_logits(logits, target)
            elif term.name == "bce":
                value = bce_loss_from_logits(logits, target)
            elif term.name == "orientation":
                value = orientation_guided_loss(
                    logits,
                    gray,
                    bins=int(term.params.get("bins", 18)),
                    sigma=float(term.params.get("sigma", 0.25)),
                    eps=float(term.params.get("eps", 1e-6)),
                )
            elif term.name == "cldice":
                value = cldice_loss(logits, target, iterations=int(term.params.get("iterations", 20)))
            elif term.name == "aux_boundary":
                value = boundary_loss(logits, target)
            else:
                raise ValueError(f"Unsupported loss term: {term.name}")
            total = total + term.weight * value
            details[term.name] = float(value.detach().cpu())
        details["total"] = float(total.detach().cpu())
        return total, details


def build_loss_from_config(config: Dict[str, object]) -> CombinedLoss:
    terms = []
    for item in config.get("terms", []):
        terms.append(
            LossTerm(
                name=str(item["name"]).lower(),
                weight=float(item.get("weight", 1.0)),
                params=dict(item.get("params", {})),
            )
        )
    return CombinedLoss(terms)
