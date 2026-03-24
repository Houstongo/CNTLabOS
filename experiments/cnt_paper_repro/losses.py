"""Paper-style loss functions for CNTSegNet reproduction."""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch

from .structural import ridge_response_from_gray, soft_skeletonize


_FFT_GRID_CACHE: Dict[Tuple[int, int, int, str, int], Tuple[torch.Tensor, torch.Tensor]] = {}


def dice_loss_from_logits(logits: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    pred = torch.sigmoid(logits).reshape(logits.shape[0], -1)
    target = target.reshape(target.shape[0], -1)
    intersection = (pred * target).sum(dim=1)
    dice = (2.0 * intersection + smooth) / (pred.sum(dim=1) + target.sum(dim=1) + smooth)
    return 1.0 - dice.mean()


def _fft_grid_indices(height: int, width: int, bins: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    key = (height, width, bins, device.type, device.index or -1)
    cached = _FFT_GRID_CACHE.get(key)
    if cached is not None:
        return cached

    ys, xs = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    cy = (height - 1) / 2.0
    cx = (width - 1) / 2.0
    ys = ys - cy
    xs = xs - cx
    radius = torch.sqrt(xs.pow(2) + ys.pow(2))
    theta = torch.atan2(ys, xs) % (2.0 * math.pi)
    bin_indices = torch.clamp((theta / (2.0 * math.pi) * bins).long(), min=0, max=bins - 1).view(1, -1)
    valid = (radius > 1.0).view(1, -1)
    _FFT_GRID_CACHE[key] = (bin_indices, valid)
    return _FFT_GRID_CACHE[key]


def orientation_histogram_from_map(x: torch.Tensor, bins: int = 360, use_power_spectrum: bool = True, eps: float = 1e-8) -> torch.Tensor:
    if x.ndim != 4 or x.shape[1] != 1:
        raise ValueError("Expected shape [B, 1, H, W] for orientation histogram input.")

    centered = x - x.mean(dim=(-2, -1), keepdim=True)
    fft_map = torch.fft.fftshift(torch.fft.fft2(centered.squeeze(1), norm="ortho"), dim=(-2, -1))
    magnitude = fft_map.abs()
    if use_power_spectrum:
        magnitude = magnitude.pow(2)

    bsz, height, width = magnitude.shape
    flat = magnitude.reshape(bsz, -1)
    bin_indices, valid = _fft_grid_indices(height, width, bins, magnitude.device)
    weighted = flat * valid.to(flat.dtype)
    hist = weighted.new_zeros((bsz, bins))
    hist.scatter_add_(1, bin_indices.expand(bsz, -1), weighted)
    return hist / (hist.sum(dim=1, keepdim=True) + eps)


def orientation_mse_loss_from_logits(logits: torch.Tensor, gray: torch.Tensor, bins: int = 360, use_power_spectrum: bool = True) -> torch.Tensor:
    pred_hist = orientation_histogram_from_map(torch.sigmoid(logits), bins=bins, use_power_spectrum=use_power_spectrum)
    gray_hist = orientation_histogram_from_map(gray, bins=bins, use_power_spectrum=use_power_spectrum)
    return ((pred_hist - gray_hist) ** 2).sum(dim=1).mean()


def cldice_loss_from_probs(probs: torch.Tensor, target: torch.Tensor, iterations: int = 10, eps: float = 1e-8) -> torch.Tensor:
    probs = probs.clamp(0.0, 1.0)
    target = target.clamp(0.0, 1.0)
    pred_skeleton = soft_skeletonize(probs, iterations=iterations)
    target_skeleton = soft_skeletonize(target, iterations=iterations)

    tprec = (pred_skeleton * target).sum(dim=(-2, -1, -3)) / (pred_skeleton.sum(dim=(-2, -1, -3)) + eps)
    trec = (target_skeleton * probs).sum(dim=(-2, -1, -3)) / (target_skeleton.sum(dim=(-2, -1, -3)) + eps)
    cl_dice = (2.0 * tprec * trec) / (tprec + trec + eps)
    return 1.0 - cl_dice.mean()


def ridge_aux_loss_from_logits(logits: torch.Tensor, gray: torch.Tensor, iterations: int = 10, eps: float = 1e-8) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    pred_skeleton = soft_skeletonize(probs, iterations=iterations)
    ridge = ridge_response_from_gray(gray)
    support = (pred_skeleton * ridge).sum(dim=(-2, -1, -3)) / (pred_skeleton.sum(dim=(-2, -1, -3)) + eps)
    return 1.0 - support.mean()


def compute_phase_loss(phase_cfg: Dict[str, object], logits: torch.Tensor, target: torch.Tensor, gray: torch.Tensor) -> tuple[torch.Tensor, Dict[str, float]]:
    dice = dice_loss_from_logits(logits, target)
    dice_weight = float(phase_cfg.get("dice_weight", 1.0))
    orientation_weight = float(phase_cfg.get("orientation_weight", 0.0))
    lambda_cl = float(phase_cfg.get("lambda_cl", 0.0))
    lambda_ridge = float(phase_cfg.get("lambda_ridge", 0.0))
    bins = int(phase_cfg.get("orientation_bins", 360))
    use_power_spectrum = bool(phase_cfg.get("orientation_use_power_spectrum", True))
    structural_iterations = int(phase_cfg.get("structural_iterations", 10))

    total = dice_weight * dice
    details = {
        "dice": float(dice.detach().cpu()),
        "orientation": 0.0,
        "cldice": 0.0,
        "ridge": 0.0,
    }

    if orientation_weight > 0.0:
        orientation = orientation_mse_loss_from_logits(logits, gray, bins=bins, use_power_spectrum=use_power_spectrum)
        total = total + orientation_weight * orientation
        details["orientation"] = float(orientation.detach().cpu())

    if lambda_cl > 0.0:
        cldice = cldice_loss_from_probs(torch.sigmoid(logits), target, iterations=structural_iterations)
        total = total + lambda_cl * cldice
        details["cldice"] = float(cldice.detach().cpu())

    if lambda_ridge > 0.0:
        ridge = ridge_aux_loss_from_logits(logits, gray, iterations=structural_iterations)
        total = total + lambda_ridge * ridge
        details["ridge"] = float(ridge.detach().cpu())

    details["total"] = float(total.detach().cpu())
    return total, details
