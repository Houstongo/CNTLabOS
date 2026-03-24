"""Structural helpers for CNT connectivity-preserving experiments."""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn.functional as F


def _require_nchw(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 4 or x.shape[1] != 1:
        raise ValueError("Expected shape [B, 1, H, W].")
    return x


def _gaussian_kernel1d(sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    radius = max(1, int(math.ceil(3.0 * float(sigma))))
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-(coords ** 2) / max(2.0 * sigma * sigma, 1e-6))
    kernel = kernel / kernel.sum().clamp_min(1e-8)
    return kernel


def _gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    x = _require_nchw(x)
    kernel_1d = _gaussian_kernel1d(sigma, device=x.device, dtype=x.dtype)
    kernel_x = kernel_1d.view(1, 1, 1, -1)
    kernel_y = kernel_1d.view(1, 1, -1, 1)
    pad = kernel_1d.numel() // 2
    x = F.conv2d(x, kernel_x, padding=(0, pad))
    x = F.conv2d(x, kernel_y, padding=(pad, 0))
    return x


def _hessian_second_derivatives(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = _require_nchw(x)
    dxx_kernel = x.new_tensor([[0.0, 0.0, 0.0], [1.0, -2.0, 1.0], [0.0, 0.0, 0.0]]).view(1, 1, 3, 3)
    dyy_kernel = x.new_tensor([[0.0, 1.0, 0.0], [0.0, -2.0, 0.0], [0.0, 1.0, 0.0]]).view(1, 1, 3, 3)
    dxy_kernel = (x.new_tensor([[1.0, 0.0, -1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 1.0]]) / 4.0).view(1, 1, 3, 3)
    dxx = F.conv2d(x, dxx_kernel, padding=1)
    dyy = F.conv2d(x, dyy_kernel, padding=1)
    dxy = F.conv2d(x, dxy_kernel, padding=1)
    return dxx, dyy, dxy


def _normalize_per_image(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mins = x.amin(dim=(-2, -1), keepdim=True)
    maxs = x.amax(dim=(-2, -1), keepdim=True)
    return (x - mins) / (maxs - mins + eps)


def ridge_response_from_gray(
    gray: torch.Tensor,
    sigmas: Iterable[float] = (1.0, 2.0),
    beta: float = 0.5,
    c: float = 0.25,
    eps: float = 1e-8,
) -> torch.Tensor:
    gray = _require_nchw(gray).clamp(0.0, 1.0)
    responses = []
    for sigma in sigmas:
        smoothed = _gaussian_blur(gray, sigma=float(sigma))
        dxx, dyy, dxy = _hessian_second_derivatives(smoothed)
        trace = dxx + dyy
        discrim = ((dxx - dyy) ** 2 + 4.0 * dxy.pow(2)).clamp_min(0.0)
        delta = torch.sqrt(discrim + eps)
        eig1 = 0.5 * (trace + delta)
        eig2 = 0.5 * (trace - delta)

        swap = eig1.abs() > eig2.abs()
        lambda1 = torch.where(swap, eig2, eig1)
        lambda2 = torch.where(swap, eig1, eig2)

        rb = lambda1.abs() / (lambda2.abs() + eps)
        s2 = lambda1.pow(2) + lambda2.pow(2)
        vesselness = torch.exp(-(rb.pow(2)) / max(2.0 * beta * beta, eps))
        vesselness = vesselness * (1.0 - torch.exp(-s2 / max(2.0 * c * c, eps)))
        vesselness = torch.where(lambda2 < 0.0, vesselness, torch.zeros_like(vesselness))
        responses.append(vesselness)

    ridge = torch.stack(responses, dim=0).amax(dim=0)
    return _normalize_per_image(ridge, eps=eps)


def soft_dilate(x: torch.Tensor) -> torch.Tensor:
    x = _require_nchw(x)
    return F.max_pool2d(x, kernel_size=3, stride=1, padding=1)


def soft_erode(x: torch.Tensor) -> torch.Tensor:
    x = _require_nchw(x)
    return -F.max_pool2d(-x, kernel_size=3, stride=1, padding=1)


def soft_open(x: torch.Tensor) -> torch.Tensor:
    return soft_dilate(soft_erode(x))


def soft_skeletonize(x: torch.Tensor, iterations: int = 10) -> torch.Tensor:
    x = _require_nchw(x).clamp(0.0, 1.0)
    opened = soft_open(x)
    skeleton = F.relu(x - opened)
    current = x
    for _ in range(max(int(iterations) - 1, 0)):
        current = soft_erode(current)
        opened = soft_open(current)
        delta = F.relu(current - opened)
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton.clamp(0.0, 1.0)
