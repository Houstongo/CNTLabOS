"""Patch dataset for the paper-reproduction pipeline."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def load_manifest_rows(path: str | Path) -> List[Dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def read_gray(path: str | Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return image


def random_flip(image: torch.Tensor, mask: torch.Tensor, gray: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if torch.rand(1).item() > 0.5:
        image = torch.flip(image, dims=[2])
        mask = torch.flip(mask, dims=[2])
        gray = torch.flip(gray, dims=[2])
    if torch.rand(1).item() > 0.5:
        image = torch.flip(image, dims=[1])
        mask = torch.flip(mask, dims=[1])
        gray = torch.flip(gray, dims=[1])
    return image, mask, gray


class CNTPatchDataset(Dataset):
    """Dataset for aligned CNT image and weak-mask patches."""

    def __init__(self, manifest_path: str | Path, augment: bool = False, normalize_mean: float = 0.5, normalize_std: float = 0.5):
        self.manifest_path = Path(manifest_path)
        self.rows = load_manifest_rows(self.manifest_path)
        self.augment = bool(augment)
        self.normalize_mean = float(normalize_mean)
        self.normalize_std = float(normalize_std)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        row = self.rows[idx]
        image = read_gray(row["patch_image_path"]).astype(np.float32) / 255.0
        mask = read_gray(row["patch_mask_path"]).astype(np.float32) / 255.0

        gray_tensor = torch.from_numpy(image).unsqueeze(0)
        image_tensor = (gray_tensor - self.normalize_mean) / max(self.normalize_std, 1e-6)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)

        if self.augment:
            image_tensor, mask_tensor, gray_tensor = random_flip(image_tensor, mask_tensor, gray_tensor)

        return {
            "image": image_tensor.float(),
            "gray": gray_tensor.float(),
            "mask": mask_tensor.float(),
            "image_id": int(row["image_id"]),
            "patch_index": int(row["patch_index"]),
            "patch_filename": row["patch_filename"],
            "sample_id": row.get("sample_id"),
        }
