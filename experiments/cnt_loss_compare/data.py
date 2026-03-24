"""Manifest-driven CNT dataset with fixed ROI cropping."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.analysis.feature_extractor import FeatureExtractor


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def load_manifest_rows(path: str | Path) -> List[Dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def resize_and_pad(image: np.ndarray, target_size: int, interpolation: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(target_size / max(h, 1), target_size / max(w, 1))
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    if image.ndim == 2:
        canvas = np.zeros((target_size, target_size), dtype=resized.dtype)
    else:
        canvas = np.zeros((target_size, target_size, image.shape[2]), dtype=resized.dtype)
    top = (target_size - new_h) // 2
    left = (target_size - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas


def random_flip(image: torch.Tensor, mask: torch.Tensor, gray: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if torch.rand(1).item() > 0.5:
        image = torch.flip(image, dims=[2])
        mask = torch.flip(mask, dims=[2])
        gray = torch.flip(gray, dims=[2])
    if torch.rand(1).item() > 0.5:
        image = torch.flip(image, dims=[1])
        mask = torch.flip(mask, dims=[1])
        gray = torch.flip(gray, dims=[1])
    return image, mask, gray


class CNTManifestDataset(Dataset):
    """Reads curated experiment manifests and applies the agreed ROI crop."""

    def __init__(self, manifest_path: str | Path, image_size: int = 512, augment: bool = False):
        self.manifest_path = Path(manifest_path)
        self.rows = load_manifest_rows(self.manifest_path)
        self.image_size = int(image_size)
        self.augment = bool(augment)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        row = self.rows[idx]
        image_path = Path(row["image_path"])
        mask_path = Path(row["mask_path"])
        image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imdecode(np.fromfile(str(mask_path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise ValueError(f"Failed to read image/mask pair: {image_path}")

        roi = FeatureExtractor.extract_roi(image)
        roi_h = roi.shape[0]
        mask_roi = mask[:roi_h, :]

        image_rgb = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)
        image_rgb = resize_and_pad(image_rgb, self.image_size, interpolation=cv2.INTER_LINEAR).astype(np.float32)
        mask_roi = resize_and_pad(mask_roi, self.image_size, interpolation=cv2.INTER_NEAREST).astype(np.float32)
        gray_roi = resize_and_pad(roi, self.image_size, interpolation=cv2.INTER_LINEAR).astype(np.float32)

        image_tensor = np.transpose(image_rgb, (2, 0, 1))
        image_tensor = (image_tensor / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        image_tensor = torch.from_numpy(image_tensor.astype(np.float32))
        mask_tensor = torch.from_numpy((mask_roi / 255.0).astype(np.float32)).unsqueeze(0)
        gray_tensor = torch.from_numpy((gray_roi / 255.0).astype(np.float32)).unsqueeze(0)

        if self.augment:
            image_tensor, mask_tensor, gray_tensor = random_flip(image_tensor, mask_tensor, gray_tensor)

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "gray": gray_tensor,
            "image_id": int(row["image_id"]),
            "image_filename": row["image_filename"],
            "sample_id": row.get("sample_id"),
            "magnification": int(float(row["magnification"])) if row.get("magnification") else None,
        }
