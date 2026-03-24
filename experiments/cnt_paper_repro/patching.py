"""Patch extraction for the paper-reproduction dataset."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import cv2
import numpy as np

from src.analysis.feature_extractor import FeatureExtractor


DEFAULT_SPLITS = ("train", "val", "test", "reserve")


@dataclass
class PatchSpec:
    top: int
    left: int
    height: int
    width: int
    patch_size: int


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_manifest_rows(path: str | Path) -> List[Dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def read_gray_image(path: str | Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return image


def write_image(path: str | Path, image: np.ndarray) -> None:
    out_path = Path(path)
    ensure_dir(out_path.parent)
    encoded = cv2.imencode(out_path.suffix, image)[1]
    out_path.write_bytes(encoded.tobytes())


def crop_roi_pair(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    roi = FeatureExtractor.extract_roi(image)
    roi_h = roi.shape[0]
    return image[:roi_h, :], mask[:roi_h, :]


def center_crop_or_pad(image: np.ndarray, patch_size: int) -> tuple[np.ndarray, PatchSpec]:
    h, w = image.shape[:2]
    crop_h = min(h, patch_size)
    crop_w = min(w, patch_size)
    top = max((h - crop_h) // 2, 0)
    left = max((w - crop_w) // 2, 0)
    cropped = image[top:top + crop_h, left:left + crop_w]

    if image.ndim == 2:
        canvas = np.zeros((patch_size, patch_size), dtype=image.dtype)
    else:
        canvas = np.zeros((patch_size, patch_size, image.shape[2]), dtype=image.dtype)

    pad_top = (patch_size - crop_h) // 2
    pad_left = (patch_size - crop_w) // 2
    canvas[pad_top:pad_top + crop_h, pad_left:pad_left + crop_w] = cropped
    return canvas, PatchSpec(top=top, left=left, height=crop_h, width=crop_w, patch_size=patch_size)


def grid_patch_specs(height: int, width: int, patch_size: int, stride: int) -> List[PatchSpec]:
    if stride <= 0:
        raise ValueError("stride must be positive")
    top_positions = [0] if height <= patch_size else list(range(0, height - patch_size + 1, stride))
    left_positions = [0] if width <= patch_size else list(range(0, width - patch_size + 1, stride))
    if height > patch_size and top_positions[-1] != height - patch_size:
        top_positions.append(height - patch_size)
    if width > patch_size and left_positions[-1] != width - patch_size:
        left_positions.append(width - patch_size)
    return [PatchSpec(top=t, left=l, height=min(height, patch_size), width=min(width, patch_size), patch_size=patch_size) for t in top_positions for l in left_positions]


def extract_patch(image: np.ndarray, spec: PatchSpec) -> np.ndarray:
    patch = image[spec.top:spec.top + spec.height, spec.left:spec.left + spec.width]
    if image.ndim == 2:
        canvas = np.zeros((spec.patch_size, spec.patch_size), dtype=image.dtype)
    else:
        canvas = np.zeros((spec.patch_size, spec.patch_size, image.shape[2]), dtype=image.dtype)
    canvas[:spec.height, :spec.width] = patch
    return canvas


def extract_patch_specs(image: np.ndarray, patch_size: int, mode: str = "center", stride: int | None = None) -> List[PatchSpec]:
    mode = mode.lower()
    if mode == "center":
        _patch, spec = center_crop_or_pad(image, patch_size)
        return [spec]
    if mode == "grid":
        return grid_patch_specs(image.shape[0], image.shape[1], patch_size, stride or patch_size)
    raise ValueError(f"Unsupported patch mode: {mode}")


def build_patch_filename(image_filename: str, index: int, spec: PatchSpec) -> str:
    stem = Path(image_filename).stem
    return f"{stem}_patch{index:03d}_r{spec.top}_c{spec.left}_{spec.patch_size}.png"


def create_patch_row(source_row: Dict[str, str], split: str, patch_filename: str, image_path: Path, mask_path: Path, spec: PatchSpec, patch_index: int) -> Dict[str, object]:
    row: Dict[str, object] = dict(source_row)
    row.update(
        {
            "split": split,
            "patch_index": patch_index,
            "patch_filename": patch_filename,
            "patch_image_path": str(image_path),
            "patch_mask_path": str(mask_path),
            "patch_top": spec.top,
            "patch_left": spec.left,
            "patch_height": spec.height,
            "patch_width": spec.width,
            "patch_size": spec.patch_size,
        }
    )
    return row


def build_contact_sheet(image_paths: Sequence[Path], output_path: Path, thumb_size: tuple[int, int] = (192, 192), cols: int = 4) -> None:
    if not image_paths:
        return
    thumbs: List[np.ndarray] = []
    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        thumb = cv2.resize(image, thumb_size, interpolation=cv2.INTER_AREA)
        thumb = cv2.cvtColor(thumb, cv2.COLOR_GRAY2BGR)
        label = image_path.stem[:28]
        cv2.rectangle(thumb, (0, thumb_size[1] - 22), (thumb_size[0], thumb_size[1]), (0, 0, 0), -1)
        cv2.putText(thumb, label, (4, thumb_size[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
        thumbs.append(thumb)
    if not thumbs:
        return
    rows = int(np.ceil(len(thumbs) / cols))
    sheet = np.full((rows * thumb_size[1], cols * thumb_size[0], 3), 255, dtype=np.uint8)
    for idx, thumb in enumerate(thumbs):
        row = idx // cols
        col = idx % cols
        y0 = row * thumb_size[1]
        x0 = col * thumb_size[0]
        sheet[y0:y0 + thumb_size[1], x0:x0 + thumb_size[0]] = thumb
    write_image(output_path, sheet)


def prepare_patch_dataset(
    source_manifest_dir: str | Path,
    output_root: str | Path,
    patch_size: int = 768,
    mode: str = "center",
    stride: int | None = None,
    splits: Sequence[str] = DEFAULT_SPLITS,
) -> Dict[str, object]:
    source_manifest_dir = Path(source_manifest_dir)
    output_root = Path(output_root)
    manifests_dir = output_root / "manifests"
    previews_dir = output_root / "previews"
    ensure_dir(manifests_dir)
    ensure_dir(previews_dir)

    summary: Dict[str, object] = {
        "output_root": str(output_root),
        "source_manifest_dir": str(source_manifest_dir),
        "patch_size": int(patch_size),
        "mode": mode,
        "stride": int(stride) if stride else None,
        "splits": {},
    }

    for split in splits:
        manifest_path = source_manifest_dir / f"{split}_manifest.csv"
        if not manifest_path.exists():
            continue
        rows = load_manifest_rows(manifest_path)
        patch_rows: List[Dict[str, object]] = []
        preview_paths: List[Path] = []
        split_image_dir = output_root / split / "images"
        split_mask_dir = output_root / split / "masks_wcntsegnet"
        ensure_dir(split_image_dir)
        ensure_dir(split_mask_dir)

        for row in rows:
            image = read_gray_image(row["image_path"])
            mask = read_gray_image(row["mask_path"])
            roi_image, roi_mask = crop_roi_pair(image, mask)
            specs = extract_patch_specs(roi_image, patch_size=patch_size, mode=mode, stride=stride)
            for patch_index, spec in enumerate(specs, start=1):
                if mode == "center":
                    patch_image, _ = center_crop_or_pad(roi_image, patch_size)
                    patch_mask, _ = center_crop_or_pad(roi_mask, patch_size)
                else:
                    patch_image = extract_patch(roi_image, spec)
                    patch_mask = extract_patch(roi_mask, spec)

                patch_filename = build_patch_filename(row["image_filename"], patch_index, spec)
                patch_image_path = split_image_dir / patch_filename
                patch_mask_path = split_mask_dir / patch_filename.replace(".png", "_mask.png")
                write_image(patch_image_path, patch_image)
                write_image(patch_mask_path, patch_mask)
                patch_rows.append(
                    create_patch_row(
                        source_row=row,
                        split=split,
                        patch_filename=patch_filename,
                        image_path=patch_image_path,
                        mask_path=patch_mask_path,
                        spec=spec,
                        patch_index=patch_index,
                    )
                )
                if len(preview_paths) < 12:
                    preview_paths.append(patch_image_path)

        write_csv(manifests_dir / f"{split}_patch_manifest.csv", patch_rows)
        build_contact_sheet(preview_paths, previews_dir / f"{split}_patch_contact_sheet.jpg")
        summary["splits"][split] = {
            "source_images": len(rows),
            "patches": len(patch_rows),
        }

    (output_root / "patch_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary
