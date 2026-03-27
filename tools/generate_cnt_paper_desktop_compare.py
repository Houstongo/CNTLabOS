"""Generate cropped nine-tile comparison panels for desktop CNT SEM images."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.cnt_paper_repro.config import load_config
from experiments.cnt_paper_repro.model import ResNet34UNet
from experiments.cnt_paper_repro.patching import center_crop_or_pad
from tools.generate_wcntsegnet_weak_labels import generate_wcntsegnet_mask

PAPER_100000_MANIFEST_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "cnt_paper_repro"
    / "datasets"
    / "zzy_mid_100000_patch768_center_paper_v1"
    / "manifests"
)
PAPER_50000_MANIFEST_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "cnt_paper_repro"
    / "datasets"
    / "zzy_mid_50000_train34_test76_paper_stage_v1"
    / "manifests"
)
PAPER_50000_DATASET_ROOT = (
    PROJECT_ROOT
    / "experiments"
    / "cnt_paper_repro"
    / "datasets"
    / "zzy_mid_50000_train34_test76_paper_stage_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate desktop comparison panels for cnt_paper_repro checkpoints.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, action="append", required=True)
    parser.add_argument("--model", action="append", nargs=2, metavar=("LABEL", "CHECKPOINT"), required=True)
    parser.add_argument("--detail-size", type=int, default=384)
    parser.add_argument("--gutter", type=int, default=24)
    parser.add_argument("--outer-pad", type=int, default=24)
    parser.add_argument("--cols", type=int, default=3)
    return parser.parse_args()


def read_gray(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return image


def write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = cv2.imencode(".png", image)[1]
    path.write_bytes(encoded.tobytes())


def ensure_color(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def build_weak_label_lookup() -> dict[str, str]:
    lookup: dict[str, str] = {}

    for manifest_name in ("train_patch_manifest.csv", "val_patch_manifest.csv", "test_patch_manifest.csv", "reserve_patch_manifest.csv"):
        manifest_path = PAPER_100000_MANIFEST_DIR / manifest_name
        if not manifest_path.exists():
            continue
        for row in load_csv_rows(manifest_path):
            key = Path(row["file_path"]).name.lower()
            lookup.setdefault(key, row["mask_path"])

    for manifest_name in ("train_manifest.csv", "test_manifest.csv", "reserve_manifest.csv", "all_candidates.csv"):
        manifest_path = PAPER_50000_MANIFEST_DIR / manifest_name
        if not manifest_path.exists():
            continue
        for row in load_csv_rows(manifest_path):
            key = Path(row["file_path"]).name.lower()
            split = row.get("split", "train")
            image_id = int(row["image_id"])
            stem = Path(row["file_path"]).stem
            mask_name = f"{image_id:05d}_{stem}_mask.png"
            mask_path = PAPER_50000_DATASET_ROOT / split / "masks_wcntsegnet" / mask_name
            lookup.setdefault(key, str(mask_path))

    return lookup


def load_model(config: dict, checkpoint_path: Path, device: torch.device) -> ResNet34UNet:
    model = ResNet34UNet(
        in_channels=int(config["model"].get("in_channels", 1)),
        num_classes=int(config["model"].get("num_classes", 1)),
        encoder_weights=None,
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def predict_mask(model: ResNet34UNet, patch_u8: np.ndarray, device: torch.device, threshold: float) -> np.ndarray:
    image = patch_u8.astype(np.float32) / 255.0
    tensor = torch.from_numpy(((image - 0.5) / 0.5).astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        prob = torch.sigmoid(model(tensor))[0, 0].detach().cpu().numpy()
    return (prob >= threshold).astype(np.uint8) * 255


def center_detail_crop(image: np.ndarray, detail_size: int) -> np.ndarray:
    h, w = image.shape[:2]
    crop_h = min(h, detail_size)
    crop_w = min(w, detail_size)
    top = max((h - crop_h) // 2, 0)
    left = max((w - crop_w) // 2, 0)
    cropped = image[top : top + crop_h, left : left + crop_w]
    if crop_h == detail_size and crop_w == detail_size:
        return cropped.copy()
    padded, _ = center_crop_or_pad(cropped, detail_size)
    return padded


def prepare_weak_tile(image_path: Path, weak_lookup: dict[str, str], patch_size: int, detail_size: int) -> np.ndarray:
    weak_path_str = weak_lookup.get(image_path.name.lower())
    if weak_path_str:
        weak_path = Path(weak_path_str)
        if weak_path.exists():
            weak_mask = read_gray(weak_path)
            weak_patch, _ = center_crop_or_pad(weak_mask, patch_size)
            weak_tile = center_detail_crop(weak_patch, detail_size)
            return ((weak_tile > 127).astype(np.uint8) * 255), "manifest"
    return np.full((detail_size, detail_size), 235, dtype=np.uint8), "missing"


def infer_magnification(image_path: Path) -> int | None:
    match = re.search(r"(?<!\d)(50000|100000)(?!\d)", image_path.stem)
    if not match:
        return None
    return int(match.group(1))


def build_generated_weak_tile(image_gray: np.ndarray, image_path: Path, patch_size: int, detail_size: int) -> np.ndarray:
    _roi_mask, full_mask, _density_roi = generate_wcntsegnet_mask(
        img_gray=image_gray,
        magnification=infer_magnification(image_path),
    )
    weak_patch, _ = center_crop_or_pad((full_mask > 0).astype(np.uint8) * 255, patch_size)
    return center_detail_crop(weak_patch, detail_size)


def draw_grid(
    title: str,
    items: list[tuple[str, np.ndarray]],
    cols: int,
    gutter: int,
    outer_pad: int,
    title_h: int = 56,
    label_h: int = 40,
) -> np.ndarray:
    tile_h, tile_w = items[0][1].shape[:2]
    rows = (len(items) + cols - 1) // cols
    canvas_w = outer_pad * 2 + cols * tile_w + (cols - 1) * gutter
    canvas_h = outer_pad * 2 + title_h + rows * label_h + rows * tile_h + (rows - 1) * gutter
    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)
    cv2.putText(canvas, title, (outer_pad, outer_pad + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (15, 15, 15), 2, cv2.LINE_AA)

    top_y = outer_pad + title_h
    for idx, (label, image) in enumerate(items):
        r = idx // cols
        c = idx % cols
        x = outer_pad + c * (tile_w + gutter)
        y = top_y + r * (label_h + tile_h + gutter)
        cv2.putText(canvas, label, (x + 10, y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2, cv2.LINE_AA)
        color = ensure_color(image)
        canvas[y + label_h : y + label_h + tile_h, x : x + tile_w] = color
    return canvas


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    threshold = float(config["inference"].get("threshold", 0.7))
    patch_size = int(config["data"].get("patch_size", 768))
    device_name = config["training"].get("device", "auto")
    device = torch.device(device_name if device_name != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

    weak_lookup = build_weak_label_lookup()
    models = [(label, load_model(config, Path(checkpoint), device)) for label, checkpoint in args.model]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    output_files: list[str] = []
    weak_stats = {"manifest": 0, "generated": 0, "missing": 0}
    for source_dir in args.source_dir:
        image_paths = sorted(
            [
                path
                for path in source_dir.iterdir()
                if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
            ]
        )
        for image_path in image_paths:
            image_gray = read_gray(image_path)
            patch, _ = center_crop_or_pad(image_gray, patch_size)
            detail_patch = center_detail_crop(patch, args.detail_size)
            weak_tile, weak_source = prepare_weak_tile(image_path, weak_lookup, patch_size, args.detail_size)
            if weak_source == "missing":
                weak_tile = build_generated_weak_tile(image_gray, image_path, patch_size, args.detail_size)
                weak_source = "generated"
            weak_stats[weak_source] += 1
            items = [("Original", detail_patch), ("WeakLabel", weak_tile)]
            items.extend(
                (label, center_detail_crop(predict_mask(model, patch, device, threshold), args.detail_size))
                for label, model in models
            )
            panel = draw_grid(
                f"{source_dir.name} | {image_path.name} | center {args.detail_size}x{args.detail_size}",
                items,
                args.cols,
                args.gutter,
                args.outer_pad,
            )
            out_name = f"{source_dir.name}__{image_path.stem}_compare.png"
            write_png(args.output_dir / out_name, panel)
            output_files.append(out_name)

    summary = {
        "device": str(device),
        "threshold": threshold,
        "patch_size": patch_size,
        "detail_size": args.detail_size,
        "gutter": args.gutter,
        "outer_pad": args.outer_pad,
        "cols": args.cols,
        "weak_label_sources": weak_stats,
        "sources": [str(path) for path in args.source_dir],
        "models": [{"label": label, "checkpoint": str(path)} for label, path in args.model],
        "count": len(output_files),
        "files": output_files,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
