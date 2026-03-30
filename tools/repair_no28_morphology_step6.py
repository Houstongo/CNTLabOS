from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = PROJECT_ROOT / "reports" / "no28_morphology_feature_steps_v4"
DEFAULT_CROP_DIR = PROJECT_ROOT / "reports" / "no28_morphology_feature_steps_v4_crop"
STEP6_NAME = "06_prediction_plus_minus_junctions.png"
STEP5_NAME = "05_cleaned_skeleton_on_prediction.png"
STEP2_NAME = "02_model_prediction_mask.png"
STEP_NAMES = (
    "01_original_sem.png",
    "02_model_prediction_mask.png",
    "03_raw_skeleton_on_prediction.png",
    "04_removed_vs_kept_on_black.png",
    "05_cleaned_skeleton_on_prediction.png",
    "06_prediction_plus_minus_junctions.png",
)
PREDICTION_RGB = np.array([245, 245, 245], dtype=np.uint8)
BACKGROUND_RGB = np.array([0, 0, 0], dtype=np.uint8)
SKELETON_RGB = np.array([255, 120, 186], dtype=np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair No28 morphology step 6 overlay and rebuild 768x768 crop outputs.")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--crop-dir", type=Path, default=DEFAULT_CROP_DIR)
    parser.add_argument("--crop-size", type=int, default=768)
    return parser.parse_args()


def read_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def write_rgb(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.astype(np.uint8)).save(path)


def find_overlay_skeleton_mask(step5_rgb: np.ndarray) -> np.ndarray:
    is_background = np.all(step5_rgb == BACKGROUND_RGB, axis=2)
    is_prediction = np.all(step5_rgb == PREDICTION_RGB, axis=2)
    return (~is_background) & (~is_prediction)


def neighbor_count(mask: np.ndarray) -> np.ndarray:
    kernel = np.array(
        [
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ],
        dtype=np.uint8,
    )
    return cv2.filter2D(mask.astype(np.uint8), ddepth=cv2.CV_16U, kernel=kernel, borderType=cv2.BORDER_CONSTANT)


def compute_minus_junction_skeleton(skeleton_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    counts = neighbor_count(skeleton_mask)
    junction_mask = skeleton_mask & (counts >= 3)
    minus_junction = skeleton_mask & (~junction_mask)
    return minus_junction, junction_mask


def render_step6(prediction_rgb: np.ndarray, minus_junction_mask: np.ndarray) -> np.ndarray:
    step6 = prediction_rgb.copy()
    step6[minus_junction_mask] = SKELETON_RGB
    return step6


def centered_crop_box_from_existing(existing_xyxy: list[int], crop_size: int, shape_hw: tuple[int, int]) -> list[int]:
    x1, y1, x2, y2 = [int(v) for v in existing_xyxy]
    old_center_x = (x1 + x2) / 2.0
    old_center_y = (y1 + y2) / 2.0
    half = crop_size / 2.0
    new_x1 = int(round(old_center_x - half))
    new_y1 = int(round(old_center_y - half))
    h, w = [int(v) for v in shape_hw]
    new_x1 = max(0, min(new_x1, w - crop_size))
    new_y1 = max(0, min(new_y1, h - crop_size))
    new_x2 = new_x1 + crop_size
    new_y2 = new_y1 + crop_size
    return [int(new_x1), int(new_y1), int(new_x2), int(new_y2)]


def crop_rgb(image_rgb: np.ndarray, crop_box_xyxy: list[int]) -> np.ndarray:
    x1, y1, x2, y2 = [int(v) for v in crop_box_xyxy]
    return image_rgb[y1:y2, x1:x2].copy()


def build_updated_summary(source_summary: dict, crop_dir: Path, crop_box_xyxy: list[int], crop_size: int) -> dict:
    updated = dict(source_summary)
    updated["crop_box_xyxy"] = [int(v) for v in crop_box_xyxy]
    updated["cropped_files"] = {
        Path(name).stem: str(crop_dir / name)
        for name in STEP_NAMES
    }
    updated["crop_size"] = [int(crop_size), int(crop_size)]
    return updated


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    crop_dir = args.crop_dir.resolve()
    crop_size = int(args.crop_size)

    source_summary_path = source_dir / "summary.json"
    source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))

    prediction_rgb = read_rgb(source_dir / STEP2_NAME)
    step5_rgb = read_rgb(source_dir / STEP5_NAME)
    cleaned_skeleton_mask = find_overlay_skeleton_mask(step5_rgb)
    minus_junction_mask, junction_mask = compute_minus_junction_skeleton(cleaned_skeleton_mask)
    repaired_step6 = render_step6(prediction_rgb, minus_junction_mask)
    write_rgb(source_dir / STEP6_NAME, repaired_step6)

    crop_box_xyxy = centered_crop_box_from_existing(
        existing_xyxy=source_summary["crop_box_xyxy"],
        crop_size=crop_size,
        shape_hw=prediction_rgb.shape[:2],
    )

    for name in STEP_NAMES:
        source_image = read_rgb(source_dir / name)
        cropped = crop_rgb(source_image, crop_box_xyxy)
        if cropped.shape[0] != crop_size or cropped.shape[1] != crop_size:
            raise ValueError(f"Unexpected crop size for {name}: {cropped.shape[:2]}")
        write_rgb(crop_dir / name, cropped)

    updated_summary = build_updated_summary(source_summary, crop_dir, crop_box_xyxy, crop_size)
    updated_summary["junction_pixel_count"] = int(np.count_nonzero(junction_mask))
    updated_summary["cleaned_minus_junction_pixel_count"] = int(np.count_nonzero(minus_junction_mask))
    source_summary_path.write_text(json.dumps(updated_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (crop_dir / "summary.json").write_text(json.dumps(updated_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(
        {
            "source_dir": str(source_dir),
            "crop_dir": str(crop_dir),
            "crop_box_xyxy": crop_box_xyxy,
            "crop_size": [crop_size, crop_size],
            "junction_pixel_count": int(np.count_nonzero(junction_mask)),
            "cleaned_minus_junction_pixel_count": int(np.count_nonzero(minus_junction_mask)),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
