from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.feature_extractor import FeatureExtractor  # noqa: E402


INPUT_ROOT = Path(r"D:\CNTDATA\coredata\u")
OUTPUT_ROOT = PROJECT_ROOT / "reports"


@dataclass
class TargetImage:
    path: Path
    magnification: int


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_gray_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return image


def select_targets(limit_per_mag: int = 3) -> List[TargetImage]:
    targets: List[TargetImage] = []
    for mag in (50000, 100000):
        mag_dir = INPUT_ROOT / str(mag)
        images = sorted(path for path in mag_dir.glob("*.png"))
        for path in images[:limit_per_mag]:
            targets.append(TargetImage(path=path, magnification=mag))
    return targets


def build_mask_base(mask: np.ndarray, fill_color=(42, 42, 42), contour_color=(220, 220, 220)) -> np.ndarray:
    canvas = np.zeros((*mask.shape, 3), dtype=np.uint8)
    canvas[mask > 0] = fill_color
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, contours, -1, contour_color, 1)
    return canvas


def random_color(seed: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(seed)
    return tuple(int(v) for v in rng.integers(80, 255, size=3))


def component_metrics(component_mask: np.ndarray, extractor: FeatureExtractor) -> Dict[str, float]:
    ys, xs = np.nonzero(component_mask)
    coords = np.column_stack([ys, xs]).astype(float)
    area_px = float(coords.shape[0])

    if coords.shape[0] < 3:
        elongation = 1.0
    else:
        centered = coords - coords.mean(axis=0)
        cov = np.cov(centered.T)
        eigvals = np.sort(np.linalg.eigvalsh(cov))
        major = float(max(eigvals[-1], 1e-6))
        minor = float(max(eigvals[0], 1e-6))
        elongation = float(np.sqrt(major / minor))

    _, component_skel = extractor.calculate_diameter((component_mask.astype(np.uint8) * 255))
    skeleton_length_px = float(np.count_nonzero(component_skel))

    return {
        "area_px": area_px,
        "elongation": elongation,
        "skeleton_length_px": skeleton_length_px,
    }


def clean_components(mask: np.ndarray, extractor: FeatureExtractor) -> Dict[str, Any]:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)

    min_area_px = max(60.0, float(extractor.expected_tube_px ** 2 * 6.0))
    min_skeleton_length_px = max(20.0, float(extractor.expected_tube_px * 8.0))
    min_elongation = 2.5

    metrics: List[Dict[str, Any]] = []
    keep_mask = np.zeros_like(mask, dtype=np.uint8)
    before_canvas = np.zeros((*mask.shape, 3), dtype=np.uint8)
    cleaned_canvas = np.zeros((*mask.shape, 3), dtype=np.uint8)

    for label_id in range(1, num_labels):
        component_mask = labels == label_id
        if not np.any(component_mask):
            continue

        comp_metrics = component_metrics(component_mask, extractor)
        keep = (
            comp_metrics["area_px"] >= min_area_px and
            comp_metrics["skeleton_length_px"] >= min_skeleton_length_px and
            comp_metrics["elongation"] >= min_elongation
        )

        metrics.append({
            "label_id": int(label_id),
            "keep": bool(keep),
            **comp_metrics,
        })

        color = random_color(label_id)
        before_canvas[component_mask] = color
        if keep:
            keep_mask[component_mask] = 255
            cleaned_canvas[component_mask] = (80, 220, 120)
        else:
            cleaned_canvas[component_mask] = (200, 80, 80)

    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(before_canvas, contours, -1, (255, 255, 255), 1)
    cv2.drawContours(cleaned_canvas, contours, -1, (220, 220, 220), 1)

    kept_count = sum(1 for item in metrics if item["keep"])
    removed_count = len(metrics) - kept_count
    return {
        "keep_mask": keep_mask,
        "before_canvas": before_canvas,
        "cleaned_canvas": cleaned_canvas,
        "metrics": metrics,
        "thresholds": {
            "min_area_px": min_area_px,
            "min_skeleton_length_px": min_skeleton_length_px,
            "min_elongation": min_elongation,
        },
        "kept_count": kept_count,
        "removed_count": removed_count,
    }


def render_panel(
    roi: np.ndarray,
    mask: np.ndarray,
    skeleton: np.ndarray,
    before_canvas: np.ndarray,
    cleaned_canvas: np.ndarray,
    summary_lines: List[str],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(21, 5.6), dpi=160, constrained_layout=True)
    panels = [
        ("ROI", cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)),
        ("Mask", cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)),
        ("Skeleton", build_mask_base(mask)),
        ("Objects Before Cleaning", before_canvas),
        ("Keep / Drop After Cleaning", cleaned_canvas),
    ]

    skeleton_rgb = panels[2][1]
    skeleton_rgb[skeleton > 0] = (255, 230, 90)
    panels[2] = ("Skeleton", skeleton_rgb)

    for ax, (title, image) in zip(axes, panels):
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else image, cmap=None if image.ndim == 3 else "gray")
        ax.set_title(title, fontsize=12)
        ax.axis("off")

    fig.text(
        0.5,
        0.02,
        " | ".join(summary_lines),
        ha="center",
        va="bottom",
        fontsize=10,
        family="monospace",
    )
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUT_ROOT / f"mask_skeleton_cleaning_panels_{timestamp}"
    ensure_dir(out_dir)

    records: List[Dict[str, Any]] = []
    for target in select_targets(limit_per_mag=3):
        image = read_gray_image(target.path)
        extractor = FeatureExtractor(magnification=target.magnification)

        roi = extractor.extract_roi(image)
        extractor._calibrate(roi.shape[1])
        processed = extractor.preprocess(roi)
        _, thresh = extractor.calculate_density(processed)
        _, skeleton = extractor.calculate_diameter(thresh)

        cleaning = clean_components(thresh, extractor)
        keep_mask = cleaning["keep_mask"]
        _, keep_skeleton = extractor.calculate_diameter(keep_mask)

        summary_lines = [
            f"file={target.path.name}",
            f"mag={target.magnification}",
            f"components={len(cleaning['metrics'])}",
            f"kept={cleaning['kept_count']}",
            f"removed={cleaning['removed_count']}",
            f"min_area={cleaning['thresholds']['min_area_px']:.1f}",
            f"min_skel={cleaning['thresholds']['min_skeleton_length_px']:.1f}",
            f"min_elong={cleaning['thresholds']['min_elongation']:.2f}",
        ]

        cleaned_canvas = cleaning["cleaned_canvas"].copy()
        cleaned_canvas[keep_skeleton > 0] = (255, 240, 120)
        stem = target.path.stem.replace(" ", "_")
        panel_path = out_dir / f"{stem}__cleaning_panel.png"
        render_panel(
            roi=roi,
            mask=thresh,
            skeleton=(skeleton > 0).astype(np.uint8),
            before_canvas=cleaning["before_canvas"],
            cleaned_canvas=cleaned_canvas,
            summary_lines=summary_lines,
            output_path=panel_path,
        )

        record = {
            "file_name": target.path.name,
            "file_path": str(target.path),
            "magnification": target.magnification,
            "components": len(cleaning["metrics"]),
            "kept_count": cleaning["kept_count"],
            "removed_count": cleaning["removed_count"],
            "thresholds": cleaning["thresholds"],
            "panel_path": str(panel_path),
            "component_metrics": cleaning["metrics"],
        }
        records.append(record)

    with (out_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump({"output_dir": str(out_dir), "count": len(records), "records": records}, fh, ensure_ascii=False, indent=2)

    print(out_dir)


if __name__ == "__main__":
    main()
