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
THRESHOLDS = [1.0, 3.0, 5.0, 7.0, 9.0]
THRESHOLD_LABELS = ["L0", "L1", "L2", "L3", "L4"]


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
        for path in sorted((INPUT_ROOT / str(mag)).glob("*.png"))[:limit_per_mag]:
            targets.append(TargetImage(path=path, magnification=mag))
    return targets


def build_mask_base(mask: np.ndarray, fill_color=(42, 42, 42), contour_color=(220, 220, 220)) -> np.ndarray:
    canvas = np.zeros((*mask.shape, 3), dtype=np.uint8)
    canvas[mask > 0] = fill_color
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, contours, -1, contour_color, 1)
    return canvas


def draw_branches(mask: np.ndarray, branches: List[Dict[str, Any]], color=(90, 255, 140)) -> np.ndarray:
    canvas = build_mask_base(mask)
    for branch in branches:
        coords = branch["coords"]
        if coords.shape[0] < 2:
            continue
        pts = np.round(coords[:, ::-1]).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], False, color, 1, lineType=cv2.LINE_AA)
    return canvas


def build_text_panel(lines: List[str]) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(6.4, 5.4), dpi=160)
    ax.set_facecolor("#0f172a")
    fig.patch.set_facecolor("#0f172a")
    ax.axis("off")
    y = 0.97
    for line in lines:
        ax.text(0.04, y, line, transform=ax.transAxes, fontsize=10.0, color="white", va="top", family="DejaVu Sans Mono")
        y -= 0.062
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8).copy()
    plt.close(fig)
    return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)


def prepare_common(image: np.ndarray, magnification: int) -> Dict[str, Any]:
    extractor = FeatureExtractor(magnification=magnification, speed_profile="accurate")
    roi = extractor.extract_roi(image)
    extractor._calibrate(roi.shape[1])
    processed = extractor.preprocess(roi)
    _, thresh = extractor.calculate_density(processed)
    _, skel = extractor.calculate_diameter(thresh)
    return {
        "extractor": extractor,
        "roi": roi,
        "mask": thresh,
        "skeleton": skel,
    }


def threshold_profile(common: Dict[str, Any], min_length_factor: float) -> Dict[str, Any]:
    extractor: FeatureExtractor = common["extractor"]
    skel = common["skeleton"]
    branches = extractor._collect_ordered_branches_v2(
        skel,
        min_points=max(extractor.V3_MIN_BRANCH_POINTS, int(round(extractor.expected_tube_px * 1.5))),
        min_length_factor=min_length_factor,
    )
    _, curvature_nm_v3 = extractor.calculate_curvature_v3(skel, ordered_branches=branches)
    waviness_v2 = extractor.calculate_waviness_v2(skel, ordered_branches=branches)
    return {
        "branches": branches,
        "branch_count": len(branches),
        "curvature_nm_v3": float(curvature_nm_v3),
        "waviness_ratio_v2": float(waviness_v2["waviness_ratio_v2"]),
        "tortuosity_v2": float(waviness_v2["tortuosity_v2"]),
    }


def render_panel(
    roi: np.ndarray,
    mask: np.ndarray,
    raw_skeleton: np.ndarray,
    profiles: Dict[str, Dict[str, Any]],
    file_name: str,
    output_path: Path,
) -> None:
    raw_canvas = build_mask_base(mask)
    raw_canvas[raw_skeleton > 0] = (255, 230, 90)

    threshold_images = [
        (label, draw_branches(mask, profiles[label]["branches"]))
        for label in THRESHOLD_LABELS
    ]

    text_lines = [f"file: {file_name}", ""]
    for label, factor in zip(THRESHOLD_LABELS, THRESHOLDS):
        record = profiles[label]
        text_lines.append(
            f"{label} len={factor:.1f}  n={record['branch_count']}  curv={record['curvature_nm_v3']:.6f}"
        )
        text_lines.append(
            f"   wav_v2={record['waviness_ratio_v2']:.4f}  tort_v2={record['tortuosity_v2']:.3f}"
        )

    text_panel = build_text_panel(text_lines)

    fig, axes = plt.subplots(2, 4, figsize=(22, 11), dpi=160, constrained_layout=True)
    panels = [
        ("Original ROI", cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)),
        ("Raw Skeleton", raw_canvas),
        (f"L0 ({profiles['L0']['branch_count']})", threshold_images[0][1]),
        (f"L1 ({profiles['L1']['branch_count']})", threshold_images[1][1]),
        (f"L2 ({profiles['L2']['branch_count']})", threshold_images[2][1]),
        (f"L3 ({profiles['L3']['branch_count']})", threshold_images[3][1]),
        (f"L4 ({profiles['L4']['branch_count']})", threshold_images[4][1]),
        ("Metrics", text_panel),
    ]

    for ax, (title, image) in zip(axes.flat, panels):
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else image, cmap=None if image.ndim == 3 else "gray")
        ax.set_title(title, fontsize=12)
        ax.axis("off")

    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUT_ROOT / f"v3_length_threshold_panels_{timestamp}"
    ensure_dir(out_dir)

    records: List[Dict[str, Any]] = []
    for target in select_targets():
        image = read_gray_image(target.path)
        common = prepare_common(image, target.magnification)
        profiles = {
            label: threshold_profile(common, factor)
            for label, factor in zip(THRESHOLD_LABELS, THRESHOLDS)
        }

        stem = target.path.stem.replace(" ", "_")
        panel_path = out_dir / f"{stem}__length_threshold_panel.png"
        render_panel(
            roi=common["roi"],
            mask=common["mask"],
            raw_skeleton=(common["skeleton"] > 0).astype(np.uint8),
            profiles=profiles,
            file_name=target.path.name,
            output_path=panel_path,
        )

        records.append(
            {
                "file_name": target.path.name,
                "file_path": str(target.path),
                "magnification": target.magnification,
                "panel_path": str(panel_path),
                "profiles": {
                    label: {
                        "min_length_factor": factor,
                        "branch_count": profiles[label]["branch_count"],
                        "curvature_nm_v3": round(profiles[label]["curvature_nm_v3"], 6),
                        "waviness_ratio_v2": round(profiles[label]["waviness_ratio_v2"], 4),
                        "tortuosity_v2": round(profiles[label]["tortuosity_v2"], 3),
                    }
                    for label, factor in zip(THRESHOLD_LABELS, THRESHOLDS)
                },
            }
        )

    with (out_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump({"output_dir": str(out_dir), "count": len(records), "records": records}, fh, ensure_ascii=False, indent=2)

    print(out_dir)


if __name__ == "__main__":
    main()
