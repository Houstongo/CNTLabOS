from __future__ import annotations

import argparse
import json
import sys
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
TOOLS_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from src.analysis.feature_extractor import FeatureExtractor


DEFAULT_ITEM_DIR = Path(
    r"D:\CNTDATA\CNTA_ML_Project\reports\desktop_expc_baseline_v2v3_report_20260328_020832\items\text10_No28_200w_15_0nm_50w_0_5nm_600_300_150_600_750_15min_180min_mid_100000-1"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "reports"

THRESHOLDS = [3.0, 5.0, 7.0, 9.0]
THRESHOLD_LABELS = ["L1", "L2", "L3", "L4"]
THRESHOLD_COLORS = [
    (0.95, 0.60, 0.20),
    (0.20, 0.80, 0.45),
    (0.20, 0.65, 0.95),
    (0.78, 0.35, 0.92),
]
BRANCH_AGGREGATIONS = ("median", "p75", "mean", "trimmed_mean")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a single CNTSegNet-SLICE V3 threshold study panel from existing Exp-C mask.")
    parser.add_argument("--item-dir", type=Path, default=DEFAULT_ITEM_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_gray_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return image


def build_bw_mask(mask: np.ndarray) -> np.ndarray:
    canvas = np.zeros((*mask.shape, 3), dtype=np.uint8)
    canvas[mask > 0] = (245, 245, 245)
    return canvas


def draw_branch_overlay(mask: np.ndarray, branches: List[Dict[str, Any]], color_rgb: tuple[float, float, float]) -> np.ndarray:
    canvas = build_bw_mask(mask)
    bgr = tuple(int(round(channel * 255)) for channel in color_rgb[::-1])
    for branch in branches:
        coords = np.asarray(branch["coords"], dtype=float)
        if coords.shape[0] < 2:
            continue
        pts = np.round(coords[:, ::-1]).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], False, bgr, 1, lineType=cv2.LINE_AA)
    return canvas


def sample_branch_diameters_nm(extractor: FeatureExtractor, distance_map: np.ndarray, branches: List[Dict[str, Any]]) -> np.ndarray:
    px_per_nm = max(extractor.px_per_um / 1000.0, 1e-6)
    diameter_values = []
    for branch in branches:
        coords = np.asarray(branch["coords"], dtype=float)
        sampled = extractor._sample_map_values(coords, distance_map)
        valid = sampled[np.isfinite(sampled) & (sampled > 0)]
        if valid.size == 0:
            continue
        diameter_values.append((valid * 2.0) / px_per_nm)
    if not diameter_values:
        return np.empty((0,), dtype=float)
    return np.concatenate(diameter_values).astype(float)


def sample_branch_curvatures_nm(extractor: FeatureExtractor, branches: List[Dict[str, Any]]) -> np.ndarray:
    px_per_nm = max(extractor.px_per_um / 1000.0, 1e-6)
    curvature_values = []
    for branch in branches:
        sampled_coords = extractor._sample_ordered_coords(np.asarray(branch["coords"], dtype=float), sample_step=1)
        point_curvature_px = extractor._compute_point_curvatures_px(sampled_coords)
        if point_curvature_px.size == 0:
            continue
        curvature_values.append(point_curvature_px * px_per_nm)
    if not curvature_values:
        return np.empty((0,), dtype=float)
    return np.concatenate(curvature_values).astype(float)


def aggregate_branch_curvature_nm(
    extractor: FeatureExtractor,
    branches: List[Dict[str, Any]],
    branch_stat: str,
    weight_mode: str,
) -> float:
    px_per_nm = max(extractor.px_per_um / 1000.0, 1e-6)
    branch_curvatures = []
    weights = []
    for branch in branches:
        sampled_coords = extractor._sample_ordered_coords(np.asarray(branch["coords"], dtype=float), sample_step=1)
        point_curvature_px = extractor._compute_point_curvatures_px(sampled_coords)
        if point_curvature_px.size == 0:
            continue
        if branch_stat == "median":
            branch_curvature_px = float(np.median(point_curvature_px))
        elif branch_stat == "p75":
            branch_curvature_px = float(np.percentile(point_curvature_px, extractor.V3_BRANCH_QUANTILE))
        elif branch_stat == "mean":
            branch_curvature_px = float(np.mean(point_curvature_px))
        elif branch_stat == "trimmed_mean":
            if point_curvature_px.size >= 5:
                low, high = np.percentile(point_curvature_px, [10, 90])
                trimmed = point_curvature_px[(point_curvature_px >= low) & (point_curvature_px <= high)]
                branch_curvature_px = float(np.mean(trimmed)) if trimmed.size > 0 else float(np.mean(point_curvature_px))
            else:
                branch_curvature_px = float(np.mean(point_curvature_px))
        else:
            raise ValueError(f"Unsupported branch_stat: {branch_stat}")

        branch_curvature_nm = branch_curvature_px * px_per_nm
        branch_curvatures.append(branch_curvature_nm)
        path_length_px = float(branch.get("path_length_px", 0.0))
        if weight_mode == "sqrt_length":
            weights.append(np.sqrt(max(path_length_px, 1.0)))
        elif weight_mode == "length":
            weights.append(max(path_length_px, 1.0))
        else:
            raise ValueError(f"Unsupported weight_mode: {weight_mode}")

    if not branch_curvatures:
        return 0.0
    return float(np.average(np.asarray(branch_curvatures, dtype=float), weights=np.asarray(weights, dtype=float)))


def build_text_block(lines: List[str]) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(4.4, 4.6), dpi=180)
    ax.set_facecolor("#0f172a")
    fig.patch.set_facecolor("#0f172a")
    ax.axis("off")
    y = 0.97
    for line in lines:
        ax.text(0.04, y, line, transform=ax.transAxes, fontsize=9.2, color="white", va="top", family="DejaVu Sans Mono")
        y -= 0.082
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8).copy()
    plt.close(fig)
    return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)


def main() -> None:
    args = parse_args()
    item_dir = args.item_dir
    features_path = item_dir / "features.json"
    mask_path = item_dir / "expc_mask.png"
    if not features_path.exists():
        raise FileNotFoundError(f"Missing features.json: {features_path}")
    if not mask_path.exists():
        raise FileNotFoundError(f"Missing expc_mask.png: {mask_path}")

    features = json.loads(features_path.read_text(encoding="utf-8"))
    image_path = Path(features["file_path"])
    magnification = int(features["magnification"])

    image_gray = read_gray_image(image_path)
    extractor = FeatureExtractor(magnification=magnification, speed_profile="accurate")
    roi = extractor.extract_roi(image_gray)
    extractor._calibrate(roi.shape[1])

    mask = read_gray_image(mask_path)
    if mask.shape != roi.shape:
        mask = cv2.resize(mask, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask = (mask > 0).astype(np.uint8) * 255

    distance_map = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    _, skeleton = extractor.calculate_diameter(mask)

    profiles: Dict[str, Dict[str, Any]] = {}
    all_curvature_values: List[np.ndarray] = []
    all_diameter_values: List[np.ndarray] = []
    for label, factor, color in zip(THRESHOLD_LABELS, THRESHOLDS, THRESHOLD_COLORS):
        branches = extractor._collect_ordered_branches_v2(
            skeleton,
            min_points=max(extractor.V3_MIN_BRANCH_POINTS, int(round(extractor.expected_tube_px * 1.5))),
            min_length_factor=factor,
        )
        curvature_label, curvature_nm_v3 = extractor.calculate_curvature_v3(skeleton, ordered_branches=branches)
        waviness_v2 = extractor.calculate_waviness_v2(skeleton, ordered_branches=branches)
        curvature_dist = sample_branch_curvatures_nm(extractor, branches)
        diameter_dist = sample_branch_diameters_nm(extractor, distance_map, branches)
        curvature_dist_um = curvature_dist * 1000.0
        all_curvature_values.append(curvature_dist_um)
        all_diameter_values.append(diameter_dist)
        aggregate_curvatures = {
            stat_name: {
                "sqrt_length": aggregate_branch_curvature_nm(extractor, branches, branch_stat=stat_name, weight_mode="sqrt_length"),
                "length": aggregate_branch_curvature_nm(extractor, branches, branch_stat=stat_name, weight_mode="length"),
            }
            for stat_name in BRANCH_AGGREGATIONS
        }

        profiles[label] = {
            "factor": factor,
            "color": color,
            "branches": branches,
            "branch_count": len(branches),
            "curvature_label": curvature_label,
            "curvature_nm_v3": float(curvature_nm_v3),
            "aggregate_curvatures_nm": aggregate_curvatures,
            "waviness_ratio_v2": float(waviness_v2["waviness_ratio_v2"]) if waviness_v2["waviness_ratio_v2"] is not None else None,
            "tortuosity_v2": float(waviness_v2["tortuosity_v2"]),
            "curvature_distribution_um": curvature_dist_um,
            "diameter_distribution_nm": diameter_dist,
        }

    non_empty_curvatures = [vals for vals in all_curvature_values if vals.size > 0]
    if non_empty_curvatures:
        curvature_stack = np.concatenate(non_empty_curvatures)
        curvature_max = float(np.percentile(curvature_stack, 99.5))
    else:
        curvature_max = 30.0
    curvature_max = max(curvature_max, 5.0)
    diameter_max = max([float(vals.max()) for vals in all_diameter_values if vals.size > 0] + [30.0])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir or (DEFAULT_OUTPUT_ROOT / f"expc_slice_v3_threshold_single_{timestamp}")
    ensure_dir(out_dir)

    fig = plt.figure(figsize=(20, 28), dpi=180, constrained_layout=True)
    grid = fig.add_gridspec(
        5,
        4,
        width_ratios=[1.2, 1.0, 1.0, 0.95],
        height_ratios=[1.0, 1, 1, 1, 1],
    )

    ax_original = fig.add_subplot(grid[0, 0])
    ax_original.imshow(roi, cmap="gray")
    ax_original.set_title("Original ROI", fontsize=14)
    ax_original.axis("off")

    ax_mask = fig.add_subplot(grid[0, 1])
    ax_mask.imshow(cv2.cvtColor(build_bw_mask(mask), cv2.COLOR_BGR2RGB))
    ax_mask.set_title("CNTSegNet-SLICE Mask", fontsize=14)
    ax_mask.axis("off")

    ax_summary = fig.add_subplot(grid[0, 2:])
    ax_summary.axis("off")
    summary_lines = [
        f"file: {image_path.name}",
        f"magnification: {magnification}x",
        f"mask source: Exp C renamed as CNTSegNet-SLICE",
        "",
        f"px_per_um: {extractor.px_per_um:.3f}",
        f"thresholds: {', '.join(f'{k}={v:.1f}' for k, v in zip(THRESHOLD_LABELS, THRESHOLDS))}",
        "",
        "standard method:",
        "CNTSegNet-SLICE + V3 pruning",
        "L1-L4 + dual aggregation",
        "curvature hist in um^-1",
    ]
    ax_summary.text(0.02, 0.98, "\n".join(summary_lines), va="top", ha="left", fontsize=12, family="DejaVu Sans Mono")

    for row_idx, label in enumerate(THRESHOLD_LABELS, start=1):
        profile = profiles[label]
        color = profile["color"]

        ax_overlay = fig.add_subplot(grid[row_idx, 0])
        overlay = draw_branch_overlay(mask, profile["branches"], color)
        ax_overlay.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        ax_overlay.set_title(f"{label} Branch Overlay  (len={profile['factor']:.1f}, n={profile['branch_count']})", fontsize=12)
        ax_overlay.axis("off")

        ax_curv = fig.add_subplot(grid[row_idx, 1])
        curv_vals = profile["curvature_distribution_um"]
        if curv_vals.size > 0:
            bins = np.linspace(0.0, curvature_max, 40)
            ax_curv.hist(curv_vals, bins=bins, color=color, alpha=0.85, edgecolor="black", linewidth=0.4)
        ax_curv.set_xlim(0.0, curvature_max)
        ax_curv.set_title(f"{label} Curvature Distribution", fontsize=12)
        ax_curv.set_xlabel("curvature (um^-1)")
        ax_curv.set_ylabel("count")
        ax_curv.grid(alpha=0.2)

        ax_diam = fig.add_subplot(grid[row_idx, 2])
        diam_vals = profile["diameter_distribution_nm"]
        if diam_vals.size > 0:
            bins = np.linspace(0.0, diameter_max, 40)
            ax_diam.hist(diam_vals, bins=bins, color=color, alpha=0.85, edgecolor="black", linewidth=0.4)
        ax_diam.set_xlim(0.0, diameter_max)
        ax_diam.set_title(f"{label} Diameter Distribution", fontsize=12)
        ax_diam.set_xlabel("diameter (nm)")
        ax_diam.set_ylabel("count")
        ax_diam.grid(alpha=0.2)

        ax_text = fig.add_subplot(grid[row_idx, 3])
        ax_text.axis("off")
        text_lines = [
            f"{label} / len={profile['factor']:.1f}",
            f"branches: {profile['branch_count']}",
            f"med s/l: {profile['aggregate_curvatures_nm']['median']['sqrt_length']:.6f} / {profile['aggregate_curvatures_nm']['median']['length']:.6f}",
            f"p75 s/l: {profile['aggregate_curvatures_nm']['p75']['sqrt_length']:.6f} / {profile['aggregate_curvatures_nm']['p75']['length']:.6f}",
            f"mean s/l: {profile['aggregate_curvatures_nm']['mean']['sqrt_length']:.6f} / {profile['aggregate_curvatures_nm']['mean']['length']:.6f}",
            f"trim s/l: {profile['aggregate_curvatures_nm']['trimmed_mean']['sqrt_length']:.6f} / {profile['aggregate_curvatures_nm']['trimmed_mean']['length']:.6f}",
            f"label: {profile['curvature_label']}",
            f"wav_v2: {profile['waviness_ratio_v2']}",
            f"tort_v2: {profile['tortuosity_v2']:.4f}",
            f"curv pts: {curv_vals.size}",
            f"diam pts: {diam_vals.size}",
        ]
        text_panel = build_text_block(text_lines)
        ax_text.imshow(cv2.cvtColor(text_panel, cv2.COLOR_BGR2RGB))

    fig.suptitle("CNTSegNet-SLICE (Exp C)  |  Standard Method  |  V3 + Length Threshold L1-L4", fontsize=18)
    panel_path = out_dir / f"{item_dir.name}__slice_v3_threshold_panel.png"
    fig.savefig(panel_path, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "item_dir": str(item_dir),
        "image_path": str(image_path),
        "panel_path": str(panel_path),
        "mask_path": str(mask_path),
        "magnification": magnification,
        "thresholds": {
            label: {
                "min_length_factor": profile["factor"],
                "branch_count": profile["branch_count"],
                "curvature_nm_v3": round(profile["curvature_nm_v3"], 6),
                "aggregate_curvatures_nm": {
                    stat_name: {
                        "sqrt_length": round(values["sqrt_length"], 6),
                        "length": round(values["length"], 6),
                    }
                    for stat_name, values in profile["aggregate_curvatures_nm"].items()
                },
                "curvature_label": profile["curvature_label"],
                "waviness_ratio_v2": round(profile["waviness_ratio_v2"], 4) if profile["waviness_ratio_v2"] is not None else None,
                "tortuosity_v2": round(profile["tortuosity_v2"], 4),
                "curvature_point_count": int(profile["curvature_distribution_um"].size),
                "diameter_point_count": int(profile["diameter_distribution_nm"].size),
            }
            for label, profile in profiles.items()
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out_dir)


if __name__ == "__main__":
    main()
