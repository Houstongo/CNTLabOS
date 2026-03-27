from __future__ import annotations

import argparse
import csv
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate branch-selection comparison panels.")
    parser.add_argument("--input-root", type=Path, default=INPUT_ROOT)
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of images to process.")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_gray_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return image


def select_targets(input_root: Path) -> List[TargetImage]:
    targets: List[TargetImage] = []
    for mag in (50000, 100000):
        for path in sorted((input_root / str(mag)).glob("*.png")):
            targets.append(TargetImage(path=path, magnification=mag))
    return targets


def build_mask_base(mask: np.ndarray, fill_color=(42, 42, 42), contour_color=(220, 220, 220)) -> np.ndarray:
    canvas = np.zeros((*mask.shape, 3), dtype=np.uint8)
    canvas[mask > 0] = fill_color
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, contours, -1, contour_color, 1)
    return canvas


def draw_branches(canvas: np.ndarray, branches: List[Dict[str, Any]], color: tuple[int, int, int], line_thickness: int = 1) -> np.ndarray:
    for branch in branches:
        coords = branch["coords"]
        if coords.shape[0] < 2:
            continue
        pts = np.round(coords[:, ::-1]).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], False, color, line_thickness, lineType=cv2.LINE_AA)
    return canvas


def build_text_panel(lines: List[str]) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(6.0, 5.2), dpi=160)
    ax.set_facecolor("#0f172a")
    fig.patch.set_facecolor("#0f172a")
    ax.axis("off")
    y = 0.97
    for line in lines:
        ax.text(0.04, y, line, transform=ax.transAxes, fontsize=10.5, color="white", va="top", family="DejaVu Sans Mono")
        y -= 0.065
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8).copy()
    plt.close(fig)
    return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)


def estimate_diameter_p30_nm(mask: np.ndarray, skel: np.ndarray, px_per_um: float) -> float | None:
    dist = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    radii = dist[skel > 0].astype(float)
    radii = radii[np.isfinite(radii) & (radii > 0)]
    if radii.size == 0:
        return None
    diameter_px_p30 = float(np.percentile(radii * 2.0, 30))
    return float((diameter_px_p30 / max(px_per_um, 1e-6)) * 1000.0)


def prepare_common(image: np.ndarray, magnification: int) -> Dict[str, Any]:
    extractor = FeatureExtractor(magnification=magnification, speed_profile="accurate")
    roi = extractor.extract_roi(image)
    extractor._calibrate(roi.shape[1])
    processed = extractor.preprocess(roi)
    density, thresh = extractor.calculate_density(processed)
    diameter_nm, skel = extractor.calculate_diameter(thresh)
    base_components = extractor._collect_components(skel)
    return {
        "extractor": extractor,
        "roi": roi,
        "processed": processed,
        "density": density,
        "thresh": thresh,
        "diameter_nm": diameter_nm,
        "skel": skel,
        "base_components": base_components,
    }


def analyze_profiles(common: Dict[str, Any], magnification: int) -> Dict[str, Any]:
    accurate = common["extractor"]
    skel = common["skel"]
    base_components = common["base_components"]

    accurate_v2_branches, accurate_v3_branches = accurate._prepare_curvature_branch_sets(
        skel,
        v2_min_points=15,
    )

    legacy_label, legacy_curvature_nm = accurate.calculate_curvature(skel, base_components=base_components)
    v2_label, v2_curvature_nm = accurate.calculate_curvature_v2(skel, ordered_branches=accurate_v2_branches)
    v3_label, v3_curvature_nm = accurate.calculate_curvature_v3(skel, ordered_branches=accurate_v3_branches)
    waviness = accurate.calculate_waviness(skel, base_components=base_components)
    waviness_v2 = accurate.calculate_waviness_v2(skel, ordered_branches=accurate_v2_branches)

    fast = FeatureExtractor(magnification=magnification, speed_profile="fast")
    fast._calibrate(common["roi"].shape[1])
    fast_v2_branches, fast_v3_branches = fast._prepare_curvature_branch_sets(
        skel,
        v2_min_points=15,
    )
    fast_legacy_label, fast_legacy_curvature_nm = fast.calculate_curvature(skel, base_components=base_components)
    fast_v2_label, fast_v2_curvature_nm = fast.calculate_curvature_v2(skel, ordered_branches=fast_v2_branches)
    fast_v3_label, fast_v3_curvature_nm = fast.calculate_curvature_v3(skel, ordered_branches=fast_v3_branches)
    fast_waviness = fast.calculate_waviness(skel, base_components=base_components)
    fast_waviness_v2 = fast.calculate_waviness_v2(skel, ordered_branches=fast_v2_branches)
    diameter_p30_nm = estimate_diameter_p30_nm(common["thresh"], skel, accurate.px_per_um)

    return {
        "accurate_v2_branches": accurate_v2_branches,
        "accurate_v3_branches": accurate_v3_branches,
        "fast_v2_branches": fast_v2_branches,
        "fast_v3_branches": fast_v3_branches,
        "metrics": {
            "density": round(common["density"], 2),
            "diameter_p30_nm": round(diameter_p30_nm, 2) if diameter_p30_nm is not None else None,
            "legacy_label": legacy_label,
            "legacy_curvature_nm": round(legacy_curvature_nm, 6),
            "v2_label": v2_label,
            "v2_curvature_nm": round(v2_curvature_nm, 6),
            "v3_label": v3_label,
            "v3_curvature_nm": round(v3_curvature_nm, 6),
            "waviness_ratio": round(waviness["waviness_ratio"], 4) if waviness["waviness_ratio"] is not None else None,
            "waviness_ratio_v2": round(waviness_v2["waviness_ratio_v2"], 4) if waviness_v2["waviness_ratio_v2"] is not None else None,
            "tortuosity": round(waviness["tortuosity"], 3),
            "tortuosity_v2": round(waviness_v2["tortuosity_v2"], 3),
            "fast_legacy_label": fast_legacy_label,
            "fast_legacy_curvature_nm": round(fast_legacy_curvature_nm, 6),
            "fast_v2_label": fast_v2_label,
            "fast_v2_curvature_nm": round(fast_v2_curvature_nm, 6),
            "fast_v3_label": fast_v3_label,
            "fast_v3_curvature_nm": round(fast_v3_curvature_nm, 6),
            "fast_waviness_ratio": round(fast_waviness["waviness_ratio"], 4) if fast_waviness["waviness_ratio"] is not None else None,
            "fast_waviness_ratio_v2": round(fast_waviness_v2["waviness_ratio_v2"], 4) if fast_waviness_v2["waviness_ratio_v2"] is not None else None,
            "fast_tortuosity": round(fast_waviness["tortuosity"], 3),
            "fast_tortuosity_v2": round(fast_waviness_v2["tortuosity_v2"], 3),
            "accurate_v2_branch_count": len(accurate_v2_branches),
            "accurate_v3_branch_count": len(accurate_v3_branches),
            "fast_v2_branch_count": len(fast_v2_branches),
            "fast_v3_branch_count": len(fast_v3_branches),
            "px_per_um": round(accurate.px_per_um, 2),
        },
    }


def render_panel(
    roi: np.ndarray,
    mask: np.ndarray,
    accurate_v2_branches: List[Dict[str, Any]],
    accurate_v3_branches: List[Dict[str, Any]],
    fast_v2_branches: List[Dict[str, Any]],
    fast_v3_branches: List[Dict[str, Any]],
    metrics: Dict[str, Any],
    file_name: str,
    output_path: Path,
) -> None:
    v2_canvas = draw_branches(build_mask_base(mask), accurate_v2_branches, (90, 255, 140))
    v3_canvas = draw_branches(build_mask_base(mask), accurate_v3_branches, (255, 190, 80))
    fast_canvas = build_mask_base(mask)
    fast_canvas = draw_branches(fast_canvas, fast_v2_branches, (80, 220, 255))
    fast_canvas = draw_branches(fast_canvas, fast_v3_branches, (255, 120, 220))

    text_lines = [
        f"file: {file_name}",
        "",
        f"density: {metrics['density']:.2f} %",
        f"diameter p30: {metrics['diameter_p30_nm']}",
        f"px_per_um: {metrics['px_per_um']}",
        "",
        f"legacy: {metrics['legacy_curvature_nm']:.6f} ({metrics['legacy_label']})",
        f"v2: {metrics['v2_curvature_nm']:.6f} ({metrics['v2_label']})",
        f"v3: {metrics['v3_curvature_nm']:.6f} ({metrics['v3_label']})",
        "",
        f"fast legacy: {metrics['fast_legacy_curvature_nm']:.6f} ({metrics['fast_legacy_label']})",
        f"fast v2: {metrics['fast_v2_curvature_nm']:.6f} ({metrics['fast_v2_label']})",
        f"fast v3: {metrics['fast_v3_curvature_nm']:.6f} ({metrics['fast_v3_label']})",
        "",
        f"waviness: {metrics['waviness_ratio']}",
        f"waviness_v2: {metrics['waviness_ratio_v2']}",
        f"fast waviness: {metrics['fast_waviness_ratio']}",
        f"fast waviness_v2: {metrics['fast_waviness_ratio_v2']}",
        "",
        f"v2 branches: {metrics['accurate_v2_branch_count']}",
        f"v3 branches: {metrics['accurate_v3_branch_count']}",
        f"fast v2 branches: {metrics['fast_v2_branch_count']}",
        f"fast v3 branches: {metrics['fast_v3_branch_count']}",
        "",
        "fast colors: cyan=V2, magenta=V3",
    ]
    text_panel = build_text_panel(text_lines)

    fig, axes = plt.subplots(1, 5, figsize=(22, 5.8), dpi=160, constrained_layout=True)
    panels = [
        ("Original ROI", cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)),
        (f"V2 Accurate ({metrics['accurate_v2_branch_count']})", v2_canvas),
        (f"V3 Accurate ({metrics['accurate_v3_branch_count']})", v3_canvas),
        (f"Fast Branches ({metrics['fast_v2_branch_count']}/{metrics['fast_v3_branch_count']})", fast_canvas),
        ("Feature Metrics", text_panel),
    ]

    for ax, (title, image) in zip(axes, panels):
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else image, cmap=None if image.ndim == 3 else "gray")
        ax.set_title(title, fontsize=12)
        ax.axis("off")

    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir or (OUTPUT_ROOT / f"branch_selection_compare_panels_{timestamp}")
    ensure_dir(out_dir)

    records: List[Dict[str, Any]] = []
    targets = select_targets(args.input_root)
    if args.limit is not None:
        targets = targets[: max(0, int(args.limit))]

    for target in targets:
        image = read_gray_image(target.path)
        common = prepare_common(image, target.magnification)
        analysis = analyze_profiles(common, target.magnification)

        stem = target.path.stem.replace(" ", "_")
        panel_path = out_dir / f"{stem}__branch_compare.png"
        render_panel(
            roi=common["roi"],
            mask=common["thresh"],
            accurate_v2_branches=analysis["accurate_v2_branches"],
            accurate_v3_branches=analysis["accurate_v3_branches"],
            fast_v2_branches=analysis["fast_v2_branches"],
            fast_v3_branches=analysis["fast_v3_branches"],
            metrics=analysis["metrics"],
            file_name=target.path.name,
            output_path=panel_path,
        )

        record = {
            "file_name": target.path.name,
            "file_path": str(target.path),
            "magnification": target.magnification,
            "panel_path": str(panel_path),
            **analysis["metrics"],
        }
        records.append(record)

    if records:
        with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)

    with (out_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump({"output_dir": str(out_dir), "count": len(records), "records": records}, fh, ensure_ascii=False, indent=2)

    print(out_dir)


if __name__ == "__main__":
    main()
