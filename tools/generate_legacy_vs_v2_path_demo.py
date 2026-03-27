"""Generate a focused visualization comparing legacy branch coords with V2 ordered paths."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

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


DEFAULT_REPORT_ROOT = PROJECT_ROOT / "reports" / "curvature_v2_comparison_20260325_061334" / "items"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "reports"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate legacy-vs-v2 branch ordering demo.")
    parser.add_argument("--image-id", type=int, default=5895)
    parser.add_argument("--magnification", type=int, default=100000)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_gray_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return image


def build_mask_base(mask: np.ndarray, fill: Tuple[int, int, int] = (40, 40, 40)) -> np.ndarray:
    canvas = np.zeros((*mask.shape, 3), dtype=np.uint8)
    canvas[mask > 0] = fill
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, contours, -1, (220, 220, 220), 1)
    return canvas


def draw_polyline_gradient(canvas: np.ndarray, coords: np.ndarray, step: int = 1) -> None:
    if coords.shape[0] < 2:
        return
    sampled = coords[:: max(1, step)]
    if sampled.shape[0] < 2:
        sampled = coords
    total = sampled.shape[0] - 1
    for idx in range(total):
        frac = idx / max(total, 1)
        color = (
            int(255 * frac),
            int(180 * (1.0 - frac)),
            int(255 * (1.0 - frac)),
        )
        p0 = tuple(np.round(sampled[idx, ::-1]).astype(int))
        p1 = tuple(np.round(sampled[idx + 1, ::-1]).astype(int))
        cv2.line(canvas, p0, p1, color, 1, lineType=cv2.LINE_AA)
    for label_idx, point_idx in enumerate(np.linspace(0, sampled.shape[0] - 1, num=min(8, sampled.shape[0]), dtype=int), start=1):
        x, y = np.round(sampled[point_idx, ::-1]).astype(int)
        cv2.circle(canvas, (x, y), 3, (255, 255, 255), -1)
        cv2.putText(canvas, str(label_idx), (x + 3, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)


def draw_ordered_branch_set(canvas: np.ndarray, branches: List[Dict[str, Any]]) -> None:
    palette = [
        (80, 220, 255),
        (120, 255, 140),
        (255, 200, 80),
        (240, 120, 255),
        (255, 120, 120),
        (120, 180, 255),
    ]
    for idx, branch in enumerate(branches):
        color = palette[idx % len(palette)]
        coords = branch["coords"]
        pts = np.round(coords[:, ::-1]).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], False, color, 1, lineType=cv2.LINE_AA)
        if coords.shape[0] >= 2:
            start = tuple(np.round(coords[0, ::-1]).astype(int))
            end = tuple(np.round(coords[-1, ::-1]).astype(int))
            cv2.circle(canvas, start, 3, (0, 255, 0), -1)
            cv2.circle(canvas, end, 3, (0, 0, 255), -1)
        if coords.shape[0] >= 6:
            p0 = tuple(np.round(coords[2, ::-1]).astype(int))
            p1 = tuple(np.round(coords[5, ::-1]).astype(int))
            cv2.arrowedLine(canvas, p0, p1, color, 1, tipLength=0.35)


def build_text_panel(lines: List[str]) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(6.0, 5.0), dpi=160)
    ax.set_facecolor("#0f172a")
    fig.patch.set_facecolor("#0f172a")
    ax.axis("off")
    y = 0.96
    for line in lines:
        ax.text(0.04, y, line, transform=ax.transAxes, fontsize=11, color="white", va="top", family="DejaVu Sans Mono")
        y -= 0.08
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8).copy()
    plt.close(fig)
    return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)


def crop_image(image: np.ndarray, top: int, left: int, bottom: int, right: int) -> np.ndarray:
    return image[top:bottom, left:right].copy()


def offset_coords(coords: np.ndarray, top: int, left: int) -> np.ndarray:
    shifted = coords.copy().astype(float)
    shifted[:, 0] -= float(top)
    shifted[:, 1] -= float(left)
    return shifted


def find_target_dir(report_root: Path, image_id: int) -> Path:
    matches = sorted(report_root.glob(f"{image_id}_*"))
    if not matches:
        raise FileNotFoundError(f"No report item found for image_id={image_id}")
    return matches[0]


def choose_representative_component(
    components: List[Tuple[np.ndarray, int]],
    neighbor_count: np.ndarray,
) -> Tuple[np.ndarray, int, int]:
    best = None
    for coords, n_points in components:
        points = np.round(coords).astype(int)
        junctions = 0
        for y, x in points:
            if 0 <= y < neighbor_count.shape[0] and 0 <= x < neighbor_count.shape[1] and neighbor_count[y, x] >= 3:
                junctions += 1
        score = (junctions, n_points)
        if best is None or score > best[0]:
            best = (score, coords, n_points, junctions)
    if best is None:
        raise ValueError("No skeleton components found")
    return best[1], int(best[2]), int(best[3])


def main() -> None:
    args = parse_args()
    item_dir = find_target_dir(args.report_root, args.image_id)
    mask_path = item_dir / "paper_repro_mask.png"
    roi_path = item_dir / "roi_original.png"

    mask = read_gray_image(mask_path)
    roi = read_gray_image(roi_path)

    extractor = FeatureExtractor(magnification=args.magnification)
    extractor._calibrate(mask.shape[1])
    _, skel = extractor.calculate_diameter(mask)
    components = extractor._collect_components(skel)
    neighbor_count = extractor._neighbor_count_map((skel > 0).astype(np.uint8))
    legacy_coords, legacy_points, legacy_junctions = choose_representative_component(components, neighbor_count)
    ordered_branches = extractor._collect_ordered_branches_v2(skel, min_points=15)

    y0 = max(0, int(np.floor(legacy_coords[:, 0].min())) - 40)
    x0 = max(0, int(np.floor(legacy_coords[:, 1].min())) - 40)
    y1 = min(mask.shape[0], int(np.ceil(legacy_coords[:, 0].max())) + 41)
    x1 = min(mask.shape[1], int(np.ceil(legacy_coords[:, 1].max())) + 41)

    relevant_v2 = []
    for branch in ordered_branches:
        coords = branch["coords"]
        inside = (
            (coords[:, 0] >= y0) & (coords[:, 0] < y1) &
            (coords[:, 1] >= x0) & (coords[:, 1] < x1)
        )
        if np.any(inside):
            clipped = coords[inside]
            if clipped.shape[0] >= 3:
                relevant_v2.append(
                    {
                        "coords": offset_coords(clipped, y0, x0),
                        "path_length_px": branch["path_length_px"],
                    }
                )

    legacy_crop = offset_coords(legacy_coords, y0, x0)
    mask_crop = crop_image(mask, y0, x0, y1, x1)
    roi_crop = crop_image(roi, y0, x0, y1, x1)
    skel_crop = crop_image((skel > 0).astype(np.uint8) * 255, y0, x0, y1, x1)

    full_mask = build_mask_base(mask)
    points_full = np.round(legacy_coords[:, ::-1]).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(full_mask, [points_full], False, (0, 200, 255), 1, lineType=cv2.LINE_AA)
    cv2.rectangle(full_mask, (x0, y0), (x1 - 1, y1 - 1), (255, 255, 255), 1)

    legacy_canvas = build_mask_base(mask_crop)
    legacy_canvas[skel_crop > 0] = (110, 110, 110)
    draw_polyline_gradient(legacy_canvas, legacy_crop, step=max(1, legacy_crop.shape[0] // 250))

    v2_canvas = build_mask_base(mask_crop)
    v2_canvas[skel_crop > 0] = (100, 100, 100)
    draw_ordered_branch_set(v2_canvas, relevant_v2[:40])

    crop_canvas = cv2.cvtColor(roi_crop, cv2.COLOR_GRAY2BGR)
    crop_canvas[skel_crop > 0] = (255, 255, 255)

    text_panel = build_text_panel(
        [
            "Legacy vs V2 Path Order",
            f"image_id        = {args.image_id}",
            f"legacy_points   = {legacy_points}",
            f"legacy_junction = {legacy_junctions}",
            f"crop_v2_branches= {len(relevant_v2)}",
            "",
            "Legacy branch:",
            "coords come from connected-component",
            "pixel collection order (np.argwhere),",
            "so adjacent coords are not guaranteed",
            "to be true path neighbors.",
            "",
            "V2 branch:",
            "junctions are cut first, then each",
            "branch is traced into an ordered path",
            "before smoothing + curvature.",
        ]
    )

    tiles = [
        (full_mask, "1. Full Mask + Selected Legacy Component", "青色折线是旧版挑中的连通组件，白框是放大区域。"),
        (crop_canvas, "2. ROI Crop + Skeleton", "同一区域的原始局部和 skeleton，仅作对照参考。"),
        (legacy_canvas, "3. Legacy Branch Coord Order", "旧版按连通域坐标顺序连线，颜色从紫/红渐变到蓝，能看到非真实路径跳线。"),
        (v2_canvas, "4. V2 Ordered Branches", "V2 先切断 junction，再分别追踪成真实路径，绿点起点，红点终点。"),
        (text_panel, "5. Why Curvature Differs", "旧版更容易把非路径相邻点拿去算三点曲率；V2 则沿真实中轴线采样。"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=130)
    axes = axes.flatten()
    for ax, (image, title, caption) in zip(axes, tiles):
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axis("off")
        ax.text(0.5, -0.08, caption, transform=ax.transAxes, ha="center", va="top", fontsize=9, color="#475569", wrap=True)
    for ax in axes[len(tiles):]:
        ax.axis("off")

    fig.suptitle(f"Legacy Branch Order vs V2 Ordered Path | image_id={args.image_id}", fontsize=16, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if args.output_dir is not None:
        output_root = args.output_dir
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_root = DEFAULT_OUTPUT_ROOT / f"legacy_vs_v2_path_demo_{args.image_id}_{stamp}"
    ensure_dir(output_root)

    panel_path = output_root / "legacy_vs_v2_path_demo.png"
    fig.savefig(panel_path, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "image_id": args.image_id,
        "magnification": args.magnification,
        "source_item_dir": str(item_dir),
        "mask_path": str(mask_path),
        "roi_path": str(roi_path),
        "selected_crop": {"top": y0, "left": x0, "bottom": y1, "right": x1},
        "legacy_component_points": legacy_points,
        "legacy_component_junction_points": legacy_junctions,
        "v2_branch_count_in_crop": len(relevant_v2),
        "panel_path": str(panel_path),
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"OUTPUT_DIR={output_root}")
    print(f"PANEL_PATH={panel_path}")


if __name__ == "__main__":
    main()
