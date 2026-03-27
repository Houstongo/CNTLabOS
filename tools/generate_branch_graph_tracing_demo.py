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
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.feature_extractor import FeatureExtractor  # noqa: E402


DEFAULT_IMAGE = Path(r"D:\CNTDATA\coredata\u\100000\No41 200w 5.0nm 10w 2.0nm 600 300 150 600 750 15min 180min mid 100000-1.png")
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "reports"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate branch graph tracing v1 demo.")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--magnification", type=int, default=100000)
    parser.add_argument("--min-length-factor", type=float, default=2.0)
    parser.add_argument("--angle-limit-deg", type=float, default=45.0)
    parser.add_argument("--angle-hard-deg", type=float, default=70.0)
    parser.add_argument("--beam-width", type=int, default=2)
    parser.add_argument("--max-paths", type=int, default=25)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_gray_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return image


def build_mask_base(mask: np.ndarray, fill_color=(42, 42, 42), contour_color=(220, 220, 220)) -> np.ndarray:
    canvas = np.zeros((*mask.shape, 3), dtype=np.uint8)
    canvas[mask > 0] = fill_color
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, contours, -1, contour_color, 1)
    return canvas


def draw_nodes(canvas: np.ndarray, nodes: Dict[str, Dict[str, Any]]) -> np.ndarray:
    result = canvas.copy()
    for node in nodes.values():
        y, x = np.round(node["coord"]).astype(int)
        color = (255, 120, 80) if node["kind"] == "junction" else (80, 220, 255)
        radius = 3 if node["kind"] == "endpoint" else 4
        cv2.circle(result, (x, y), radius, color, -1)
    return result


def draw_branches(canvas: np.ndarray, branches: Dict[str, Dict[str, Any]], color=(150, 150, 150)) -> np.ndarray:
    result = canvas.copy()
    for branch in branches.values():
        coords = np.asarray(branch["coords"], dtype=float)
        if coords.shape[0] < 2:
            continue
        pts = np.round(coords[:, ::-1]).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(result, [pts], False, color, 1, lineType=cv2.LINE_AA)
    return result


def draw_paths(canvas: np.ndarray, paths: List[Dict[str, Any]]) -> np.ndarray:
    palette = [
        (80, 220, 255),
        (120, 255, 120),
        (255, 200, 80),
        (240, 120, 255),
        (255, 120, 120),
        (160, 180, 255),
    ]
    result = canvas.copy()
    for idx, path in enumerate(paths):
        color = palette[idx % len(palette)]
        coords = np.asarray(path["coords"], dtype=float)
        if coords.shape[0] < 2:
            continue
        pts = np.round(coords[:, ::-1]).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(result, [pts], False, color, 2, lineType=cv2.LINE_AA)
        y, x = np.round(coords[0]).astype(int)
        cv2.putText(result, str(path["path_id"]), (x + 3, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return result


def build_text_panel(lines: List[str]) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(6.3, 5.3), dpi=160)
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


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir or (DEFAULT_OUTPUT_ROOT / f"branch_graph_tracing_demo_{timestamp}")
    ensure_dir(out_dir)

    image = read_gray_image(args.image)
    extractor = FeatureExtractor(magnification=args.magnification, speed_profile="accurate")
    result = extractor.extract_branch_graph_paths_v1(
        image,
        min_length_factor=args.min_length_factor,
        angle_limit_deg=args.angle_limit_deg,
        angle_hard_deg=args.angle_hard_deg,
        beam_width=args.beam_width,
        max_paths=args.max_paths,
    )

    roi = result["roi"]
    mask = result["mask"]
    skeleton = (result["skeleton"] > 0).astype(np.uint8)
    nodes = result["nodes"]
    branches = result["branches"]
    paths = result["reconstructed_paths"]

    skeleton_canvas = build_mask_base(mask)
    skeleton_canvas[skeleton > 0] = (255, 230, 90)
    graph_canvas = draw_nodes(draw_branches(build_mask_base(mask), branches), nodes)
    path_canvas = draw_paths(draw_nodes(build_mask_base(mask), nodes), paths[: min(20, len(paths))])

    text_lines = [
        f"file: {args.image.name}",
        f"mag: {args.magnification}",
        f"angle_soft/hard: {args.angle_limit_deg:.1f}/{args.angle_hard_deg:.1f}",
        f"beam_width: {args.beam_width}",
        f"min_len_factor: {args.min_length_factor:.1f}",
        "",
        f"nodes: {len(nodes)}",
        f"branches: {len(branches)}",
        f"paths: {len(paths)}",
        "",
    ]
    for path in paths[:10]:
        text_lines.append(
            f"P{path['path_id']:02d} conf={path['confidence']:.3f} L/D={path['ld_ratio']:.3f} nB={len(path['branch_ids'])}"
        )
        text_lines.append(
            f"   curv={path['mean_curvature_nm']:.6f} p90={path['p90_curvature_nm']:.6f}"
        )
    text_panel = build_text_panel(text_lines)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=160, constrained_layout=True)
    panels = [
        ("Original ROI", cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)),
        ("Raw Skeleton", skeleton_canvas),
        ("Branch Graph + Nodes", graph_canvas),
        ("Reconstructed Paths", path_canvas),
    ]
    for ax, (title, image_panel) in zip(axes.flat[:4], panels):
        ax.imshow(cv2.cvtColor(image_panel, cv2.COLOR_BGR2RGB) if image_panel.ndim == 3 else image_panel, cmap=None if image_panel.ndim == 3 else "gray")
        ax.set_title(title, fontsize=12)
        ax.axis("off")

    fig.savefig(out_dir / "branch_graph_demo.png", bbox_inches="tight")
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(7, 8), dpi=160)
    ax2.imshow(cv2.cvtColor(text_panel, cv2.COLOR_BGR2RGB))
    ax2.axis("off")
    fig2.savefig(out_dir / "path_metrics_panel.png", bbox_inches="tight")
    plt.close(fig2)

    payload = {
        "image": str(args.image),
        "magnification": args.magnification,
        "angle_limit_deg": args.angle_limit_deg,
        "angle_hard_deg": args.angle_hard_deg,
        "beam_width": args.beam_width,
        "min_length_factor": args.min_length_factor,
        "node_count": len(nodes),
        "branch_count": len(branches),
        "path_count": len(paths),
        "paths": [
            {
                key: value
                for key, value in path.items()
                if key != "coords"
            }
            for path in paths
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out_dir)


if __name__ == "__main__":
    main()
