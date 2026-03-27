from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

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


DEFAULT_COMPARISON_CSV = PROJECT_ROOT / "reports" / "curvature_v2_comparison_20260325_061334" / "comparison.csv"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "reports"
COLOR_CYCLE = [
    "#ff6b6b",
    "#4ecdc4",
    "#ffe66d",
    "#5dade2",
    "#58d68d",
    "#f5b041",
    "#ec7063",
    "#a569bd",
    "#48c9b0",
    "#f1948a",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Kappa-lite top-segment validation panel.")
    parser.add_argument("--comparison-csv", type=Path, default=DEFAULT_COMPARISON_CSV)
    parser.add_argument("--magnification", type=int, default=100000)
    parser.add_argument("--image-id", type=int, default=None)
    parser.add_argument("--image-path", type=Path, default=None)
    parser.add_argument("--mask-path", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_gray_image(path: str | Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return image


def slugify(text: str) -> str:
    safe = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_"}:
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe).strip("_") or "image"


def pick_target_from_comparison(csv_path: Path, magnification: int, image_id: Optional[int]) -> Dict[str, Any]:
    rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8-sig", newline="")))
    candidates = []
    for row in rows:
        if row.get("status") != "success":
            continue
        if int(row["magnification"]) != int(magnification):
            continue
        if image_id is not None and int(row["image_id"]) != int(image_id):
            continue
        image_path = Path(row["file_path"])
        mask_path = Path(row.get("new_mask_path") or row.get("baseline_mask_path") or "")
        if not image_path.exists() or not mask_path.exists():
            continue
        candidates.append(row)

    if not candidates:
        raise FileNotFoundError("No matching 100000x comparison record with accessible image/mask was found.")
    candidates.sort(key=lambda item: int(item["image_id"]))
    return candidates[0]


def plot_branch_overlay(ax, background: np.ndarray, branches: List[Dict[str, Any]], title: str, annotate: bool = False) -> None:
    ax.imshow(background, cmap="gray")
    for idx, branch in enumerate(branches):
        coords = np.asarray(branch["coords"], dtype=float)
        if coords.size == 0:
            continue
        color = COLOR_CYCLE[idx % len(COLOR_CYCLE)]
        ax.plot(coords[:, 1], coords[:, 0], color=color, linewidth=1.2 if not annotate else 2.2, alpha=0.95)
        if annotate:
            mid_idx = coords.shape[0] // 2
            ax.text(
                coords[mid_idx, 1],
                coords[mid_idx, 0],
                str(branch["segment_id"]),
                color="white",
                fontsize=9,
                ha="center",
                va="center",
                bbox={"boxstyle": "round,pad=0.18", "facecolor": color, "edgecolor": "none", "alpha": 0.95},
            )
    ax.set_title(title, fontsize=12)
    ax.axis("off")


def serialize_segment(segment: Dict[str, Any]) -> Dict[str, Any]:
    record = {}
    for key, value in segment.items():
        if key.startswith("_"):
            continue
        if isinstance(value, np.ndarray):
            record[key] = value.tolist()
        elif isinstance(value, (np.floating, np.integer)):
            record[key] = value.item()
        else:
            record[key] = value
    return record


def render_panel(
    roi: np.ndarray,
    mask: np.ndarray,
    ordered_branches: List[Dict[str, Any]],
    selected_segments: List[Dict[str, Any]],
    metadata: Dict[str, Any],
    output_path: Path,
) -> None:
    fig = plt.figure(figsize=(16, 12), dpi=180, constrained_layout=True)
    grid = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 0.92])

    ax_roi = fig.add_subplot(grid[0, 0])
    ax_mask = fig.add_subplot(grid[0, 1])
    ax_branches = fig.add_subplot(grid[1, 0])
    ax_selected = fig.add_subplot(grid[1, 1])
    ax_table = fig.add_subplot(grid[2, :])

    ax_roi.imshow(roi, cmap="gray")
    ax_roi.set_title("Original ROI", fontsize=12)
    ax_roi.axis("off")

    ax_mask.imshow(mask, cmap="gray")
    ax_mask.set_title("Binary Mask", fontsize=12)
    ax_mask.axis("off")

    plot_branch_overlay(
        ax_branches,
        mask,
        ordered_branches[: min(len(ordered_branches), 32)],
        title=f"Ordered Branch Reference ({len(ordered_branches)} branches)",
        annotate=False,
    )
    plot_branch_overlay(
        ax_selected,
        roi,
        selected_segments,
        title=f"Selected Top {len(selected_segments)} Segments",
        annotate=True,
    )

    ax_table.axis("off")
    column_defs = [
        ("segment_id", "ID"),
        ("score", "Score"),
        ("ld_ratio", "L/D"),
        ("path_length_nm", "Length (nm)"),
        ("mean_curvature_nm", "Mean Curv."),
        ("p90_curvature_nm", "P90 Curv."),
        ("mean_width_nm", "Mean Width"),
        ("width_cv", "Width CV"),
        ("junction_distance_px", "Junc Dist"),
    ]
    table_rows = []
    for segment in selected_segments:
        table_rows.append(
            [
                str(segment["segment_id"]),
                f"{segment['score']:.3f}",
                f"{segment['ld_ratio']:.3f}",
                f"{segment['path_length_nm']:.1f}",
                f"{segment['mean_curvature_nm']:.4f}",
                f"{segment['p90_curvature_nm']:.4f}",
                f"{segment['mean_width_nm']:.2f}",
                f"{segment['width_cv']:.3f}",
                f"{segment['junction_distance_px']:.1f}",
            ]
        )

    if not table_rows:
        table_rows = [["-", "-", "-", "-", "-", "-", "-", "-", "-"]]
    table = ax_table.table(
        cellText=table_rows,
        colLabels=[label for _, label in column_defs],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.35)

    fig.suptitle(
        f"Kappa-lite Segment Selection\n"
        f"{metadata['label']} | density={metadata['density']:.2f}% | px_per_um={metadata['px_per_um']:.2f}",
        fontsize=15,
    )
    ensure_dir(output_path.parent)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.output_dir is None:
        output_dir = DEFAULT_OUTPUT_ROOT / f"kappa_lite_segment_panel_{timestamp}"
    else:
        output_dir = args.output_dir
    ensure_dir(output_dir)

    if args.image_path is not None:
        if args.mask_path is None:
            raise ValueError("--mask-path is required when --image-path is provided.")
        image_path = args.image_path
        mask_path = args.mask_path
        label = image_path.stem
        magnification = int(args.magnification)
        source_row = {
            "image_id": None,
            "sample_id": label,
            "file_path": str(image_path),
            "new_mask_path": str(mask_path),
            "magnification": magnification,
        }
    else:
        source_row = pick_target_from_comparison(args.comparison_csv, args.magnification, args.image_id)
        image_path = Path(source_row["file_path"])
        mask_path = Path(source_row["new_mask_path"])
        label = f"{source_row['sample_id']} (image_id={source_row['image_id']})"
        magnification = int(source_row["magnification"])

    image_gray = read_gray_image(image_path)
    mask_gray = read_gray_image(mask_path)

    extractor = FeatureExtractor(magnification=magnification)
    result = extractor.extract_kappa_lite_segments(
        image_gray,
        external_binary_mask=mask_gray,
        top_k=args.top_k,
    )

    selected_segments = [serialize_segment(segment) for segment in result["selected_segments"]]
    ordered_branches = [
        {"coords": np.asarray(branch["coords"], dtype=float)}
        for branch in result["ordered_branches"]
    ]

    stem = slugify(source_row.get("sample_id") or image_path.stem)
    panel_path = output_dir / f"{stem}__kappa_lite_panel.png"
    json_path = output_dir / f"{stem}__segments.json"
    csv_path = output_dir / f"{stem}__segments.csv"

    render_panel(
        roi=result["roi"],
        mask=result["mask"],
        ordered_branches=ordered_branches,
        selected_segments=selected_segments,
        metadata={
            "label": label,
            "density": result["density"],
            "px_per_um": result["px_per_um"],
        },
        output_path=panel_path,
    )

    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "source": {
                    "image_path": str(image_path),
                    "mask_path": str(mask_path),
                    "image_id": source_row.get("image_id"),
                    "sample_id": source_row.get("sample_id"),
                    "magnification": magnification,
                },
                "density": result["density"],
                "px_per_um": result["px_per_um"],
                "ordered_branch_count": len(result["ordered_branches"]),
                "candidate_count": len(result["candidate_segments"]),
                "selected_count": len(selected_segments),
                "segments": selected_segments,
            },
            fh,
            ensure_ascii=False,
            indent=2,
        )

    csv_fields = [
        "segment_id",
        "score",
        "length_score",
        "junction_distance_score",
        "width_consistency_score",
        "path_length_px",
        "path_length_nm",
        "span_px",
        "span_nm",
        "ld_ratio",
        "mean_curvature_nm",
        "p90_curvature_nm",
        "mean_width_nm",
        "width_cv",
        "border_distance_px",
        "junction_distance_px",
        "n_points",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=csv_fields)
        writer.writeheader()
        for segment in selected_segments:
            writer.writerow({field: segment.get(field) for field in csv_fields})

    print(output_dir)


if __name__ == "__main__":
    main()
