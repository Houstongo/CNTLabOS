from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.generate_xr_slice_standard_batch import build_standard_summary_sections, ensure_dir, slugify

LEFT_PANEL_RATIO = 0.42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rerender standard-method panels from cached features/panels without rerunning analysis.")
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def load_records(report_dir: Path) -> list[dict]:
    records: list[dict] = []
    for features_path in sorted((report_dir / "items").glob("*/features.json")):
        try:
            records.append(json.loads(features_path.read_text(encoding="utf-8")))
        except Exception:
            continue
    return records


def render_cached_panel(src_panel_path: Path, record: dict, output_path: Path) -> None:
    panel_bgr = cv2.imread(str(src_panel_path), cv2.IMREAD_COLOR)
    if panel_bgr is None:
        raise ValueError(f"Failed to read cached panel: {src_panel_path}")
    panel_rgb = cv2.cvtColor(panel_bgr, cv2.COLOR_BGR2RGB)
    left_width = max(1, int(round(panel_rgb.shape[1] * LEFT_PANEL_RATIO)))
    left_crop = panel_rgb[:, :left_width]

    fig = plt.figure(figsize=(18, 12), dpi=170, constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=[left_crop.shape[1], max(1, panel_rgb.shape[1] - left_width)])

    ax_left = fig.add_subplot(grid[0, 0])
    ax_left.imshow(left_crop)
    ax_left.axis("off")

    ax_right = fig.add_subplot(grid[0, 1])
    ax_right.set_facecolor("white")
    ax_right.axis("off")
    y = 0.985
    for section in build_standard_summary_sections(record):
        ax_right.text(
            0.02,
            y,
            section["title"],
            va="top",
            ha="left",
            fontsize=12.6,
            fontweight="bold",
            color=section["color"],
            family="DejaVu Sans Mono",
        )
        y -= 0.04
        for line in section["lines"]:
            ax_right.text(
                0.03,
                y,
                str(line),
                va="top",
                ha="left",
                fontsize=10.0,
                color="#111827",
                family="DejaVu Sans Mono",
            )
            y -= 0.032
        y -= 0.018

    fig.suptitle("CNTSegNet-SLICE  |  XR Standard Method  |  V3 + Length Threshold L1-L4", fontsize=18)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def copy_grouped_panel(report_dir: Path, record: dict, panel_path: Path) -> None:
    magnification = int(record["magnification"])
    grouped_dir = report_dir / f"panels_{magnification}x"
    ensure_dir(grouped_dir)
    dst = grouped_dir / f"{slugify(record['sample_id'])}__panel.png"
    dst.write_bytes(panel_path.read_bytes())


def main() -> None:
    args = parse_args()
    records = load_records(args.report_dir)
    if args.limit is not None:
        records = records[: max(0, int(args.limit))]

    for record in records:
        panel_path = Path(record["panel_path"])
        if not panel_path.exists():
            continue
        render_cached_panel(panel_path, record, panel_path)
        copy_grouped_panel(args.report_dir, record, panel_path)

    print(args.report_dir)


if __name__ == "__main__":
    main()
