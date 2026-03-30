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
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.feature_extractor import FeatureExtractor
from tools.generate_slice_spur_prune_demo import (
    DEFAULT_REFERENCE_SUMMARY,
    DEFAULT_SOURCE_DIR,
    EXPC_SPEC,
    MODEL_LABEL,
    DemoImage,
    build_bw_mask,
    choose_demo_images,
    draw_skeleton_overlay,
    ensure_dir,
    load_model,
    predict_roi_mask,
    prune_terminal_spurs,
    read_gray_image,
    remove_short_isolated_components,
    render_panel,
    slugify,
)


DEFAULT_SPUR_FACTORS = [3.0, 4.0, 5.0]
FACTOR_COLORS = {
    3.0: (72, 72, 255),
    4.0: (64, 180, 255),
    5.0: (200, 120, 255),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate spur-pruning strength sweep panels and single views.")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--reference-summary", type=Path, default=DEFAULT_REFERENCE_SUMMARY)
    parser.add_argument("--spur-factors", type=float, nargs="+", default=DEFAULT_SPUR_FACTORS)
    return parser.parse_args()


def render_sweep_panel(
    output_path: Path,
    label: str,
    sample_id: str,
    roi: np.ndarray,
    mask: np.ndarray,
    raw_skeleton: np.ndarray,
    isolated_removed_mask: np.ndarray,
    isolated_meta: Dict[str, Any],
    factor_results: List[Dict[str, Any]],
) -> None:
    n_rows = 1 + len(factor_results)
    fig_height = max(6.5, 2.5 * n_rows)
    fig = plt.figure(figsize=(18, fig_height), dpi=180, constrained_layout=True)
    grid = fig.add_gridspec(n_rows, 3, width_ratios=[1, 1, 1], height_ratios=[1] * n_rows)

    top_views = [
        ("Original ROI", roi, "gray"),
        (f"{MODEL_LABEL} Mask", cv2.cvtColor(build_bw_mask(mask), cv2.COLOR_BGR2RGB), None),
        ("Mask + Raw Skeleton", cv2.cvtColor(draw_skeleton_overlay(mask, raw_skeleton, (80, 170, 255)), cv2.COLOR_BGR2RGB), None),
    ]
    for col, (title, image, cmap) in enumerate(top_views):
        ax = fig.add_subplot(grid[0, col])
        if cmap:
            ax.imshow(image, cmap=cmap)
        else:
            ax.imshow(image)
        ax.set_title(title, fontsize=12)
        ax.axis("off")

    for row_idx, result in enumerate(factor_results, start=1):
        factor = float(result["spur_factor"])
        color = FACTOR_COLORS.get(factor, (72, 72, 255))
        removed_overlay = draw_skeleton_overlay(mask, isolated_removed_mask | result["removed_mask"], color)
        pruned_overlay = draw_skeleton_overlay(mask, result["pruned_skeleton"], color)

        ax_removed = fig.add_subplot(grid[row_idx, 0])
        ax_removed.imshow(cv2.cvtColor(removed_overlay, cv2.COLOR_BGR2RGB))
        ax_removed.set_title(f"F{factor:.1f} Removed", fontsize=12)
        ax_removed.axis("off")

        ax_pruned = fig.add_subplot(grid[row_idx, 1])
        ax_pruned.imshow(cv2.cvtColor(pruned_overlay, cv2.COLOR_BGR2RGB))
        ax_pruned.set_title(f"F{factor:.1f} Pruned Skeleton", fontsize=12)
        ax_pruned.axis("off")

        ax_text = fig.add_subplot(grid[row_idx, 2])
        ax_text.axis("off")
        lines = [
            f"spur_factor: {factor:.1f}",
            f"limit_px: {result['spur_length_limit_px']:.2f}",
            f"removed_short: {isolated_meta['removed_component_count']}",
            f"removed_spurs: {result['removed_spur_count']}",
            f"removed_pixels: {isolated_meta['removed_pixel_count'] + result['removed_pixel_count']}",
            f"iterations: {result['iterations']}",
        ]
        y = 0.94
        ax_text.text(0.02, y, f"Strength F{factor:.1f}", fontsize=12.5, fontweight="bold", color="#111827")
        y -= 0.12
        for line in lines:
            ax_text.text(0.03, y, line, fontsize=10.8, color="#111827", family="DejaVu Sans Mono")
            y -= 0.12

    fig.suptitle(
        f"{label}  |  {sample_id}\nshort-isolated cleanup first, then spur pruning sweep",
        fontsize=15,
    )
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def write_rgb_png(path: Path, image_rgb: np.ndarray) -> None:
    ensure_dir(path.parent)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    encoded = cv2.imencode(".png", image_bgr)[1]
    path.write_bytes(encoded.tobytes())


def process_one(
    image: DemoImage,
    model,
    config: Dict[str, Any],
    device: torch.device,
    batch_size: int | None,
    spur_factors: List[float],
    out_dir: Path,
) -> Dict[str, Any]:
    image_gray = read_gray_image(image.path)
    extractor = FeatureExtractor(magnification=image.magnification, speed_profile="accurate")
    roi = extractor.extract_roi(image_gray)
    extractor._calibrate(roi.shape[1])

    mask, _, patch_count = predict_roi_mask(
        model=model,
        roi_gray=roi,
        patch_size=int(config["data"].get("patch_size", 768)),
        stride=int(config["inference"].get("stride", config["data"].get("patch_size", 768) // 2)),
        threshold=float(config["inference"].get("threshold", 0.7)),
        normalize_mean=float(config["data"].get("normalize_mean", 0.5)),
        normalize_std=float(config["data"].get("normalize_std", 0.5)),
        device=device,
        batch_size=batch_size,
    )
    mask = (mask > 0).astype(np.uint8) * 255
    _, raw_skeleton = extractor.calculate_diameter(mask)
    raw_skeleton = (raw_skeleton > 0).astype(np.uint8) * 255
    isolated = remove_short_isolated_components(extractor, raw_skeleton, min_length_factor=2.0, min_points_factor=1.2)
    cleaned_skeleton = isolated["cleaned_skeleton"]

    image_slug = slugify(f"{image.label}_{image.path.stem}")
    item_dir = out_dir / "items" / image_slug
    ensure_dir(item_dir)
    mask_plus_raw_path = item_dir / "mask_plus_raw_skeleton.png"
    mask_plus_raw_rgb = cv2.cvtColor(
        draw_skeleton_overlay(mask, raw_skeleton, (80, 170, 255)),
        cv2.COLOR_BGR2RGB,
    )
    write_rgb_png(mask_plus_raw_path, mask_plus_raw_rgb)

    factor_results: List[Dict[str, Any]] = []
    for factor in spur_factors:
        pruned = prune_terminal_spurs(extractor, cleaned_skeleton, spur_factor=factor)
        single_dir = item_dir / f"factor_{str(factor).replace('.', '_')}"
        ensure_dir(single_dir)
        panel_path = single_dir / "spur_prune_panel.png"
        mask_plus_pruned_path = single_dir / "mask_plus_pruned.png"
        mask_plus_removed_path = single_dir / "mask_plus_removed.png"
        render_panel(
            output_path=panel_path,
            label=f"{image.label} | F{factor:.1f}",
            sample_id=image.path.stem,
            roi=roi,
            mask=mask,
            raw_skeleton=raw_skeleton,
            cleaned_skeleton=cleaned_skeleton,
            isolated_removed_mask=isolated["removed_mask"],
            removed_mask=pruned["removed_mask"],
            pruned_skeleton=pruned["pruned_skeleton"],
            meta={
                "spur_factor": factor,
                "spur_length_limit_px": pruned["spur_length_limit_px"],
                "removed_spur_count": pruned["removed_spur_count"],
                "removed_pixel_count": pruned["removed_pixel_count"],
                "removed_component_count": isolated["removed_component_count"],
                "isolated_removed_pixel_count": isolated["removed_pixel_count"],
            },
        )
        mask_plus_removed_rgb = cv2.cvtColor(
            draw_skeleton_overlay(mask, isolated["removed_mask"] | pruned["removed_mask"], FACTOR_COLORS.get(float(factor), (72, 72, 255))),
            cv2.COLOR_BGR2RGB,
        )
        mask_plus_pruned_rgb = cv2.cvtColor(
            draw_skeleton_overlay(mask, pruned["pruned_skeleton"], FACTOR_COLORS.get(float(factor), (200, 120, 255))),
            cv2.COLOR_BGR2RGB,
        )
        write_rgb_png(mask_plus_removed_path, mask_plus_removed_rgb)
        write_rgb_png(mask_plus_pruned_path, mask_plus_pruned_rgb)
        result = {
            "spur_factor": float(factor),
            "spur_length_limit_px": float(pruned["spur_length_limit_px"]),
            "removed_spur_count": int(pruned["removed_spur_count"]),
            "removed_pixel_count": int(pruned["removed_pixel_count"]),
            "iterations": int(pruned["iterations"]),
            "panel_path": str(panel_path),
            "mask_plus_removed_path": str(mask_plus_removed_path),
            "mask_plus_pruned_path": str(mask_plus_pruned_path),
            "removed_mask": pruned["removed_mask"],
            "pruned_skeleton": pruned["pruned_skeleton"],
        }
        factor_results.append(result)

    combined_panel_path = item_dir / "spur_prune_strength_sweep.png"
    render_sweep_panel(
        output_path=combined_panel_path,
        label=image.label,
        sample_id=image.path.stem,
        roi=roi,
        mask=mask,
        raw_skeleton=raw_skeleton,
        isolated_removed_mask=isolated["removed_mask"],
        isolated_meta=isolated,
        factor_results=factor_results,
    )

    serializable_factors = []
    for result in factor_results:
        serializable_factors.append(
            {
                "spur_factor": result["spur_factor"],
                "spur_length_limit_px": round(result["spur_length_limit_px"], 4),
                "removed_spur_count": result["removed_spur_count"],
                "removed_pixel_count": result["removed_pixel_count"],
                "iterations": result["iterations"],
                "panel_path": result["panel_path"],
                "mask_plus_removed_path": result["mask_plus_removed_path"],
                "mask_plus_pruned_path": result["mask_plus_pruned_path"],
            }
        )

    record = {
        "label": image.label,
        "file_path": str(image.path),
        "sample_id": image.path.stem,
        "magnification": int(image.magnification),
        "patch_count": int(patch_count),
        "runtime_device": device.type,
        "removed_short_component_count": int(isolated["removed_component_count"]),
        "removed_short_pixel_count": int(isolated["removed_pixel_count"]),
        "isolated_min_length_px": round(float(isolated["min_length_px"]), 4),
        "isolated_min_points": int(isolated["min_points"]),
        "mask_plus_raw_path": str(mask_plus_raw_path),
        "combined_panel_path": str(combined_panel_path),
        "factors": serializable_factors,
    }
    (item_dir / "summary.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir or (PROJECT_ROOT / "reports" / f"slice_spur_prune_strength_sweep_{timestamp}")
    ensure_dir(out_dir)

    selected = choose_demo_images(args.source_dir, args.reference_summary)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model, config = load_model(EXPC_SPEC, device)

    spur_factors = [float(v) for v in args.spur_factors]
    results = []
    for image in selected:
        results.append(
            process_one(
                image=image,
                model=model,
                config=config,
                device=device,
                batch_size=args.batch_size,
                spur_factors=spur_factors,
                out_dir=out_dir,
            )
        )

    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "source_dir": str(args.source_dir),
                "model": MODEL_LABEL,
                "spur_factors": spur_factors,
                "count": len(results),
                "items": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(out_dir)


if __name__ == "__main__":
    main()
