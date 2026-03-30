from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping

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

from experiments.cnt_paper_repro.model import ResNet34UNet
from src.analysis.feature_extractor import FeatureExtractor
from tools.generate_xr_slice_standard_batch import (
    EXPC_SPEC,
    DEFAULT_OUTPUT_ROOT,
    MODEL_LABEL,
    ensure_dir,
    infer_magnification_from_name,
    load_model,
    predict_roi_mask,
    read_gray_image,
    slugify,
)


DEFAULT_SOURCE_DIR = Path(r"C:\Users\clearlove\Desktop\text10")
DEFAULT_REFERENCE_SUMMARY = PROJECT_ROOT / "reports" / "text10_slice_standard_batch_mean" / "summary.csv"
STEP_FILENAMES = {
    "roi": "01_roi.png",
    "mask": "02_mask.png",
    "raw_skeleton": "03_raw_skeleton.png",
    "removed_short": "04_removed_short.png",
    "removed_spur": "05_removed_spur.png",
    "cleaned_skeleton": "06_cleaned_skeleton.png",
    "v3_metrics_overlay": "07_v3_metrics_overlay.png",
}


@dataclass(frozen=True)
class DemoImage:
    label: str
    path: Path
    magnification: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate minimal CNTSegNet-SLICE -> skeleton -> spur-pruned demo panels.")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--spur-factor", type=float, default=3.0)
    parser.add_argument("--reference-summary", type=Path, default=DEFAULT_REFERENCE_SUMMARY)
    return parser.parse_args()


def build_bw_mask(mask: np.ndarray) -> np.ndarray:
    canvas = np.zeros((*mask.shape, 3), dtype=np.uint8)
    canvas[mask > 0] = (245, 245, 245)
    return canvas


def build_bw_skeleton(skeleton: np.ndarray) -> np.ndarray:
    canvas = np.zeros((*skeleton.shape, 3), dtype=np.uint8)
    canvas[skeleton > 0] = (245, 245, 245)
    return canvas


def draw_skeleton_overlay(mask: np.ndarray, skeleton: np.ndarray, color_bgr: tuple[int, int, int]) -> np.ndarray:
    canvas = build_bw_mask(mask)
    canvas[skeleton > 0] = color_bgr
    return canvas


def build_removed_overlay(
    mask: np.ndarray,
    base_skeleton: np.ndarray,
    removed_mask: np.ndarray,
    kept_color_bgr: tuple[int, int, int] = (110, 110, 110),
    removed_color_bgr: tuple[int, int, int] = (72, 72, 255),
) -> np.ndarray:
    canvas = build_bw_mask(mask)
    canvas[base_skeleton > 0] = kept_color_bgr
    canvas[removed_mask > 0] = removed_color_bgr
    return canvas


def load_reference_ranking(summary_path: Path, source_dir: Path) -> List[DemoImage]:
    if not summary_path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with summary_path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            file_path = Path(row["file_path"])
            try:
                if file_path.parent.resolve() != source_dir.resolve():
                    continue
            except Exception:
                continue
            try:
                curvature = float(row["l2_curvature_sqrt_length_nm"])
                magnification = int(row["magnification"])
            except Exception:
                continue
            rows.append(
                {
                    "file_path": file_path,
                    "sample_id": row["sample_id"],
                    "magnification": magnification,
                    "curvature": curvature,
                }
            )
    if len(rows) < 3:
        return []
    rows.sort(key=lambda item: item["curvature"])
    picks = [rows[0], rows[len(rows) // 2], rows[-1]]
    labels = ["Straighter", "Medium", "Complex"]
    return [
        DemoImage(label=label, path=Path(item["file_path"]), magnification=int(item["magnification"]))
        for label, item in zip(labels, picks)
    ]


def fallback_select_images(source_dir: Path) -> List[DemoImage]:
    files = []
    for path in sorted(source_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
            continue
        magnification = infer_magnification_from_name(path)
        if magnification is None:
            continue
        files.append((path, magnification))
    if len(files) < 3:
        raise ValueError(f"Need at least 3 images in {source_dir}")
    indices = [0, len(files) // 2, len(files) - 1]
    labels = ["Sample-A", "Sample-B", "Sample-C"]
    return [
        DemoImage(label=label, path=files[idx][0], magnification=files[idx][1])
        for label, idx in zip(labels, indices)
    ]


def choose_demo_images(source_dir: Path, summary_path: Path) -> List[DemoImage]:
    ranked = load_reference_ranking(summary_path, source_dir)
    if ranked:
        return ranked
    return fallback_select_images(source_dir)


def save_image(output_path: Path, image: np.ndarray) -> None:
    ensure_dir(output_path.parent)
    array = np.asarray(image)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim == 2:
        cv2.imwrite(str(output_path), array)
        return
    cv2.imwrite(str(output_path), cv2.cvtColor(array, cv2.COLOR_RGB2BGR))


def format_value(value: Any, precision: int = 4, suffix: str = "") -> str:
    if value is None:
        return "N/A"
    if isinstance(value, str):
        return value
    if isinstance(value, (int, np.integer)):
        return f"{int(value)}{suffix}"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{precision}f}{suffix}"
    return f"{value}{suffix}"


def build_metric_lines(feature_metrics: Mapping[str, Any], meta: Mapping[str, Any]) -> List[str]:
    return [
        "Core Metrics",
        f"density: {format_value(feature_metrics.get('density'), 2, ' %')}",
        f"alignment: {format_value(feature_metrics.get('alignment'), 4)}",
        f"diameter: {format_value(feature_metrics.get('diameter'), 2, ' nm')}",
        "",
        "V3 Curvature",
        f"label: {format_value(feature_metrics.get('curvature_v3'))}",
        f"main p75 sqrtL: {format_value(feature_metrics.get('curvature_nm_v3_p75_sqrt_length'), 6)}",
        f"p50  len/sqrtL: {format_value(feature_metrics.get('curvature_nm_v3_p50_length'), 6)} / {format_value(feature_metrics.get('curvature_nm_v3_p50_sqrt_length'), 6)}",
        f"p75  len/sqrtL: {format_value(feature_metrics.get('curvature_nm_v3_p75_length'), 6)} / {format_value(feature_metrics.get('curvature_nm_v3_p75_sqrt_length'), 6)}",
        f"mean len/sqrtL: {format_value(feature_metrics.get('curvature_nm_v3_mean_length'), 6)} / {format_value(feature_metrics.get('curvature_nm_v3_mean_sqrt_length'), 6)}",
        f"trim len/sqrtL: {format_value(feature_metrics.get('curvature_nm_v3_trimmed_mean_length'), 6)} / {format_value(feature_metrics.get('curvature_nm_v3_trimmed_mean_sqrt_length'), 6)}",
        f"branches: {format_value(feature_metrics.get('curvature_v3_branch_count'), 0)}",
        "",
        "Waviness",
        f"ratio: {format_value(feature_metrics.get('waviness_ratio'), 4)}",
        f"tortuosity: {format_value(feature_metrics.get('tortuosity'), 3)}",
        f"height/wavelength: {format_value(feature_metrics.get('waviness_height_nm'), 2, ' nm')} / {format_value(feature_metrics.get('waviness_wavelength_nm'), 2, ' nm')}",
        "",
        "Topology-Clean",
        f"removed short/spur: {format_value(meta.get('removed_short_component_count'), 0)} / {format_value(meta.get('removed_spur_count'), 0)}",
        f"removed px short/spur: {format_value(meta.get('removed_short_pixel_count'), 0)} / {format_value(meta.get('removed_spur_pixel_count'), 0)}",
        f"spur factor/limit: {format_value(meta.get('spur_factor'), 1)} / {format_value(meta.get('spur_length_limit_px'), 2, ' px')}",
        f"isolated min len/pts: {format_value(meta.get('isolated_min_length_px'), 2, ' px')} / {format_value(meta.get('isolated_min_points'), 0)}",
        f"spur iterations: {format_value(meta.get('spur_iterations'), 0)}",
    ]


def render_metric_card(ax: plt.Axes, metric_lines: List[str]) -> None:
    ax.axis("off")
    y = 0.98
    line_height = 0.045
    for line in metric_lines:
        if not line:
            y -= line_height * 0.7
            continue
        if line in {"Core Metrics", "V3 Curvature", "Waviness", "Topology-Clean"}:
            ax.text(0.0, y, line, fontsize=11, fontweight="bold", va="top", ha="left")
            y -= line_height
            continue
        ax.text(0.0, y, line, fontsize=9.2, family="monospace", va="top", ha="left")
        y -= line_height


def render_metrics_overlay(
    output_path: Path,
    sample_id: str,
    overlay_rgb: np.ndarray,
    metric_lines: List[str],
) -> None:
    fig = plt.figure(figsize=(12.8, 5.4), dpi=180, constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.2, 0.9])

    ax_image = fig.add_subplot(grid[0, 0])
    ax_image.imshow(overlay_rgb)
    ax_image.set_title(f"{MODEL_LABEL} Cleaned Overlay", fontsize=12)
    ax_image.axis("off")

    ax_text = fig.add_subplot(grid[0, 1])
    render_metric_card(ax_text, metric_lines)

    fig.suptitle(sample_id, fontsize=13)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def render_overview_panel(
    output_path: Path,
    label: str,
    sample_id: str,
    roi: np.ndarray,
    mask: np.ndarray,
    raw_skeleton: np.ndarray,
    isolated_removed_mask: np.ndarray,
    isolated_skeleton: np.ndarray,
    removed_spur_mask: np.ndarray,
    cleaned_skeleton: np.ndarray,
    metrics_overlay_rgb: np.ndarray,
    feature_metrics: Mapping[str, Any],
    meta: Mapping[str, Any],
) -> None:
    fig = plt.figure(figsize=(18, 9.6), dpi=180, constrained_layout=True)
    grid = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1.15])

    views = [
        ("Original ROI", roi, "gray"),
        (f"{MODEL_LABEL} Mask", cv2.cvtColor(build_bw_mask(mask), cv2.COLOR_BGR2RGB), None),
        ("Mask + Raw Skeleton", cv2.cvtColor(draw_skeleton_overlay(mask, raw_skeleton, (80, 170, 255)), cv2.COLOR_BGR2RGB), None),
        ("Mask + Removed Short", cv2.cvtColor(build_removed_overlay(mask, raw_skeleton, isolated_removed_mask), cv2.COLOR_BGR2RGB), None),
        ("Mask + Removed Spur", cv2.cvtColor(build_removed_overlay(mask, isolated_skeleton, removed_spur_mask), cv2.COLOR_BGR2RGB), None),
        ("Mask + Cleaned Skeleton", cv2.cvtColor(draw_skeleton_overlay(mask, cleaned_skeleton, (186, 120, 255)), cv2.COLOR_BGR2RGB), None),
        ("V3 Metrics Overlay", metrics_overlay_rgb, None),
    ]

    for idx, (title, image, cmap) in enumerate(views):
        row, col = divmod(idx, 4)
        ax = fig.add_subplot(grid[row, col])
        if cmap:
            ax.imshow(image, cmap=cmap)
        else:
            ax.imshow(image)
        ax.set_title(title, fontsize=12)
        ax.axis("off")

    ax_text = fig.add_subplot(grid[1, 3])
    render_metric_card(ax_text, build_metric_lines(feature_metrics, meta))

    fig.suptitle(
        f"{label}  |  {sample_id}\n"
        f"spur_factor={meta['spur_factor']:.1f}  limit_px={meta['spur_length_limit_px']:.2f} px  "
        f"removed_spurs={meta['removed_spur_count']}  removed_short={meta['removed_short_component_count']}  "
        f"branch_count={feature_metrics.get('curvature_v3_branch_count', 0)}",
        fontsize=14,
    )
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_step_images(
    steps_dir: Path,
    sample_id: str,
    roi: np.ndarray,
    mask: np.ndarray,
    raw_skeleton: np.ndarray,
    isolated_removed_mask: np.ndarray,
    isolated_skeleton: np.ndarray,
    removed_spur_mask: np.ndarray,
    cleaned_skeleton: np.ndarray,
    metrics_overlay_rgb: np.ndarray,
    metric_lines: List[str],
) -> Dict[str, str]:
    ensure_dir(steps_dir)
    step_paths = {
        "roi": steps_dir / STEP_FILENAMES["roi"],
        "mask": steps_dir / STEP_FILENAMES["mask"],
        "raw_skeleton": steps_dir / STEP_FILENAMES["raw_skeleton"],
        "removed_short": steps_dir / STEP_FILENAMES["removed_short"],
        "removed_spur": steps_dir / STEP_FILENAMES["removed_spur"],
        "cleaned_skeleton": steps_dir / STEP_FILENAMES["cleaned_skeleton"],
        "v3_metrics_overlay": steps_dir / STEP_FILENAMES["v3_metrics_overlay"],
    }

    save_image(step_paths["roi"], roi)
    save_image(step_paths["mask"], cv2.cvtColor(build_bw_mask(mask), cv2.COLOR_BGR2RGB))
    save_image(step_paths["raw_skeleton"], cv2.cvtColor(draw_skeleton_overlay(mask, raw_skeleton, (80, 170, 255)), cv2.COLOR_BGR2RGB))
    save_image(step_paths["removed_short"], cv2.cvtColor(build_removed_overlay(mask, raw_skeleton, isolated_removed_mask), cv2.COLOR_BGR2RGB))
    save_image(step_paths["removed_spur"], cv2.cvtColor(build_removed_overlay(mask, isolated_skeleton, removed_spur_mask), cv2.COLOR_BGR2RGB))
    save_image(step_paths["cleaned_skeleton"], cv2.cvtColor(draw_skeleton_overlay(mask, cleaned_skeleton, (186, 120, 255)), cv2.COLOR_BGR2RGB))
    render_metrics_overlay(
        output_path=step_paths["v3_metrics_overlay"],
        sample_id=sample_id,
        overlay_rgb=metrics_overlay_rgb,
        metric_lines=metric_lines,
    )
    return {key: str(path) for key, path in step_paths.items()}


def process_one(
    image: DemoImage,
    model: ResNet34UNet,
    config: Dict[str, Any],
    device: torch.device,
    batch_size: int | None,
    spur_factor: float,
    out_dir: Path,
) -> Dict[str, Any]:
    image_gray = read_gray_image(image.path)
    extractor = FeatureExtractor(magnification=image.magnification, speed_profile="accurate")
    extractor.BRANCH_CLEANUP_SPUR_LENGTH_FACTOR = float(spur_factor)
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
    isolated = extractor._remove_short_isolated_skeleton_components(
        raw_skeleton,
        min_length_factor=2.0,
        min_points_factor=1.2,
    )
    isolated_skeleton = isolated["cleaned_skeleton"].astype(np.uint8) * 255
    pruned = extractor._prune_terminal_spurs(
        isolated["cleaned_skeleton"],
        spur_factor=spur_factor,
    )
    cleaned_skeleton = pruned["cleaned_skeleton"].astype(np.uint8) * 255
    removed_spur_mask = pruned["removed_spur_mask"].astype(np.uint8) * 255
    isolated_removed_mask = isolated["removed_mask"].astype(np.uint8) * 255
    feature_metrics = extractor.extract_all(image_gray, external_binary_mask=mask)

    image_slug = slugify(f"{image.label}_{image.path.stem}")
    item_dir = out_dir / "items" / image_slug
    ensure_dir(item_dir)
    steps_dir = item_dir / "steps"
    metrics_overlay_rgb = cv2.cvtColor(draw_skeleton_overlay(mask, cleaned_skeleton, (186, 120, 255)), cv2.COLOR_BGR2RGB)
    cleanup_meta = {
        "spur_factor": spur_factor,
        "removed_short_component_count": feature_metrics["removed_short_component_count"],
        "removed_short_pixel_count": feature_metrics["removed_short_pixel_count"],
        "removed_spur_count": feature_metrics["removed_spur_count"],
        "removed_spur_pixel_count": feature_metrics["removed_spur_pixel_count"],
        "spur_length_limit_px": feature_metrics["spur_length_limit_px"],
        "isolated_min_length_px": feature_metrics["isolated_min_length_px"],
        "isolated_min_points": feature_metrics["isolated_min_points"],
        "spur_iterations": feature_metrics["spur_iterations"],
    }
    metric_lines = build_metric_lines(feature_metrics, cleanup_meta)
    step_paths = save_step_images(
        steps_dir=steps_dir,
        sample_id=image.path.stem,
        roi=roi,
        mask=mask,
        raw_skeleton=raw_skeleton,
        isolated_removed_mask=isolated_removed_mask,
        isolated_skeleton=isolated_skeleton,
        removed_spur_mask=removed_spur_mask,
        cleaned_skeleton=cleaned_skeleton,
        metrics_overlay_rgb=metrics_overlay_rgb,
        metric_lines=metric_lines,
    )

    overview_panel_path = item_dir / "overview_panel.png"
    render_overview_panel(
        output_path=overview_panel_path,
        label=image.label,
        sample_id=image.path.stem,
        roi=roi,
        mask=mask,
        raw_skeleton=raw_skeleton,
        cleaned_skeleton=cleaned_skeleton,
        isolated_removed_mask=isolated_removed_mask,
        isolated_skeleton=isolated_skeleton,
        removed_spur_mask=removed_spur_mask,
        metrics_overlay_rgb=metrics_overlay_rgb,
        feature_metrics=feature_metrics,
        meta=cleanup_meta,
    )

    metrics_path = item_dir / "metrics.json"
    metrics_payload = {
        "label": image.label,
        "file_path": str(image.path),
        "sample_id": image.path.stem,
        "magnification": int(image.magnification),
        "patch_count": int(patch_count),
        "runtime_device": device.type,
        "spur_factor": float(spur_factor),
        "feature_metrics": feature_metrics,
        "artifacts": {
            "overview_panel_path": str(overview_panel_path),
            "steps_dir": str(steps_dir),
            "step_paths": step_paths,
        },
    }
    metrics_path.write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    result = {
        "label": image.label,
        "file_path": str(image.path),
        "sample_id": image.path.stem,
        "magnification": int(image.magnification),
        "patch_count": int(patch_count),
        "runtime_device": device.type,
        "spur_factor": float(spur_factor),
        "density": feature_metrics["density"],
        "alignment": feature_metrics["alignment"],
        "diameter": feature_metrics["diameter"],
        "curvature_v3": feature_metrics["curvature_v3"],
        "curvature_nm_v3": feature_metrics["curvature_nm_v3"],
        "curvature_nm_v3_p50_length": feature_metrics["curvature_nm_v3_p50_length"],
        "curvature_nm_v3_p50_sqrt_length": feature_metrics["curvature_nm_v3_p50_sqrt_length"],
        "curvature_nm_v3_p75_length": feature_metrics["curvature_nm_v3_p75_length"],
        "curvature_nm_v3_p75_sqrt_length": feature_metrics["curvature_nm_v3_p75_sqrt_length"],
        "curvature_nm_v3_mean_length": feature_metrics["curvature_nm_v3_mean_length"],
        "curvature_nm_v3_mean_sqrt_length": feature_metrics["curvature_nm_v3_mean_sqrt_length"],
        "curvature_nm_v3_trimmed_mean_length": feature_metrics["curvature_nm_v3_trimmed_mean_length"],
        "curvature_nm_v3_trimmed_mean_sqrt_length": feature_metrics["curvature_nm_v3_trimmed_mean_sqrt_length"],
        "curvature_v3_branch_count": feature_metrics["curvature_v3_branch_count"],
        "waviness_ratio": feature_metrics["waviness_ratio"],
        "waviness_height_nm": feature_metrics["waviness_height_nm"],
        "waviness_wavelength_nm": feature_metrics["waviness_wavelength_nm"],
        "tortuosity": feature_metrics["tortuosity"],
        "isolated_min_length_px": feature_metrics["isolated_min_length_px"],
        "isolated_min_points": feature_metrics["isolated_min_points"],
        "removed_short_component_count": feature_metrics["removed_short_component_count"],
        "removed_short_pixel_count": feature_metrics["removed_short_pixel_count"],
        "spur_length_limit_px": feature_metrics["spur_length_limit_px"],
        "removed_spur_count": feature_metrics["removed_spur_count"],
        "removed_spur_pixel_count": feature_metrics["removed_spur_pixel_count"],
        "spur_iterations": feature_metrics["spur_iterations"],
        "panel_path": str(overview_panel_path),
        "overview_panel_path": str(overview_panel_path),
        "metrics_path": str(metrics_path),
        "steps_dir": str(steps_dir),
        "step_paths": step_paths,
    }
    (item_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir or (DEFAULT_OUTPUT_ROOT / f"slice_topoclean_v3_demo_{timestamp}")
    ensure_dir(out_dir)

    selected = choose_demo_images(args.source_dir, args.reference_summary)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model, config = load_model(EXPC_SPEC, device)

    results = []
    for image in selected:
        results.append(
            process_one(
                image=image,
                model=model,
                config=config,
                device=device,
                batch_size=args.batch_size,
                spur_factor=args.spur_factor,
                out_dir=out_dir,
            )
        )

    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "source_dir": str(args.source_dir),
                "model": MODEL_LABEL,
                "spur_factor": float(args.spur_factor),
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
