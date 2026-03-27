"""Render annotated V2 feature-calculation panels for representative SEM images."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

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

from experiments.cnt_paper_repro.config import load_config  # noqa: E402
from experiments.cnt_paper_repro.model import ResNet34UNet  # noqa: E402
from experiments.cnt_paper_repro.patching import extract_patch, extract_patch_specs  # noqa: E402
from src.analysis.feature_extractor import FeatureExtractor  # noqa: E402


DEFAULT_COMPARISON_CSV = PROJECT_ROOT / "reports" / "curvature_v2_comparison_20260325_061334" / "comparison.csv"
DEFAULT_CONFIG = PROJECT_ROOT / "experiments" / "cnt_paper_repro" / "configs" / "paper_100000x_cldice.yaml"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "experiments" / "cnt_paper_repro" / "runs" / "cnt_paper_repro_100000x_center768_cldice_seed42" / "best_model.pth"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "reports"


@dataclass
class TargetImage:
    image_id: int
    sample_id: str
    file_path: str
    magnification: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate annotated V2 feature visualization panels.")
    parser.add_argument("--comparison-csv", type=Path, default=DEFAULT_COMPARISON_CSV)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--magnification", type=int, default=100000)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--patch-size", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_gray_image(path: str | Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return image


def write_image(path: str | Path, image: np.ndarray) -> None:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    encoded = cv2.imencode(output_path.suffix or ".png", image)[1]
    output_path.write_bytes(encoded.tobytes())


def safe_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def slugify(text: str) -> str:
    safe = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_"}:
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe).strip("_") or "image"


def format_metric(value: Any, fmt: str = "{:.4f}") -> str:
    numeric = safe_float(value)
    if numeric is None:
        return "N/A" if value in (None, "") else str(value)
    return fmt.format(numeric)


def select_target_images(csv_path: Path, magnification: int, limit: int) -> List[TargetImage]:
    rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8-sig", newline="")))
    selected: List[TargetImage] = []
    seen_groups = set()

    for row in rows:
        if int(row["magnification"]) != int(magnification):
            continue
        group = row["sample_id"].split("-")[0]
        if group in seen_groups:
            continue
        seen_groups.add(group)
        selected.append(
            TargetImage(
                image_id=int(row["image_id"]),
                sample_id=row["sample_id"],
                file_path=row["file_path"],
                magnification=int(row["magnification"]),
            )
        )
        if len(selected) >= limit:
            break

    if len(selected) < limit:
        for row in rows:
            if int(row["magnification"]) != int(magnification):
                continue
            image_id = int(row["image_id"])
            if any(item.image_id == image_id for item in selected):
                continue
            selected.append(
                TargetImage(
                    image_id=image_id,
                    sample_id=row["sample_id"],
                    file_path=row["file_path"],
                    magnification=int(row["magnification"]),
                )
            )
            if len(selected) >= limit:
                break
    return selected


def resolve_device(config: Dict[str, Any], requested: str) -> torch.device:
    if requested and requested.lower() != "auto":
        return torch.device(requested)
    configured = str(config["training"].get("device", "auto"))
    if configured != "auto":
        return torch.device(configured)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(config: Dict[str, Any], checkpoint_path: Path, device: torch.device) -> ResNet34UNet:
    model = ResNet34UNet(
        in_channels=int(config["model"].get("in_channels", 1)),
        num_classes=int(config["model"].get("num_classes", 1)),
        encoder_weights=None,
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict_roi_mask(
    model: ResNet34UNet,
    roi_gray: np.ndarray,
    patch_size: int,
    stride: int,
    threshold: float,
    normalize_mean: float,
    normalize_std: float,
    device: torch.device,
    batch_size: int | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    specs = extract_patch_specs(roi_gray, patch_size=patch_size, mode="grid", stride=stride)
    accum = np.zeros(roi_gray.shape, dtype=np.float32)
    counts = np.zeros(roi_gray.shape, dtype=np.float32)
    if batch_size is None:
        batch_size = 8 if device.type == "cuda" else 2
    batch_size = max(1, int(batch_size))

    with torch.no_grad():
        for start in range(0, len(specs), batch_size):
            batch_specs = specs[start : start + batch_size]
            batch_patches = []
            for spec in batch_specs:
                patch = extract_patch(roi_gray, spec).astype(np.float32) / 255.0
                batch_patches.append(((patch - normalize_mean) / max(normalize_std, 1e-6)).astype(np.float32))

            tensor = torch.from_numpy(np.stack(batch_patches, axis=0)).unsqueeze(1).to(device)
            probs = torch.sigmoid(model(tensor)).detach().cpu().numpy()[:, 0]

            for spec, prob in zip(batch_specs, probs):
                valid_prob = prob[: spec.height, : spec.width]
                accum[spec.top : spec.top + spec.height, spec.left : spec.left + spec.width] += valid_prob
                counts[spec.top : spec.top + spec.height, spec.left : spec.left + spec.width] += 1.0

    prob_map = accum / np.maximum(counts, 1.0)
    mask = (prob_map >= threshold).astype(np.uint8) * 255
    return mask, prob_map, len(specs)


def colorize_probability(prob_map: np.ndarray) -> np.ndarray:
    prob_u8 = np.clip(prob_map * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(prob_u8, cv2.COLORMAP_VIRIDIS)


def draw_mask_contours(canvas: np.ndarray, mask: np.ndarray, color: tuple[int, int, int] = (220, 220, 220), thickness: int = 1) -> None:
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, contours, -1, color, thickness)


def build_mask_base(
    mask: np.ndarray,
    fill_color: tuple[int, int, int] = (42, 42, 42),
    contour_color: tuple[int, int, int] = (220, 220, 220),
    contour_thickness: int = 1,
) -> np.ndarray:
    canvas = np.zeros((*mask.shape, 3), dtype=np.uint8)
    canvas[mask > 0] = fill_color
    draw_mask_contours(canvas, mask, color=contour_color, thickness=contour_thickness)
    return canvas


def build_probability_visual(prob_map: np.ndarray, mask: np.ndarray) -> np.ndarray:
    heat = colorize_probability(prob_map)
    canvas = np.zeros_like(heat)
    active = prob_map > 0.05
    canvas[active] = heat[active]
    draw_mask_contours(canvas, mask, color=(255, 255, 255), thickness=1)
    return canvas


def build_binary_mask_visual(mask: np.ndarray) -> np.ndarray:
    canvas = np.zeros((*mask.shape, 3), dtype=np.uint8)
    canvas[mask > 0] = (245, 245, 245)
    draw_mask_contours(canvas, mask, color=(255, 255, 255), thickness=1)
    return canvas


def build_skeleton_visual(mask: np.ndarray, skel: np.ndarray) -> np.ndarray:
    canvas = build_mask_base(mask)
    canvas[skel > 0] = (80, 255, 160)
    return canvas


def build_branch_split_visual(mask: np.ndarray, skel: np.ndarray, extractor: FeatureExtractor) -> np.ndarray:
    from skimage.measure import label

    canvas = build_mask_base(mask)
    skel_mask = (skel > 0).astype(np.uint8)
    neighbor_count = extractor._neighbor_count_map(skel_mask)
    junction_mask = (skel_mask > 0) & (neighbor_count >= 3)
    branch_mask = (skel_mask > 0) & np.logical_not(junction_mask)
    labeled = label(branch_mask, connectivity=2)

    palette = [
        (255, 210, 80),
        (90, 220, 255),
        (255, 120, 120),
        (170, 255, 120),
        (210, 120, 255),
        (120, 255, 220),
    ]
    for branch_id in range(1, int(labeled.max()) + 1):
        color = palette[(branch_id - 1) % len(palette)]
        canvas[labeled == branch_id] = color

    canvas[junction_mask] = (0, 0, 255)
    return canvas


def draw_polyline(canvas: np.ndarray, coords: np.ndarray, color: tuple[int, int, int], thickness: int = 1) -> None:
    if coords.shape[0] < 2:
        return
    pts = np.round(coords[:, ::-1]).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(canvas, [pts], False, color, thickness=thickness, lineType=cv2.LINE_AA)


def build_centerline_visual(mask: np.ndarray, ordered_branches: List[Dict[str, Any]]) -> np.ndarray:
    canvas = build_mask_base(mask)
    palette = [
        (90, 220, 255),
        (120, 255, 140),
        (255, 200, 80),
        (240, 120, 255),
        (255, 120, 120),
        (120, 180, 255),
    ]
    for idx, branch in enumerate(ordered_branches[:60]):
        coords = branch["coords"]
        color = palette[idx % len(palette)]
        draw_polyline(canvas, coords, color, thickness=1)
        start = tuple(np.round(coords[0, ::-1]).astype(int))
        end = tuple(np.round(coords[-1, ::-1]).astype(int))
        cv2.circle(canvas, start, 3, (0, 255, 0), -1)
        cv2.circle(canvas, end, 3, (0, 0, 255), -1)
        if coords.shape[0] >= 6:
            mid = coords[min(5, coords.shape[0] - 1)]
            prev = coords[max(0, min(3, coords.shape[0] - 2))]
            start_arrow = tuple(np.round(prev[::-1]).astype(int))
            end_arrow = tuple(np.round(mid[::-1]).astype(int))
            cv2.arrowedLine(canvas, start_arrow, end_arrow, color, 1, tipLength=0.35)
    return canvas


def build_distance_transform_visual(mask: np.ndarray) -> np.ndarray:
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    if dist.max() <= 0:
        return np.zeros((*mask.shape, 3), dtype=np.uint8)
    dist_u8 = np.clip(dist / dist.max() * 255.0, 0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(dist_u8, cv2.COLORMAP_TURBO)
    canvas = np.zeros_like(heat)
    canvas[mask > 0] = heat[mask > 0]
    draw_mask_contours(canvas, mask, color=(255, 255, 255), thickness=1)
    return canvas


def build_diameter_sampling_visual(mask: np.ndarray, skel: np.ndarray, extractor: FeatureExtractor) -> tuple[np.ndarray, Dict[str, Any]]:
    canvas = build_mask_base(mask, fill_color=(24, 30, 56), contour_color=(240, 240, 240))

    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    skel_points = np.argwhere(skel > 0)
    diameters_px = dist[skel > 0] * 2.0 if np.any(skel > 0) else np.array([], dtype=float)
    diameters_px = diameters_px[diameters_px > 0]

    diameter_raw, _ = extractor.calculate_diameter(mask)
    stats = {
        "diameter_raw_nm": diameter_raw if diameter_raw >= 0 else None,
        "samples": int(diameters_px.size),
    }

    if skel_points.size > 0:
        radii = dist[skel > 0]
        valid = radii > 0
        valid_points = skel_points[valid]
        radii = radii[valid]
        if radii.size > 0:
            order = np.argsort(radii)[::-1]
            sampled_indices = order[: min(40, order.size)]
            for idx in sampled_indices:
                y, x = valid_points[idx]
                radius = radii[idx]
                cv2.circle(canvas, (int(x), int(y)), int(round(radius)), (255, 220, 60), 1)
                cv2.circle(canvas, (int(x), int(y)), 1, (255, 255, 255), -1)

    return canvas, stats


def extract_curvature_samples(extractor: FeatureExtractor, ordered_branches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    px_per_nm = max(extractor.px_per_um / 1000.0, 1e-6)
    for branch in ordered_branches:
        coords = extractor._sample_ordered_coords(branch["coords"], sample_step=1)
        if coords.shape[0] < 3:
            continue
        for idx in range(1, coords.shape[0] - 1):
            p_prev = coords[idx - 1]
            p_curr = coords[idx]
            p_next = coords[idx + 1]
            ab = p_curr - p_prev
            bc = p_next - p_curr
            ca = p_next - p_prev
            a = float(np.linalg.norm(ab))
            b = float(np.linalg.norm(bc))
            c = float(np.linalg.norm(ca))
            if min(a, b, c) <= 1e-6:
                continue
            cross = abs(ab[0] * bc[1] - ab[1] * bc[0])
            curvature_px = (2.0 * cross) / max(a * b * c, 1e-6)
            curvature_nm = curvature_px * px_per_nm
            samples.append({"point": p_curr, "curvature_nm": curvature_nm})
    return samples


def build_curvature_visual(mask: np.ndarray, ordered_branches: List[Dict[str, Any]], curvature_samples: List[Dict[str, Any]]) -> np.ndarray:
    canvas = build_mask_base(mask)
    for branch in ordered_branches[:80]:
        draw_polyline(canvas, branch["coords"], (70, 220, 220), thickness=1)

    for sample in curvature_samples[:: max(1, len(curvature_samples) // 600)]:
        curvature_nm = float(sample["curvature_nm"])
        if curvature_nm < 5e-4:
            color = (0, 255, 0)
        elif curvature_nm < 2.5e-3:
            color = (0, 255, 255)
        else:
            color = (0, 64, 255)
        x, y = np.round(sample["point"][1]).astype(int), np.round(sample["point"][0]).astype(int)
        cv2.circle(canvas, (x, y), 2, color, -1)
    return canvas


def analyze_waviness_branch(extractor: FeatureExtractor, coords: np.ndarray) -> Optional[Dict[str, Any]]:
    if coords.shape[0] < 20:
        return None

    centered = coords - coords.mean(axis=0)
    cov = np.cov(centered.T)
    if cov.ndim < 2:
        return None

    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, np.argmax(eigvals)]
    axis_norm = np.linalg.norm(axis)
    if axis_norm == 0:
        return None
    axis = axis / axis_norm
    normal = np.array([-axis[1], axis[0]])

    longitudinal = centered @ axis
    lateral = centered @ normal
    order = np.argsort(longitudinal)
    longitudinal = longitudinal[order]
    lateral = lateral[order]
    coords_sorted = coords[order]

    bins = np.round(longitudinal - longitudinal.min()).astype(np.int32)
    if bins.size == 0:
        return None
    sums_s = np.bincount(bins, weights=longitudinal)
    sums_d = np.bincount(bins, weights=lateral)
    counts = np.bincount(bins)
    valid = counts > 0
    if np.count_nonzero(valid) < 12:
        return None
    s_vals = sums_s[valid] / counts[valid]
    d_vals = sums_d[valid] / counts[valid]

    linear = np.polyfit(s_vals, d_vals, deg=1)
    detrended = d_vals - np.polyval(linear, s_vals)
    smoothed = extractor._smooth_signal(detrended, window=5)
    if np.ptp(smoothed) < 2.0:
        return None

    extrema = []
    min_spacing = max(3.0, extractor.expected_tube_px * 2.0)
    valid_indices = np.flatnonzero(valid)

    for idx in range(1, smoothed.size - 1):
        prev_val = smoothed[idx - 1]
        curr_val = smoothed[idx]
        next_val = smoothed[idx + 1]
        kind = None
        if curr_val >= prev_val and curr_val > next_val:
            kind = "peak"
        elif curr_val <= prev_val and curr_val < next_val:
            kind = "trough"
        if kind is None:
            continue

        extremum = {
            "kind": kind,
            "s": float(s_vals[idx]),
            "value": float(curr_val),
            "coord": coords_sorted[min(valid_indices[idx], coords_sorted.shape[0] - 1)],
        }

        if extrema and kind == extrema[-1]["kind"]:
            more_extreme = (kind == "peak" and curr_val > extrema[-1]["value"]) or (kind == "trough" and curr_val < extrema[-1]["value"])
            if more_extreme:
                extrema[-1] = extremum
            continue

        if extrema and abs(s_vals[idx] - extrema[-1]["s"]) < min_spacing:
            more_extreme = (kind == "peak" and curr_val > extrema[-1]["value"]) or (kind == "trough" and curr_val < extrema[-1]["value"])
            if more_extreme:
                extrema[-1] = extremum
            continue
        extrema.append(extremum)

    metrics = extractor._calculate_component_waviness(coords)
    if metrics is None:
        return None

    return {
        "coords_sorted": coords_sorted,
        "mean": coords.mean(axis=0),
        "axis": axis,
        "extrema": extrema,
        "metrics": metrics,
    }


def build_waviness_visual(mask: np.ndarray, ordered_branches: List[Dict[str, Any]], extractor: FeatureExtractor) -> tuple[np.ndarray, Optional[Dict[str, Any]]]:
    canvas = build_mask_base(mask)
    if not ordered_branches:
        return canvas, None

    representative = max(ordered_branches, key=lambda item: item["path_length_px"])
    detail = analyze_waviness_branch(extractor, representative["coords"])
    draw_polyline(canvas, representative["coords"], (80, 220, 255), thickness=2)
    if detail is None:
        return canvas, None

    mean = detail["mean"]
    axis = detail["axis"]
    start = mean - axis * 160
    end = mean + axis * 160
    cv2.line(canvas, tuple(np.round(start[::-1]).astype(int)), tuple(np.round(end[::-1]).astype(int)), (255, 255, 255), 1)

    for extremum in detail["extrema"]:
        x, y = np.round(extremum["coord"][1]).astype(int), np.round(extremum["coord"][0]).astype(int)
        color = (0, 0, 255) if extremum["kind"] == "peak" else (255, 0, 0)
        cv2.circle(canvas, (x, y), 4, color, -1)

    return canvas, detail


def build_text_panel(lines: Iterable[str], facecolor: str = "#0f172a", textcolor: str = "white") -> np.ndarray:
    fig, ax = plt.subplots(figsize=(5.0, 4.5), dpi=160)
    ax.set_facecolor(facecolor)
    fig.patch.set_facecolor(facecolor)
    ax.axis("off")
    y = 0.96
    for line in lines:
        ax.text(0.04, y, line, transform=ax.transAxes, fontsize=11, color=textcolor, va="top", family="DejaVu Sans Mono")
        y -= 0.075
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8).copy()
    plt.close(fig)
    return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)


def add_caption(ax, title: str, caption: str) -> None:
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axis("off")
    ax.text(0.5, -0.10, caption, transform=ax.transAxes, ha="center", va="top", fontsize=9, color="#475569", wrap=True)


def render_panel(output_path: Path, target: TargetImage, tiles: List[tuple[np.ndarray, str, str]]) -> None:
    fig, axes = plt.subplots(4, 3, figsize=(19, 23), dpi=120)
    axes = axes.flatten()

    for ax, (image, title, caption) in zip(axes, tiles):
        if image.ndim == 2:
            ax.imshow(image, cmap="gray")
        else:
            ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        add_caption(ax, title, caption)

    for ax in axes[len(tiles):]:
        ax.axis("off")

    fig.suptitle(
        f"V2 Feature Visualization | image_id={target.image_id} | {target.sample_id} | {Path(target.file_path).name}",
        fontsize=16,
        y=0.992,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    ensure_dir(output_path.parent)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def build_final_note_lines(extractor: FeatureExtractor, target: TargetImage) -> List[str]:
    guard_mag = int(extractor.MIN_MAG_FOR_DIAMETER)
    if int(target.magnification) < guard_mag:
        return [
            "Note",
            f"{target.magnification}x < {guard_mag}x guard:",
            "system suppresses diameter / curvature /",
            "waviness in final reporting to avoid",
            "pretending low-mag geometry is reliable.",
        ]
    return [
        "Note",
        f"{target.magnification}x >= {guard_mag}x guard:",
        "final reported values are the real outputs",
        "from extract_all on this latest V2 flow,",
        "so the numbers are not zeroed by policy.",
    ]


def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_builtin(v) for v in value]
    if isinstance(value, tuple):
        return [to_builtin(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = resolve_device(config, args.device)
    threshold = float(args.threshold if args.threshold is not None else config["inference"].get("threshold", 0.7))
    patch_size = int(args.patch_size if args.patch_size is not None else config["data"].get("patch_size", 768))
    stride = int(args.stride if args.stride is not None else max(1, patch_size // 2))
    normalize_mean = float(config["data"].get("normalize_mean", 0.5))
    normalize_std = float(config["data"].get("normalize_std", 0.5))

    targets = select_target_images(args.comparison_csv, magnification=args.magnification, limit=args.limit)
    if not targets:
        raise SystemExit("No matching target images found.")

    if args.output_dir is not None:
        output_root = args.output_dir
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_root = DEFAULT_OUTPUT_ROOT / f"v2_feature_visualization_{args.magnification}x_{stamp}"
    ensure_dir(output_root)

    model = load_model(config, args.checkpoint, device)
    summary_items: List[Dict[str, Any]] = []

    for index, target in enumerate(targets, start=1):
        print(f"[{index}/{len(targets)}] image_id={target.image_id} sample={target.sample_id}")
        item_dir = output_root / f"{target.image_id}_{slugify(Path(target.file_path).stem)}"
        ensure_dir(item_dir)

        image_gray = read_gray_image(target.file_path)
        extractor = FeatureExtractor(magnification=target.magnification)
        roi_gray = extractor.extract_roi(image_gray)
        extractor._calibrate(roi_gray.shape[1])
        processed = extractor.preprocess(roi_gray)

        mask, prob_map, patch_count = predict_roi_mask(
            model=model,
            roi_gray=roi_gray,
            patch_size=patch_size,
            stride=stride,
            threshold=threshold,
            normalize_mean=normalize_mean,
            normalize_std=normalize_std,
            device=device,
        )
        thresh = (mask > 0).astype(np.uint8) * 255
        density = float(np.count_nonzero(thresh) / max(thresh.size, 1) * 100.0)

        diameter_raw_nm, skel = extractor.calculate_diameter(thresh)
        base_components = extractor._collect_components(skel)
        alignment_metrics = extractor.calculate_hof_skeleton_adaptive(skel, processed=processed, base_components=base_components)
        legacy_label_raw, legacy_curvature_raw = extractor.calculate_curvature(skel, base_components=base_components)
        ordered_branches = extractor._collect_ordered_branches_v2(skel, min_points=15)
        v2_label_raw, v2_curvature_raw = extractor.calculate_curvature_v2(skel, ordered_branches=ordered_branches)
        waviness_raw = extractor.calculate_waviness(skel, base_components=base_components)
        waviness_v2_raw = extractor.calculate_waviness_v2(skel, ordered_branches=ordered_branches)
        final_results = extractor.extract_all(image_gray, external_binary_mask=mask)

        probability_vis = build_probability_visual(prob_map, mask)
        mask_vis = build_binary_mask_visual(mask)
        skeleton_vis = build_skeleton_visual(mask, skel)
        branch_split_vis = build_branch_split_visual(mask, skel, extractor)
        centerline_vis = build_centerline_visual(mask, ordered_branches)
        distance_vis = build_distance_transform_visual(thresh)
        diameter_vis, diameter_stats = build_diameter_sampling_visual(thresh, skel, extractor)
        curvature_samples = extract_curvature_samples(extractor, ordered_branches)
        curvature_vis = build_curvature_visual(thresh, ordered_branches, curvature_samples)
        waviness_vis, waviness_detail = build_waviness_visual(thresh, ordered_branches, extractor)

        raw_lines = [
            "Raw Geometry (pre-guard)",
            f"density           = {density:.2f}%",
            f"diameter_raw_nm   = {format_metric(diameter_stats['diameter_raw_nm'], '{:.2f}')}",
            f"legacy_curv_nm    = {format_metric(legacy_curvature_raw)} ({legacy_label_raw})",
            f"v2_curv_nm        = {format_metric(v2_curvature_raw)} ({v2_label_raw})",
            f"legacy_wavy_ratio = {format_metric(waviness_raw.get('waviness_ratio'))}",
            f"v2_wavy_ratio     = {format_metric(waviness_v2_raw.get('waviness_ratio_v2'))}",
            f"legacy_tortuosity = {format_metric(waviness_raw.get('tortuosity'))}",
            f"v2_tortuosity     = {format_metric(waviness_v2_raw.get('tortuosity_v2'))}",
            f"ordered_branches  = {len(ordered_branches)}",
            f"patch_count       = {patch_count}",
            f"px_per_um         = {extractor.px_per_um:.2f}",
        ]
        final_lines = [
            "Final Reported (extract_all)",
            f"density           = {format_metric(final_results.get('density'), '{:.2f}')}",
            f"diameter          = {format_metric(final_results.get('diameter'), '{:.2f}')}",
            f"curvature_nm      = {format_metric(final_results.get('curvature_nm'))}",
            f"curvature_nm_v2   = {format_metric(final_results.get('curvature_nm_v2'))}",
            f"waviness_ratio    = {format_metric(final_results.get('waviness_ratio'))}",
            f"waviness_ratio_v2 = {format_metric(final_results.get('waviness_ratio_v2'))}",
            f"tortuosity        = {format_metric(final_results.get('tortuosity'))}",
            f"tortuosity_v2     = {format_metric(final_results.get('tortuosity_v2'))}",
            f"alignment         = {format_metric(final_results.get('alignment'))}",
            "",
            *build_final_note_lines(extractor, target),
        ]
        raw_panel = build_text_panel(raw_lines, facecolor="#0f172a", textcolor="white")
        final_panel = build_text_panel(final_lines, facecolor="#1e293b", textcolor="white")

        tiles = [
            (roi_gray, "1. ROI Original", "仅作参考；后续几何计算都不再叠回原图。"),
            (probability_vis, "2. Probability Map", "黑底上显示模型概率场，白线是最终阈值边界。"),
            (mask_vis, "3. Final Binary Mask", "黑底白 mask 是后续所有几何计算的真实输入。"),
            (skeleton_vis, "4. Skeleton", "在同一张 mask 上骨架化，绿色中心线直接对应预测边界。"),
            (branch_split_vis, "5. Junction Cut / Branch Split", "暗灰是 mask 内部，红色 junction 被切断，彩色 branch 是 V2 候选。"),
            (centerline_vis, "6. Ordered + Smoothed Centerlines", "每条 branch 在 mask 上排序并平滑，绿点起点，红点终点。"),
            (distance_vis, "7. Distance Transform", "只在 mask 内部计算到边界的距离，颜色越热局部半径越大。"),
            (diameter_vis, "8. Diameter Sampling", "直径圆直接贴着 mask 边界展开，展示局部厚度如何被采样。"),
            (curvature_vis, "9. Curvature Sampling", "曲率点直接落在平滑中心线上，颜色表示局部弯曲强弱。"),
            (waviness_vis, "10. Waviness Branch", "代表 branch 的主轴和峰谷都画在 mask 内，方便看波形来源。"),
            (raw_panel, "11. Raw Geometry", "低倍率保护前的原始几何计算候选值。"),
            (
                final_panel,
                "12. Final Reported Values",
                "系统真正返回的值；若低于倍率保护阈值会被抑制，否则这里就是最新 V2 链路的正式输出。",
            ),
        ]

        render_panel(item_dir / "v2_feature_panel.png", target, tiles)

        if waviness_detail is not None:
            detail_payload = {
                "ratio": waviness_detail["metrics"]["ratio"],
                "height_px": waviness_detail["metrics"]["height_px"],
                "wavelength_px": waviness_detail["metrics"]["wavelength_px"],
                "tortuosity": waviness_detail["metrics"]["tortuosity"],
                "extrema_count": len(waviness_detail["extrema"]),
            }
        else:
            detail_payload = None

        summary_items.append(
            {
                "image_id": target.image_id,
                "sample_id": target.sample_id,
                "file_path": target.file_path,
                "magnification": target.magnification,
                "artifacts": {
                    "item_dir": str(item_dir),
                    "panel_path": str(item_dir / "v2_feature_panel.png"),
                },
                "raw": to_builtin(
                    {
                        "density": density,
                        "diameter_raw_nm": diameter_stats["diameter_raw_nm"],
                        "legacy_curvature_nm": legacy_curvature_raw,
                        "legacy_curvature_label": legacy_label_raw,
                        "v2_curvature_nm": v2_curvature_raw,
                        "v2_curvature_label": v2_label_raw,
                        "legacy_waviness": waviness_raw,
                        "v2_waviness": waviness_v2_raw,
                        "patch_count": patch_count,
                        "ordered_branches": len(ordered_branches),
                        "alignment_raw": alignment_metrics.get("alignment"),
                        "waviness_detail": detail_payload,
                    }
                ),
                "final": to_builtin(final_results),
            }
        )

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "output_root": str(output_root),
        "comparison_csv": str(args.comparison_csv),
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "magnification": args.magnification,
        "threshold": threshold,
        "patch_size": patch_size,
        "stride": stride,
        "count": len(summary_items),
        "items": summary_items,
    }
    (output_root / "summary.json").write_text(json.dumps(to_builtin(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"OUTPUT_DIR={output_root}")
    print(f"SUMMARY_PATH={output_root / 'summary.json'}")


if __name__ == "__main__":
    main()
