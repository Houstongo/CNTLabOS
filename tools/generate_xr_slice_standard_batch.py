from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List
import re

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

from experiments.cnt_paper_repro.config import load_config
from experiments.cnt_paper_repro.model import ResNet34UNet
from experiments.cnt_paper_repro.patching import extract_patch, extract_patch_specs
from src.analysis.feature_extractor import FeatureExtractor
from tools.generate_branch_selection_compare_panels import estimate_diameter_p30_nm


DB_PATH = PROJECT_ROOT / "database" / "cnta_experiments.sqlite"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "reports"
MODEL_LABEL = "CNTSegNet-SLICE"
THRESHOLDS = [("L1", 3.0), ("L2", 5.0), ("L3", 7.0), ("L4", 9.0)]
REFERENCE_THRESHOLD_LABEL = "L2"
THRESHOLD_COLORS = {
    "L1": (0.95, 0.60, 0.20),
    "L2": (0.20, 0.80, 0.45),
    "L3": (0.20, 0.65, 0.95),
    "L4": (0.78, 0.35, 0.92),
}


@dataclass(frozen=True)
class ModelSpec:
    label: str
    config_path: Path
    checkpoint_path: Path


EXPC_SPEC = ModelSpec(
    label=MODEL_LABEL,
    config_path=Path(
        r"D:\CNTDATA\CNTA_ML_Project\experiments\cnt_paper_repro\runs\cnt_paper_repro_100000x_center768_cldice_seed42\config_snapshot.yaml"
    ),
    checkpoint_path=Path(
        r"D:\CNTDATA\CNTA_ML_Project\experiments\cnt_paper_repro\runs\cnt_paper_repro_100000x_center768_cldice_seed42\best_model.pth"
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate CNTSegNet-SLICE standard-method report for XR rows or a local folder.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--image-id", type=int, action="append", default=None)
    parser.add_argument("--source-dir", type=Path, action="append", default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def slugify(text: str) -> str:
    safe = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_"}:
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe).strip("_") or "image"


def read_gray_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return image


def infer_magnification_from_name(path: Path) -> int | None:
    match = re.search(r"(?<!\d)(50000|100000)(?!\d)", path.stem)
    if not match:
        return None
    return int(match.group(1))


def write_png(path: Path, image: np.ndarray) -> None:
    ensure_dir(path.parent)
    encoded = cv2.imencode(".png", image)[1]
    path.write_bytes(encoded.tobytes())


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


def summarize_values(values: np.ndarray) -> Dict[str, float | None]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "mean": None,
            "std": None,
            "min": None,
            "p25": None,
            "p30": None,
            "p50": None,
            "p75": None,
            "max": None,
        }
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p25": float(np.percentile(values, 25)),
        "p30": float(np.percentile(values, 30)),
        "p50": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "max": float(np.max(values)),
    }


def sample_branch_diameters_nm(extractor: FeatureExtractor, distance_map: np.ndarray, branches: List[Dict[str, Any]]) -> np.ndarray:
    diameter_values = []
    for branch in branches:
        cached = branch.get("_diameter_distribution_nm")
        if cached is not None:
            if cached.size > 0:
                diameter_values.append(cached)
            continue
        px_per_nm = max(extractor.px_per_um / 1000.0, 1e-6)
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
    curvature_values = []
    for branch in branches:
        cached = branch.get("_curvature_distribution_nm")
        if cached is not None:
            if cached.size > 0:
                curvature_values.append(cached)
            continue
        px_per_nm = max(extractor.px_per_um / 1000.0, 1e-6)
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
        cached_stats = branch.get("_curvature_stats_nm")
        if cached_stats is not None:
            branch_curvature_nm = float(cached_stats.get(branch_stat, 0.0))
        else:
            cached_nm = branch.get("_curvature_distribution_nm")
            if cached_nm is not None:
                point_curvature_nm = cached_nm
                point_curvature_px = point_curvature_nm / px_per_nm if point_curvature_nm.size > 0 else np.empty((0,), dtype=float)
            else:
                sampled_coords = extractor._sample_ordered_coords(np.asarray(branch["coords"], dtype=float), sample_step=1)
                point_curvature_px = extractor._compute_point_curvatures_px(sampled_coords)
            if point_curvature_px.size == 0:
                continue

            if branch_stat == "median":
                branch_curvature_nm = float(np.median(point_curvature_px)) * px_per_nm
            elif branch_stat == "p75":
                branch_curvature_nm = float(np.percentile(point_curvature_px, extractor.V3_BRANCH_QUANTILE)) * px_per_nm
            elif branch_stat == "mean":
                branch_curvature_nm = float(np.mean(point_curvature_px)) * px_per_nm
            elif branch_stat == "trimmed_mean":
                if point_curvature_px.size >= 5:
                    low, high = np.percentile(point_curvature_px, [10, 90])
                    trimmed = point_curvature_px[(point_curvature_px >= low) & (point_curvature_px <= high)]
                    branch_curvature_nm = float(np.mean(trimmed if trimmed.size > 0 else point_curvature_px)) * px_per_nm
                else:
                    branch_curvature_nm = float(np.mean(point_curvature_px)) * px_per_nm
            else:
                raise ValueError(f"Unsupported branch_stat: {branch_stat}")

        if branch_curvature_nm <= 0:
            continue
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


def cache_branch_measurements(
    extractor: FeatureExtractor,
    branches: List[Dict[str, Any]],
    distance_map: np.ndarray,
) -> List[Dict[str, Any]]:
    px_per_nm = max(extractor.px_per_um / 1000.0, 1e-6)
    for branch in branches:
        coords = np.asarray(branch["coords"], dtype=float)
        sampled_coords = extractor._sample_ordered_coords(coords, sample_step=1)
        point_curvature_px = extractor._compute_point_curvatures_px(sampled_coords)
        curvature_nm = point_curvature_px * px_per_nm if point_curvature_px.size > 0 else np.empty((0,), dtype=float)
        branch["_curvature_distribution_nm"] = curvature_nm
        if curvature_nm.size > 0:
            if curvature_nm.size >= 5:
                low, high = np.percentile(curvature_nm, [10, 90])
                trimmed = curvature_nm[(curvature_nm >= low) & (curvature_nm <= high)]
                trimmed_mean = float(np.mean(trimmed)) if trimmed.size > 0 else float(np.mean(curvature_nm))
            else:
                trimmed_mean = float(np.mean(curvature_nm))
            branch["_curvature_stats_nm"] = {
                "median": float(np.median(curvature_nm)),
                "p75": float(np.percentile(curvature_nm, extractor.V3_BRANCH_QUANTILE)),
                "mean": float(np.mean(curvature_nm)),
                "trimmed_mean": trimmed_mean,
            }
        else:
            branch["_curvature_stats_nm"] = {}

        sampled_diameters = extractor._sample_map_values(coords, distance_map)
        valid = sampled_diameters[np.isfinite(sampled_diameters) & (sampled_diameters > 0)]
        branch["_diameter_distribution_nm"] = (
            (valid * 2.0) / px_per_nm if valid.size > 0 else np.empty((0,), dtype=float)
        )
    return branches


def cache_branch_waviness_metrics(extractor: FeatureExtractor, branches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for branch in branches:
        metric = extractor._calculate_component_waviness(np.asarray(branch["coords"], dtype=float), fast_mode=False)
        if metric is not None:
            metric["weight"] = max(float(branch.get("path_length_px", 0.0)), float(metric.get("weight", 0.0)))
        branch["_waviness_metric_v2"] = metric
    return branches


def aggregate_cached_waviness(branches: List[Dict[str, Any]]) -> Dict[str, float | None]:
    metrics = [branch.get("_waviness_metric_v2") for branch in branches]
    metrics = [metric for metric in metrics if metric is not None]
    if not metrics:
        return {"waviness_ratio_v2": None, "tortuosity_v2": 1.0}

    weights = np.asarray([float(metric["weight"]) for metric in metrics], dtype=float)
    return {
        "waviness_ratio_v2": float(np.average([metric["ratio"] for metric in metrics], weights=weights)),
        "tortuosity_v2": float(np.average([metric["tortuosity"] for metric in metrics], weights=weights)),
    }


def classify_v3_curvature(curvature_nm: float) -> str:
    if curvature_nm < 8e-4:
        return "Straight"
    if curvature_nm < 4e-3:
        return "Wavy"
    return "Coiled"


def load_model(spec: ModelSpec, device: torch.device) -> tuple[ResNet34UNet, dict]:
    config = load_config(spec.config_path)
    model = ResNet34UNet(
        in_channels=int(config["model"].get("in_channels", 1)),
        num_classes=int(config["model"].get("num_classes", 1)),
        encoder_weights=None,
    ).to(device)
    checkpoint = torch.load(spec.checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model, config


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


def query_active_xr_rows(limit: int | None = None, image_ids: List[int] | None = None) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    where = ["source = 'XR'", "COALESCE(is_deleted, 0) = 0"]
    params: List[Any] = []
    if image_ids:
        placeholders = ",".join("?" for _ in image_ids)
        where.append(f"id IN ({placeholders})")
        params.extend(int(v) for v in image_ids)

    sql = f"""
        SELECT id, sample_id, file_path, magnification, processed
        FROM images
        WHERE {' AND '.join(where)}
        ORDER BY id DESC
    """
    if limit is not None:
        sql += " LIMIT ?"
        params.append(int(limit))

    rows = [dict(row) for row in cur.execute(sql, params).fetchall()]
    conn.close()
    return rows


def collect_source_dir_rows(source_dirs: List[Path], limit: int | None = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    next_id = 1
    for source_dir in source_dirs:
        if not source_dir.exists():
            continue
        for path in sorted(source_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
                continue
            magnification = infer_magnification_from_name(path)
            if magnification is None:
                continue
            rows.append(
                {
                    "id": next_id,
                    "sample_id": path.stem,
                    "file_path": str(path),
                    "magnification": magnification,
                    "processed": 0,
                }
            )
            next_id += 1
    if limit is not None:
        rows = rows[: max(0, int(limit))]
    return rows


def make_manifest_row(row: Dict[str, Any]) -> Dict[str, Any]:
    path = Path(str(row["file_path"]))
    image_slug = slugify(f"xr_{row['id']}_{path.stem}")
    return {
        "image_id": int(row["id"]),
        "image_slug": image_slug,
        "sample_id": row.get("sample_id"),
        "file_name": path.name,
        "file_path": str(path),
        "magnification": int(row["magnification"]) if row.get("magnification") is not None else None,
        "processed": int(row.get("processed") or 0),
    }


def write_manifest(manifest_rows: List[Dict[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    if not manifest_rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)


def flatten_summary_record(record: Dict[str, Any]) -> Dict[str, Any]:
    flat = {
        "image_id": record["image_id"],
        "sample_id": record["sample_id"],
        "file_name": record["file_name"],
        "file_path": record["file_path"],
        "magnification": record["magnification"],
        "panel_path": record["panel_path"],
        "mask_path": record["mask_path"],
        "patch_count": record["patch_count"],
        "threshold": record["threshold"],
        "density": record["density"],
        "alignment": record["alignment"],
        "alignment_raw": record["alignment_raw"],
        "mean_phi_deg": record["mean_phi_deg"],
        "mean_phi_raw_deg": record["mean_phi_raw_deg"],
        "hof_method": record["hof_method"],
        "diameter_nm": record["diameter_nm"],
        "diameter_p30_nm": record["diameter_p30_nm"],
        "diameter_mean_nm": record["diameter_stats_nm"]["mean"],
        "diameter_std_nm": record["diameter_stats_nm"]["std"],
        "diameter_min_nm": record["diameter_stats_nm"]["min"],
        "diameter_p25_nm": record["diameter_stats_nm"]["p25"],
        "diameter_p30_full_nm": record["diameter_stats_nm"]["p30"],
        "diameter_p50_nm": record["diameter_stats_nm"]["p50"],
        "diameter_p75_nm": record["diameter_stats_nm"]["p75"],
        "diameter_max_nm": record["diameter_stats_nm"]["max"],
    }
    for label, threshold_data in record["thresholds"].items():
        prefix = label.lower()
        flat[f"{prefix}_min_length_factor"] = threshold_data["min_length_factor"]
        flat[f"{prefix}_branch_count"] = threshold_data["branch_count"]
        flat[f"{prefix}_curvature_label"] = threshold_data["curvature_label"]
        flat[f"{prefix}_waviness_ratio_v2"] = threshold_data["waviness_ratio_v2"]
        flat[f"{prefix}_tortuosity_v2"] = threshold_data["tortuosity_v2"]
        flat[f"{prefix}_curvature_point_count"] = threshold_data["curvature_point_count"]
        flat[f"{prefix}_diameter_point_count"] = threshold_data["diameter_point_count"]
        flat[f"{prefix}_diameter_mean_nm"] = threshold_data["diameter_stats_nm"]["mean"]
        flat[f"{prefix}_diameter_p30_nm"] = threshold_data["diameter_stats_nm"]["p30"]
        flat[f"{prefix}_diameter_p50_nm"] = threshold_data["diameter_stats_nm"]["p50"]
        flat[f"{prefix}_diameter_p75_nm"] = threshold_data["diameter_stats_nm"]["p75"]
        flat[f"{prefix}_curvature_sqrt_length_nm"] = threshold_data["curvature_nm_v3_sqrt_length"]
        flat[f"{prefix}_curvature_length_nm"] = threshold_data["curvature_nm_v3_length"]
        flat[f"{prefix}_curvature_mean_sqrt_length_nm"] = threshold_data.get("curvature_nm_v3_mean_sqrt_length")
        flat[f"{prefix}_curvature_mean_length_nm"] = threshold_data.get("curvature_nm_v3_mean_length")
    return flat


def plot_text_block(ax, lines: Iterable[str]) -> None:
    ax.set_facecolor("#0f172a")
    ax.axis("off")
    y = 0.97
    for line in lines:
        ax.text(0.04, y, line, transform=ax.transAxes, fontsize=8.8, color="white", va="top", family="DejaVu Sans Mono")
        y -= 0.075


def format_value(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def build_standard_summary_sections(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    sections: List[Dict[str, Any]] = [
        {
            "title": "Global",
            "color": "#111827",
            "lines": [
                f"sample_id: {record['sample_id']}",
                f"magnification: {record['magnification']}x",
                f"model/device: {MODEL_LABEL} / {record.get('runtime_device', 'unknown')}",
                f"patches/threshold: {record['patch_count']} / {format_value(record['threshold'], 2)}",
                f"density: {format_value(record['density'], 2)} %",
                f"alignment: {format_value(record['alignment'], 4)}",
                f"mean_phi_deg: {format_value(record['mean_phi_deg'], 2)}",
                f"diameter: {format_value(record['diameter_nm'], 4)}",
                f"diam p30: {format_value(record['diameter_p30_nm'], 4)}",
                "diam mean/p50/p75: "
                f"{format_value(record['diameter_stats_nm']['mean'], 4)} / "
                f"{format_value(record['diameter_stats_nm']['p50'], 4)} / "
                f"{format_value(record['diameter_stats_nm']['p75'], 4)}",
            ],
        }
    ]

    for label, _ in THRESHOLDS:
        profile = record["thresholds"][label]
        sections.append(
            {
                "title": f"{label}  |  len={format_value(profile['min_length_factor'], 1)}",
                "color": matplotlib.colors.to_hex(THRESHOLD_COLORS[label]),
                "lines": [
                    f"branches: {profile['branch_count']}  label: {profile['curvature_label']}",
                    "p75 sqrt/len: "
                    f"{format_value(profile['curvature_nm_v3_sqrt_length'], 6)} / "
                    f"{format_value(profile['curvature_nm_v3_length'], 6)}",
                    "mean sqrt/len: "
                    f"{format_value(profile.get('curvature_nm_v3_mean_sqrt_length'), 6)} / "
                    f"{format_value(profile.get('curvature_nm_v3_mean_length'), 6)}",
                    "waviness/tort: "
                    f"{format_value(profile['waviness_ratio_v2'], 6)} / "
                    f"{format_value(profile['tortuosity_v2'], 6)}",
                    "diam p30/p50/p75: "
                    f"{format_value(profile['diameter_stats_nm']['p30'], 4)} / "
                    f"{format_value(profile['diameter_stats_nm']['p50'], 4)} / "
                    f"{format_value(profile['diameter_stats_nm']['p75'], 4)}",
                    "curv pts/diam pts: "
                    f"{profile['curvature_point_count']} / {profile['diameter_point_count']}",
                ],
            }
        )
    return sections


def analyze_threshold_profiles(
    roi: np.ndarray,
    mask: np.ndarray,
    magnification: int,
    stage_callback=None,
) -> Dict[str, Any]:
    extractor = FeatureExtractor(magnification=magnification, speed_profile="accurate")
    extractor._calibrate(roi.shape[1])
    processed = extractor.preprocess(roi)
    density = float(np.count_nonzero(mask) / max(mask.size, 1) * 100.0)
    if stage_callback:
        stage_callback("analysis_started", {"density": round(density, 4)})

    diameter_nm, skeleton = extractor.calculate_diameter(mask)
    distance_map = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    base_components = extractor._collect_components(skeleton)
    alignment_metrics = extractor.calculate_hof_skeleton_adaptive(
        skeleton,
        processed=processed,
        base_components=base_components,
    )
    if stage_callback:
        stage_callback(
            "alignment_ready",
            {
                "alignment": round(float(alignment_metrics["alignment"]), 6),
                "mean_phi_deg": round(float(alignment_metrics["mean_phi_deg"]), 4),
            },
        )

    base_min_points = max(extractor.V3_MIN_BRANCH_POINTS, int(round(extractor.expected_tube_px * 1.5)))
    relaxed_branches = extractor._collect_ordered_branches_v2(
        skeleton,
        min_points=base_min_points,
        min_length_factor=extractor.V3_MIN_BRANCH_LENGTH_FACTOR,
    )
    cache_branch_measurements(extractor, relaxed_branches, distance_map)
    cache_branch_waviness_metrics(extractor, relaxed_branches)
    if stage_callback:
        stage_callback("relaxed_branches_ready", {"branch_count": len(relaxed_branches)})

    global_diameters_nm = sample_branch_diameters_nm(extractor, distance_map, relaxed_branches)
    diameter_stats_nm = summarize_values(global_diameters_nm)
    diameter_p30_nm = estimate_diameter_p30_nm(mask, skeleton, extractor.px_per_um)

    thresholds: Dict[str, Any] = {}
    for label, factor in THRESHOLDS:
        branches = extractor._filter_ordered_branches(
            relaxed_branches,
            min_points=base_min_points,
            min_length_factor=factor,
        )
        curvature_distribution_nm = sample_branch_curvatures_nm(extractor, branches)
        diameter_distribution_nm = sample_branch_diameters_nm(extractor, distance_map, branches)
        curvature_nm_v3_sqrt_length = aggregate_branch_curvature_nm(extractor, branches, "p75", "sqrt_length")
        curvature_nm_v3_length = aggregate_branch_curvature_nm(extractor, branches, "p75", "length")
        curvature_nm_v3_mean_sqrt_length = aggregate_branch_curvature_nm(extractor, branches, "mean", "sqrt_length")
        curvature_nm_v3_mean_length = aggregate_branch_curvature_nm(extractor, branches, "mean", "length")
        curvature_nm_v3 = float(curvature_nm_v3_sqrt_length)
        curvature_label = classify_v3_curvature(curvature_nm_v3) if branches else "Unknown"
        waviness_v2 = aggregate_cached_waviness(branches)
        thresholds[label] = {
            "min_length_factor": factor,
            "branch_count": len(branches),
            "branches": branches,
            "curvature_label": curvature_label,
            "curvature_nm_v3": float(curvature_nm_v3),
            "curvature_nm_v3_sqrt_length": float(curvature_nm_v3_sqrt_length),
            "curvature_nm_v3_length": float(curvature_nm_v3_length),
            "curvature_nm_v3_mean_sqrt_length": float(curvature_nm_v3_mean_sqrt_length),
            "curvature_nm_v3_mean_length": float(curvature_nm_v3_mean_length),
            "waviness_ratio_v2": float(waviness_v2["waviness_ratio_v2"]) if waviness_v2["waviness_ratio_v2"] is not None else None,
            "tortuosity_v2": float(waviness_v2["tortuosity_v2"]),
            "curvature_distribution_um": curvature_distribution_nm * 1000.0,
            "diameter_distribution_nm": diameter_distribution_nm,
            "diameter_stats_nm": summarize_values(diameter_distribution_nm),
        }
        if stage_callback:
            stage_callback(
                f"threshold_{label}_ready",
                {
                    "branch_count": len(branches),
                    "curvature_sqrt_length_nm": round(curvature_nm_v3_sqrt_length, 6),
                    "curvature_length_nm": round(curvature_nm_v3_length, 6),
                    "curvature_mean_sqrt_length_nm": round(curvature_nm_v3_mean_sqrt_length, 6),
                    "curvature_mean_length_nm": round(curvature_nm_v3_mean_length, 6),
                },
            )

    return {
        "density": density,
        "diameter_nm": float(diameter_nm) if diameter_nm is not None else None,
        "diameter_p30_nm": diameter_p30_nm,
        "diameter_stats_nm": diameter_stats_nm,
        "alignment": float(alignment_metrics["alignment"]),
        "alignment_raw": float(alignment_metrics["alignment_raw"]),
        "mean_phi_deg": float(alignment_metrics["mean_phi_deg"]),
        "mean_phi_raw_deg": float(alignment_metrics["mean_phi_raw_deg"]),
        "hof_method": alignment_metrics["hof_method"],
        "skeleton": skeleton,
        "thresholds": thresholds,
    }


def render_panel(
    output_path: Path,
    roi: np.ndarray,
    mask: np.ndarray,
    record: Dict[str, Any],
    threshold_profiles: Dict[str, Any],
) -> None:
    threshold_entries = list(threshold_profiles.items())
    fig = plt.figure(figsize=(18, 12), dpi=170, constrained_layout=True)
    grid = fig.add_gridspec(3, 3, width_ratios=[1.05, 1.05, 1.15], height_ratios=[1.0, 1.0, 1.0])

    ax_original = fig.add_subplot(grid[0, 0])
    ax_original.imshow(roi, cmap="gray")
    ax_original.set_title("Original ROI", fontsize=14)
    ax_original.axis("off")

    ax_mask = fig.add_subplot(grid[0, 1])
    ax_mask.imshow(cv2.cvtColor(build_bw_mask(mask), cv2.COLOR_BGR2RGB))
    ax_mask.set_title(f"{MODEL_LABEL} Mask", fontsize=14)
    ax_mask.axis("off")

    ax_summary = fig.add_subplot(grid[:, 2])
    ax_summary.set_facecolor("white")
    ax_summary.axis("off")
    y = 0.985
    for section in build_standard_summary_sections(record):
        ax_summary.text(
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
            ax_summary.text(
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

    overlay_positions = {
        "L1": (1, 0),
        "L2": (1, 1),
        "L3": (2, 0),
        "L4": (2, 1),
    }

    for label, profile in threshold_entries:
        color = THRESHOLD_COLORS[label]
        row_idx, col_idx = overlay_positions[label]
        ax_overlay = fig.add_subplot(grid[row_idx, col_idx])
        overlay = draw_branch_overlay(mask, profile["branches"], color)
        ax_overlay.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        ax_overlay.set_title(
            f"{label} Overlay (len={profile['min_length_factor']:.1f}, n={profile['branch_count']})\n"
            f"curv s/l={profile['curvature_nm_v3_sqrt_length']:.6f} / {profile['curvature_nm_v3_length']:.6f}",
            fontsize=11,
        )
        ax_overlay.axis("off")

    fig.suptitle(f"{MODEL_LABEL}  |  XR Standard Method  |  V3 + Length Threshold L1-L4", fontsize=18)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def load_completed_records(out_dir: Path) -> List[Dict[str, Any]]:
    items_dir = out_dir / "items"
    if not items_dir.exists():
        return []
    records = []
    for features_path in sorted(items_dir.glob("*/features.json")):
        try:
            records.append(json.loads(features_path.read_text(encoding="utf-8")))
        except Exception:
            continue
    return records


def write_summary_files(out_dir: Path, records: List[Dict[str, Any]]) -> None:
    summary_json = out_dir / "summary.json"
    summary_csv = out_dir / "summary.csv"
    flat_panels_dir = out_dir / "all_panels"
    ensure_dir(flat_panels_dir)

    panel_names = set()
    for record in records:
        src = Path(record["panel_path"])
        if not src.exists():
            continue
        panel_name = src.name
        if panel_name in panel_names:
            panel_name = f"{record['image_id']}_{panel_name}"
        panel_names.add(panel_name)
        dst = flat_panels_dir / panel_name
        if not dst.exists():
            dst.write_bytes(src.read_bytes())

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "count": len(records),
        "items": records,
    }
    summary_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    flat_rows = [flatten_summary_record(record) for record in records]
    if flat_rows:
        fieldnames = list(flat_rows[0].keys())
        with summary_csv.open("w", newline="", encoding="utf-8-sig") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(flat_rows)


def process_one(
    row: Dict[str, Any],
    model: ResNet34UNet,
    config: Dict[str, Any],
    device: torch.device,
    batch_size: int | None,
    out_dir: Path,
    skip_existing: bool,
) -> Dict[str, Any] | None:
    manifest = make_manifest_row(row)
    item_dir = out_dir / "items" / manifest["image_slug"]
    ensure_dir(item_dir)
    features_path = item_dir / "features.json"
    panel_path = item_dir / "panel.png"
    mask_path = item_dir / "mask.png"
    progress_path = item_dir / "progress.json"

    if skip_existing and features_path.exists() and panel_path.exists():
        return json.loads(features_path.read_text(encoding="utf-8"))

    timings: Dict[str, float] = {}

    def write_progress(stage: str, extra: Dict[str, Any] | None = None) -> None:
        payload = {
            "stage": stage,
            "image_id": manifest["image_id"],
            "sample_id": manifest["sample_id"],
            "file_path": manifest["file_path"],
            "timings": timings,
        }
        if extra:
            payload["extra"] = extra
        progress_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    image_path = Path(manifest["file_path"])
    write_progress("reading_image")
    image_gray = read_gray_image(image_path)
    extractor = FeatureExtractor(magnification=manifest["magnification"], speed_profile="accurate")
    roi = extractor.extract_roi(image_gray)

    infer_started = datetime.now().timestamp()
    write_progress("predicting_mask")
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
    timings["mask_inference_s"] = round(datetime.now().timestamp() - infer_started, 3)
    write_png(mask_path, mask.astype(np.uint8))
    write_progress("mask_saved", {"patch_count": int(patch_count)})

    analysis_started = datetime.now().timestamp()
    analysis = analyze_threshold_profiles(
        roi,
        mask.astype(np.uint8),
        manifest["magnification"],
        stage_callback=write_progress,
    )
    timings["analysis_s"] = round(datetime.now().timestamp() - analysis_started, 3)
    write_progress("analysis_done")

    record = {
        **manifest,
        "model_label": MODEL_LABEL,
        "panel_path": str(panel_path),
        "mask_path": str(mask_path),
        "runtime_device": device.type,
        "patch_count": int(patch_count),
        "threshold": float(config["inference"].get("threshold", 0.7)),
        "density": round(analysis["density"], 4),
        "alignment": round(analysis["alignment"], 6),
        "alignment_raw": round(analysis["alignment_raw"], 6),
        "mean_phi_deg": round(analysis["mean_phi_deg"], 4),
        "mean_phi_raw_deg": round(analysis["mean_phi_raw_deg"], 4),
        "hof_method": analysis["hof_method"],
        "diameter_nm": round(analysis["diameter_nm"], 4) if analysis["diameter_nm"] is not None else None,
        "diameter_p30_nm": round(analysis["diameter_p30_nm"], 4) if analysis["diameter_p30_nm"] is not None else None,
        "diameter_stats_nm": {k: (round(v, 4) if v is not None else None) for k, v in analysis["diameter_stats_nm"].items()},
        "thresholds": {},
        "timings": timings,
    }

    for label, profile in analysis["thresholds"].items():
        record["thresholds"][label] = {
            "min_length_factor": profile["min_length_factor"],
            "branch_count": int(profile["branch_count"]),
            "curvature_label": profile["curvature_label"],
            "curvature_nm_v3": round(profile["curvature_nm_v3"], 6),
            "curvature_nm_v3_sqrt_length": round(profile["curvature_nm_v3_sqrt_length"], 6),
            "curvature_nm_v3_length": round(profile["curvature_nm_v3_length"], 6),
            "curvature_nm_v3_mean_sqrt_length": round(profile["curvature_nm_v3_mean_sqrt_length"], 6),
            "curvature_nm_v3_mean_length": round(profile["curvature_nm_v3_mean_length"], 6),
            "waviness_ratio_v2": round(profile["waviness_ratio_v2"], 6) if profile["waviness_ratio_v2"] is not None else None,
            "tortuosity_v2": round(profile["tortuosity_v2"], 6),
            "curvature_point_count": int(profile["curvature_distribution_um"].size),
            "diameter_point_count": int(profile["diameter_distribution_nm"].size),
            "diameter_stats_nm": {k: (round(v, 4) if v is not None else None) for k, v in profile["diameter_stats_nm"].items()},
        }

    features_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    write_progress("features_saved")

    render_started = datetime.now().timestamp()
    render_panel(panel_path, roi, mask.astype(np.uint8), record, analysis["thresholds"])
    timings["render_panel_s"] = round(datetime.now().timestamp() - render_started, 3)
    record["timings"] = timings
    features_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    write_progress("panel_saved", {"panel_path": str(panel_path)})
    return record


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir or (DEFAULT_OUTPUT_ROOT / f"slice_standard_batch_{timestamp}")
    ensure_dir(out_dir)

    if args.source_dir:
        rows = collect_source_dir_rows(args.source_dir, limit=args.limit)
    else:
        rows = query_active_xr_rows(limit=args.limit, image_ids=args.image_id)
    manifest_rows = [make_manifest_row(row) for row in rows]
    write_manifest(manifest_rows, out_dir / "manifest.csv")

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model, config = load_model(EXPC_SPEC, device)

    completed_records = load_completed_records(out_dir)
    completed_by_id = {int(record["image_id"]): record for record in completed_records}

    for row in rows:
        processed = process_one(
            row=row,
            model=model,
            config=config,
            device=device,
            batch_size=args.batch_size,
            out_dir=out_dir,
            skip_existing=args.skip_existing,
        )
        if processed is not None:
            completed_by_id[int(row["id"])] = processed

    write_summary_files(out_dir, list(completed_by_id.values()))
    print(out_dir)


if __name__ == "__main__":
    main()
