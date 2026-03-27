from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

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
SMOOTH_PROFILE_NAMES = ("raw", "conservative", "balanced", "visual")
SMOOTH_PROFILE_CONFIGS = {
    "raw": {"window_scale": 1.0, "window_delta": 0, "min_window": 3, "passes": 0, "curvature_window_px": 6.0},
    "conservative": {"window_scale": 0.7, "window_delta": -2, "min_window": 3, "passes": 1, "curvature_window_px": 8.0},
    "balanced": {"window_scale": 1.0, "window_delta": 0, "min_window": 5, "passes": 1, "curvature_window_px": 12.0},
    "visual": {"window_scale": 1.4, "window_delta": 4, "min_window": 7, "passes": 2, "curvature_window_px": 18.0},
}


def parse_args():
    parser = argparse.ArgumentParser(description="Probe intact skeleton tracing across junctions.")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--mask-image", type=Path, default=None)
    parser.add_argument("--mask-threshold", type=int, default=127)
    parser.add_argument("--magnification", type=int, default=100000)
    parser.add_argument("--angle-limit-deg", type=float, default=45.0)
    parser.add_argument("--angle-hard-deg", type=float, default=70.0)
    parser.add_argument("--cluster-exit-relax-deg", type=float, default=12.0)
    parser.add_argument("--cluster-backtrack-hook-penalty-max", type=float, default=0.06)
    parser.add_argument("--cluster-backtrack-min-forward-ratio", type=float, default=0.55)
    parser.add_argument("--cluster-backtrack-min-straightness", type=float, default=0.80)
    parser.add_argument("--cluster-backtrack-min-target-improve-px", type=float, default=4.0)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--top-seeds", type=int, default=100)
    parser.add_argument("--top-candidates", type=int, default=100)
    parser.add_argument("--edge-seed-margin-px", type=float, default=None)
    parser.add_argument("--seed-candidate-margin-px", type=float, default=None)
    parser.add_argument("--bridge-gap-px", type=float, default=5.0)
    parser.add_argument("--bridge-direction-cos-min", type=float, default=0.60)
    parser.add_argument("--gray-window-path-px", type=float, default=48.0)
    parser.add_argument("--gray-window-candidate-px", type=float, default=28.0)
    parser.add_argument("--seed-walk-hops", type=int, default=14)
    parser.add_argument("--max-junction-visits", type=int, default=12)
    parser.add_argument("--max-cumulative-turn-deg", type=float, default=260.0)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def read_gray(path: Path):
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return image


def normalize(vec):
    arr = np.asarray(vec, dtype=float)
    norm = float(np.linalg.norm(arr))
    return np.zeros((2,), dtype=float) if norm <= 1e-8 else arr / norm


def build_mask_base(mask, fill_color=(42, 42, 42), contour_color=(220, 220, 220)):
    canvas = np.zeros((*mask.shape, 3), dtype=np.uint8)
    canvas[mask > 0] = fill_color
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, contours, -1, contour_color, 1)
    return canvas


def build_binary_mask_canvas(mask, foreground_color=(182, 182, 182)):
    canvas = np.zeros((*mask.shape, 3), dtype=np.uint8)
    canvas[mask > 0] = foreground_color
    return canvas


def cumulative_arclength(coords):
    coords = np.asarray(coords, dtype=float)
    if coords.shape[0] == 0:
        return np.zeros((0,), dtype=float)
    arc = np.zeros((coords.shape[0],), dtype=float)
    if coords.shape[0] >= 2:
        deltas = np.diff(coords, axis=0)
        arc[1:] = np.cumsum(np.hypot(deltas[:, 0], deltas[:, 1]))
    return arc


def resample_coords(coords, target_count=None, step_px=1.0):
    coords = np.asarray(coords, dtype=float)
    if coords.shape[0] <= 1:
        return coords.copy()
    arc = cumulative_arclength(coords)
    total = float(arc[-1])
    if total <= 1e-6:
        return coords.copy()
    if target_count is None:
        target_count = max(2, int(round(total / max(float(step_px), 1e-6))) + 1)
    target_count = max(2, int(target_count))
    sample_positions = np.linspace(0.0, total, target_count)
    y_coords = np.interp(sample_positions, arc, coords[:, 0])
    x_coords = np.interp(sample_positions, arc, coords[:, 1])
    return np.column_stack([y_coords, x_coords])


def bounded_odd_window(window, point_count, minimum=3):
    window = max(int(minimum), int(window))
    window = min(window, max(3, int(point_count)))
    if window % 2 == 0:
        window -= 1
    if window < 3:
        window = 3 if int(point_count) >= 3 else int(point_count)
    return max(1, int(window))


def default_smoothing_window(extractor, point_count):
    base = 3 if extractor.speed_profile == "fast" else 5
    dynamic = int(round(extractor.expected_tube_px * 1.5)) | 1
    return bounded_odd_window(max(base, dynamic), point_count, minimum=3)


def profile_smoothing_window(extractor, point_count, profile_name):
    config = SMOOTH_PROFILE_CONFIGS[profile_name]
    base_window = default_smoothing_window(extractor, point_count)
    suggested = int(round(base_window * float(config["window_scale"]))) + int(config["window_delta"])
    return bounded_odd_window(suggested, point_count, minimum=int(config["min_window"]))


def smooth_profile_coords(extractor, coords, profile_name, target_count=None):
    coords = resample_coords(coords, target_count=target_count, step_px=1.0)
    if coords.shape[0] < 3 or profile_name == "raw":
        return coords
    result = coords.copy()
    window = profile_smoothing_window(extractor, result.shape[0], profile_name)
    for _ in range(max(1, int(SMOOTH_PROFILE_CONFIGS[profile_name]["passes"]))):
        result = extractor._smooth_path_coords(result, window=window)
        result[0] = coords[0]
        result[-1] = coords[-1]
    return result


def compute_fiji_like_curvature(coords, tangent_window_points):
    coords = np.asarray(coords, dtype=float)
    n_points = int(coords.shape[0])
    if n_points == 0:
        return np.zeros((0,), dtype=float)
    if n_points < 3:
        return np.zeros((n_points,), dtype=float)
    tangent_window_points = max(1, int(round(tangent_window_points)))
    tangent_angles = np.zeros((n_points,), dtype=float)
    for idx in range(n_points):
        left = max(0, idx - tangent_window_points)
        right = min(n_points - 1, idx + tangent_window_points)
        if right <= left:
            tangent_angles[idx] = tangent_angles[idx - 1] if idx > 0 else 0.0
            continue
        delta = coords[right] - coords[left]
        if float(np.linalg.norm(delta)) <= 1e-8:
            tangent_angles[idx] = tangent_angles[idx - 1] if idx > 0 else 0.0
            continue
        tangent_angles[idx] = float(np.arctan2(delta[0], delta[1]))
    tangent_angles = np.unwrap(tangent_angles)
    arc = cumulative_arclength(coords)
    curvature = np.zeros((n_points,), dtype=float)
    for idx in range(1, n_points - 1):
        ds = float(arc[idx + 1] - arc[idx - 1])
        if ds <= 1e-6:
            continue
        curvature[idx] = abs(float(tangent_angles[idx + 1] - tangent_angles[idx - 1])) / ds
    curvature[0] = curvature[1]
    curvature[-1] = curvature[-2]
    return curvature


def summarize_curvature_series(curvature_px, px_per_nm, path_length_px, span_px):
    curvature_px = np.asarray(curvature_px, dtype=float)
    curvature_nm = curvature_px * float(px_per_nm)
    finite_px = curvature_px[np.isfinite(curvature_px)]
    finite_nm = curvature_nm[np.isfinite(curvature_nm)]
    if finite_px.size == 0:
        finite_px = np.zeros((1,), dtype=float)
        finite_nm = np.zeros((1,), dtype=float)
    return {
        "mean_curvature_px": float(np.mean(finite_px)),
        "median_curvature_px": float(np.median(finite_px)),
        "p95_curvature_px": float(np.percentile(finite_px, 95)),
        "max_curvature_px": float(np.max(finite_px)),
        "mean_curvature_nm": float(np.mean(finite_nm)),
        "median_curvature_nm": float(np.median(finite_nm)),
        "p95_curvature_nm": float(np.percentile(finite_nm, 95)),
        "max_curvature_nm": float(np.max(finite_nm)),
        "path_length_px": float(path_length_px),
        "span_px": float(span_px),
        "tortuosity": float(path_length_px / max(span_px, 1e-6)) if path_length_px > 0 else 0.0,
    }


def build_path_smoothing_profiles(extractor, coords_raw, px_per_um):
    coords_raw = np.asarray(coords_raw, dtype=float)
    if coords_raw.shape[0] < 2:
        return {}
    sample_count = max(2, int(round(extractor._path_length(coords_raw))) + 1)
    px_per_nm = max(float(px_per_um) / 1000.0, 1e-6)
    profiles = {}
    for profile_name in SMOOTH_PROFILE_NAMES:
        coords = smooth_profile_coords(extractor, coords_raw, profile_name, target_count=sample_count)
        arc = cumulative_arclength(coords)
        curvature_px = compute_fiji_like_curvature(coords, SMOOTH_PROFILE_CONFIGS[profile_name]["curvature_window_px"])
        span_px = float(np.linalg.norm(coords[-1] - coords[0])) if coords.shape[0] >= 2 else 0.0
        path_length_px = float(extractor._path_length(coords))
        profiles[profile_name] = {
            "coords": coords,
            "s_px": arc,
            "curvature_px": curvature_px,
            "curvature_nm": curvature_px * px_per_nm,
            "summary": summarize_curvature_series(curvature_px, px_per_nm, path_length_px, span_px),
            "point_count": int(coords.shape[0]),
            "curvature_window_px": float(SMOOTH_PROFILE_CONFIGS[profile_name]["curvature_window_px"]),
        }
    return profiles


def serialize_profile_summaries(profiles):
    payload = {}
    for name, profile in profiles.items():
        payload[name] = {
            "point_count": int(profile["point_count"]),
            "curvature_window_px": float(profile["curvature_window_px"]),
            **{key: float(value) for key, value in profile["summary"].items()},
        }
    return payload


def export_path_profile_csv(csv_path, profiles):
    if not profiles:
        return
    ordered_names = [name for name in SMOOTH_PROFILE_NAMES if name in profiles]
    row_count = max(int(profiles[name]["coords"].shape[0]) for name in ordered_names)
    headers = ["sample_index", "path_fraction"]
    for name in ordered_names:
        headers.extend(
            [
                f"{name}_s_px",
                f"{name}_y_px",
                f"{name}_x_px",
                f"{name}_curvature_px",
                f"{name}_curvature_nm",
            ]
        )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for idx in range(row_count):
            denom = max(row_count - 1, 1)
            row = [idx, float(idx) / float(denom)]
            for name in ordered_names:
                coords = profiles[name]["coords"]
                if idx < coords.shape[0]:
                    row.extend(
                        [
                            float(profiles[name]["s_px"][idx]),
                            float(coords[idx, 0]),
                            float(coords[idx, 1]),
                            float(profiles[name]["curvature_px"][idx]),
                            float(profiles[name]["curvature_nm"][idx]),
                        ]
                    )
                else:
                    row.extend(["", "", "", "", ""])
            writer.writerow(row)


def export_curvature_plot(plot_path, path_label, profiles):
    if not profiles:
        return
    colors = {
        "raw": "#94a3b8",
        "conservative": "#22c55e",
        "balanced": "#f59e0b",
        "visual": "#ef4444",
    }
    fig, ax = plt.subplots(figsize=(7.8, 4.4), dpi=160, constrained_layout=True)
    for name in SMOOTH_PROFILE_NAMES:
        if name not in profiles:
            continue
        ax.plot(
            profiles[name]["s_px"],
            profiles[name]["curvature_nm"],
            label=name,
            color=colors.get(name, None),
            linewidth=1.6 if name != "raw" else 1.2,
            alpha=0.95 if name != "raw" else 0.75,
        )
    ax.set_title(f"{path_label} Curvature")
    ax.set_xlabel("Arc Length (px)")
    ax.set_ylabel("Curvature (1/nm)")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)


def path_summary_record(path):
    record = {}
    for key, value in path.items():
        if key in {"coords", "coords_raw", "smooth_profiles"}:
            continue
        record[key] = value
    if "smooth_profiles" in path:
        record["smooth_profiles"] = serialize_profile_summaries(path["smooth_profiles"])
    return record


def export_curvature_artifacts(out_dir, candidates):
    csv_dir = out_dir / "path_profile_csv"
    plot_dir = out_dir / "path_curvature_plots"
    ensure_dir(csv_dir)
    ensure_dir(plot_dir)
    summary_rows = []
    summary_json = []
    for path in candidates:
        path_id = int(path["path_id"])
        profiles = path.get("smooth_profiles", {})
        if not profiles:
            continue
        export_path_profile_csv(csv_dir / f"path_{path_id:03d}_profiles.csv", profiles)
        export_curvature_plot(plot_dir / f"path_{path_id:03d}_curvature.png", f"Path {path_id:03d}", profiles)
        row = {
            "path_id": path_id,
            "seed_id": path.get("seed_id"),
            "seed_border_name": path.get("seed_border_name"),
            "end_border_name": path.get("end_border_name"),
            "main_score": path.get("main_score"),
            "target_band_hit": path.get("target_band_hit"),
        }
        profile_payload = {}
        for name, profile in profiles.items():
            stats = profile["summary"]
            row[f"{name}_length_px"] = float(stats["path_length_px"])
            row[f"{name}_span_px"] = float(stats["span_px"])
            row[f"{name}_tortuosity"] = float(stats["tortuosity"])
            row[f"{name}_mean_curvature_nm"] = float(stats["mean_curvature_nm"])
            row[f"{name}_median_curvature_nm"] = float(stats["median_curvature_nm"])
            row[f"{name}_p95_curvature_nm"] = float(stats["p95_curvature_nm"])
            row[f"{name}_max_curvature_nm"] = float(stats["max_curvature_nm"])
            profile_payload[name] = {
                **{k: float(v) for k, v in stats.items()},
                "point_count": int(profile["point_count"]),
                "curvature_window_px": float(profile["curvature_window_px"]),
            }
        summary_rows.append(row)
        summary_json.append({"path_id": path_id, "profiles": profile_payload})
    if summary_rows:
        fieldnames = list(summary_rows[0].keys())
        with (out_dir / "path_curvature_summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
    (out_dir / "path_curvature_summary.json").write_text(json.dumps(summary_json, ensure_ascii=False, indent=2), encoding="utf-8")


def path_coords(path, profile_name=None):
    if profile_name and "smooth_profiles" in path and profile_name in path["smooth_profiles"]:
        return np.asarray(path["smooth_profiles"][profile_name]["coords"], dtype=float)
    return np.asarray(path["coords"], dtype=float)


def summarize_image_curvature(candidates, profile_name="balanced"):
    rows = []
    for path in candidates:
        profiles = path.get("smooth_profiles", {})
        if profile_name not in profiles:
            continue
        stats = profiles[profile_name]["summary"]
        rows.append(
            {
                "path_id": int(path["path_id"]),
                "path_length_px": float(stats["path_length_px"]),
                "mean_curvature_nm": float(stats["mean_curvature_nm"]),
                "median_curvature_nm": float(stats["median_curvature_nm"]),
                "p95_curvature_nm": float(stats["p95_curvature_nm"]),
                "max_curvature_nm": float(stats["max_curvature_nm"]),
                "tortuosity": float(stats["tortuosity"]),
            }
        )
    if not rows:
        return {
            "profile_name": profile_name,
            "path_count": 0,
            "mean_curvature_nm": 0.0,
            "weighted_mean_curvature_nm": 0.0,
            "median_of_path_mean_curvature_nm": 0.0,
            "mean_tortuosity": 0.0,
            "distribution_bins_nm": [],
            "distribution_counts": [],
            "path_rows": [],
        }
    mean_values = np.asarray([row["mean_curvature_nm"] for row in rows], dtype=float)
    lengths = np.asarray([row["path_length_px"] for row in rows], dtype=float)
    tortuosities = np.asarray([row["tortuosity"] for row in rows], dtype=float)
    if float(np.sum(lengths)) <= 1e-6:
        weighted_mean = float(np.mean(mean_values))
    else:
        weighted_mean = float(np.average(mean_values, weights=lengths))
    hist_counts, hist_edges = np.histogram(mean_values, bins=min(12, max(4, len(rows))))
    return {
        "profile_name": profile_name,
        "path_count": int(len(rows)),
        "mean_curvature_nm": float(np.mean(mean_values)),
        "weighted_mean_curvature_nm": weighted_mean,
        "median_of_path_mean_curvature_nm": float(np.median(mean_values)),
        "mean_tortuosity": float(np.mean(tortuosities)),
        "distribution_bins_nm": [float(v) for v in hist_edges.tolist()],
        "distribution_counts": [int(v) for v in hist_counts.tolist()],
        "path_rows": rows,
    }


def export_curvature_distribution_plot(plot_path, curvature_summary):
    values = np.asarray([row["mean_curvature_nm"] for row in curvature_summary["path_rows"]], dtype=float)
    fig, ax = plt.subplots(figsize=(7.8, 4.8), dpi=160, constrained_layout=True)
    if values.size > 0:
        ax.hist(values, bins=min(12, max(4, values.size)), color="#60a5fa", edgecolor="#0f172a", alpha=0.85)
        ax.axvline(float(curvature_summary["mean_curvature_nm"]), color="#f59e0b", linewidth=2.0, label="Mean")
        ax.axvline(float(curvature_summary["weighted_mean_curvature_nm"]), color="#ef4444", linewidth=2.0, label="Weighted Mean")
    ax.set_title(f"Curvature Distribution ({curvature_summary['profile_name']})")
    ax.set_xlabel("Mean Curvature (1/nm)")
    ax.set_ylabel("Path Count")
    ax.grid(alpha=0.25, linestyle="--")
    if values.size > 0:
        ax.legend(frameon=False, fontsize=8)
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)


def border_info(coord, shape):
    y, x = [float(v) for v in coord]
    h, w = int(shape[0]), int(shape[1])
    distances = {"top": y, "bottom": (h - 1) - y, "left": x, "right": (w - 1) - x}
    min_distance = float(min(distances.values()))
    near = [name for name, value in distances.items() if value <= min_distance + 1.0]
    normals = {"top": np.array([1.0, 0.0]), "bottom": np.array([-1.0, 0.0]), "left": np.array([0.0, 1.0]), "right": np.array([0.0, -1.0])}
    return {
        "border_distance_px": min_distance,
        "border_name": "+".join(near),
        "distances": distances,
        "inward_normal": normalize(np.sum([normals[name] for name in near], axis=0)),
    }


def opposite(border_names):
    mapping = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
    return [mapping[name] for name in border_names if name in mapping]


def progress_from_borders(coord, shape, border_names):
    info = border_info(coord, shape)
    h, w = int(shape[0]), int(shape[1])
    scales = {"top": max(h - 1, 1), "bottom": max(h - 1, 1), "left": max(w - 1, 1), "right": max(w - 1, 1)}
    values = [float(info["distances"][name]) / float(scales[name]) for name in border_names]
    return float(np.mean(values)) if values else 0.0


def progress_to_borders(coord, shape, border_names):
    info = border_info(coord, shape)
    h, w = int(shape[0]), int(shape[1])
    scales = {"top": max(h - 1, 1), "bottom": max(h - 1, 1), "left": max(w - 1, 1), "right": max(w - 1, 1)}
    values = [1.0 - float(info["distances"][name]) / float(scales[name]) for name in border_names]
    return float(np.mean(values)) if values else 0.0


def distance_to_borders(coord, shape, border_names):
    info = border_info(coord, shape)
    values = [float(info["distances"][name]) for name in border_names]
    return min(values) if values else None


def find_gap_bridge_candidates(adjacency, max_gap_px, direction_cos_min, direction_hops=10):
    t0 = perf_counter()
    if max_gap_px is None or float(max_gap_px) <= 1.5:
        return [], {"endpoint_count": 0, "candidate_pairs_considered": 0, "elapsed_s": 0.0}
    endpoints = [node for node, neighbors in adjacency.items() if len(neighbors) == 1]
    endpoint_dirs = {node: estimate_seed_direction(node, adjacency, direction_hops) for node in endpoints}
    existing_edges = {edge_key(node, neighbor) for node, neighbors in adjacency.items() for neighbor in neighbors}
    cell_size = max(float(max_gap_px), 1.0)
    buckets = {}
    for idx, endpoint in enumerate(endpoints):
        key = (int(endpoint[0] // cell_size), int(endpoint[1] // cell_size))
        buckets.setdefault(key, []).append((idx, endpoint))
    candidates = []
    candidate_pairs_considered = 0
    for idx, first in enumerate(endpoints):
        dir_first = endpoint_dirs[first]
        if float(np.linalg.norm(dir_first)) <= 1e-8:
            continue
        first_vec = np.asarray(first, dtype=float)
        bucket_key = (int(first[0] // cell_size), int(first[1] // cell_size))
        nearby = []
        for by in range(bucket_key[0] - 1, bucket_key[0] + 2):
            for bx in range(bucket_key[1] - 1, bucket_key[1] + 2):
                nearby.extend(buckets.get((by, bx), []))
        for second_idx, second in nearby:
            if second_idx <= idx:
                continue
            if edge_key(first, second) in existing_edges:
                continue
            candidate_pairs_considered += 1
            second_vec = np.asarray(second, dtype=float)
            delta = second_vec - first_vec
            distance = float(np.linalg.norm(delta))
            if distance <= 1.5 or distance > float(max_gap_px):
                continue
            unit = delta / max(distance, 1e-6)
            dir_second = endpoint_dirs[second]
            if float(np.linalg.norm(dir_second)) <= 1e-8:
                continue
            cos_first = float(np.dot(dir_first, unit))
            cos_second = float(np.dot(dir_second, -unit))
            if min(cos_first, cos_second) < float(direction_cos_min):
                continue
            score = 0.55 * (cos_first + cos_second) + 0.45 * max(0.0, 1.0 - distance / max(float(max_gap_px), 1e-6))
            candidates.append((score, first, second, distance, cos_first, cos_second))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates, {
        "endpoint_count": int(len(endpoints)),
        "candidate_pairs_considered": int(candidate_pairs_considered),
        "elapsed_s": float(perf_counter() - t0),
    }


def heal_skeleton_gaps(skeleton_mask, max_gap_px, direction_cos_min):
    t0 = perf_counter()
    adjacency = build_adjacency(skeleton_mask)
    candidates, candidate_stats = find_gap_bridge_candidates(adjacency, max_gap_px=max_gap_px, direction_cos_min=direction_cos_min)
    healed = skeleton_mask.copy().astype(np.uint8)
    used_endpoints = set()
    bridges = []
    for score, first, second, distance, cos_first, cos_second in candidates:
        if first in used_endpoints or second in used_endpoints:
            continue
        cv2.line(healed, (int(first[1]), int(first[0])), (int(second[1]), int(second[0])), 1, 1, lineType=cv2.LINE_8)
        used_endpoints.add(first)
        used_endpoints.add(second)
        bridges.append(
            {
                "first": list(first),
                "second": list(second),
                "distance_px": float(distance),
                "cos_first": float(cos_first),
                "cos_second": float(cos_second),
                "bridge_score": float(score),
            }
        )
    return (healed > 0).astype(np.uint8), bridges, {
        "endpoint_count": int(candidate_stats["endpoint_count"]),
        "candidate_pairs_considered": int(candidate_stats["candidate_pairs_considered"]),
        "candidate_search_s": float(candidate_stats["elapsed_s"]),
        "elapsed_s": float(perf_counter() - t0),
    }


def build_probe_data(extractor, image, skeleton_heal_gap_px, skeleton_heal_direction_cos_min):
    t0 = perf_counter()
    roi = extractor.extract_roi(image)
    extractor._calibrate(roi.shape[1])
    processed = extractor.preprocess(roi)
    _, thresh = extractor.calculate_density(processed)
    _, skeleton = extractor.calculate_diameter(thresh)
    skeleton_mask = (skeleton > 0).astype(np.uint8)
    skeleton_mask, skeleton_bridges, bridge_stats = heal_skeleton_gaps(
        skeleton_mask,
        max_gap_px=skeleton_heal_gap_px,
        direction_cos_min=skeleton_heal_direction_cos_min,
    )
    adjacency = build_adjacency(skeleton_mask)
    neighbor_count = extractor._neighbor_count_map(skeleton_mask)
    num_clusters, cluster_labels = cv2.connectedComponents(((skeleton_mask > 0) & (neighbor_count >= 3)).astype(np.uint8), connectivity=8)
    cluster_pixels_cache, cluster_exit_cache = build_junction_cluster_cache(cluster_labels, adjacency)
    segment_cache = build_segment_cache(adjacency)
    processed_p10 = float(np.percentile(processed, 10))
    processed_p90 = float(np.percentile(processed, 90))
    return {
        "roi": roi,
        "processed": processed,
        "processed_p10": processed_p10,
        "processed_p90": processed_p90,
        "mask": thresh,
        "skeleton": skeleton_mask,
        "adjacency": adjacency,
        "segment_cache": segment_cache,
        "neighbor_count": neighbor_count,
        "endpoint_mask": (skeleton_mask > 0) & (neighbor_count <= 1),
        "junction_mask": (skeleton_mask > 0) & (neighbor_count >= 3),
        "junction_cluster_labels": cluster_labels,
        "junction_cluster_pixels_cache": cluster_pixels_cache,
        "junction_cluster_exit_cache": cluster_exit_cache,
        "junction_cluster_count": int(max(num_clusters - 1, 0)),
        "skeleton_bridge_count": len(skeleton_bridges),
        "skeleton_bridge_edges": skeleton_bridges,
        "bridge_candidate_pairs_considered": int(bridge_stats["candidate_pairs_considered"]),
        "bridge_candidate_endpoint_count": int(bridge_stats["endpoint_count"]),
        "bridge_candidate_search_s": float(bridge_stats["candidate_search_s"]),
        "skeleton_heal_s": float(bridge_stats["elapsed_s"]),
        "segment_cache_count": int(len(segment_cache)),
        "build_probe_data_s": float(perf_counter() - t0),
        "distance_map": cv2.distanceTransform((thresh > 0).astype(np.uint8), cv2.DIST_L2, 5),
        "px_per_um": float(extractor.px_per_um),
    }


def build_probe_data_from_external_mask(
    extractor,
    image,
    mask_image,
    mask_threshold,
    skeleton_heal_gap_px,
    skeleton_heal_direction_cos_min,
):
    t0 = perf_counter()
    roi = extractor.extract_roi(image)
    extractor._calibrate(roi.shape[1])
    processed = extractor.preprocess(roi)
    mask_roi = mask_image[: roi.shape[0], : roi.shape[1]]
    thresh = ((mask_roi >= int(mask_threshold)).astype(np.uint8)) * 255
    _, skeleton = extractor.calculate_diameter(thresh)
    skeleton_mask = (skeleton > 0).astype(np.uint8)
    skeleton_mask, skeleton_bridges, bridge_stats = heal_skeleton_gaps(
        skeleton_mask,
        max_gap_px=skeleton_heal_gap_px,
        direction_cos_min=skeleton_heal_direction_cos_min,
    )
    adjacency = build_adjacency(skeleton_mask)
    neighbor_count = extractor._neighbor_count_map(skeleton_mask)
    num_clusters, cluster_labels = cv2.connectedComponents(((skeleton_mask > 0) & (neighbor_count >= 3)).astype(np.uint8), connectivity=8)
    cluster_pixels_cache, cluster_exit_cache = build_junction_cluster_cache(cluster_labels, adjacency)
    segment_cache = build_segment_cache(adjacency)
    processed_p10 = float(np.percentile(processed, 10))
    processed_p90 = float(np.percentile(processed, 90))
    return {
        "roi": roi,
        "processed": processed,
        "processed_p10": processed_p10,
        "processed_p90": processed_p90,
        "mask": thresh,
        "skeleton": skeleton_mask,
        "adjacency": adjacency,
        "segment_cache": segment_cache,
        "neighbor_count": neighbor_count,
        "endpoint_mask": (skeleton_mask > 0) & (neighbor_count <= 1),
        "junction_mask": (skeleton_mask > 0) & (neighbor_count >= 3),
        "junction_cluster_labels": cluster_labels,
        "junction_cluster_pixels_cache": cluster_pixels_cache,
        "junction_cluster_exit_cache": cluster_exit_cache,
        "junction_cluster_count": int(max(num_clusters - 1, 0)),
        "skeleton_bridge_count": len(skeleton_bridges),
        "skeleton_bridge_edges": skeleton_bridges,
        "bridge_candidate_pairs_considered": int(bridge_stats["candidate_pairs_considered"]),
        "bridge_candidate_endpoint_count": int(bridge_stats["endpoint_count"]),
        "bridge_candidate_search_s": float(bridge_stats["candidate_search_s"]),
        "skeleton_heal_s": float(bridge_stats["elapsed_s"]),
        "segment_cache_count": int(len(segment_cache)),
        "build_probe_data_s": float(perf_counter() - t0),
        "distance_map": cv2.distanceTransform((thresh > 0).astype(np.uint8), cv2.DIST_L2, 5),
        "px_per_um": float(extractor.px_per_um),
        "mask_source": "external_mask",
    }


def build_adjacency(skeleton_mask):
    points = np.argwhere(skeleton_mask > 0)
    point_set = {tuple(int(v) for v in p) for p in points}
    adjacency = {}
    for y, x in point_set:
        neighbors = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                candidate = (y + dy, x + dx)
                if candidate in point_set:
                    neighbors.append(candidate)
        adjacency[(y, x)] = sorted(neighbors)
    return adjacency


def build_junction_cluster_cache(cluster_labels, adjacency):
    cluster_pixels = {}
    cluster_exits = {}
    max_cluster_id = int(cluster_labels.max()) if cluster_labels.size else 0
    for cluster_id in range(1, max_cluster_id + 1):
        ys, xs = np.where(cluster_labels == cluster_id)
        pixels = [tuple(int(v) for v in item) for item in np.column_stack([ys, xs])]
        cluster_pixels[cluster_id] = pixels
        cluster_set = set(pixels)
        exits = {}
        for cluster_pixel in pixels:
            for neighbor in adjacency.get(cluster_pixel, []):
                if neighbor in cluster_set:
                    continue
                exits.setdefault(neighbor, cluster_pixel)
        cluster_exits[cluster_id] = exits
    return cluster_pixels, cluster_exits


def build_segment_cache(adjacency):
    special_nodes = {node for node, neighbors in adjacency.items() if len(neighbors) != 2}
    if not special_nodes:
        special_nodes = set(adjacency.keys())
    cache = {}
    for start in special_nodes:
        for next_pixel in adjacency.get(start, []):
            key = (start, next_pixel)
            if key in cache:
                continue
            pixels = [next_pixel]
            traversed = [edge_key(start, next_pixel)]
            prev = start
            curr = next_pixel
            while curr not in special_nodes:
                next_candidates = [pix for pix in adjacency.get(curr, []) if pix != prev]
                if len(next_candidates) != 1:
                    break
                nxt = next_candidates[0]
                traversed.append(edge_key(curr, nxt))
                prev, curr = curr, nxt
                pixels.append(curr)
            coords = np.asarray([start] + pixels, dtype=float)
            cache[key] = {
                "pixels": tuple(pixels),
                "stop_pixel": curr,
                "previous_pixel": prev,
                "traversed_edges": frozenset(traversed),
                "direction": normalize(coords[-1] - coords[0]),
            }
    return cache


def junction_cluster_id(probe_data, pixel):
    y, x = [int(v) for v in pixel]
    labels = probe_data["junction_cluster_labels"]
    if y < 0 or x < 0 or y >= labels.shape[0] or x >= labels.shape[1]:
        return 0
    return int(labels[y, x])


def junction_cluster_pixels(probe_data, cluster_id):
    if cluster_id <= 0:
        return []
    return list(probe_data["junction_cluster_pixels_cache"].get(int(cluster_id), []))


def junction_cluster_exit_neighbors(probe_data, adjacency, cluster_id):
    return dict(probe_data["junction_cluster_exit_cache"].get(int(cluster_id), {}))


def edge_key(a, b):
    return (a, b) if a <= b else (b, a)


def estimate_seed_direction(endpoint, adjacency, max_hops):
    neighbors = adjacency.get(endpoint, [])
    if not neighbors:
        return np.zeros((2,), dtype=float)
    prev = endpoint
    curr = neighbors[0]
    walked = [endpoint, curr]
    for _ in range(max(1, int(max_hops)) - 1):
        next_candidates = [pix for pix in adjacency.get(curr, []) if pix != prev]
        if len(next_candidates) != 1:
            break
        prev, curr = curr, next_candidates[0]
        walked.append(curr)
    return normalize(np.asarray(walked[-1], dtype=float) - np.asarray(endpoint, dtype=float))


def trace_seed_walk(endpoint, adjacency, max_hops):
    neighbors = adjacency.get(endpoint, [])
    if not neighbors:
        return np.asarray([endpoint], dtype=float)
    prev = endpoint
    curr = neighbors[0]
    walked = [endpoint, curr]
    for _ in range(max(1, int(max_hops)) - 1):
        next_candidates = [pix for pix in adjacency.get(curr, []) if pix != prev]
        if len(next_candidates) != 1:
            break
        prev, curr = curr, next_candidates[0]
        walked.append(curr)
    return np.asarray(walked, dtype=float)


def collect_seeds(extractor, probe_data, adjacency, edge_seed_margin_px, seed_candidate_margin_px, top_seeds, seed_walk_hops):
    shape = probe_data["roi"].shape
    diagonal = float(np.hypot(shape[0], shape[1]))
    seeds = []
    for endpoint, neighbors in adjacency.items():
        if len(neighbors) != 1:
            continue
        direction = estimate_seed_direction(endpoint, adjacency, seed_walk_hops)
        if float(np.linalg.norm(direction)) <= 1e-8:
            continue
        edge = border_info(np.asarray(endpoint, dtype=float), shape)
        if edge["border_distance_px"] > seed_candidate_margin_px or "+" in edge["border_name"]:
            continue
        if edge["border_name"] not in {"top", "bottom"}:
            continue
        inward = float(np.dot(direction, edge["inward_normal"]))
        if inward <= 0.05:
            continue
        walk_coords = trace_seed_walk(endpoint, adjacency, seed_walk_hops)
        gray_values = sample_map_values_along_coords(walk_coords, probe_data["processed"])
        gray_mean = float(np.mean(gray_values)) if gray_values.size else 0.0
        gray_norm = float(
            np.clip(
                (gray_mean - float(probe_data["processed_p10"])) / max(float(probe_data["processed_p90"]) - float(probe_data["processed_p10"]), 1e-6),
                0.0,
                1.0,
            )
        )
        edge_priority = max(0.0, 1.0 - edge["border_distance_px"] / max(seed_candidate_margin_px, 1e-6))
        edge_band_bonus = max(0.0, 1.0 - edge["border_distance_px"] / max(edge_seed_margin_px, 1e-6))
        score = (
            0.14 * edge_priority
            + 0.08 * edge_band_bonus
            + 0.42 * inward
            + 0.18 * min(float(seed_walk_hops) / max(0.25 * diagonal, 1e-6), 1.0)
            + 0.18 * gray_norm
        )
        seeds.append(
            {
                "seed_id": f"seed_{len(seeds) + 1}",
                "pixel": endpoint,
                "coord": np.asarray(endpoint, dtype=float),
                "seed_direction": direction,
                "entry_neighbor": neighbors[0],
                "border_name": edge["border_name"],
                "target_borders": opposite([edge["border_name"]]),
                "border_distance_px": float(edge["border_distance_px"]),
                "inward_score": inward,
                "edge_priority": float(edge_priority),
                "seed_gray_mean": gray_mean,
                "seed_gray_norm": gray_norm,
                "seed_score": float(score),
            }
        )
    seeds.sort(key=lambda item: (item["seed_score"], item["seed_gray_norm"], item["inward_score"]), reverse=True)
    deduped = []
    for seed in seeds:
        keep = True
        for chosen in deduped:
            delta = np.asarray(seed["coord"], dtype=float) - np.asarray(chosen["coord"], dtype=float)
            similarity = float(np.dot(seed["seed_direction"], chosen["seed_direction"]))
            if float(np.linalg.norm(delta)) <= 10.0 and similarity >= 0.95:
                keep = False
                break
        if keep:
            deduped.append(seed)
    return deduped[: max(1, int(top_seeds))]


def advance_segment(adjacency, start_pixel, next_pixel, used_edges, probe_data=None, max_hops=2000):
    segment_cache = None if probe_data is None else probe_data.get("segment_cache")
    if segment_cache is not None:
        cached = segment_cache.get((start_pixel, next_pixel))
        if cached is not None:
            if not cached["traversed_edges"].isdisjoint(used_edges):
                return None
            return cached
    first = edge_key(start_pixel, next_pixel)
    if first in used_edges:
        return None
    pixels = [next_pixel]
    traversed = {first}
    prev = start_pixel
    curr = next_pixel
    for _ in range(max_hops):
        next_candidates = [pix for pix in adjacency.get(curr, []) if pix != prev]
        available = [pix for pix in next_candidates if edge_key(curr, pix) not in used_edges and edge_key(curr, pix) not in traversed]
        if len(adjacency.get(curr, [])) != 2 or len(available) != 1:
            break
        nxt = available[0]
        traversed.add(edge_key(curr, nxt))
        prev, curr = curr, nxt
        pixels.append(curr)
    coords = np.asarray([start_pixel] + pixels, dtype=float)
    return {
        "pixels": tuple(pixels),
        "stop_pixel": curr,
        "previous_pixel": prev,
        "traversed_edges": frozenset(traversed),
        "direction": normalize(coords[-1] - coords[0]),
    }


def smooth_coords(extractor, coords):
    coords = np.asarray(coords, dtype=float)
    if coords.shape[0] < 5:
        return coords
    return extractor._smooth_path_coords(coords)


def tail_coords_by_length(extractor, coords, window_px):
    coords = np.asarray(coords, dtype=float)
    if coords.shape[0] <= 1:
        return coords
    total = float(extractor._path_length(coords))
    if total <= float(window_px):
        return coords
    arc = cumulative_arclength(coords)
    start_s = max(0.0, float(arc[-1]) - float(window_px))
    sample_positions = arc[arc >= start_s]
    if sample_positions.size == 0 or sample_positions[0] > start_s + 1e-6:
        sample_positions = np.concatenate([[start_s], sample_positions])
    if sample_positions[-1] < arc[-1] - 1e-6:
        sample_positions = np.concatenate([sample_positions, [arc[-1]]])
    y_coords = np.interp(sample_positions, arc, coords[:, 0])
    x_coords = np.interp(sample_positions, arc, coords[:, 1])
    return np.column_stack([y_coords, x_coords])


def sample_map_values_along_coords(coords, value_map):
    coords = np.asarray(coords, dtype=float)
    if coords.size == 0:
        return np.zeros((0,), dtype=float)
    h, w = int(value_map.shape[0]), int(value_map.shape[1])
    pixels = np.round(coords).astype(int)
    pixels[:, 0] = np.clip(pixels[:, 0], 0, h - 1)
    pixels[:, 1] = np.clip(pixels[:, 1], 0, w - 1)
    return value_map[pixels[:, 0], pixels[:, 1]].astype(float)


def gray_signature_for_coords(extractor, coords, value_map, window_px):
    tail = tail_coords_by_length(extractor, coords, window_px)
    values = sample_map_values_along_coords(tail, value_map)
    if values.size == 0:
        return {"mean": 0.0, "std": 0.0, "count": 0}
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "count": int(values.size),
    }


def gray_consistency_metrics(reference_signature, candidate_signature):
    ref_mean = float(reference_signature.get("mean", 0.0))
    ref_std = float(reference_signature.get("std", 0.0))
    cand_mean = float(candidate_signature.get("mean", 0.0))
    gray_diff = abs(cand_mean - ref_mean)
    tolerance = max(12.0, 1.8 * ref_std + 8.0)
    score = float(np.clip(1.0 - gray_diff / max(tolerance, 1e-6), 0.0, 1.0))
    return {
        "reference_gray_mean": ref_mean,
        "reference_gray_std": ref_std,
        "candidate_gray_mean": cand_mean,
        "candidate_gray_std": float(candidate_signature.get("std", 0.0)),
        "gray_diff": float(gray_diff),
        "gray_tolerance": float(tolerance),
        "gray_consistency_score": score,
    }


def smoothed_terminal_direction(extractor, coords, fallback, lookback=12):
    coords = smooth_coords(extractor, coords)
    if coords.shape[0] < 2:
        return normalize(fallback)
    span = min(max(2, int(lookback)), coords.shape[0] - 1)
    direction = normalize(coords[-1] - coords[-span - 1])
    if float(np.linalg.norm(direction)) <= 1e-8:
        return normalize(fallback)
    return direction


def state_geometry(extractor, state):
    coords = smooth_coords(extractor, np.asarray(state["coords"], dtype=float))
    if coords.shape[0] < 2:
        return {"span_px": 0.0, "path_length_px": 0.0, "projected_progress_px": 0.0, "lateral_drift_px": 0.0}
    displacement = coords[-1] - coords[0]
    perpendicular = np.array([-state["seed_direction"][1], state["seed_direction"][0]], dtype=float)
    return {
        "span_px": float(np.linalg.norm(displacement)),
        "path_length_px": float(extractor._path_length(coords)),
        "projected_progress_px": float(np.dot(displacement, state["seed_direction"])),
        "lateral_drift_px": float(abs(np.dot(displacement, perpendicular))),
    }


def recent_motion_metrics(extractor, coords, reference_direction, window_points=24):
    coords = smooth_coords(extractor, np.asarray(coords, dtype=float))
    if coords.shape[0] < 3:
        return {
            "recent_path_length_px": 0.0,
            "recent_forward_progress_px": 0.0,
            "recent_forward_ratio": 1.0,
            "recent_straightness": 1.0,
            "recent_reverse_progress_px": 0.0,
        }
    span = min(max(3, int(window_points)), coords.shape[0])
    tail = coords[-span:]
    path_length = float(extractor._path_length(tail))
    if path_length <= 1e-6:
        return {
            "recent_path_length_px": 0.0,
            "recent_forward_progress_px": 0.0,
            "recent_forward_ratio": 1.0,
            "recent_straightness": 1.0,
            "recent_reverse_progress_px": 0.0,
        }
    disp = tail[-1] - tail[0]
    ref = normalize(reference_direction)
    forward_progress = float(np.dot(disp, ref))
    reverse_progress = float(max(0.0, -forward_progress))
    forward_ratio = float(max(0.0, forward_progress) / max(path_length, 1e-6))
    straightness = float(np.linalg.norm(disp) / max(path_length, 1e-6))
    return {
        "recent_path_length_px": path_length,
        "recent_forward_progress_px": forward_progress,
        "recent_forward_ratio": forward_ratio,
        "recent_straightness": straightness,
        "recent_reverse_progress_px": reverse_progress,
    }


def anti_hook_penalty(recent_metrics):
    path_length = float(recent_metrics["recent_path_length_px"])
    if path_length < 12.0:
        return 0.0
    penalty = 0.0
    penalty += 0.90 * max(0.0, 0.58 - float(recent_metrics["recent_forward_ratio"]))
    penalty += 0.70 * max(0.0, 0.72 - float(recent_metrics["recent_straightness"]))
    penalty += 0.06 * float(recent_metrics["recent_reverse_progress_px"])
    if path_length >= 18.0 and float(recent_metrics["recent_forward_progress_px"]) < 8.0:
        penalty += 0.45
    return float(penalty)


def beam_rank(extractor, state, shape):
    diagonal = float(np.hypot(shape[0], shape[1]))
    geom = state_geometry(extractor, state)
    progress = max(0.0, geom["projected_progress_px"]) / max(diagonal, 1e-6)
    span = geom["span_px"] / max(diagonal, 1e-6)
    length = geom["path_length_px"] / max(diagonal, 1e-6)
    straight = geom["span_px"] / max(geom["path_length_px"], 1e-6)
    stability = 1.0 / (1.0 + state["cumulative_turn_deg"] / 120.0 + 0.7 * state["soft_turn_count"])
    hook_penalty = float(state.get("hook_penalty", 0.0))
    gray_score = float(state.get("gray_consistency_running_mean", 0.5))
    return float(0.23 * progress + 0.22 * span + 0.14 * length + 0.14 * straight + 0.09 * stability + 0.06 * state["probe_score"] + 0.12 * gray_score - 0.18 * hook_penalty)


def final_score(extractor, state, shape, target_hit_margin_px):
    diagonal = float(np.hypot(shape[0], shape[1]))
    coords = smooth_coords(extractor, np.asarray(state["coords"], dtype=float))
    geom = state_geometry(extractor, state)
    start_borders = [name for name in state["start_border_name"].split("+") if name]
    target_borders = list(state["target_borders"])
    end = border_info(coords[-1], shape)
    target_progress = progress_to_borders(coords[-1], shape, target_borders)
    escape_progress = progress_from_borders(coords[-1], shape, start_borders)
    target_distances = [float(end["distances"][name]) for name in target_borders]
    distance_to_target_border_px = min(target_distances) if target_distances else None
    target_band_hit = bool(
        distance_to_target_border_px is not None and distance_to_target_border_px <= float(target_hit_margin_px)
    )
    span = geom["span_px"] / max(diagonal, 1e-6)
    progress = max(0.0, geom["projected_progress_px"]) / max(diagonal, 1e-6)
    length = geom["path_length_px"] / max(diagonal, 1e-6)
    straight = geom["span_px"] / max(geom["path_length_px"], 1e-6)
    stability = 1.0 / (1.0 + state["cumulative_turn_deg"] / 120.0 + 0.7 * state["soft_turn_count"])
    hook_penalty = float(state.get("hook_penalty", 0.0))
    recent_forward_ratio = float(state.get("recent_forward_ratio", 1.0))
    recent_straightness = float(state.get("recent_straightness", 1.0))
    target_hit_bonus = 0.20 if target_band_hit else 0.0
    same_border_penalty = 0.18 if end["border_name"] in start_borders else 0.0
    weak_target_penalty = 0.25 * max(0.0, 0.40 - target_progress)
    weak_escape_penalty = 0.20 * max(0.0, 0.30 - escape_progress)
    score = float(
        0.26 * target_progress
        + 0.20 * escape_progress
        + 0.18 * span
        + 0.12 * progress
        + 0.10 * length
        + 0.08 * straight
        + 0.06 * stability
        + target_hit_bonus
        - same_border_penalty
        - 0.12 * hook_penalty
        - weak_target_penalty
        - weak_escape_penalty
    )
    return {
        "main_score": score,
        "target_progress_norm": float(target_progress),
        "escape_progress_norm": float(escape_progress),
        "span_norm": float(span),
        "progress_norm": float(progress),
        "length_norm": float(length),
        "straightness": float(straight),
        "direction_stability": float(stability),
        "hook_penalty": hook_penalty,
        "recent_forward_ratio": recent_forward_ratio,
        "recent_straightness": recent_straightness,
        "target_hit_bonus": float(target_hit_bonus),
        "target_band_hit": target_band_hit,
        "target_hit_margin_px": float(target_hit_margin_px),
        "same_border_penalty": float(same_border_penalty),
        "weak_target_penalty": float(weak_target_penalty),
        "weak_escape_penalty": float(weak_escape_penalty),
        "end_border_name": end["border_name"],
        "end_border_distance_px": float(end["border_distance_px"]),
        "distance_to_target_border_px": distance_to_target_border_px,
    }


def diagnose_terminal_options(extractor, probe_data, adjacency, state):
    current = tuple(int(v) for v in state["current_pixel"])
    cluster_id = junction_cluster_id(probe_data, current)
    target_borders = list(state["target_borders"])
    options = []
    viable_count = 0

    if cluster_id > 0:
        candidate_specs = []
        for exit_neighbor, boundary_pixel in junction_cluster_exit_neighbors(probe_data, adjacency, cluster_id).items():
            if exit_neighbor == state["previous_pixel"]:
                continue
            candidate_specs.append((boundary_pixel, exit_neighbor, True))
    else:
        neighbors = adjacency.get(current, [])
        candidate_specs = [(current, nxt, False) for nxt in neighbors if nxt != state["previous_pixel"]]

    for boundary_pixel, nxt, from_cluster in candidate_specs:
        is_used = edge_key(boundary_pixel, nxt) in state["used_edges"]
        segment = None if is_used else advance_segment(adjacency, boundary_pixel, nxt, state["used_edges"], probe_data=probe_data)
        if segment is None:
            options.append(
                {
                    "next_pixel": list(nxt),
                    "boundary_pixel": list(boundary_pixel),
                    "from_cluster": bool(from_cluster),
                    "blocked_by": "used_edge" if is_used else "unavailable_segment",
                }
            )
            continue

        similarity = float(np.clip(np.dot(state["current_direction"], segment["direction"]), -1.0, 1.0))
        turn_deg = float(np.degrees(np.arccos(similarity)))
        effective_hard_deg = float(probe_data["angle_hard_deg"] + (probe_data.get("cluster_exit_relax_deg", 0.0) if from_cluster else 0.0))
        hard_blocked = bool(turn_deg > effective_hard_deg)
        if not hard_blocked:
            viable_count += 1
        stop_pixel = segment["stop_pixel"]
        stop_degree = int(len(adjacency.get(stop_pixel, [])))
        stop_target_progress = progress_to_borders(np.asarray(stop_pixel, dtype=float), probe_data["roi"].shape, target_borders)
        segment_coords = np.asarray([current] + list(segment["pixels"]), dtype=float)
        options.append(
            {
                "next_pixel": list(nxt),
                "boundary_pixel": list(boundary_pixel),
                "from_cluster": bool(from_cluster),
                "turn_deg": turn_deg,
                "effective_hard_deg": effective_hard_deg,
                "hard_blocked": hard_blocked,
                "segment_length_px": float(extractor._path_length(segment_coords)),
                "stop_pixel": list(stop_pixel),
                "stop_degree": stop_degree,
                "stop_target_progress": float(stop_target_progress),
            }
        )

    if cluster_id <= 0 and len(adjacency.get(current, [])) < 3:
        reason = "non_junction_stop"
    elif viable_count > 0:
        reason = "junction_with_viable_exits"
    elif options:
        reason = "junction_exits_blocked"
    else:
        reason = "junction_no_exits"

    return {
        "termination_reason": reason,
        "terminal_options": options,
    }


def viable_junction_extensions(extractor, probe_data, adjacency, state, angle_limit_deg, angle_hard_deg):
    current = tuple(int(v) for v in state["current_pixel"])
    cluster_id = junction_cluster_id(probe_data, current)
    candidate_specs = []

    if cluster_id > 0:
        exit_neighbors = junction_cluster_exit_neighbors(probe_data, adjacency, cluster_id)
        for exit_neighbor, boundary_pixel in exit_neighbors.items():
            if exit_neighbor == state["previous_pixel"]:
                continue
            if edge_key(boundary_pixel, exit_neighbor) in state["used_edges"]:
                continue
            candidate_specs.append((boundary_pixel, exit_neighbor, True))
    else:
        neighbors = adjacency.get(current, [])
        if len(neighbors) < 3:
            return []
        for nxt in neighbors:
            if nxt == state["previous_pixel"]:
                continue
            if edge_key(current, nxt) in state["used_edges"]:
                continue
            candidate_specs.append((current, nxt, False))

    candidates = []
    reference_signature = gray_signature_for_coords(
        extractor,
        np.asarray(state["coords"], dtype=float),
        probe_data["processed"],
        probe_data["gray_window_path_px"],
    )
    for boundary_pixel, nxt, from_cluster in candidate_specs:
        segment = advance_segment(adjacency, boundary_pixel, nxt, state["used_edges"], probe_data=probe_data)
        if segment is None:
            continue

        similarity = float(np.clip(np.dot(state["current_direction"], segment["direction"]), -1.0, 1.0))
        turn = float(np.degrees(np.arccos(similarity)))
        effective_hard_deg = float(angle_hard_deg + (probe_data.get("cluster_exit_relax_deg", 0.0) if from_cluster else 0.0))
        if turn > effective_hard_deg:
            continue

        append_parts = []
        if from_cluster and boundary_pixel != current:
            append_parts.append(np.asarray([boundary_pixel], dtype=float))
        append_parts.append(np.asarray(list(segment["pixels"]), dtype=float))
        appended = np.vstack(append_parts)
        candidate_signature = gray_signature_for_coords(
            extractor,
            appended,
            probe_data["processed"],
            probe_data["gray_window_candidate_px"],
        )
        gray_metrics = gray_consistency_metrics(reference_signature, candidate_signature)

        new_state = dict(state)
        new_state["coords"] = np.vstack([state["coords"], appended])
        new_state["used_edges"] = set(state["used_edges"]) | set(segment["traversed_edges"])
        new_state["current_pixel"] = segment["stop_pixel"]
        new_state["previous_pixel"] = segment["previous_pixel"]
        new_state["current_direction"] = smoothed_terminal_direction(extractor, new_state["coords"], segment["direction"])
        new_state["cumulative_turn_deg"] = float(state["cumulative_turn_deg"] + turn)
        new_state["soft_turn_count"] = int(state["soft_turn_count"] + int(turn > angle_limit_deg))
        recent_metrics = recent_motion_metrics(extractor, new_state["coords"], state["current_direction"])
        hook_penalty = anti_hook_penalty(recent_metrics)
        new_state["recent_forward_ratio"] = float(recent_metrics["recent_forward_ratio"])
        new_state["recent_straightness"] = float(recent_metrics["recent_straightness"])
        new_state["hook_penalty"] = hook_penalty
        new_state["gray_consistency_score"] = float(gray_metrics["gray_consistency_score"])
        prev_gray = float(state.get("gray_consistency_running_mean", 0.5))
        new_state["gray_consistency_running_mean"] = float(0.75 * prev_gray + 0.25 * gray_metrics["gray_consistency_score"])

        geom = state_geometry(extractor, new_state)
        diagonal = float(np.hypot(probe_data["roi"].shape[0], probe_data["roi"].shape[1]))
        target_progress = progress_to_borders(new_state["coords"][-1], probe_data["roi"].shape, state["target_borders"])
        local_gain = (
            0.90 * max(0.0, geom["projected_progress_px"]) / max(diagonal, 1e-6)
            + 0.80 * geom["span_px"] / max(diagonal, 1e-6)
            + 0.35 * target_progress
            + 0.18 * gray_metrics["gray_consistency_score"]
            - 0.55 * (turn / max(effective_hard_deg, 1e-6))
            - hook_penalty
        )
        new_state["probe_score"] = float(state["probe_score"] + local_gain)
        candidates.append(
            {
                "state": new_state,
                "turn_deg": turn,
                "target_progress": float(target_progress),
                "beam_rank": beam_rank(extractor, new_state, probe_data["roi"].shape),
                "from_cluster": bool(from_cluster),
                "boundary_pixel": list(boundary_pixel),
                "next_pixel": list(nxt),
                "effective_hard_deg": effective_hard_deg,
                **gray_metrics,
            }
        )

    candidates.sort(
        key=lambda item: (item["beam_rank"], item["gray_consistency_score"], item["target_progress"], -item["turn_deg"]),
        reverse=True,
    )
    return candidates


def evaluate_cluster_backtrack(extractor, probe_data, prior_state, candidate):
    next_state = candidate["state"]
    reasons = []
    hook_penalty = float(next_state.get("hook_penalty", 0.0))
    recent_forward_ratio = float(next_state.get("recent_forward_ratio", 1.0))
    recent_straightness = float(next_state.get("recent_straightness", 1.0))
    before_target_dist = distance_to_borders(prior_state["coords"][-1], probe_data["roi"].shape, prior_state["target_borders"])
    after_target_dist = distance_to_borders(next_state["coords"][-1], probe_data["roi"].shape, prior_state["target_borders"])
    target_improve_px = None
    if before_target_dist is not None and after_target_dist is not None:
        target_improve_px = float(before_target_dist - after_target_dist)

    hook_fail = hook_penalty > float(probe_data["cluster_backtrack_hook_penalty_max"])
    forward_fail = recent_forward_ratio < float(probe_data["cluster_backtrack_min_forward_ratio"])
    straight_fail = recent_straightness < float(probe_data["cluster_backtrack_min_straightness"])
    target_fail = target_improve_px is not None and target_improve_px < float(probe_data["cluster_backtrack_min_target_improve_px"])

    if hook_fail:
        reasons.append("hook")
    if forward_fail:
        reasons.append("forward")
    if straight_fail:
        reasons.append("straightness")
    if target_fail:
        reasons.append("target_progress")

    reject = bool((hook_fail and (forward_fail or target_fail)) or (forward_fail and straight_fail and target_fail))

    return {
        "accepted": not reject,
        "reasons": reasons,
        "hook_penalty": hook_penalty,
        "recent_forward_ratio": recent_forward_ratio,
        "recent_straightness": recent_straightness,
        "target_improve_px": target_improve_px,
    }


def force_continue_viable_junctions(extractor, probe_data, adjacency, state, angle_limit_deg, angle_hard_deg):
    current_state = dict(state)
    forced_steps = 0
    cluster_backtrack_rejections = 0
    cluster_backtrack_rejection_events = []
    failed_cluster_exits = set()
    safety_limit = max(32, int(probe_data["junction_cluster_count"]) + 8)
    while forced_steps < safety_limit:
        current_cluster_id = junction_cluster_id(probe_data, current_state["current_pixel"])
        candidates = viable_junction_extensions(
            extractor=extractor,
            probe_data=probe_data,
            adjacency=adjacency,
            state=current_state,
            angle_limit_deg=angle_limit_deg,
            angle_hard_deg=angle_hard_deg,
        )
        if not candidates:
            break
        selected = None
        if current_cluster_id > 0:
            for candidate in candidates:
                exit_key = (
                    int(current_cluster_id),
                    tuple(int(v) for v in candidate["boundary_pixel"]),
                    tuple(int(v) for v in candidate["next_pixel"]),
                )
                if exit_key in failed_cluster_exits:
                    continue
                evaluation = evaluate_cluster_backtrack(extractor, probe_data, current_state, candidate)
                if not evaluation["accepted"]:
                    failed_cluster_exits.add(exit_key)
                    cluster_backtrack_rejections += 1
                    cluster_backtrack_rejection_events.append(
                        {
                            "cluster_id": int(current_cluster_id),
                            "boundary_pixel": list(candidate["boundary_pixel"]),
                            "next_pixel": list(candidate["next_pixel"]),
                            "reasons": list(evaluation["reasons"]),
                            "target_improve_px": evaluation["target_improve_px"],
                            "hook_penalty": evaluation["hook_penalty"],
                            "recent_forward_ratio": evaluation["recent_forward_ratio"],
                            "recent_straightness": evaluation["recent_straightness"],
                        }
                    )
                    continue
                selected = candidate
                break
        else:
            selected = candidates[0]
        if selected is None:
            break
        current_state = selected["state"]
        forced_steps += 1
    current_state["cluster_backtrack_rejections"] = int(cluster_backtrack_rejections)
    current_state["cluster_backtrack_rejection_events"] = list(cluster_backtrack_rejection_events[-12:])
    return current_state, forced_steps


def trace_seed(extractor, probe_data, adjacency, seed, args):
    start = tuple(int(v) for v in seed["pixel"])
    first_neighbor = tuple(int(v) for v in seed["entry_neighbor"])
    if first_neighbor not in adjacency.get(start, []):
        return []
    initial = advance_segment(adjacency, start, first_neighbor, used_edges=set(), probe_data=probe_data)
    if initial is None:
        return []
    initial_coords = np.asarray([start] + list(initial["pixels"]), dtype=float)
    beams = [{
        "seed_id": seed["seed_id"],
        "start_border_name": seed["border_name"],
        "start_border_distance_px": float(seed["border_distance_px"]),
        "target_borders": list(seed["target_borders"]),
        "coords": initial_coords,
        "used_edges": set(initial["traversed_edges"]),
        "current_pixel": initial["stop_pixel"],
        "previous_pixel": initial["previous_pixel"],
        "seed_direction": np.asarray(seed["seed_direction"], dtype=float),
        "current_direction": smoothed_terminal_direction(extractor, initial_coords, initial["direction"]),
        "cumulative_turn_deg": 0.0,
        "soft_turn_count": 0,
        "probe_score": 0.0,
        "hook_penalty": 0.0,
        "recent_forward_ratio": 1.0,
        "recent_straightness": 1.0,
        "gray_consistency_score": 0.5,
        "gray_consistency_running_mean": 0.5,
    }]

    for _ in range(max(1, int(args.max_junction_visits))):
        expanded = []
        any_expansion = False
        for state in beams:
            current = tuple(int(v) for v in state["current_pixel"])
            neighbors = adjacency.get(current, [])
            if len(neighbors) < 3:
                expanded.append(state)
                continue
            candidates = [pix for pix in neighbors if pix != state["previous_pixel"] and edge_key(current, pix) not in state["used_edges"]]
            if not candidates:
                expanded.append(state)
                continue
            scored = []
            reference_signature = gray_signature_for_coords(
                extractor,
                np.asarray(state["coords"], dtype=float),
                probe_data["processed"],
                probe_data["gray_window_path_px"],
            )
            for nxt in candidates:
                segment = advance_segment(adjacency, current, nxt, state["used_edges"], probe_data=probe_data)
                if segment is None:
                    continue
                similarity = float(np.clip(np.dot(state["current_direction"], segment["direction"]), -1.0, 1.0))
                turn = float(np.degrees(np.arccos(similarity)))
                if turn > args.angle_hard_deg:
                    continue
                segment_pixels = np.asarray(list(segment["pixels"]), dtype=float)
                candidate_signature = gray_signature_for_coords(
                    extractor,
                    segment_pixels,
                    probe_data["processed"],
                    probe_data["gray_window_candidate_px"],
                )
                gray_metrics = gray_consistency_metrics(reference_signature, candidate_signature)
                new_state = dict(state)
                new_state["coords"] = np.vstack([state["coords"], segment_pixels])
                new_state["used_edges"] = set(state["used_edges"]) | set(segment["traversed_edges"])
                new_state["current_pixel"] = segment["stop_pixel"]
                new_state["previous_pixel"] = segment["previous_pixel"]
                new_state["current_direction"] = smoothed_terminal_direction(extractor, new_state["coords"], segment["direction"])
                new_state["cumulative_turn_deg"] = float(state["cumulative_turn_deg"] + turn)
                new_state["soft_turn_count"] = int(state["soft_turn_count"] + int(turn > args.angle_limit_deg))
                recent_metrics = recent_motion_metrics(extractor, new_state["coords"], state["current_direction"])
                hook_penalty = anti_hook_penalty(recent_metrics)
                new_state["recent_forward_ratio"] = float(recent_metrics["recent_forward_ratio"])
                new_state["recent_straightness"] = float(recent_metrics["recent_straightness"])
                new_state["hook_penalty"] = hook_penalty
                new_state["gray_consistency_score"] = float(gray_metrics["gray_consistency_score"])
                new_state["gray_consistency_running_mean"] = float(0.75 * float(state.get("gray_consistency_running_mean", 0.5)) + 0.25 * gray_metrics["gray_consistency_score"])
                if new_state["cumulative_turn_deg"] > args.max_cumulative_turn_deg:
                    continue
                geom = state_geometry(extractor, new_state)
                diagonal = float(np.hypot(probe_data["roi"].shape[0], probe_data["roi"].shape[1]))
                local_gain = (
                    0.90 * max(0.0, geom["projected_progress_px"]) / max(diagonal, 1e-6)
                    + 0.80 * geom["span_px"] / max(diagonal, 1e-6)
                    + 0.35 * progress_to_borders(new_state["coords"][-1], probe_data["roi"].shape, seed["target_borders"])
                    + 0.18 * gray_metrics["gray_consistency_score"]
                    - 0.55 * (turn / max(args.angle_hard_deg, 1e-6))
                    - hook_penalty
                )
                new_state["probe_score"] = float(state["probe_score"] + local_gain)
                scored.append((new_state, beam_rank(extractor, new_state, probe_data["roi"].shape)))
            if not scored:
                expanded.append(state)
                continue
            any_expansion = True
            scored.sort(key=lambda item: item[1], reverse=True)
            expanded.extend(state_item for state_item, _ in scored[: max(1, int(args.beam_width))])
        expanded.sort(key=lambda item: beam_rank(extractor, item, probe_data["roi"].shape), reverse=True)
        beams = expanded[: max(1, int(args.beam_width))]
        if not any_expansion:
            break

    candidates = []
    for state in beams:
        finalized_state, forced_steps = force_continue_viable_junctions(
            extractor=extractor,
            probe_data=probe_data,
            adjacency=adjacency,
            state=state,
            angle_limit_deg=args.angle_limit_deg,
            angle_hard_deg=args.angle_hard_deg,
        )
        coords_raw = np.asarray(finalized_state["coords"], dtype=float)
        coords = smooth_coords(extractor, coords_raw)
        metrics = {
            "coords": coords,
            "coords_raw": coords_raw,
            "path_length_px": float(extractor._path_length(coords)),
            "seed_id": seed["seed_id"],
            "seed_border_name": seed["border_name"],
            "seed_border_distance_px": float(seed["border_distance_px"]),
            "projected_progress_px": float(state_geometry(extractor, finalized_state)["projected_progress_px"]),
            "cumulative_turn_deg": float(finalized_state["cumulative_turn_deg"]),
            "soft_turn_count": int(finalized_state["soft_turn_count"]),
            "probe_score": float(finalized_state["probe_score"]),
            "forced_rollout_steps": int(forced_steps),
            "cluster_backtrack_rejections": int(finalized_state.get("cluster_backtrack_rejections", 0)),
            "cluster_backtrack_rejection_events": list(finalized_state.get("cluster_backtrack_rejection_events", [])),
            "gray_consistency_score": float(finalized_state.get("gray_consistency_score", 0.5)),
            "gray_consistency_running_mean": float(finalized_state.get("gray_consistency_running_mean", 0.5)),
            "terminal_pixel": [int(round(v)) for v in coords[-1]],
            "terminal_degree": int(len(adjacency.get(tuple(int(round(v)) for v in coords[-1]), []))),
            "terminal_is_junction": bool(len(adjacency.get(tuple(int(round(v)) for v in coords[-1]), [])) >= 3),
            "terminal_untraversed_neighbors": int(
                len(
                    [
                        pix
                        for pix in adjacency.get(tuple(int(round(v)) for v in coords[-1]), [])
                        if edge_key(tuple(int(round(v)) for v in coords[-1]), pix) not in finalized_state["used_edges"]
                    ]
                )
            ),
        }
        span_px = float(np.linalg.norm(coords[-1] - coords[0])) if coords.shape[0] >= 2 else 0.0
        px_per_nm = max(probe_data["px_per_um"] / 1000.0, 1e-6)
        metrics["span_px"] = span_px
        metrics["path_length_nm"] = float(metrics["path_length_px"] / px_per_nm)
        metrics["span_nm"] = float(span_px / px_per_nm)
        metrics["ld_ratio"] = float(metrics["path_length_px"] / max(span_px, 1e-6)) if metrics["path_length_px"] > 0 else 0.0
        metrics.update(final_score(extractor, finalized_state, probe_data["roi"].shape, probe_data["target_hit_margin_px"]))
        metrics.update(diagnose_terminal_options(extractor, probe_data, adjacency, finalized_state))
        candidates.append(metrics)
    candidates.sort(key=lambda item: (item["main_score"], item["span_px"], item["path_length_px"]), reverse=True)
    return candidates


def draw_points(canvas, points, color, radius):
    result = canvas.copy()
    for y, x in points:
        cv2.circle(result, (int(x), int(y)), radius, color, -1)
    return result


def draw_seeds(canvas, seeds):
    result = canvas.copy()
    for idx, seed in enumerate(seeds, start=1):
        y, x = np.round(seed["coord"]).astype(int)
        end = np.round(seed["coord"] + 18.0 * normalize(seed["seed_direction"])).astype(int)
        cv2.circle(result, (x, y), 5, (0, 220, 255), -1)
        cv2.arrowedLine(result, (x, y), (int(end[1]), int(end[0])), (0, 220, 255), 2, tipLength=0.25)
        cv2.putText(result, str(idx), (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return result


def draw_paths(canvas, paths, highlight_first=False, profile_name="balanced"):
    palette = [(80, 220, 255), (120, 255, 120), (255, 200, 80), (240, 120, 255), (255, 120, 120), (160, 180, 255)]
    result = canvas.copy()
    for idx, path in enumerate(paths):
        coords = path_coords(path, profile_name=profile_name)
        if coords.shape[0] < 2:
            continue
        color = (60, 80, 255) if highlight_first and idx == 0 else palette[idx % len(palette)]
        thickness = 4 if highlight_first and idx == 0 else 2
        pts = np.round(coords[:, ::-1]).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(result, [pts], False, color, thickness, lineType=cv2.LINE_AA)
        y, x = np.round(coords[0]).astype(int)
        path_label = path.get("path_id", path.get("rejected_path_id", idx + 1))
        cv2.putText(result, str(path_label), (x + 3, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return result


def draw_path_overlay_on_roi(roi, kept_paths, rejected_paths, profile_name="balanced"):
    base = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
    overlay = base.copy()
    for path in rejected_paths:
        coords = path_coords(path, profile_name=profile_name)
        if coords.shape[0] < 2:
            continue
        pts = np.round(coords[:, ::-1]).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(overlay, [pts], False, (100, 70, 220), 1, lineType=cv2.LINE_AA)
    palette = [(80, 220, 255), (120, 255, 120), (255, 200, 80), (240, 120, 255), (255, 120, 120), (160, 180, 255)]
    for idx, path in enumerate(kept_paths):
        coords = path_coords(path, profile_name=profile_name)
        if coords.shape[0] < 2:
            continue
        pts = np.round(coords[:, ::-1]).astype(np.int32).reshape(-1, 1, 2)
        color = palette[idx % len(palette)]
        cv2.polylines(overlay, [pts], False, color, 2, lineType=cv2.LINE_AA)
        y, x = np.round(coords[0]).astype(int)
        cv2.putText(overlay, str(path.get("path_id", idx + 1)), (x + 3, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return cv2.addWeighted(base, 0.72, overlay, 0.88, 0.0)


def draw_path_overlay(canvas, paths, profile_name="balanced", palette=None, thickness=2, label_paths=True):
    if palette is None:
        palette = [(80, 220, 255), (120, 255, 120), (255, 200, 80), (240, 120, 255), (255, 120, 120), (160, 180, 255)]
    result = canvas.copy()
    for idx, path in enumerate(paths):
        coords = path_coords(path, profile_name=profile_name)
        if coords.shape[0] < 2:
            continue
        color = palette[idx % len(palette)]
        pts = np.round(coords[:, ::-1]).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(result, [pts], False, color, thickness, lineType=cv2.LINE_AA)
        if label_paths:
            y, x = np.round(coords[0]).astype(int)
            cv2.putText(result, str(path.get("path_id", path.get("rejected_path_id", idx + 1))), (x + 3, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return result


def select_long_paths(paths, min_ratio=0.40):
    if not paths:
        return []
    best_length = max(float(path["path_length_px"]) for path in paths)
    threshold = float(best_length) * float(min_ratio)
    return [path for path in paths if float(path["path_length_px"]) >= threshold]


def coords_to_pixel_index_set(coords, shape, radius=2):
    h, w = int(shape[0]), int(shape[1])
    expanded = set()
    rounded = np.round(np.asarray(coords, dtype=float)).astype(int)
    for y, x in rounded:
        for dy in range(-int(radius), int(radius) + 1):
            for dx in range(-int(radius), int(radius) + 1):
                yy = int(y + dy)
                xx = int(x + dx)
                if yy < 0 or xx < 0 or yy >= h or xx >= w:
                    continue
                expanded.add(yy * w + xx)
    return expanded


def dedupe_candidates_by_overlap(paths, shape, overlap_threshold=0.72, pixel_radius=2):
    selected = []
    selected_sets = []
    rejected = []
    for path in paths:
        pixel_set = coords_to_pixel_index_set(path["coords"], shape, radius=pixel_radius)
        if not pixel_set:
            continue
        max_overlap = 0.0
        for chosen_set in selected_sets:
            overlap = float(len(pixel_set & chosen_set)) / max(1, min(len(pixel_set), len(chosen_set)))
            max_overlap = max(max_overlap, overlap)
        if max_overlap >= float(overlap_threshold):
            rejected_path = dict(path)
            rejected_path["overlap_ratio_max"] = float(max_overlap)
            rejected.append(rejected_path)
            continue
        kept = dict(path)
        kept["overlap_ratio_max"] = float(max_overlap)
        selected.append(kept)
        selected_sets.append(pixel_set)
    return selected, rejected


def build_text_panel(lines):
    fig, ax = plt.subplots(figsize=(7.2, 7.6), dpi=160)
    ax.set_facecolor("#0f172a")
    fig.patch.set_facecolor("#0f172a")
    ax.axis("off")
    y = 0.97
    for line in lines:
        ax.text(0.04, y, line, transform=ax.transAxes, fontsize=10.0, color="white", va="top", family="DejaVu Sans Mono")
        y -= 0.052
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8).copy()
    plt.close(fig)
    return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)


def main():
    total_t0 = perf_counter()
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir or (DEFAULT_OUTPUT_ROOT / f"intact_skeleton_probe_{timestamp}")
    ensure_dir(out_dir)

    image = read_gray(args.image)
    extractor = FeatureExtractor(magnification=args.magnification, speed_profile="accurate")
    probe_t0 = perf_counter()
    if args.mask_image is not None:
        mask_image = read_gray(args.mask_image)
        probe_data = build_probe_data_from_external_mask(
            extractor,
            image,
            mask_image,
            mask_threshold=args.mask_threshold,
            skeleton_heal_gap_px=args.bridge_gap_px,
            skeleton_heal_direction_cos_min=args.bridge_direction_cos_min,
        )
    else:
        probe_data = build_probe_data(
            extractor,
            image,
            skeleton_heal_gap_px=args.bridge_gap_px,
            skeleton_heal_direction_cos_min=args.bridge_direction_cos_min,
        )
    probe_stage_s = float(perf_counter() - probe_t0)
    probe_data["angle_hard_deg"] = float(args.angle_hard_deg)
    adjacency = probe_data["adjacency"]
    shape = probe_data["roi"].shape
    edge_seed_margin_px = args.edge_seed_margin_px
    if edge_seed_margin_px is None:
        edge_seed_margin_px = max(2.0 * extractor.expected_tube_px, 0.03 * min(shape[0], shape[1]))
    seed_candidate_margin_px = args.seed_candidate_margin_px
    if seed_candidate_margin_px is None:
        seed_candidate_margin_px = max(2.5 * edge_seed_margin_px, edge_seed_margin_px + 18.0)
    probe_data["target_hit_margin_px"] = float(edge_seed_margin_px)
    probe_data["cluster_exit_relax_deg"] = float(args.cluster_exit_relax_deg)
    probe_data["cluster_backtrack_hook_penalty_max"] = float(args.cluster_backtrack_hook_penalty_max)
    probe_data["cluster_backtrack_min_forward_ratio"] = float(args.cluster_backtrack_min_forward_ratio)
    probe_data["cluster_backtrack_min_straightness"] = float(args.cluster_backtrack_min_straightness)
    probe_data["cluster_backtrack_min_target_improve_px"] = float(args.cluster_backtrack_min_target_improve_px)
    probe_data["seed_candidate_margin_px"] = float(seed_candidate_margin_px)
    probe_data["gray_window_path_px"] = float(args.gray_window_path_px)
    probe_data["gray_window_candidate_px"] = float(args.gray_window_candidate_px)
    seed_t0 = perf_counter()
    seeds = collect_seeds(
        extractor,
        probe_data,
        adjacency,
        edge_seed_margin_px,
        seed_candidate_margin_px,
        args.top_seeds,
        args.seed_walk_hops,
    )
    seed_stage_s = float(perf_counter() - seed_t0)

    trace_t0 = perf_counter()
    raw_candidates = []
    for seed in seeds:
        raw_candidates.extend(trace_seed(extractor, probe_data, adjacency, seed, args))
    raw_candidates.sort(key=lambda item: (item["main_score"], item["span_px"], item["path_length_px"]), reverse=True)
    raw_candidates = raw_candidates[: max(1, int(args.top_candidates))]
    trace_stage_s = float(perf_counter() - trace_t0)
    dedup_t0 = perf_counter()
    candidates, dedup_rejected = dedupe_candidates_by_overlap(raw_candidates, shape, overlap_threshold=0.72, pixel_radius=2)
    dedup_stage_s = float(perf_counter() - dedup_t0)
    for idx, candidate in enumerate(candidates, start=1):
        candidate["path_id"] = idx
        candidate["smooth_profiles"] = build_path_smoothing_profiles(extractor, candidate.get("coords_raw", candidate["coords"]), probe_data["px_per_um"])
    for idx, candidate in enumerate(dedup_rejected, start=1):
        candidate["rejected_path_id"] = idx
    target_band_hits = [item for item in candidates if item.get("target_band_hit")]
    long_paths = select_long_paths(candidates, min_ratio=0.30)
    rejected_long_paths = select_long_paths(dedup_rejected, min_ratio=0.30)
    curvature_summary = summarize_image_curvature(candidates, profile_name="balanced")
    export_curvature_distribution_plot(out_dir / "curvature_distribution.png", curvature_summary)
    original_roi_canvas = cv2.cvtColor(probe_data["roi"], cv2.COLOR_GRAY2BGR)
    binary_mask_canvas = build_binary_mask_canvas(probe_data["mask"])
    selected_paths_canvas = draw_path_overlay(original_roi_canvas, long_paths, profile_name="balanced", thickness=2, label_paths=True)
    rejected_paths_canvas = draw_path_overlay(
        original_roi_canvas,
        rejected_long_paths,
        profile_name="balanced",
        palette=[(100, 70, 220)],
        thickness=1,
        label_paths=True,
    )

    diagonal = float(np.hypot(shape[0], shape[1]))
    lines = [
        f"file: {args.image.name}",
        f"mag: {args.magnification}",
        f"angle_soft/hard: {args.angle_limit_deg:.1f}/{args.angle_hard_deg:.1f}",
        f"cluster_exit_relax_deg: {args.cluster_exit_relax_deg:.1f}",
        f"cluster_backtrack_min_target_improve_px: {args.cluster_backtrack_min_target_improve_px:.1f}",
        f"beam_width: {args.beam_width}",
        f"edge_seed_margin_px: {edge_seed_margin_px:.1f}",
        f"seed_candidate_margin_px: {probe_data['seed_candidate_margin_px']:.1f}",
        f"target_hit_margin_px: {probe_data['target_hit_margin_px']:.1f}",
        f"skeleton_heal_gap_px: {args.bridge_gap_px:.1f}",
        f"skeleton_bridge_count: {probe_data['skeleton_bridge_count']}",
        f"bridge_pairs_considered: {probe_data['bridge_candidate_pairs_considered']}",
        f"segment_cache_count: {probe_data['segment_cache_count']}",
        f"max_junction_visits: {args.max_junction_visits}",
        f"gray_window_path_px: {args.gray_window_path_px:.1f}",
        f"gray_window_candidate_px: {args.gray_window_candidate_px:.1f}",
        "",
        f"skeleton_pixels: {len(adjacency)}",
        f"endpoints: {int(np.count_nonzero(probe_data['endpoint_mask']))}",
        f"junction_pixels: {int(np.count_nonzero(probe_data['junction_mask']))}",
        f"seeds: {len(seeds)}",
        f"raw_candidates: {len(raw_candidates)}",
        f"candidates: {len(candidates)}",
        f"dedup_rejected: {len(dedup_rejected)}",
        f"target_band_hits: {len(target_band_hits)}",
        f"long_paths_drawn: {len(long_paths)}",
        f"rejected_long_paths: {len(rejected_long_paths)}",
        f"curvature_profile: balanced",
        f"mean_curvature_nm: {curvature_summary['mean_curvature_nm']:.6f}",
        f"weighted_mean_curvature_nm: {curvature_summary['weighted_mean_curvature_nm']:.6f}",
        f"mean_tortuosity: {curvature_summary['mean_tortuosity']:.4f}",
        f"time_probe_s: {probe_stage_s:.2f}",
        f"time_seed_s: {seed_stage_s:.2f}",
        f"time_trace_s: {trace_stage_s:.2f}",
        f"time_dedup_s: {dedup_stage_s:.2f}",
        "",
    ]
    for seed in seeds[:6]:
        lines.append(
            f"{seed['seed_id']} {seed['border_name']} d={seed['border_distance_px']:.1f} "
            f"inward={seed['inward_score']:.3f} gray={seed.get('seed_gray_norm', 0.0):.3f}"
        )
    text_panel = build_text_panel(lines)

    output_panels = (
        ("original_roi.png", "Original ROI", original_roi_canvas),
        ("binary_mask.png", "Binary Mask", binary_mask_canvas),
        ("selected_paths.png", "Selected Paths on ROI", selected_paths_canvas),
        ("dedup_rejected.png", "Dedup Rejected on ROI", rejected_paths_canvas),
    )
    for filename, title, image_panel in output_panels:
        fig_panel, ax_panel = plt.subplots(figsize=(10, 12), dpi=160, constrained_layout=True)
        ax_panel.imshow(cv2.cvtColor(image_panel, cv2.COLOR_BGR2RGB))
        ax_panel.set_title(title, fontsize=12)
        ax_panel.axis("off")
        fig_panel.savefig(out_dir / filename, bbox_inches="tight")
        plt.close(fig_panel)

    fig2, ax2 = plt.subplots(figsize=(7.4, 9.2), dpi=160)
    ax2.imshow(cv2.cvtColor(text_panel, cv2.COLOR_BGR2RGB))
    ax2.axis("off")
    fig2.savefig(out_dir / "path_metrics_panel.png", bbox_inches="tight")
    plt.close(fig2)

    payload = {
        "image": str(args.image),
        "mask_image": None if args.mask_image is None else str(args.mask_image),
        "mask_threshold": int(args.mask_threshold),
        "mask_source": probe_data.get("mask_source", "internal_threshold"),
        "magnification": args.magnification,
        "angle_limit_deg": args.angle_limit_deg,
        "angle_hard_deg": args.angle_hard_deg,
        "cluster_exit_relax_deg": args.cluster_exit_relax_deg,
        "cluster_backtrack_hook_penalty_max": args.cluster_backtrack_hook_penalty_max,
        "cluster_backtrack_min_forward_ratio": args.cluster_backtrack_min_forward_ratio,
        "cluster_backtrack_min_straightness": args.cluster_backtrack_min_straightness,
        "cluster_backtrack_min_target_improve_px": args.cluster_backtrack_min_target_improve_px,
        "gray_window_path_px": args.gray_window_path_px,
        "gray_window_candidate_px": args.gray_window_candidate_px,
        "beam_width": args.beam_width,
        "edge_seed_margin_px": edge_seed_margin_px,
        "seed_candidate_margin_px": probe_data["seed_candidate_margin_px"],
        "target_hit_margin_px": probe_data["target_hit_margin_px"],
        "bridge_gap_px": args.bridge_gap_px,
        "bridge_direction_cos_min": args.bridge_direction_cos_min,
        "bridge_count": probe_data["skeleton_bridge_count"],
        "bridge_candidate_pairs_considered": probe_data["bridge_candidate_pairs_considered"],
        "bridge_candidate_endpoint_count": probe_data["bridge_candidate_endpoint_count"],
        "bridge_candidate_search_s": probe_data["bridge_candidate_search_s"],
        "skeleton_heal_s": probe_data["skeleton_heal_s"],
        "segment_cache_count": probe_data["segment_cache_count"],
        "build_probe_data_s": probe_data["build_probe_data_s"],
        "stage_probe_s": probe_stage_s,
        "stage_seed_s": seed_stage_s,
        "stage_trace_s": trace_stage_s,
        "stage_dedup_s": dedup_stage_s,
        "total_runtime_s": float(perf_counter() - total_t0),
        "skeleton_pixel_count": len(adjacency),
        "endpoint_count": int(np.count_nonzero(probe_data["endpoint_mask"])),
        "junction_pixel_count": int(np.count_nonzero(probe_data["junction_mask"])),
        "seed_count": len(seeds),
        "raw_candidate_count": len(raw_candidates),
        "candidate_count": len(candidates),
        "dedup_rejected_count": len(dedup_rejected),
        "target_band_hit_count": len(target_band_hits),
        "long_path_count": len(long_paths),
        "long_path_min_ratio": 0.30,
        "long_path_ids": [int(path["path_id"]) for path in long_paths],
        "rejected_long_path_count": len(rejected_long_paths),
        "rejected_long_path_ids": [int(path["rejected_path_id"]) for path in rejected_long_paths],
        "dedup_overlap_threshold": 0.72,
        "smooth_profiles": list(SMOOTH_PROFILE_NAMES),
        "curvature_summary": {key: value for key, value in curvature_summary.items() if key != "path_rows"},
        "seeds": [
            {
                "seed_id": s["seed_id"],
                "pixel": list(s["pixel"]),
                "border_name": s["border_name"],
                "border_distance_px": s["border_distance_px"],
                "inward_score": s["inward_score"],
                "edge_priority": s["edge_priority"],
                "seed_gray_mean": s.get("seed_gray_mean"),
                "seed_gray_norm": s.get("seed_gray_norm"),
                "seed_score": s["seed_score"],
            }
            for s in seeds
        ],
        "bridge_edges": probe_data["skeleton_bridge_edges"][: min(50, len(probe_data["skeleton_bridge_edges"]))],
        "curvature_distribution_plot": str(out_dir / "curvature_distribution.png"),
        "original_roi_plot": str(out_dir / "original_roi.png"),
        "binary_mask_plot": str(out_dir / "binary_mask.png"),
        "selected_paths_plot": str(out_dir / "selected_paths.png"),
        "dedup_rejected_plot": str(out_dir / "dedup_rejected.png"),
        "main_path": None if not candidates else path_summary_record(candidates[0]),
        "candidates": [path_summary_record(c) for c in candidates],
        "dedup_rejected_candidates": [path_summary_record(c) for c in dedup_rejected[: min(80, len(dedup_rejected))]],
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out_dir)


if __name__ == "__main__":
    main()
