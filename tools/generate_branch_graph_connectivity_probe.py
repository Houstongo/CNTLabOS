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
    parser = argparse.ArgumentParser(description="Generate branch-graph connectivity probe demo.")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--magnification", type=int, default=100000)
    parser.add_argument("--min-length-factor", type=float, default=2.0)
    parser.add_argument("--angle-limit-deg", type=float, default=45.0)
    parser.add_argument("--angle-hard-deg", type=float, default=70.0)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--top-seeds", type=int, default=8)
    parser.add_argument("--top-candidates", type=int, default=12)
    parser.add_argument("--edge-seed-margin-px", type=float, default=None)
    parser.add_argument("--max-branch-steps", type=int, default=8)
    parser.add_argument("--max-cumulative-turn-deg", type=float, default=220.0)
    parser.add_argument("--min-span-gain-px", type=float, default=10.0)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_gray_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return image


def normalize(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=float)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-8:
        return np.zeros((2,), dtype=float)
    return arr / norm


def build_mask_base(mask: np.ndarray, fill_color=(42, 42, 42), contour_color=(220, 220, 220)) -> np.ndarray:
    canvas = np.zeros((*mask.shape, 3), dtype=np.uint8)
    canvas[mask > 0] = fill_color
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, contours, -1, contour_color, 1)
    return canvas


def border_info(coord: np.ndarray, image_shape) -> dict:
    y, x = [float(v) for v in coord]
    height, width = int(image_shape[0]), int(image_shape[1])
    distances = {
        "top": y,
        "bottom": (height - 1) - y,
        "left": x,
        "right": (width - 1) - x,
    }
    min_distance = float(min(distances.values()))
    near_borders = [name for name, distance in distances.items() if distance <= min_distance + 1.0]
    normals = {
        "top": np.array([1.0, 0.0], dtype=float),
        "bottom": np.array([-1.0, 0.0], dtype=float),
        "left": np.array([0.0, 1.0], dtype=float),
        "right": np.array([0.0, -1.0], dtype=float),
    }
    inward_normal = normalize(np.sum([normals[name] for name in near_borders], axis=0))
    return {
        "border_distance_px": min_distance,
        "border_name": "+".join(near_borders),
        "inward_normal": inward_normal,
        "distances": distances,
    }


def opposite_borders(border_names: List[str]) -> List[str]:
    mapping = {
        "top": "bottom",
        "bottom": "top",
        "left": "right",
        "right": "left",
    }
    targets = []
    for name in border_names:
        target = mapping.get(name)
        if target and target not in targets:
            targets.append(target)
    return targets


def normalized_progress_away_from_borders(coord: np.ndarray, image_shape, border_names: List[str]) -> float:
    info = border_info(coord, image_shape)
    height, width = int(image_shape[0]), int(image_shape[1])
    scales = {
        "top": max(height - 1, 1),
        "bottom": max(height - 1, 1),
        "left": max(width - 1, 1),
        "right": max(width - 1, 1),
    }
    values = [
        float(info["distances"][border_name]) / max(float(scales[border_name]), 1.0)
        for border_name in border_names
    ]
    if not values:
        return 0.0
    return float(np.mean(values))


def normalized_progress_toward_borders(coord: np.ndarray, image_shape, border_names: List[str]) -> float:
    info = border_info(coord, image_shape)
    height, width = int(image_shape[0]), int(image_shape[1])
    scales = {
        "top": max(height - 1, 1),
        "bottom": max(height - 1, 1),
        "left": max(width - 1, 1),
        "right": max(width - 1, 1),
    }
    values = [
        1.0 - float(info["distances"][border_name]) / max(float(scales[border_name]), 1.0)
        for border_name in border_names
    ]
    if not values:
        return 0.0
    return float(np.mean(values))


def branch_graph_from_image(extractor: FeatureExtractor, image: np.ndarray, min_length_factor: float) -> dict:
    roi = extractor.extract_roi(image)
    extractor._calibrate(roi.shape[1])
    processed = extractor.preprocess(roi)
    _, thresh = extractor.calculate_density(processed)
    _, skeleton = extractor.calculate_diameter(thresh)
    width_distance_map = cv2.distanceTransform((thresh > 0).astype(np.uint8), cv2.DIST_L2, 5)
    graph = extractor._build_branch_graph_v1(
        skeleton=skeleton,
        processed=processed,
        width_distance_map=width_distance_map,
        min_length_factor=min_length_factor,
        junction_link_radius=4,
    )
    graph.update(
        {
            "roi": roi,
            "processed": processed,
            "mask": thresh,
            "skeleton": skeleton,
            "px_per_um": float(extractor.px_per_um),
            "magnification": int(extractor.mag) if extractor.mag else None,
        }
    )
    return graph


def collect_boundary_seeds(
    extractor: FeatureExtractor,
    graph: dict,
    edge_seed_margin_px: float,
    top_seeds: int,
) -> List[dict]:
    nodes = graph["nodes"]
    branches = graph["branches"]
    image_shape = graph["roi"].shape
    image_diagonal_px = float(np.hypot(image_shape[0], image_shape[1]))
    seeds: List[dict] = []

    for node in nodes.values():
        if node["kind"] != "endpoint" or node.get("degree", 0) != 1:
            continue
        branch_id = node["branch_ids"][0]
        branch = branches[branch_id]
        oriented = extractor._branch_orientation_from_node(branch, node["node_id"])
        seed_direction = normalize(oriented["node_to_interior"])
        if float(np.linalg.norm(seed_direction)) <= 1e-8:
            continue

        edge = border_info(node["coord"], image_shape)
        if edge["border_distance_px"] > edge_seed_margin_px:
            continue
        if "+" in edge["border_name"]:
            continue

        inward_score = float(np.dot(seed_direction, edge["inward_normal"]))
        if inward_score <= 0.05:
            continue

        if branch["length_px"] < max(18.0, 4.0 * extractor.expected_tube_px):
            continue

        normalized_edge = max(0.0, 1.0 - edge["border_distance_px"] / max(edge_seed_margin_px, 1e-6))
        normalized_length = min(float(branch["length_px"]) / max(0.25 * image_diagonal_px, 1e-6), 1.0)
        width_stability = 1.0 / (1.0 + max(float(branch.get("width_cv", 0.0)), 0.0))
        seed_score = 0.45 * normalized_edge + 0.35 * inward_score + 0.15 * normalized_length + 0.05 * width_stability
        seeds.append(
            {
                "seed_id": f"seed_{len(seeds) + 1}",
                "node_id": node["node_id"],
                "branch_id": branch_id,
                "coord": np.asarray(node["coord"], dtype=float),
                "seed_direction": seed_direction,
                "border_name": edge["border_name"],
                "target_borders": opposite_borders([edge["border_name"]]),
                "border_distance_px": float(edge["border_distance_px"]),
                "inward_score": inward_score,
                "branch_length_px": float(branch["length_px"]),
                "seed_score": float(seed_score),
                "other_node_id": oriented["other_node"],
                "coords_forward": np.asarray(oriented["coords_forward"], dtype=float),
            }
        )

    seeds.sort(
        key=lambda item: (
            item["seed_score"],
            item["inward_score"],
            item["branch_length_px"],
        ),
        reverse=True,
    )

    deduped: List[dict] = []
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


def state_geometry(state: dict) -> dict:
    coords = np.asarray(state["coords"], dtype=float)
    if coords.shape[0] < 2:
        return {
            "span_px": 0.0,
            "path_length_px": 0.0,
            "projected_progress_px": 0.0,
            "lateral_drift_px": 0.0,
        }

    start = coords[0]
    end = coords[-1]
    displacement = end - start
    seed_direction = np.asarray(state["seed_direction"], dtype=float)
    perpendicular = np.array([-seed_direction[1], seed_direction[0]], dtype=float)
    return {
        "span_px": float(np.linalg.norm(displacement)),
        "path_length_px": float(state["path_length_px"]),
        "projected_progress_px": float(np.dot(displacement, seed_direction)),
        "lateral_drift_px": float(abs(np.dot(displacement, perpendicular))),
    }


def current_direction(extractor: FeatureExtractor, coords: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    tangent = extractor._estimate_endpoint_tangent(np.asarray(coords, dtype=float), at_start=False)
    tangent = normalize(tangent)
    if float(np.linalg.norm(tangent)) <= 1e-8:
        return normalize(fallback)
    return tangent


def beam_rank(state: dict, image_shape) -> float:
    diagonal_px = float(np.hypot(image_shape[0], image_shape[1]))
    geometry = state_geometry(state)
    progress_norm = max(0.0, geometry["projected_progress_px"]) / max(diagonal_px, 1e-6)
    span_norm = geometry["span_px"] / max(diagonal_px, 1e-6)
    length_norm = geometry["path_length_px"] / max(diagonal_px, 1e-6)
    straightness = geometry["span_px"] / max(geometry["path_length_px"], 1e-6)
    stability = 1.0 / (1.0 + state["cumulative_turn_deg"] / 120.0 + 0.6 * state["soft_turn_count"])
    return float(
        0.32 * progress_norm
        + 0.26 * span_norm
        + 0.16 * length_norm
        + 0.16 * straightness
        + 0.10 * stability
        + 0.08 * state["probe_score"]
    )


def final_candidate_score(state: dict, image_shape, edge_seed_margin_px: float) -> dict:
    diagonal_px = float(np.hypot(image_shape[0], image_shape[1]))
    coords = np.asarray(state["coords"], dtype=float)
    geometry = state_geometry(state)
    end_info = border_info(coords[-1], image_shape)
    start_distance = float(state["start_border_distance_px"])
    end_distance = float(end_info["border_distance_px"])
    start_border_name = str(state["start_border_name"])
    end_border_name = str(end_info["border_name"])
    start_border_names = [name for name in start_border_name.split("+") if name]
    target_border_names = list(state.get("target_borders", opposite_borders(start_border_names)))
    edge_reward = max(0.0, 1.0 - start_distance / max(edge_seed_margin_px, 1e-6)) + max(
        0.0, 1.0 - end_distance / max(edge_seed_margin_px, 1e-6)
    )
    span_norm = geometry["span_px"] / max(diagonal_px, 1e-6)
    progress_norm = max(0.0, geometry["projected_progress_px"]) / max(diagonal_px, 1e-6)
    length_norm = geometry["path_length_px"] / max(diagonal_px, 1e-6)
    escape_progress_norm = normalized_progress_away_from_borders(coords[-1], image_shape, start_border_names)
    target_progress_norm = normalized_progress_toward_borders(coords[-1], image_shape, target_border_names)
    stability = 1.0 / (1.0 + state["cumulative_turn_deg"] / 120.0 + 0.6 * state["soft_turn_count"])
    zigzag_penalty = max(0.0, geometry["path_length_px"] - geometry["span_px"]) / max(diagonal_px, 1e-6)
    target_hit_bonus = 0.18 if end_border_name in target_border_names else 0.0
    same_border_penalty = 0.16 if end_border_name in start_border_names else 0.0
    residual_start_border_penalty = 0.22 * max(0.0, 1.0 - escape_progress_norm)
    weak_target_progress_penalty = 0.20 * max(0.0, 0.35 - target_progress_norm)
    score = float(
        0.04 * edge_reward
        + 0.20 * span_norm
        + 0.12 * progress_norm
        + 0.18 * escape_progress_norm
        + 0.28 * target_progress_norm
        + 0.10 * stability
        + 0.08 * length_norm
        + target_hit_bonus
        - same_border_penalty
        - residual_start_border_penalty
        - weak_target_progress_penalty
        - 0.18 * zigzag_penalty
    )
    return {
        "main_score": score,
        "edge_reward": float(edge_reward),
        "span_norm": float(span_norm),
        "progress_norm": float(progress_norm),
        "length_norm": float(length_norm),
        "escape_progress_norm": float(escape_progress_norm),
        "target_progress_norm": float(target_progress_norm),
        "direction_stability": float(stability),
        "zigzag_penalty": float(zigzag_penalty),
        "target_hit_bonus": float(target_hit_bonus),
        "same_border_penalty": float(same_border_penalty),
        "residual_start_border_penalty": float(residual_start_border_penalty),
        "weak_target_progress_penalty": float(weak_target_progress_penalty),
        "end_border_distance_px": float(end_distance),
    }


def probe_seed_paths(
    extractor: FeatureExtractor,
    graph: dict,
    seed: dict,
    angle_limit_deg: float,
    angle_hard_deg: float,
    beam_width: int,
    max_branch_steps: int,
    max_cumulative_turn_deg: float,
    min_span_gain_px: float,
    edge_seed_margin_px: float,
) -> List[dict]:
    branches = graph["branches"]
    nodes = graph["nodes"]
    image_shape = graph["roi"].shape
    diagonal_px = float(np.hypot(image_shape[0], image_shape[1]))

    initial_state = {
        "seed_id": seed["seed_id"],
        "start_node_id": seed["node_id"],
        "start_border_distance_px": float(seed["border_distance_px"]),
        "start_border_name": seed["border_name"],
        "target_borders": list(seed.get("target_borders", [])),
        "branch_ids": [seed["branch_id"]],
        "used_branches": {seed["branch_id"]},
        "coords": np.asarray(seed["coords_forward"], dtype=float),
        "transition_costs": [],
        "current_branch_id": seed["branch_id"],
        "current_node_id": seed["other_node_id"],
        "seed_direction": np.asarray(seed["seed_direction"], dtype=float),
        "current_direction": np.asarray(seed["seed_direction"], dtype=float),
        "path_length_px": float(branches[seed["branch_id"]]["length_px"]),
        "cumulative_turn_deg": 0.0,
        "soft_turn_count": 0,
        "probe_score": 0.0,
    }
    beams = [initial_state]

    for _ in range(max(0, int(max_branch_steps) - 1)):
        expanded: List[dict] = []
        any_expansion = False
        for state in beams:
            current_node_id = state["current_node_id"]
            current_branch = branches[state["current_branch_id"]]
            if current_node_id not in nodes or nodes[current_node_id]["kind"] != "junction":
                expanded.append(state)
                continue

            candidate_ids = [
                branch_id
                for branch_id in nodes[current_node_id]["branch_ids"]
                if branch_id != current_branch["branch_id"] and branch_id not in state["used_branches"]
            ]
            if not candidate_ids:
                expanded.append(state)
                continue

            geometry_before = state_geometry(state)
            scored_extensions = []
            for candidate_id in candidate_ids:
                scored = extractor._score_branch_graph_transition_v1(
                    current_branch=current_branch,
                    current_node_id=current_node_id,
                    candidate_branch=branches[candidate_id],
                    angle_soft_deg=angle_limit_deg,
                    angle_hard_deg=angle_hard_deg,
                )
                if scored is None:
                    continue

                next_coords = np.asarray(scored["candidate_coords_forward"], dtype=float)
                if next_coords.shape[0] <= 1:
                    continue

                candidate_branch = branches[candidate_id]
                new_coords = np.vstack([state["coords"], next_coords[1:]])
                new_state = dict(state)
                new_state["branch_ids"] = state["branch_ids"] + [candidate_id]
                new_state["used_branches"] = set(state["used_branches"]) | {candidate_id}
                new_state["coords"] = new_coords
                new_state["transition_costs"] = state["transition_costs"] + [float(scored["total_cost"])]
                new_state["current_branch_id"] = candidate_id
                new_state["current_node_id"] = scored["candidate_other_node"]
                new_state["path_length_px"] = float(state["path_length_px"] + candidate_branch["length_px"])
                new_state["cumulative_turn_deg"] = float(state["cumulative_turn_deg"] + scored["turning_angle_deg"])
                new_state["soft_turn_count"] = int(state["soft_turn_count"] + int(scored["turning_angle_deg"] > angle_limit_deg))
                new_state["current_direction"] = current_direction(extractor, new_coords, state["current_direction"])

                geometry_after = state_geometry(new_state)
                span_gain_px = float(geometry_after["span_px"] - geometry_before["span_px"])
                forward_gain_px = float(geometry_after["projected_progress_px"] - geometry_before["projected_progress_px"])
                if candidate_branch["length_px"] < max(12.0, 3.0 * extractor.expected_tube_px) and span_gain_px < min_span_gain_px:
                    continue
                if new_state["cumulative_turn_deg"] > max_cumulative_turn_deg:
                    continue

                lateral_norm = geometry_after["lateral_drift_px"] / max(diagonal_px, 1e-6)
                local_probe_gain = float(
                    1.30 * max(0.0, forward_gain_px) / max(diagonal_px, 1e-6)
                    + 1.10 * max(0.0, span_gain_px) / max(diagonal_px, 1e-6)
                    + 0.18 * min(candidate_branch["length_px"] / max(diagonal_px, 1e-6), 1.0)
                    - 0.95 * float(scored["angle_cost"])
                    - 0.22 * lateral_norm
                    - 0.08 * float(scored["width_cost"])
                    - 0.05 * float(scored["intensity_cost"])
                )
                new_state["probe_score"] = float(state["probe_score"] + local_probe_gain)
                scored_extensions.append((new_state, beam_rank(new_state, image_shape)))

            if not scored_extensions:
                expanded.append(state)
                continue

            any_expansion = True
            scored_extensions.sort(key=lambda item: item[1], reverse=True)
            expanded.extend(state_item for state_item, _ in scored_extensions[: max(1, int(beam_width))])

        expanded.sort(key=lambda item: beam_rank(item, image_shape), reverse=True)
        beams = expanded[: max(1, int(beam_width))]
        if not any_expansion:
            break

    candidates: List[dict] = []
    for state in beams:
        geometry = state_geometry(state)
        metrics = extractor._compute_reconstructed_path_metrics_v1(
            path_id=0,
            path_coords=np.asarray(state["coords"], dtype=float),
            branch_ids=list(state["branch_ids"]),
            transition_costs=list(state["transition_costs"]),
            distance_map=graph["distance_map"],
        )
        final_score = final_candidate_score(state, image_shape, edge_seed_margin_px)
        end_border = border_info(state["coords"][-1], image_shape)
        metrics.update(
            {
                "seed_id": seed["seed_id"],
                "seed_border_name": seed["border_name"],
                "seed_border_distance_px": float(seed["border_distance_px"]),
                "end_border_name": end_border["border_name"],
                "end_border_distance_px": float(end_border["border_distance_px"]),
                "projected_progress_px": float(geometry["projected_progress_px"]),
                "lateral_drift_px": float(geometry["lateral_drift_px"]),
                "soft_turn_count": int(state["soft_turn_count"]),
                "cumulative_turn_deg": float(state["cumulative_turn_deg"]),
                "probe_score": float(state["probe_score"]),
            }
        )
        metrics.update(final_score)
        candidates.append(metrics)

    candidates.sort(
        key=lambda item: (item["main_score"], item["span_px"], item["path_length_px"]),
        reverse=True,
    )
    return candidates


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


def draw_seed_overlay(canvas: np.ndarray, seeds: List[dict]) -> np.ndarray:
    result = canvas.copy()
    for idx, seed in enumerate(seeds, start=1):
        y, x = np.round(seed["coord"]).astype(int)
        direction = normalize(seed["seed_direction"])
        endpoint = np.round(seed["coord"] + 18.0 * direction).astype(int)
        cv2.circle(result, (x, y), 5, (0, 220, 255), -1)
        cv2.arrowedLine(result, (x, y), (int(endpoint[1]), int(endpoint[0])), (0, 220, 255), 2, tipLength=0.25)
        cv2.putText(result, str(idx), (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return result


def draw_paths(canvas: np.ndarray, paths: List[Dict[str, Any]], highlight_first: bool = False) -> np.ndarray:
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
        coords = np.asarray(path["coords"], dtype=float)
        if coords.shape[0] < 2:
            continue
        color = (60, 80, 255) if highlight_first and idx == 0 else palette[idx % len(palette)]
        thickness = 4 if highlight_first and idx == 0 else 2
        pts = np.round(coords[:, ::-1]).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(result, [pts], False, color, thickness, lineType=cv2.LINE_AA)
        y, x = np.round(coords[0]).astype(int)
        cv2.putText(result, str(path["path_id"]), (x + 3, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return result


def build_text_panel(lines: List[str]) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(7.2, 7.4), dpi=160)
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


def serialize_seed(seed: dict) -> dict:
    return {
        "seed_id": seed["seed_id"],
        "node_id": seed["node_id"],
        "branch_id": seed["branch_id"],
        "border_name": seed["border_name"],
        "border_distance_px": round(float(seed["border_distance_px"]), 3),
        "inward_score": round(float(seed["inward_score"]), 6),
        "branch_length_px": round(float(seed["branch_length_px"]), 3),
        "seed_score": round(float(seed["seed_score"]), 6),
    }


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir or (DEFAULT_OUTPUT_ROOT / f"branch_graph_connectivity_probe_{timestamp}")
    ensure_dir(out_dir)

    image = read_gray_image(args.image)
    extractor = FeatureExtractor(magnification=args.magnification, speed_profile="accurate")
    graph = branch_graph_from_image(extractor, image, min_length_factor=args.min_length_factor)
    image_shape = graph["roi"].shape
    edge_seed_margin_px = args.edge_seed_margin_px
    if edge_seed_margin_px is None:
        edge_seed_margin_px = max(2.0 * extractor.expected_tube_px, 0.03 * min(image_shape[0], image_shape[1]))

    seeds = collect_boundary_seeds(
        extractor=extractor,
        graph=graph,
        edge_seed_margin_px=edge_seed_margin_px,
        top_seeds=args.top_seeds,
    )

    candidates: List[dict] = []
    for seed in seeds:
        candidates.extend(
            probe_seed_paths(
                extractor=extractor,
                graph=graph,
                seed=seed,
                angle_limit_deg=args.angle_limit_deg,
                angle_hard_deg=args.angle_hard_deg,
                beam_width=args.beam_width,
                max_branch_steps=args.max_branch_steps,
                max_cumulative_turn_deg=args.max_cumulative_turn_deg,
                min_span_gain_px=args.min_span_gain_px,
                edge_seed_margin_px=edge_seed_margin_px,
            )
        )
    candidates.sort(key=lambda item: (item["main_score"], item["span_px"], item["path_length_px"]), reverse=True)
    candidates = candidates[: max(1, int(args.top_candidates))]
    for idx, candidate in enumerate(candidates, start=1):
        candidate["path_id"] = idx

    roi = graph["roi"]
    mask = graph["mask"]
    skeleton = (graph["skeleton"] > 0).astype(np.uint8)
    nodes = graph["nodes"]
    branches = graph["branches"]
    skeleton_canvas = build_mask_base(mask)
    skeleton_canvas[skeleton > 0] = (255, 230, 90)
    graph_canvas = draw_seed_overlay(draw_nodes(draw_branches(build_mask_base(mask), branches), nodes), seeds)
    candidates_canvas = draw_paths(draw_seed_overlay(draw_branches(build_mask_base(mask), branches), seeds), candidates[: min(8, len(candidates))])
    main_canvas = draw_paths(draw_seed_overlay(draw_branches(build_mask_base(mask), branches), seeds), candidates[:1], highlight_first=True)

    diagonal_px = float(np.hypot(image_shape[0], image_shape[1]))
    text_lines = [
        f"file: {args.image.name}",
        f"mag: {args.magnification}",
        f"angle_soft/hard: {args.angle_limit_deg:.1f}/{args.angle_hard_deg:.1f}",
        f"beam_width: {args.beam_width}",
        f"edge_seed_margin_px: {edge_seed_margin_px:.1f}",
        f"max_branch_steps: {args.max_branch_steps}",
        "",
        f"nodes: {len(nodes)}",
        f"branches: {len(branches)}",
        f"seeds: {len(seeds)}",
        f"probe_candidates: {len(candidates)}",
        "",
    ]
    for seed in seeds[:6]:
        text_lines.append(
            f"{seed['seed_id']} {seed['border_name']} d={seed['border_distance_px']:.1f} inward={seed['inward_score']:.3f}"
        )
    text_lines.append("")
    for path in candidates[:6]:
        span_ratio = path["span_px"] / max(diagonal_px, 1e-6)
        text_lines.append(
            f"P{path['path_id']:02d} score={path['main_score']:.3f} spanR={span_ratio:.3f} nB={len(path['branch_ids'])}"
        )
        text_lines.append(
            f"   {path['seed_border_name']}->{path['end_border_name']} prog={path['progress_norm']:.3f} edge={path['edge_reward']:.3f}"
        )
    text_panel = build_text_panel(text_lines)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12), dpi=160, constrained_layout=True)
    panels = [
        ("Original ROI", cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)),
        ("Branch Graph + Edge Seeds", graph_canvas),
        ("Top Probe Candidates", candidates_canvas),
        ("Winning Connectivity Path", main_canvas),
    ]
    for ax, (title, image_panel) in zip(axes.flat[:4], panels):
        ax.imshow(cv2.cvtColor(image_panel, cv2.COLOR_BGR2RGB) if image_panel.ndim == 3 else image_panel)
        ax.set_title(title, fontsize=12)
        ax.axis("off")

    fig.savefig(out_dir / "branch_graph_probe.png", bbox_inches="tight")
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(7.4, 9.2), dpi=160)
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
        "edge_seed_margin_px": edge_seed_margin_px,
        "node_count": len(nodes),
        "branch_count": len(branches),
        "seed_count": len(seeds),
        "candidate_count": len(candidates),
        "seeds": [serialize_seed(seed) for seed in seeds],
        "main_path": None if not candidates else {key: value for key, value in candidates[0].items() if key != "coords"},
        "candidates": [{key: value for key, value in candidate.items() if key != "coords"} for candidate in candidates],
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out_dir)


if __name__ == "__main__":
    main()
