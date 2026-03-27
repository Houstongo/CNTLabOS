"""Minimal offline test for curvature V2 relaxed on the 3 selected 100000x samples."""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
from skimage.measure import label

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.feature_extractor import FeatureExtractor  # noqa: E402


COMPARISON_CSV = PROJECT_ROOT / "reports" / "curvature_v2_comparison_20260325_061334" / "comparison.csv"
ITEMS_ROOT = PROJECT_ROOT / "reports" / "curvature_v2_comparison_20260325_061334" / "items"
OUTPUT_ROOT = PROJECT_ROOT / "reports"


def read_gray_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return image


def select_target_rows(magnification: int = 100000, limit: int = 3) -> List[Dict[str, Any]]:
    rows = list(csv.DictReader(COMPARISON_CSV.open("r", encoding="utf-8-sig", newline="")))
    selected: List[Dict[str, Any]] = []
    seen_groups = set()
    for row in rows:
        if int(row["magnification"]) != magnification:
            continue
        group = row["sample_id"].split("-")[0]
        if group in seen_groups:
            continue
        seen_groups.add(group)
        selected.append(row)
        if len(selected) >= limit:
            break
    return selected


def find_item_dir(image_id: int) -> Path:
    matches = sorted(ITEMS_ROOT.glob(f"{image_id}_*"))
    if not matches:
        raise FileNotFoundError(f"No item dir found for image_id={image_id}")
    return matches[0]


def collect_ordered_branches_v2_relaxed(extractor: FeatureExtractor, skel: np.ndarray) -> List[Dict[str, Any]]:
    skel_mask = (skel > 0).astype(np.uint8)
    if not np.any(skel_mask):
        return []

    neighbor_count = extractor._neighbor_count_map(skel_mask)
    junction_mask = (skel_mask > 0) & (neighbor_count >= 3)
    branch_mask = (skel_mask > 0) & np.logical_not(junction_mask)
    labeled = label(branch_mask, connectivity=2)
    if labeled.max() == 0:
        return []

    branches: List[Dict[str, Any]] = []
    for branch_id in range(1, int(labeled.max()) + 1):
        component_mask = labeled == branch_id
        ordered = extractor._trace_ordered_component_path(component_mask)
        if ordered.shape[0] < 3:
            continue
        ordered = extractor._smooth_path_coords(ordered)
        path_length_px = extractor._path_length(ordered)
        branches.append(
            {
                "coords": ordered,
                "n_points": int(ordered.shape[0]),
                "path_length_px": float(path_length_px),
            }
        )

    branches.sort(key=lambda item: (item["path_length_px"], item["n_points"]), reverse=True)
    return branches


def calculate_curvature_from_branches(
    extractor: FeatureExtractor,
    branches: List[Dict[str, Any]],
    sample_step: int = 1,
) -> Dict[str, Any]:
    px_per_nm = max(extractor.px_per_um / 1000.0, 1e-6)
    branch_curvatures = []
    weights = []
    contributing_points = 0
    contributing_branches = 0

    for branch in branches:
        sampled_coords = extractor._sample_ordered_coords(branch["coords"], sample_step=sample_step)
        if sampled_coords.shape[0] < 3:
            continue

        curvature_values_px = []
        for idx in range(1, sampled_coords.shape[0] - 1):
            p_prev = sampled_coords[idx - 1]
            p_curr = sampled_coords[idx]
            p_next = sampled_coords[idx + 1]

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
            if np.isfinite(curvature_px) and curvature_px > 0:
                curvature_values_px.append(curvature_px)

        if not curvature_values_px:
            continue

        contributing_points += len(curvature_values_px)
        contributing_branches += 1
        branch_curvatures.append(float(np.median(curvature_values_px)) * px_per_nm)
        weights.append(max(branch["path_length_px"], 1.0))

    if not branch_curvatures:
        return {
            "curvature_nm": 0.0,
            "contributing_branches": 0,
            "contributing_points": 0,
            "total_branches": len(branches),
            "total_path_points": int(sum(branch["n_points"] for branch in branches)),
        }

    curvature_nm = float(np.average(branch_curvatures, weights=np.asarray(weights, dtype=float)))
    return {
        "curvature_nm": curvature_nm,
        "contributing_branches": contributing_branches,
        "contributing_points": contributing_points,
        "total_branches": len(branches),
        "total_path_points": int(sum(branch["n_points"] for branch in branches)),
    }


def main() -> None:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_ROOT / f"curvature_v2_relaxed_test_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for row in select_target_rows():
        image_id = int(row["image_id"])
        sample_id = row["sample_id"]
        magnification = int(row["magnification"])
        item_dir = find_item_dir(image_id)
        mask = read_gray_image(item_dir / "paper_repro_mask.png")

        extractor = FeatureExtractor(magnification=magnification)
        extractor._calibrate(mask.shape[1])
        _, skel = extractor.calculate_diameter(mask)

        _, current_v2_nm = extractor.calculate_curvature_v2(skel)
        standard_branches = extractor._collect_ordered_branches_v2(skel, min_points=15)
        relaxed_branches = collect_ordered_branches_v2_relaxed(extractor, skel)
        relaxed_stats = calculate_curvature_from_branches(extractor, relaxed_branches, sample_step=1)

        results.append(
            {
                "image_id": image_id,
                "sample_id": sample_id,
                "magnification": magnification,
                "current_v2_curvature_nm": round(float(current_v2_nm), 6),
                "v2_relaxed_curvature_nm": round(float(relaxed_stats["curvature_nm"]), 6),
                "delta_relaxed_minus_v2": round(float(relaxed_stats["curvature_nm"] - current_v2_nm), 6),
                "standard_branch_count": len(standard_branches),
                "relaxed_branch_count": int(relaxed_stats["total_branches"]),
                "relaxed_contributing_branches": int(relaxed_stats["contributing_branches"]),
                "relaxed_contributing_points": int(relaxed_stats["contributing_points"]),
                "relaxed_total_path_points": int(relaxed_stats["total_path_points"]),
                "source_item_dir": str(item_dir),
            }
        )

    csv_path = output_dir / "results.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "comparison_csv": str(COMPARISON_CSV),
        "items_root": str(ITEMS_ROOT),
        "count": len(results),
        "results": results,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"OUTPUT_DIR={output_dir}")
    print(f"CSV_PATH={csv_path}")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
