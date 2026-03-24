"""Build triplet comparison panels for baseline, Exp-C, and Exp-D paper-repro runs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, List

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build side-by-side comparison panels for three paper-repro runs.")
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--expc-dir", type=Path, required=True)
    parser.add_argument("--expd-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    ensure_dir(path.parent)
    encoded = cv2.imencode(path.suffix, image)[1]
    path.write_bytes(encoded.tobytes())


def split_visual_panel(panel: np.ndarray) -> List[np.ndarray]:
    height, width = panel.shape[:2]
    tile_width = width // 4
    return [panel[:, i * tile_width:(i + 1) * tile_width] for i in range(4)]


def label_tile(tile: np.ndarray, label: str, model_name: str | None = None) -> np.ndarray:
    banner_h = 34
    canvas = np.zeros((tile.shape[0] + banner_h, tile.shape[1], 3), dtype=np.uint8)
    canvas[:banner_h, :] = 245
    canvas[banner_h:, :] = tile
    text = label if not model_name else f"{model_name} | {label}"
    cv2.putText(canvas, text, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 1, cv2.LINE_AA)
    return canvas


def build_row(original: np.ndarray, weak: np.ndarray, prob: np.ndarray, pred: np.ndarray, model_name: str) -> np.ndarray:
    tiles = [
        label_tile(original, "Original", model_name),
        label_tile(weak, "Weak", model_name),
        label_tile(prob, "Probability", model_name),
        label_tile(pred, "0.7 Mask", model_name),
    ]
    return np.concatenate(tiles, axis=1)


def intersect_names(paths: Iterable[Path]) -> List[str]:
    name_sets = []
    for path in paths:
        name_sets.append({p.name for p in path.glob("*.png")})
    names = sorted(set.intersection(*name_sets))
    return names


def write_csv(path: Path, rows: List[dict]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["filename"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    baseline_dir = args.baseline_dir
    expc_dir = args.expc_dir
    expd_dir = args.expd_dir
    output_root = args.output_root
    panel_dir = output_root / "comparison_panels"
    ensure_dir(panel_dir)

    names = intersect_names([baseline_dir, expc_dir, expd_dir])
    manifest_rows = []

    for idx, name in enumerate(names, 1):
        baseline_tiles = split_visual_panel(read_image(baseline_dir / name))
        expc_tiles = split_visual_panel(read_image(expc_dir / name))
        expd_tiles = split_visual_panel(read_image(expd_dir / name))

        row_baseline = build_row(baseline_tiles[0], baseline_tiles[1], baseline_tiles[2], baseline_tiles[3], "Baseline")
        row_expc = build_row(expc_tiles[0], expc_tiles[1], expc_tiles[2], expc_tiles[3], "Exp-C")
        row_expd = build_row(expd_tiles[0], expd_tiles[1], expd_tiles[2], expd_tiles[3], "Exp-D")

        panel = np.concatenate([row_baseline, row_expc, row_expd], axis=0)
        out_path = panel_dir / name.replace("_panel.png", "_triplet_compare.png")
        write_image(out_path, panel)
        manifest_rows.append(
            {
                "filename": out_path.name,
                "baseline_panel": str(baseline_dir / name),
                "expc_panel": str(expc_dir / name),
                "expd_panel": str(expd_dir / name),
                "output_panel": str(out_path),
            }
        )
        print(f"[{idx}/{len(names)}] {out_path.name}")

    write_csv(output_root / "triplet_compare_manifest.csv", manifest_rows)


if __name__ == "__main__":
    main()
