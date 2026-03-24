"""Build concise comparison panels for baseline, Exp-C, and Exp-D masks."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, List

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build concise comparison panels for three paper-repro runs.")
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
    _, width = panel.shape[:2]
    tile_width = width // 4
    return [panel[:, i * tile_width:(i + 1) * tile_width] for i in range(4)]


def label_tile(tile: np.ndarray, label: str) -> np.ndarray:
    banner_h = 34
    canvas = np.zeros((tile.shape[0] + banner_h, tile.shape[1], 3), dtype=np.uint8)
    canvas[:banner_h, :] = 245
    canvas[banner_h:, :] = tile
    cv2.putText(canvas, label, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 1, cv2.LINE_AA)
    return canvas


def info_tile_like(reference_tile: np.ndarray, text: str) -> np.ndarray:
    tile = np.full_like(reference_tile, 250)
    h, w = tile.shape[:2]
    cv2.rectangle(tile, (0, 0), (w - 1, h - 1), (220, 220, 220), 1)
    words = text.replace("_panel", "").split("_")
    lines = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if len(candidate) > 28:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    y = 60
    for line in lines[:8]:
        cv2.putText(tile, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (40, 40, 40), 1, cv2.LINE_AA)
        y += 34
    return tile


def intersect_names(paths: Iterable[Path]) -> List[str]:
    name_sets = [{p.name for p in path.glob("*.png")} for path in paths]
    return sorted(set.intersection(*name_sets))


def write_csv(path: Path, rows: List[dict]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["filename"])
        writer.writeheader()
        writer.writerows(rows)


def compose_grid(tiles: List[np.ndarray], rows: int = 2, cols: int = 3, gap: int = 18, outer: int = 18) -> np.ndarray:
    tile_h, tile_w = tiles[0].shape[:2]
    canvas_h = outer * 2 + rows * tile_h + (rows - 1) * gap
    canvas_w = outer * 2 + cols * tile_w + (cols - 1) * gap
    canvas = np.full((canvas_h, canvas_w, 3), 235, dtype=np.uint8)
    for idx, tile in enumerate(tiles):
        r = idx // cols
        c = idx % cols
        y = outer + r * (tile_h + gap)
        x = outer + c * (tile_w + gap)
        canvas[y:y + tile_h, x:x + tile_w] = tile
    return canvas


def main() -> None:
    args = parse_args()
    panel_dir = args.output_root / "comparison_panels"
    ensure_dir(panel_dir)

    names = intersect_names([args.baseline_dir, args.expc_dir, args.expd_dir])
    manifest_rows = []

    for idx, name in enumerate(names, 1):
        baseline_tiles = split_visual_panel(read_image(args.baseline_dir / name))
        expc_tiles = split_visual_panel(read_image(args.expc_dir / name))
        expd_tiles = split_visual_panel(read_image(args.expd_dir / name))

        tiles = [
            label_tile(baseline_tiles[0], "Original"),
            label_tile(baseline_tiles[1], "Weak"),
            label_tile(baseline_tiles[3], "Baseline 0.7"),
            label_tile(expc_tiles[3], "Exp-C 0.7"),
            label_tile(expd_tiles[3], "Exp-D 0.7"),
            info_tile_like(label_tile(baseline_tiles[0], "Original"), Path(name).stem),
        ]
        panel = compose_grid(tiles, rows=2, cols=3, gap=18, outer=18)
        out_path = panel_dir / name.replace("_panel.png", "_mask_compare.png")
        write_image(out_path, panel)

        manifest_rows.append(
            {
                "filename": out_path.name,
                "baseline_panel": str(args.baseline_dir / name),
                "expc_panel": str(args.expc_dir / name),
                "expd_panel": str(args.expd_dir / name),
                "output_panel": str(out_path),
            }
        )
        print(f"[{idx}/{len(names)}] {out_path.name}")

    write_csv(args.output_root / "mask_compare_manifest.csv", manifest_rows)


if __name__ == "__main__":
    main()
