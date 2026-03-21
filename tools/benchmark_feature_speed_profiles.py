from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(r"D:\CNTDATA\CNTA_ML_Project")
DB_PATH = PROJECT_ROOT / "database" / "cnta_experiments.sqlite"
OUTPUT_ROOT = PROJECT_ROOT / "reports"

import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.feature_extractor import FeatureExtractor


@dataclass
class ImageRow:
    image_id: int
    file_path: str
    magnification: int
    source: str


STEP_KEYS = ["roi", "preprocess", "density", "diameter", "alignment", "curvature", "waviness", "done"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark FeatureExtractor accurate vs fast speed profiles.")
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--source", type=str, default="XR")
    parser.add_argument("--min-mag", type=int, default=20000)
    parser.add_argument("--diameter-method", type=str, default="standard", choices=["standard", "enhanced"])
    parser.add_argument("--processed-only", action="store_true", help="Only benchmark rows with processed=1.")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def fetch_rows(db_path: Path, source: str, limit: int, min_mag: int, processed_only: bool) -> list[ImageRow]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.cursor()
        where = [
            "source = ?",
            "COALESCE(is_deleted, 0) = 0",
            "COALESCE(magnification, 0) >= ?",
        ]
        params: list[object] = [source, min_mag]
        if processed_only:
            where.append("COALESCE(processed, 0) = 1")
        cursor.execute(
            f"""
            SELECT id, file_path, magnification, source
            FROM images
            WHERE {" AND ".join(where)}
            ORDER BY id DESC
            LIMIT ?
            """,
            (*params, limit),
        )
        rows = cursor.fetchall()
    finally:
        conn.close()

    return [
        ImageRow(
            image_id=int(row["id"]),
            file_path=str(row["file_path"]),
            magnification=int(row["magnification"] or 0),
            source=str(row["source"] or ""),
        )
        for row in rows
    ]


def read_gray(path: str) -> np.ndarray | None:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        return img
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    except Exception:
        return None


def run_one_profile(img: np.ndarray, magnification: int, diameter_method: str, speed_profile: str) -> dict:
    extractor = FeatureExtractor(
        magnification=magnification if magnification > 0 else None,
        diameter_method=diameter_method,
        speed_profile=speed_profile,
    )
    progress_events: list[dict] = []

    def on_progress(step_name: str, elapsed_s: float, payload: dict):
        progress_events.append(
            {
                "step": step_name,
                "elapsed_s": float(elapsed_s),
                "payload": payload or {},
            }
        )

    started = time.perf_counter()
    features = extractor.extract_all(img, progress_callback=on_progress)
    total_s = time.perf_counter() - started

    cumulative = {key: None for key in STEP_KEYS}
    incremental = {key: None for key in STEP_KEYS}
    previous = 0.0
    for event in progress_events:
        step = event["step"]
        elapsed = float(event["elapsed_s"])
        if step in cumulative:
            cumulative[step] = elapsed
            incremental[step] = max(0.0, elapsed - previous)
        previous = elapsed

    return {
        "features": features,
        "progress_events": progress_events,
        "total_s": float(total_s),
        "step_cumulative_s": cumulative,
        "step_incremental_s": incremental,
    }


def safe_num(value):
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def build_output_dir(output_dir: Path | None) -> Path:
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = OUTPUT_ROOT / f"feature_speed_compare_{stamp}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def write_runs_csv(path: Path, runs: list[dict]) -> None:
    fields = [
        "image_id",
        "file_name",
        "magnification",
        "speed_profile",
        "total_s",
        "roi_s",
        "preprocess_s",
        "density_s",
        "diameter_s",
        "alignment_s",
        "curvature_s",
        "waviness_s",
        "density",
        "diameter",
        "alignment",
        "curvature",
        "curvature_nm",
        "waviness_ratio",
        "waviness_height_nm",
        "waviness_wavelength_nm",
        "hof_method",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in runs:
            inc = row["benchmark"]["step_incremental_s"]
            ft = row["benchmark"]["features"]
            writer.writerow(
                {
                    "image_id": row["image_id"],
                    "file_name": Path(row["file_path"]).name,
                    "magnification": row["magnification"],
                    "speed_profile": row["speed_profile"],
                    "total_s": round(row["benchmark"]["total_s"], 6),
                    "roi_s": inc.get("roi"),
                    "preprocess_s": inc.get("preprocess"),
                    "density_s": inc.get("density"),
                    "diameter_s": inc.get("diameter"),
                    "alignment_s": inc.get("alignment"),
                    "curvature_s": inc.get("curvature"),
                    "waviness_s": inc.get("waviness"),
                    "density": ft.get("density"),
                    "diameter": ft.get("diameter"),
                    "alignment": ft.get("alignment"),
                    "curvature": ft.get("curvature"),
                    "curvature_nm": ft.get("curvature_nm"),
                    "waviness_ratio": ft.get("waviness_ratio"),
                    "waviness_height_nm": ft.get("waviness_height_nm"),
                    "waviness_wavelength_nm": ft.get("waviness_wavelength_nm"),
                    "hof_method": ft.get("hof_method"),
                }
            )


def write_pairs_csv(path: Path, pairs: list[dict]) -> None:
    fields = [
        "image_id",
        "file_name",
        "magnification",
        "accurate_total_s",
        "fast_total_s",
        "speedup_x",
        "delta_density_abs",
        "delta_diameter_abs_nm",
        "delta_alignment_abs",
        "delta_curvature_nm_abs",
        "delta_waviness_ratio_abs",
        "curvature_label_changed",
        "hof_method_accurate",
        "hof_method_fast",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in pairs:
            writer.writerow(row)


def main() -> int:
    args = parse_args()
    rows = fetch_rows(
        db_path=args.db,
        source=args.source,
        limit=args.limit,
        min_mag=args.min_mag,
        processed_only=args.processed_only,
    )
    if not rows:
        print("No rows matched benchmark query.")
        return 1

    output_dir = build_output_dir(args.output_dir)
    print(f"Benchmark rows: {len(rows)}")
    print(f"Output dir: {output_dir}")

    run_records: list[dict] = []
    for idx, row in enumerate(rows, start=1):
        img = read_gray(row.file_path)
        if img is None:
            print(f"[{idx}/{len(rows)}] SKIP unreadable: {row.file_path}")
            continue

        for profile in ("accurate", "fast"):
            print(
                f"[{idx}/{len(rows)}] image_id={row.image_id} mag={row.magnification} "
                f"profile={profile} file={Path(row.file_path).name}",
                flush=True,
            )
            bench = run_one_profile(
                img=img,
                magnification=row.magnification,
                diameter_method=args.diameter_method,
                speed_profile=profile,
            )
            run_records.append(
                {
                    "image_id": row.image_id,
                    "file_path": row.file_path,
                    "magnification": row.magnification,
                    "source": row.source,
                    "speed_profile": profile,
                    "benchmark": bench,
                }
            )

    runs_csv = output_dir / "runs.csv"
    write_runs_csv(runs_csv, run_records)

    by_image: dict[int, dict[str, dict]] = {}
    for record in run_records:
        by_image.setdefault(record["image_id"], {})[record["speed_profile"]] = record

    pair_rows: list[dict] = []
    speedups: list[float] = []
    delta_alignment: list[float] = []
    delta_waviness: list[float] = []
    delta_curv: list[float] = []
    delta_diameter: list[float] = []
    changed_curvature_labels = 0

    for image_id, item in by_image.items():
        if "accurate" not in item or "fast" not in item:
            continue
        a = item["accurate"]
        f = item["fast"]
        fa = a["benchmark"]["features"]
        ff = f["benchmark"]["features"]

        a_total = float(a["benchmark"]["total_s"])
        f_total = float(f["benchmark"]["total_s"])
        speedup = (a_total / f_total) if f_total > 1e-12 else None
        if speedup is not None:
            speedups.append(speedup)

        d_align = abs((safe_num(fa.get("alignment")) or 0.0) - (safe_num(ff.get("alignment")) or 0.0))
        d_wave = abs((safe_num(fa.get("waviness_ratio")) or 0.0) - (safe_num(ff.get("waviness_ratio")) or 0.0))
        d_curv = abs((safe_num(fa.get("curvature_nm")) or 0.0) - (safe_num(ff.get("curvature_nm")) or 0.0))
        d_diam = abs((safe_num(fa.get("diameter")) or 0.0) - (safe_num(ff.get("diameter")) or 0.0))
        d_dens = abs((safe_num(fa.get("density")) or 0.0) - (safe_num(ff.get("density")) or 0.0))
        label_changed = int(str(fa.get("curvature")) != str(ff.get("curvature")))
        changed_curvature_labels += label_changed

        delta_alignment.append(d_align)
        delta_waviness.append(d_wave)
        delta_curv.append(d_curv)
        delta_diameter.append(d_diam)

        pair_rows.append(
            {
                "image_id": image_id,
                "file_name": Path(a["file_path"]).name,
                "magnification": a["magnification"],
                "accurate_total_s": round(a_total, 6),
                "fast_total_s": round(f_total, 6),
                "speedup_x": round(speedup, 6) if speedup is not None else None,
                "delta_density_abs": round(d_dens, 6),
                "delta_diameter_abs_nm": round(d_diam, 6),
                "delta_alignment_abs": round(d_align, 6),
                "delta_curvature_nm_abs": round(d_curv, 6),
                "delta_waviness_ratio_abs": round(d_wave, 6),
                "curvature_label_changed": label_changed,
                "hof_method_accurate": fa.get("hof_method"),
                "hof_method_fast": ff.get("hof_method"),
            }
        )

    pairs_csv = output_dir / "pairs.csv"
    write_pairs_csv(pairs_csv, pair_rows)

    def stats(values: list[float]) -> dict:
        if not values:
            return {"mean": None, "median": None, "max": None}
        arr = np.asarray(values, dtype=float)
        return {
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "max": float(arr.max()),
        }

    summary = {
        "sample_count": len(pair_rows),
        "diameter_method": args.diameter_method,
        "speedup_x": stats(speedups),
        "delta_alignment_abs": stats(delta_alignment),
        "delta_waviness_ratio_abs": stats(delta_waviness),
        "delta_curvature_nm_abs": stats(delta_curv),
        "delta_diameter_abs_nm": stats(delta_diameter),
        "curvature_label_changed_count": changed_curvature_labels,
        "runs_csv": str(runs_csv),
        "pairs_csv": str(pairs_csv),
    }

    summary_json = output_dir / "summary.json"
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

