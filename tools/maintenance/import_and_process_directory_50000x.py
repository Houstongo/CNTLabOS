from __future__ import annotations

import argparse
import csv
import json
import shutil
import sqlite3
import tempfile
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import cv2
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "database" / "cnta_experiments.sqlite"
REPORT_ROOT = PROJECT_ROOT / "reports"
VALID_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
KNOWN_MAGS = (10000, 20000, 50000, 100000)
CALIBRATION_CONSTANT = 269792.0


import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.core.batch_processor import _get_cldice_segmenter, _make_feature_extractor  # noqa: E402
from tools.batch_zzy_feature_panels import augment_features_with_junction_metrics  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import and process arbitrary directory 50000X images into images table.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--db-path", type=Path, default=DB_PATH)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--diameter-method", default="enhanced", choices=["standard", "enhanced"])
    parser.add_argument("--source", default="ZZY")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def init_filename_pattern():
    import re

    return re.compile(
        r"^(?P<sample>No\d+)\s+"
        r"(?P<al2o3_power>\d+)w\s+(?P<al2o3_thickness>[\d.]+)nm\s+"
        r"(?P<fe_power>\d+)w\s+(?P<fe_thickness>[\d.]+)nm\s+"
        r"(?P<ar>\d+)\s+(?P<h2>\d+)\s+(?P<c2h4>\d+)\s+"
        r"(?P<anneal_temp>\d+)\s+(?P<growth_temp>\d+)\s+"
        r"(?P<anneal_time>\d+)min\s+(?P<growth_time>\d+)min\s+"
        r"(?P<tail>.+?)$"
    )


FILENAME_PATTERN = init_filename_pattern()


def read_gray(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return image


def to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def to_int(value: Any) -> Optional[int]:
    numeric = to_float(value)
    if numeric is None:
        return None
    return int(round(numeric))


def minutes_to_hours(value: Any) -> Optional[float]:
    numeric = to_float(value)
    if numeric is None:
        return None
    return numeric / 60.0


def curvature_nm_to_um(value: Any) -> Optional[float]:
    numeric = to_float(value)
    if numeric is None:
        return None
    return numeric * 1000.0


def load_fei_xml(path: Path) -> Optional[ET.Element]:
    try:
        info = Image.open(path).info
        xml_text = info.get("34683") or info.get(34683)
        if not xml_text:
            return None
        return ET.fromstring(xml_text)
    except Exception:
        return None


def infer_magnification(path: Path) -> Optional[int]:
    root = load_fei_xml(path)
    if root is None:
        return None

    pixel_width = None
    databar_label = ""
    for elem in root.iter():
        tag = elem.tag.split("}")[-1]
        text = (elem.text or "").strip()
        if tag == "pixelWidth" and text:
            pixel_width = to_float(text)
        elif tag == "databarLabel" and text:
            databar_label = text

    if databar_label:
        for mag in KNOWN_MAGS:
            if str(mag) in databar_label:
                return mag

    if pixel_width and pixel_width > 0:
        estimated = CALIBRATION_CONSTANT / pixel_width
        return min(KNOWN_MAGS, key=lambda mag: abs(mag - estimated))
    return None


def parse_filename(stem: str) -> Dict[str, Any]:
    match = FILENAME_PATTERN.match(stem)
    if not match:
        raise ValueError(f"Cannot parse filename: {stem}")
    gd = match.groupdict()
    tail = gd["tail"].strip()
    position_label = tail
    repeat_id = None
    import re

    m = re.match(r"^(?P<pos>.+)-(?P<repeat>\d+)$", tail)
    if m:
        position_label = m.group("pos").strip()
        repeat_id = int(m.group("repeat"))
    elif tail.isdigit():
        repeat_id = int(tail)

    sample_id = (
        f"{gd['sample']}_{gd['al2o3_power']}W_{gd['al2o3_thickness']}nm_"
        f"{gd['fe_power']}W_{gd['fe_thickness']}nm_{gd['ar']}_{gd['h2']}_{gd['c2h4']}_"
        f"{gd['anneal_temp']}_{gd['growth_temp']}_{gd['anneal_time']}min_{gd['growth_time']}min"
    )
    return {
        "sample_id": sample_id,
        "position_label": position_label,
        "repeat_id": repeat_id,
        "al2o3_power": to_float(gd["al2o3_power"]),
        "al2o3_thickness": to_float(gd["al2o3_thickness"]),
        "fe_power": to_float(gd["fe_power"]),
        "fe_thickness": to_float(gd["fe_thickness"]),
        "ar_flow": to_float(gd["ar"]),
        "h2_flow": to_float(gd["h2"]),
        "c2h4_flow": to_float(gd["c2h4"]),
        "anneal_temp": to_float(gd["anneal_temp"]),
        "growth_temp": to_float(gd["growth_temp"]),
        "anneal_time": minutes_to_hours(gd["anneal_time"]),
        "growth_time": minutes_to_hours(gd["growth_time"]),
    }


def discover_50000_images(input_dir: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for path in sorted(input_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in VALID_SUFFIXES:
            continue
        mag = infer_magnification(path)
        if mag != 50000:
            continue
        meta = parse_filename(path.stem)
        items.append({"image_path": path, "magnification": mag, **meta})
    return items


def existing_columns(conn: sqlite3.Connection) -> set[str]:
    cur = conn.cursor()
    return {row[1] for row in cur.execute("PRAGMA table_info(images)").fetchall()}


def build_insert_payload(item: Dict[str, Any], source: str) -> Dict[str, Any]:
    payload = {
        "file_path": str(item["image_path"]),
        "source": source,
        "sample_id": item["sample_id"],
        "growth_temp": item["growth_temp"],
        "growth_time": item["growth_time"],
        "ar_flow": item["ar_flow"],
        "h2_flow": item["h2_flow"],
        "c2h4_flow": item["c2h4_flow"],
        "al2o3_power": item["al2o3_power"],
        "al2o3_thickness": item["al2o3_thickness"],
        "fe_power": item["fe_power"],
        "fe_thickness": item["fe_thickness"],
        "anneal_temp": item["anneal_temp"],
        "anneal_time": item["anneal_time"],
        "position_label": item["position_label"],
        "magnification": item["magnification"],
        "repeat_id": item["repeat_id"],
        "processed": 0,
        "is_deleted": 0,
    }
    return payload


def build_feature_update(features: Dict[str, Any]) -> Dict[str, Any]:
    branch_count = to_int(features.get("curvature_v3_branch_count"))
    diameter = to_float(features.get("diameter"))
    payload = {
        "diameter": diameter,
        "diameter_mean": diameter,
        "density": to_float(features.get("density")),
        "alignment": to_float(features.get("alignment")),
        "curvature": curvature_nm_to_um(features.get("curvature_nm_v3_trimmed_mean_sqrt_length")),
        "tortuosity": to_float(features.get("tortuosity_v2")),
        "waviness_ratio": to_float(features.get("waviness_ratio_v2")),
        "junction_count": to_float(features.get("junction_count")),
        "junction_ratio": to_float(features.get("junction_ratio")),
        "skeleton_length_px": to_float(features.get("skeleton_length_px")),
        "skeleton_length_um": to_float(features.get("skeleton_length_um")),
        "junctions_per_100um": to_float(features.get("junctions_per_100um")),
        "branch_count": branch_count,
        "l2_branch_count": branch_count,
        "curvature_label": features.get("curvature_v3"),
        "l2_curvature_label": features.get("curvature_v3"),
        "curvature_mean": curvature_nm_to_um(features.get("curvature_nm_v3_mean_sqrt_length")),
        "curvature_trimmed_mean": curvature_nm_to_um(features.get("curvature_nm_v3_trimmed_mean_sqrt_length")),
        "l2_curvature_mean_sqrt_length_nm": to_float(features.get("curvature_nm_v3_mean_sqrt_length")),
        "l2_curvature_trimmed_mean_sqrt_length_nm": to_float(features.get("curvature_nm_v3_trimmed_mean_sqrt_length")),
        "l2_waviness_ratio_v2": to_float(features.get("waviness_ratio_v2")),
        "l2_tortuosity_v2": to_float(features.get("tortuosity_v2")),
        "processed": 1,
    }
    return payload


def insert_or_get_row_id(cur: sqlite3.Cursor, payload: Dict[str, Any], columns: Iterable[str]) -> int:
    cur.execute("SELECT id FROM images WHERE file_path = ?", (payload["file_path"],))
    row = cur.fetchone()
    if row is not None:
        return int(row[0])

    cols = [col for col in payload.keys() if col in columns]
    placeholders = ", ".join(["?"] * len(cols))
    sql = f"INSERT INTO images ({', '.join(cols)}) VALUES ({placeholders})"
    cur.execute(sql, [payload[col] for col in cols])
    return int(cur.lastrowid)


def update_row(cur: sqlite3.Cursor, row_id: int, payload: Dict[str, Any], columns: Iterable[str]) -> None:
    cols = [col for col in payload.keys() if col in columns]
    assignments = ", ".join(f"{col} = ?" for col in cols)
    sql = f"UPDATE images SET {assignments} WHERE id = ?"
    cur.execute(sql, [payload[col] for col in cols] + [row_id])


def process_items(
    db_path: Path,
    items: List[Dict[str, Any]],
    source: str,
    output_dir: Path,
    device: str,
    diameter_method: str,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    features_dir = output_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    backup_path = db_path.with_name(f"{db_path.stem}.before_no49_import_{datetime.now().strftime('%Y%m%d_%H%M%S')}{db_path.suffix}")
    shutil.copy2(db_path, backup_path)

    tmp_dir = Path(tempfile.mkdtemp(prefix="import_process_50000x_"))
    work_db = tmp_dir / db_path.name
    shutil.copy2(db_path, work_db)

    conn = sqlite3.connect(work_db)
    cols = existing_columns(conn)
    cur = conn.cursor()
    segmenter = _get_cldice_segmenter(device=device)

    rows: List[Dict[str, Any]] = []
    inserted = 0
    updated = 0
    t0 = time.perf_counter()

    for idx, item in enumerate(items, start=1):
        image_path = item["image_path"]
        print(f"[{idx}/{len(items)}] {image_path.name}", flush=True)
        insert_payload = build_insert_payload(item, source=source)
        cur.execute("SELECT id FROM images WHERE file_path = ?", (str(image_path),))
        existed = cur.fetchone() is not None
        row_id = insert_or_get_row_id(cur, insert_payload, cols)
        if existed:
            update_row(cur, row_id, insert_payload, cols)
            updated += 1
        else:
            inserted += 1

        img_gray = read_gray(image_path)
        extractor = _make_feature_extractor(magnification=50000, diameter_method=diameter_method)
        roi = extractor.extract_roi(img_gray)
        mask = segmenter.predict_mask(roi)
        features = extractor.extract_all(img_gray, external_binary_mask=mask)
        features = augment_features_with_junction_metrics(extractor, mask, features)

        feature_update = build_feature_update(features)
        update_row(cur, row_id, feature_update, cols)

        feature_path = features_dir / f"{image_path.stem}__features.json"
        feature_path.write_text(
            json.dumps(
                {
                    "row_id": row_id,
                    "file_name": image_path.name,
                    "source_path": str(image_path),
                    "insert_payload": insert_payload,
                    "features": features,
                },
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

        rows.append(
            {
                "id": row_id,
                "file_name": image_path.name,
                "source_path": str(image_path),
                "sample_id": item["sample_id"],
                "position_label": item["position_label"],
                "repeat_id": item["repeat_id"],
                "magnification": 50000,
                "fe_power": item["fe_power"],
                "fe_thickness": item["fe_thickness"],
                "al2o3_power": item["al2o3_power"],
                "al2o3_thickness": item["al2o3_thickness"],
                "tortuosity_v2": features.get("tortuosity_v2"),
                "waviness_ratio_v2": features.get("waviness_ratio_v2"),
                "diameter": features.get("diameter"),
                "alignment": features.get("alignment"),
                "density": features.get("density"),
                "curvature_nm_v3_trimmed_mean_length": features.get("curvature_nm_v3_trimmed_mean_length"),
            }
        )

    conn.commit()
    conn.close()
    shutil.copy2(work_db, db_path)

    csv_path = output_dir / "summary.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "input_count": len(items),
        "inserted": inserted,
        "updated": updated,
        "device": device,
        "diameter_method": diameter_method,
        "backup_db": str(backup_path),
        "elapsed_s": round(time.perf_counter() - t0, 1),
        "rows": rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (REPORT_ROOT / f"directory_50000_import_process_{stamp}")
    items = discover_50000_images(args.input_dir)
    if args.limit > 0:
        items = items[: args.limit]
    if not items:
        raise SystemExit("No 50000X images found in input directory.")

    print(f"50000X images found: {len(items)}")
    summary = process_items(
        db_path=args.db_path,
        items=items,
        source=args.source,
        output_dir=output_dir,
        device=args.device,
        diameter_method=args.diameter_method,
    )
    print(f"OUTPUT_DIR={output_dir}")
    print(f"INSERTED={summary['inserted']}")
    print(f"UPDATED={summary['updated']}")


if __name__ == "__main__":
    main()
