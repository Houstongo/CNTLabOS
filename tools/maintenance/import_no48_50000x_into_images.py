from __future__ import annotations

import argparse
import csv
import shutil
import sqlite3
import tempfile
from pathlib import Path
from typing import Dict, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = PROJECT_ROOT / "database" / "cnta_experiments.sqlite"
DEFAULT_INPUT_CSV = (
    PROJECT_ROOT
    / "reports"
    / "50000x_feature_extract_20260401"
    / "no48_50000x_modeling_table_extended.csv"
)
DEFAULT_IMAGE_ROOT = Path(r"D:\CNTDATA\4.1_by_true_magnification\50000x")


def to_float(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text or text.lower() in {"none", "nan", "null", "na", "n/a"}:
        return None
    return float(text)


def to_int(value):
    numeric = to_float(value)
    if numeric is None:
        return None
    return int(round(numeric))


def nm_minutes_to_hours(value):
    numeric = to_float(value)
    if numeric is None:
        return None
    return float(numeric) / 60.0


def um_to_nm_inverse(value):
    numeric = to_float(value)
    if numeric is None:
        return None
    return float(numeric) / 1000.0


def load_rows(input_csv: Path) -> list[dict]:
    with input_csv.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def existing_columns(conn: sqlite3.Connection) -> set[str]:
    cur = conn.cursor()
    return {row[1] for row in cur.execute("PRAGMA table_info(images)").fetchall()}


def sanitize_sample_id(field_token: str) -> str:
    return field_token.replace(" ", "_")


def build_actual_file_path(row: dict, image_root: Path) -> Path:
    basename = str(row["file_name"])
    if basename.startswith("50000x__"):
        basename = basename[len("50000x__") :]
    return image_root / basename


def build_payload(row: dict, image_path: Path) -> Dict[str, object]:
    fe_power = to_int(row.get("fe_power_w"))
    fe_thickness = to_float(row.get("fe_thickness_nm"))
    field_token = str(row.get("field_token") or "").strip()
    field_group = str(row.get("field_group") or field_token).strip() or field_token
    sample_id = f"No48-50000-{fe_power}W-{fe_thickness:g}nm-{sanitize_sample_id(field_token)}"

    curvature_um = to_float(row.get("mean_curvature_um_inv"))
    curvature_p70_um = to_float(row.get("curvature_p70_um_inv"))
    diameter_mean = to_float(row.get("diameter_mean_nm"))
    branch_count = to_int(row.get("branch_count"))

    payload = {
        "file_path": str(image_path),
        "source": "ZZY",
        "sample_id": sample_id,
        "al2o3_power": to_float(row.get("al2o3_power_w")),
        "al2o3_thickness": to_float(row.get("al2o3_thickness_nm")),
        "fe_power": fe_power,
        "fe_thickness": fe_thickness,
        "ar_flow": to_float(row.get("ar_flow")),
        "h2_flow": to_float(row.get("h2_flow")),
        "c2h4_flow": to_float(row.get("c2h4_flow")),
        "anneal_temp": to_float(row.get("anneal_temp_c")),
        "growth_temp": to_float(row.get("growth_temp_c")),
        "anneal_time": nm_minutes_to_hours(row.get("anneal_time_min")),
        "growth_time": nm_minutes_to_hours(row.get("growth_time_min")),
        "position_label": field_group,
        "magnification": to_int(row.get("magnification")),
        "repeat_id": to_int(row.get("repeat_id")),
        "diameter": diameter_mean,
        "diameter_mean": diameter_mean,
        "density": to_float(row.get("density")),
        "alignment": to_float(row.get("alignment")),
        "curvature": curvature_um,
        "curvature_trimmed_mean": curvature_um,
        "curvature_p70": curvature_p70_um,
        "curvature_label": row.get("curvature_label"),
        "tortuosity": to_float(row.get("tortuosity")),
        "waviness_ratio": to_float(row.get("waviness_ratio")),
        "junction_count": to_int(row.get("junction_count")),
        "junction_ratio": to_float(row.get("junction_ratio")),
        "skeleton_length_um": to_float(row.get("skeleton_length_um")),
        "branch_count": branch_count,
        "diameter_p30_nm_v2": to_float(row.get("diameter_p30_nm")),
        "l2_branch_count": branch_count,
        "l2_curvature_label": row.get("curvature_label"),
        "l2_curvature_p70_sqrt_length_nm": um_to_nm_inverse(row.get("curvature_p70_um_inv")),
        "l2_curvature_trimmed_mean_sqrt_length_nm": um_to_nm_inverse(row.get("mean_curvature_um_inv")),
        "l2_waviness_ratio_v2": to_float(row.get("waviness_ratio")),
        "l2_tortuosity_v2": to_float(row.get("tortuosity")),
        "processed": 1,
    }
    return payload


def insert_row(cur: sqlite3.Cursor, payload: Dict[str, object], columns: Iterable[str]) -> None:
    cols = [col for col in payload.keys() if col in columns]
    placeholders = ", ".join(["?"] * len(cols))
    sql = f"INSERT INTO images ({', '.join(cols)}) VALUES ({placeholders})"
    cur.execute(sql, [payload[col] for col in cols])


def update_row(cur: sqlite3.Cursor, row_id: int, payload: Dict[str, object], columns: Iterable[str]) -> None:
    cols = [col for col in payload.keys() if col in columns and col != "file_path"]
    assignments = ", ".join(f"{col} = ?" for col in cols)
    sql = f"UPDATE images SET {assignments} WHERE id = ?"
    cur.execute(sql, [payload[col] for col in cols] + [row_id])


def sync_rows(conn: sqlite3.Connection, rows: list[dict], image_root: Path) -> dict:
    cols = existing_columns(conn)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    inserted = 0
    updated = 0
    missing_files = []

    for row in rows:
        image_path = build_actual_file_path(row, image_root)
        if not image_path.exists():
            missing_files.append(str(image_path))
            continue

        payload = build_payload(row, image_path)
        cur.execute("SELECT id FROM images WHERE file_path = ?", (str(image_path),))
        existing = cur.fetchone()
        if existing is None:
            insert_row(cur, payload, cols)
            inserted += 1
        else:
            update_row(cur, int(existing["id"]), payload, cols)
            updated += 1

    conn.commit()
    return {
        "input_rows": len(rows),
        "inserted": inserted,
        "updated": updated,
        "missing_files": len(missing_files),
        "missing_file_examples": missing_files[:10],
    }


def run_import(db_path: Path, input_csv: Path, image_root: Path) -> dict:
    rows = load_rows(input_csv)
    tmp_dir = Path(tempfile.mkdtemp(prefix="no48_50000x_import_"))
    work_db = tmp_dir / db_path.name
    shutil.copy2(db_path, work_db)

    conn = sqlite3.connect(work_db)
    try:
        stats = sync_rows(conn, rows, image_root)
    finally:
        conn.close()

    shutil.copy2(work_db, db_path)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Import No48 + 50000x images and extracted features into images table.")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    args = parser.parse_args()

    stats = run_import(args.db_path, args.input_csv, args.image_root)
    print(f"Imported into: {args.db_path}")
    for key, value in stats.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
