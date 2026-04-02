import argparse
import csv
import shutil
import sqlite3
import tempfile
from pathlib import Path
from typing import Dict, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = PROJECT_ROOT / "database" / "cnta_experiments.sqlite"
DEFAULT_SUMMARY_CSV = (
    PROJECT_ROOT
    / "reports"
    / "slice_standard_batch_20260331_005741"
    / "summary.csv"
)

OVERWRITE_MAPPING = {
    "density": "density",
    "alignment": "alignment",
    "diameter": "diameter_mean_nm",
    "curvature": "l2_curvature_trimmed_mean_sqrt_length_nm",
    "tortuosity": "l2_tortuosity_v2",
    "waviness_ratio": "l2_waviness_ratio_v2",
}

FILL_EXISTING_MAPPING = {
    "diameter_mean": "diameter_mean_nm",
    "diameter_std": "diameter_std_nm",
    "diameter_min": "diameter_min_nm",
    "diameter_max": "diameter_max_nm",
    "diameter_p50": "diameter_p50_nm",
    "diameter_p75": "diameter_p75_nm",
}

NEW_COLUMNS = {
    "junction_count": "REAL",
    "junction_ratio": "REAL",
    "skeleton_length_px": "REAL",
    "skeleton_length_um": "REAL",
    "diameter_p30_nm_v2": "REAL",
    "l2_branch_count": "INTEGER",
    "l2_curvature_label": "TEXT",
    "l2_curvature_p70_sqrt_length_nm": "REAL",
    "l2_curvature_mean_sqrt_length_nm": "REAL",
    "l2_curvature_trimmed_mean_sqrt_length_nm": "REAL",
    "l2_waviness_ratio_v2": "REAL",
    "l2_tortuosity_v2": "REAL",
    "branch_count": "INTEGER",
    "curvature_label": "TEXT",
    "curvature_p70": "REAL",
    "curvature_mean": "REAL",
    "curvature_trimmed_mean": "REAL",
    "xr_feature_report_tag": "TEXT",
}

NEW_COLUMN_MAPPING = {
    "junction_count": "junction_count",
    "junction_ratio": "junction_ratio",
    "skeleton_length_px": "skeleton_length_px",
    "skeleton_length_um": "skeleton_length_um",
    "diameter_p30_nm_v2": "diameter_p30_nm",
    "l2_branch_count": "l2_branch_count",
    "l2_curvature_label": "l2_curvature_label",
    "l2_curvature_p70_sqrt_length_nm": "l2_curvature_p70_sqrt_length_nm",
    "l2_curvature_mean_sqrt_length_nm": "l2_curvature_mean_sqrt_length_nm",
    "l2_curvature_trimmed_mean_sqrt_length_nm": "l2_curvature_trimmed_mean_sqrt_length_nm",
    "l2_waviness_ratio_v2": "l2_waviness_ratio_v2",
    "l2_tortuosity_v2": "l2_tortuosity_v2",
}

CONCISE_MAPPING = {
    "branch_count": "l2_branch_count",
    "curvature_label": "l2_curvature_label",
    "curvature_p70": "l2_curvature_p70_sqrt_length_nm",
    "curvature_mean": "l2_curvature_mean_sqrt_length_nm",
    "curvature_trimmed_mean": "l2_curvature_trimmed_mean_sqrt_length_nm",
}

CURVATURE_UM_FIELDS = {
    "curvature",
    "curvature_p70",
    "curvature_mean",
    "curvature_trimmed_mean",
    "l2_curvature_p70_sqrt_length_nm",
    "l2_curvature_mean_sqrt_length_nm",
    "l2_curvature_trimmed_mean_sqrt_length_nm",
}


def to_float(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text or text.lower() in {"none", "nan", "null", "na", "n/a"}:
        return None
    return float(text)


def maybe_convert_curvature_to_um(field_name: str, value):
    if value is None:
        return None
    if field_name in CURVATURE_UM_FIELDS:
        return float(value) * 1000.0
    return value


def load_summary_rows(summary_csv: Path) -> Dict[int, dict]:
    with summary_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    filtered = {}
    for row in rows:
        if str(row.get("status") or "").strip().lower() != "success":
            continue
        image_id = row.get("image_id")
        if not image_id:
            continue
        filtered[int(image_id)] = row
    return filtered


def ensure_columns(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    existing = {row[1] for row in cur.execute("PRAGMA table_info(images)").fetchall()}
    for col_name, col_type in NEW_COLUMNS.items():
        if col_name in existing:
            continue
        cur.execute(f"ALTER TABLE images ADD COLUMN {col_name} {col_type}")
    conn.commit()


def build_update_payload(row: dict) -> Dict[str, object]:
    payload = {}
    for db_col, summary_col in OVERWRITE_MAPPING.items():
        payload[db_col] = maybe_convert_curvature_to_um(
            db_col,
            to_float(row.get(summary_col)),
        )
    for db_col, summary_col in FILL_EXISTING_MAPPING.items():
        payload[db_col] = to_float(row.get(summary_col))
    for db_col, summary_col in NEW_COLUMN_MAPPING.items():
        if db_col == "l2_curvature_label":
            payload[db_col] = row.get(summary_col)
        else:
            payload[db_col] = maybe_convert_curvature_to_um(
                db_col,
                to_float(row.get(summary_col)),
            )
    for db_col, summary_col in CONCISE_MAPPING.items():
        if db_col == "curvature_label":
            payload[db_col] = row.get(summary_col)
        else:
            payload[db_col] = maybe_convert_curvature_to_um(
                db_col,
                to_float(row.get(summary_col)),
            )
    payload["xr_feature_report_tag"] = "slice_standard_batch_20260331_005741"
    return payload


def sync_summary_into_images(conn: sqlite3.Connection, summary_rows: Dict[int, dict]) -> Dict[str, int]:
    ensure_columns(conn)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    updated = 0
    skipped_non_xr = 0
    missing_images = 0

    for image_id, row in summary_rows.items():
        found = cur.execute(
            "SELECT id, source FROM images WHERE id = ?",
            (image_id,),
        ).fetchone()
        if found is None:
            missing_images += 1
            continue
        if str(found["source"] or "").upper() != "XR":
            skipped_non_xr += 1
            continue

        payload = build_update_payload(row)
        cols = list(payload.keys())
        assignments = ", ".join(f"{col} = ?" for col in cols)
        values = [payload[col] for col in cols]
        values.append(image_id)
        cur.execute(
            f"UPDATE images SET {assignments} WHERE id = ?",
            values,
        )
        updated += 1

    conn.commit()
    return {
        "updated": updated,
        "skipped_non_xr": skipped_non_xr,
        "missing_images": missing_images,
    }


def run_sync(db_path: Path, summary_csv: Path) -> Tuple[Path, Dict[str, int]]:
    summary_rows = load_summary_rows(summary_csv)
    tmp_dir = Path(tempfile.mkdtemp(prefix="xr_summary_sync_"))
    work_db = tmp_dir / db_path.name
    shutil.copy2(db_path, work_db)

    conn = sqlite3.connect(work_db)
    try:
        stats = sync_summary_into_images(conn, summary_rows)
    finally:
        conn.close()

    shutil.copy2(work_db, db_path)
    return db_path, stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sync latest XR summary features into images table."
    )
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--summary-csv", default=str(DEFAULT_SUMMARY_CSV))
    args = parser.parse_args()

    db_path, stats = run_sync(Path(args.db_path), Path(args.summary_csv))
    print(f"XR summary sync completed: {db_path}")
    print(f"updated={stats['updated']}")
    print(f"skipped_non_xr={stats['skipped_non_xr']}")
    print(f"missing_images={stats['missing_images']}")


if __name__ == "__main__":
    main()
