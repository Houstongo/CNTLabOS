import argparse
import json
import shutil
import sqlite3
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = PROJECT_ROOT / "database" / "cnta_experiments.sqlite"
DEFAULT_SUMMARY_JSON = (
    PROJECT_ROOT
    / "reports"
    / "zzy_feature_panels_cldice_20260331_gt10000_with_junction"
    / "summary.json"
)

NEW_COLUMNS = {
    "junctions_per_100um": "REAL",
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


def to_int(value):
    numeric = to_float(value)
    if numeric is None:
        return None
    return int(round(numeric))


def curvature_nm_to_um(value):
    numeric = to_float(value)
    if numeric is None:
        return None
    return numeric * 1000.0


def load_summary_rows(summary_json: Path) -> List[dict]:
    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    return [row for row in payload.get("rows", []) if str(row.get("status") or "").lower() == "ok"]


def ensure_columns(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    existing = {row[1] for row in cur.execute("PRAGMA table_info(images)").fetchall()}
    for col_name, col_type in NEW_COLUMNS.items():
        if col_name in existing:
            continue
        cur.execute(f"ALTER TABLE images ADD COLUMN {col_name} {col_type}")
    conn.commit()


def fetch_candidate_rows(conn: sqlite3.Connection) -> List[sqlite3.Row]:
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    return cur.execute(
        """
        SELECT id, file_path, source, magnification, COALESCE(is_deleted, 0) AS is_deleted
        FROM images
        WHERE source = 'ZZY'
          AND magnification > 10000
        ORDER BY id
        """
    ).fetchall()


def basename(path_text: str) -> str:
    return Path(path_text).name.lower()


def align_rows(candidate_rows: List[sqlite3.Row], summary_rows: List[dict]) -> Tuple[List[Tuple[sqlite3.Row, dict]], int]:
    pairs: List[Tuple[sqlite3.Row, dict]] = []
    db_index = 0
    summary_index = 0
    skipped_deleted_unmatched = 0

    while db_index < len(candidate_rows) and summary_index < len(summary_rows):
        db_row = candidate_rows[db_index]
        summary_row = summary_rows[summary_index]
        db_name = basename(str(db_row["file_path"]))
        summary_name = str(summary_row["file_name"]).lower()

        if db_name == summary_name:
            pairs.append((db_row, summary_row))
            db_index += 1
            summary_index += 1
            continue

        if int(db_row["is_deleted"]) == 1:
            skipped_deleted_unmatched += 1
            db_index += 1
            continue

        raise RuntimeError(
            "Summary-to-database alignment failed: "
            f"db id={db_row['id']} basename='{Path(str(db_row['file_path'])).name}' "
            f"!= summary file_name='{summary_row['file_name']}'"
        )

    while db_index < len(candidate_rows):
        db_row = candidate_rows[db_index]
        if int(db_row["is_deleted"]) != 1:
            raise RuntimeError(
                "Unpaired active database row remains after alignment: "
                f"id={db_row['id']} path={db_row['file_path']}"
            )
        skipped_deleted_unmatched += 1
        db_index += 1

    if summary_index != len(summary_rows):
        raise RuntimeError(
            f"Summary rows remain unmatched after alignment: {len(summary_rows) - summary_index}"
        )

    return pairs, skipped_deleted_unmatched


def build_update_payload(summary_row: dict) -> Dict[str, object]:
    branch_count = to_int(summary_row.get("curvature_v3_branch_count"))
    payload = {
        "density": to_float(summary_row.get("density")),
        "alignment": to_float(summary_row.get("alignment")),
        "diameter": to_float(summary_row.get("diameter")),
        "curvature": curvature_nm_to_um(summary_row.get("curvature_nm_v3_trimmed_mean_sqrt_length")),
        "tortuosity": to_float(summary_row.get("tortuosity_v2")),
        "waviness_ratio": to_float(summary_row.get("waviness_ratio_v2")),
        "processed": 1,
        "junction_count": to_float(summary_row.get("junction_count")),
        "junction_ratio": to_float(summary_row.get("junction_ratio")),
        "skeleton_length_px": to_float(summary_row.get("skeleton_length_px")),
        "skeleton_length_um": to_float(summary_row.get("skeleton_length_um")),
        "junctions_per_100um": to_float(summary_row.get("junctions_per_100um")),
        "l2_branch_count": branch_count,
        "l2_curvature_label": summary_row.get("curvature_v3"),
        "l2_curvature_mean_sqrt_length_nm": to_float(summary_row.get("curvature_nm_v3_mean_sqrt_length")),
        "l2_curvature_trimmed_mean_sqrt_length_nm": to_float(
            summary_row.get("curvature_nm_v3_trimmed_mean_sqrt_length")
        ),
        "l2_waviness_ratio_v2": to_float(summary_row.get("waviness_ratio_v2")),
        "l2_tortuosity_v2": to_float(summary_row.get("tortuosity_v2")),
        "branch_count": branch_count,
        "curvature_label": summary_row.get("curvature_v3"),
        "curvature_mean": curvature_nm_to_um(summary_row.get("curvature_nm_v3_mean_sqrt_length")),
        "curvature_trimmed_mean": curvature_nm_to_um(
            summary_row.get("curvature_nm_v3_trimmed_mean_sqrt_length")
        ),
    }
    return payload


def sync_summary_into_images(conn: sqlite3.Connection, summary_rows: List[dict]) -> Dict[str, int]:
    ensure_columns(conn)
    candidate_rows = fetch_candidate_rows(conn)
    pairs, skipped_deleted_unmatched = align_rows(candidate_rows, summary_rows)

    cur = conn.cursor()
    updated = 0
    skipped_deleted_paired = 0

    for db_row, summary_row in pairs:
        if int(db_row["is_deleted"]) == 1:
            skipped_deleted_paired += 1
            continue

        payload = build_update_payload(summary_row)
        cols = list(payload.keys())
        assignments = ", ".join(f"{col} = ?" for col in cols)
        values = [payload[col] for col in cols]
        values.append(int(db_row["id"]))
        cur.execute(f"UPDATE images SET {assignments} WHERE id = ?", values)
        updated += 1

    conn.commit()
    return {
        "candidate_rows": len(candidate_rows),
        "summary_rows": len(summary_rows),
        "paired_rows": len(pairs),
        "updated_active_rows": updated,
        "skipped_deleted_paired": skipped_deleted_paired,
        "skipped_deleted_unmatched": skipped_deleted_unmatched,
    }


def run_sync(db_path: Path, summary_json: Path) -> Tuple[Path, Dict[str, int]]:
    summary_rows = load_summary_rows(summary_json)
    tmp_dir = Path(tempfile.mkdtemp(prefix="zzy_summary_sync_"))
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
        description="Sync latest ZZY summary features into images table."
    )
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--summary-json", default=str(DEFAULT_SUMMARY_JSON))
    args = parser.parse_args()

    db_path, stats = run_sync(Path(args.db_path), Path(args.summary_json))
    print(f"ZZY summary sync completed: {db_path}")
    for key, value in stats.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
