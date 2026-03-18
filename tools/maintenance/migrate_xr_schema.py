import argparse
import os
import re
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = PROJECT_ROOT / "database" / "cnta_experiments.sqlite"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from backend.core.data_manager import CNTADataParser


def _to_experiment_date(folder_name: str) -> str | None:
    match = re.match(r"^(\d{6})", folder_name or "")
    if not match:
        return None
    yymmdd = match.group(1)
    return f"20{yymmdd[0:2]}-{yymmdd[2:4]}-{yymmdd[4:6]}"


def create_xr_tables(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS xr_runs (
            run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            folder_name TEXT UNIQUE NOT NULL,
            experiment_date TEXT,
            source TEXT NOT NULL DEFAULT 'XR',
            set_temp_c REAL NOT NULL,
            growth_time_h REAL,
            ar_flow REAL NOT NULL,
            catalyst_concentration REAL NOT NULL,
            carbon_source TEXT NOT NULL DEFAULT 'toluene',
            catalyst_precursor TEXT NOT NULL DEFAULT 'ferrocene',
            notes TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS xr_images (
            image_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            file_path TEXT UNIQUE NOT NULL,
            sample_id TEXT,
            membrane_id INTEGER,
            position_label TEXT,
            horizontal_pos TEXT,
            vertical_pos INTEGER,
            membrane_pos_cm REAL,
            magnification INTEGER,
            actual_temp_c REAL,
            processed INTEGER DEFAULT 0,
            FOREIGN KEY (run_id) REFERENCES xr_runs(run_id)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS xr_targets (
            target_id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id INTEGER NOT NULL,
            diameter REAL,
            density REAL,
            alignment REAL,
            curvature REAL,
            tortuosity REAL,
            label_source TEXT,
            label_version TEXT,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (image_id) REFERENCES xr_images(image_id)
        )
        """
    )
    conn.commit()


def migrate_xr_data(conn: sqlite3.Connection) -> None:
    parser = CNTADataParser()
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("DELETE FROM xr_targets")
    cur.execute("DELETE FROM xr_images")
    cur.execute("DELETE FROM xr_runs")

    xr_rows = cur.execute(
        "SELECT * FROM images WHERE source = 'XR' ORDER BY file_path"
    ).fetchall()

    run_cache: dict[str, int] = {}
    for row in xr_rows:
        file_path = row["file_path"]
        folder_name = os.path.basename(os.path.dirname(file_path))
        if folder_name in run_cache:
            continue

        folder_meta = parser.parse_xr_folder_metadata(folder_name)
        set_temp_c = (
            folder_meta.get("growth_temp")
            if folder_meta.get("growth_temp") is not None
            else row["growth_temp"]
        )
        ar_flow = (
            folder_meta.get("ar_flow")
            if folder_meta.get("ar_flow") is not None
            else row["ar_flow"]
        )
        catalyst_concentration = (
            folder_meta.get("catalyst_weight")
            if folder_meta.get("catalyst_weight") is not None
            else row["catalyst_weight"]
        )

        if set_temp_c is None or ar_flow is None or catalyst_concentration is None:
            raise ValueError(f"Missing XR run params for folder: {folder_name}")

        cur.execute(
            """
            INSERT INTO xr_runs (
                folder_name,
                experiment_date,
                set_temp_c,
                growth_time_h,
                ar_flow,
                catalyst_concentration,
                notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                folder_name,
                _to_experiment_date(folder_name),
                float(set_temp_c),
                row["growth_time"],
                float(ar_flow),
                float(catalyst_concentration),
                "migrated_from_images",
            ),
        )
        run_cache[folder_name] = cur.lastrowid

    for row in xr_rows:
        file_path = row["file_path"]
        folder_name = os.path.basename(os.path.dirname(file_path))
        run_id = run_cache[folder_name]

        cur.execute(
            """
            INSERT INTO xr_images (
                run_id,
                file_path,
                sample_id,
                membrane_id,
                position_label,
                horizontal_pos,
                vertical_pos,
                membrane_pos_cm,
                magnification,
                actual_temp_c,
                processed
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                file_path,
                row["sample_id"],
                row["membrane_id"],
                row["position_label"],
                row["horizontal_pos"],
                row["vertical_pos"],
                row["membrane_pos_cm"],
                row["magnification"],
                row["actual_temp"],
                row["processed"],
            ),
        )
        image_id = cur.lastrowid

        has_target = any(
            row[name] is not None
            for name in ("diameter", "density", "alignment", "curvature", "tortuosity")
        )
        if has_target:
            cur.execute(
                """
                INSERT INTO xr_targets (
                    image_id,
                    diameter,
                    density,
                    alignment,
                    curvature,
                    tortuosity,
                    label_source,
                    label_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    image_id,
                    row["diameter"],
                    row["density"],
                    row["alignment"],
                    row["curvature"],
                    row["tortuosity"],
                    "images_table",
                    "v1",
                ),
            )

    conn.commit()


def run_migration(source_db: Path, output_db: Path | None = None) -> Path:
    source_db = Path(source_db)
    target_db = source_db if output_db is None else Path(output_db)
    target_db.parent.mkdir(parents=True, exist_ok=True)

    # Write all SQLite transactions in a temp directory, then copy back.
    # This avoids environment-specific disk I/O issues on workspace drives.
    tmp_dir = Path(tempfile.mkdtemp(prefix="xr_schema_migrate_"))
    work_db = tmp_dir / target_db.name
    shutil.copy2(source_db, work_db)

    conn = sqlite3.connect(work_db)
    try:
        create_xr_tables(conn)
        migrate_xr_data(conn)
    finally:
        conn.close()

    shutil.copy2(work_db, target_db)
    return target_db


def main() -> None:
    parser = argparse.ArgumentParser(description="Create XR normalized schema and migrate data.")
    parser.add_argument(
        "--source-db",
        default=str(DEFAULT_DB_PATH),
        help="Path to source sqlite database.",
    )
    parser.add_argument(
        "--output-db",
        default=None,
        help="Optional path for migrated copy. If omitted, source DB is modified in-place.",
    )
    args = parser.parse_args()

    target_db = run_migration(Path(args.source_db), Path(args.output_db) if args.output_db else None)
    print(f"XR migration completed: {target_db}")


if __name__ == "__main__":
    main()
