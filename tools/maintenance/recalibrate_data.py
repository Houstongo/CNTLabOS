import os
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

# Add project root to sys.path.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from backend.core.calibrator import calibrator
from backend.core.data_manager import CNTADataParser

DB_PATH = r'd:\CNTDATA\CNTA_ML_Project\database\cnta_experiments.sqlite'
parser = CNTADataParser()


def migrate_db(db_path: str = DB_PATH):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    columns = [
        ("actual_temp", "REAL"),
        ("membrane_pos_cm", "REAL"),
    ]

    for col_name, col_type in columns:
        try:
            cursor.execute(f"ALTER TABLE images ADD COLUMN {col_name} {col_type}")
            print(f"Added column: {col_name}")
        except sqlite3.OperationalError:
            pass

    conn.commit()
    conn.close()


def recalibrate(db_path: str = DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM images")
    rows = cursor.fetchall()

    print(f"Recalibrating {len(rows)} records using DataCalibrator...")

    for row in rows:
        data = dict(row)

        if data.get("source") == "XR" and data.get("file_path"):
            folder_name = os.path.basename(os.path.dirname(data["file_path"]))
            data.update(parser.parse_xr_folder_metadata(folder_name))

        data["actual_temp"] = None
        data["membrane_pos_cm"] = None
        calibrated = calibrator.calibrate(data)

        cursor.execute(
            """
            UPDATE images
            SET growth_temp = ?,
                ar_flow = ?,
                catalyst_weight = ?,
                actual_temp = ?,
                membrane_pos_cm = ?
            WHERE id = ?
            """,
            (
                calibrated.get("growth_temp"),
                calibrated.get("ar_flow"),
                calibrated.get("catalyst_weight"),
                calibrated.get("actual_temp"),
                calibrated.get("membrane_pos_cm"),
                data["id"],
            ),
        )

    conn.commit()
    conn.close()
    print("Recalibration complete.")


def run_recalibration(source_db: str = DB_PATH, output_db: str | None = None) -> str:
    source = Path(source_db)
    target = source if output_db is None else Path(output_db)
    target.parent.mkdir(parents=True, exist_ok=True)

    # Run SQLite write operations in temp directory to avoid drive-specific I/O issues.
    tmp_dir = Path(tempfile.mkdtemp(prefix="xr_recalibrate_"))
    work_db = tmp_dir / source.name
    shutil.copy2(source, work_db)

    migrate_db(str(work_db))
    recalibrate(str(work_db))
    shutil.copy2(work_db, target)
    return str(target)


if __name__ == "__main__":
    out = run_recalibration()
    print(f"Recalibrated DB written to: {out}")
