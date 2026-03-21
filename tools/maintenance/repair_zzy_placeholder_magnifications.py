from __future__ import annotations

import sqlite3
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

from backend.core.data_manager import CNTADataParser
from backend.core.sem_magnification import build_zzy_filename_with_magnification
from tools.maintenance.db_mag_updater import extract_mag


DB_PATH = Path(r"D:\CNTDATA\CNTA_ML_Project\database\cnta_experiments.sqlite")
ZZY_ROOT = Path(r"D:\CNTDATA\ZZY")
TARGET_FOLDERS = (
    "20260318No34不同厚度1.25-2.25",
    "20260318No35不同厚度1.5-2.75",
    "20260319No39调流速全样品·0.75-2.75 5-20w",
    "20260319No40调流速间隔20s全样品·0.75-2.75 5w",
)


def _normalize_path_key(path: Path | str) -> str:
    return str(path).lower()


def collect_repairs() -> Tuple[List[Dict[str, object]], List[Path]]:
    parser = CNTADataParser()
    planned: List[Dict[str, object]] = []
    skipped: List[Path] = []

    for folder_name in TARGET_FOLDERS:
        folder = ZZY_ROOT / folder_name
        if not folder.exists():
            continue

        for image_path in sorted(folder.glob("*.png")):
            parsed = parser.parse_zzy_filename(image_path.name)
            if not parsed:
                continue
            if not parser.is_zzy_mid_position(str(parsed.get("position_label", ""))):
                continue

            actual_mag = extract_mag(str(image_path))
            if actual_mag is None:
                skipped.append(image_path)
                continue

            repaired_path = build_zzy_filename_with_magnification(image_path, int(actual_mag))
            repaired_parsed = parser.parse_zzy_filename(repaired_path.name)
            if not repaired_parsed:
                raise ValueError(f"Repaired name is no longer parseable: {repaired_path.name}")

            planned.append(
                {
                    "folder": folder_name,
                    "old_path": image_path,
                    "new_path": repaired_path,
                    "old_mag": int(parsed["magnification"]),
                    "new_mag": int(actual_mag),
                    "new_sample_id": repaired_parsed["sample_id"],
                    "restore": parser.should_include_zzy_record(repaired_parsed),
                }
            )

    return planned, skipped


def print_summary(planned: List[Dict[str, object]], skipped: List[Path]) -> None:
    by_folder = Counter()
    restore_by_folder = Counter()
    changes_by_mag = Counter()

    for item in planned:
        folder = str(item["folder"])
        by_folder[folder] += 1
        if item["restore"]:
            restore_by_folder[folder] += 1
        changes_by_mag[(int(item["old_mag"]), int(item["new_mag"]))] += 1

    print("Planned file repairs:")
    for folder in TARGET_FOLDERS:
        if folder not in by_folder:
            continue
        print(
            f"  {folder}: total_mid={by_folder[folder]} "
            f"restore_after_fix={restore_by_folder[folder]}"
        )

    print("Magnification transitions:")
    for (old_mag, new_mag), count in sorted(changes_by_mag.items()):
        print(f"  {old_mag} -> {new_mag}: {count}")
    if skipped:
        print("Skipped files without recoverable magnification:")
        for path in skipped:
            print(f"  {path}")


def apply_repairs(planned: List[Dict[str, object]]) -> None:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    renamed_pairs = []

    try:
        cursor.execute("BEGIN")

        for item in planned:
            old_path = Path(item["old_path"])
            new_path = Path(item["new_path"])
            if old_path != new_path:
                if new_path.exists():
                    raise FileExistsError(f"Target path already exists: {new_path}")
                old_path.rename(new_path)
                renamed_pairs.append((new_path, old_path))

            cursor.execute(
                """
                UPDATE images
                SET file_path = ?, sample_id = ?, magnification = ?, is_deleted = ?
                WHERE source = 'ZZY' AND lower(file_path) = lower(?)
                """,
                (
                    str(new_path),
                    str(item["new_sample_id"]),
                    int(item["new_mag"]),
                    0 if bool(item["restore"]) else 1,
                    str(old_path),
                ),
            )

        conn.commit()
    except Exception:
        conn.rollback()
        for new_path, old_path in reversed(renamed_pairs):
            if new_path.exists() and not old_path.exists():
                new_path.rename(old_path)
        raise
    finally:
        conn.close()


def main() -> None:
    planned, skipped = collect_repairs()
    print_summary(planned, skipped)
    apply_repairs(planned)
    restored = sum(1 for item in planned if item["restore"])
    print(
        f"Applied {len(planned)} repairs; restored {restored} rows that now meet the >=9000 rule; "
        f"skipped {len(skipped)} files."
    )


if __name__ == "__main__":
    main()
