"""Add waviness-related columns to the images table."""

import sqlite3


DB_PATH = r"d:\CNTDATA\CNTA_ML_Project\database\cnta_experiments.sqlite"
NEW_COLUMNS = [
    ("waviness_ratio", "REAL"),
    ("waviness_height_nm", "REAL"),
    ("waviness_wavelength_nm", "REAL"),
    ("waviness_branches", "INTEGER"),
]


def add_waviness_fields(db_path: str = DB_PATH):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    print("=" * 70)
    print("添加 waviness 相关字段")
    print("=" * 70)

    existing_columns = {
        row[1]
        for row in cursor.execute("PRAGMA table_info(images)").fetchall()
    }

    for column_name, column_type in NEW_COLUMNS:
        if column_name in existing_columns:
            print(f"  列 {column_name} 已存在，跳过")
            continue
        cursor.execute(f"ALTER TABLE images ADD COLUMN {column_name} {column_type}")
        print(f"  添加列 {column_name} ({column_type})")

    conn.commit()
    conn.close()
    print("数据库修改完成")


if __name__ == "__main__":
    add_waviness_fields()
