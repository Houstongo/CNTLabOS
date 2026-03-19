"""添加 is_deleted 字段用于逻辑删除"""
import sqlite3

DB_PATH = r"d:\CNTDATA\CNTA_ML_Project\database\cnta_experiments.sqlite"

def add_is_deleted_field():
    """添加is_deleted字段"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    print("=" * 70)
    print("添加 is_deleted 字段")
    print("=" * 70)

    try:
        cursor.execute("""
            ALTER TABLE images ADD COLUMN is_deleted INTEGER DEFAULT 0
        """)
        print("  添加列: is_deleted (INTEGER DEFAULT 0)")
        conn.commit()
        print("  数据库修改完成")
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e).lower():
            print("  列 is_deleted 已存在，跳过")
        else:
            print(f"  错误: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    add_is_deleted_field()
