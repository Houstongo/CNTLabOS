import sqlite3
from pathlib import Path

FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR if (FILE_DIR / "backend").exists() else FILE_DIR.parents[1]
DB_PATH = PROJECT_ROOT / "database" / "cnta_experiments.sqlite"
BACKEND_MAIN = PROJECT_ROOT / "backend" / "main.py"

# Check backend API
with BACKEND_MAIN.open('r', encoding='utf-8', errors='ignore') as f:
    content = f.read()

# Check DELETE and RESTORE endpoints
if '@app.put("/api/images/{image_id}/delete")' in content:
    print('[OK] DELETE endpoint added')
else:
    print('[FAIL] DELETE endpoint NOT found')

if '@app.put("/api/images/{image_id}/restore")' in content:
    print('[OK] RESTORE endpoint added')
else:
    print('[FAIL] RESTORE endpoint NOT found')

# Check is_deleted query logic
if 'COALESCE(is_deleted, 0) = 0' in content:
    print('[OK] Query logic updated (exclude is_deleted=1)')
else:
    print('[FAIL] Query logic NOT updated')

# Check is_deleted field
conn = sqlite3.connect(str(DB_PATH))
cursor = conn.cursor()
cursor.execute('PRAGMA table_info(images)')
columns = [row[1] for row in cursor.fetchall()]
if 'is_deleted' in columns:
    print('[OK] is_deleted field exists')
else:
    print('[FAIL] is_deleted field does NOT exist')

conn.close()

# Test query
conn = sqlite3.connect(str(DB_PATH))
cursor = conn.cursor()
cursor.execute('SELECT COUNT(*) FROM images WHERE file_path LIKE "%XR%" AND is_deleted = 1')
deleted_count = cursor.fetchone()[0]
cursor.execute('SELECT COUNT(*) FROM images WHERE file_path LIKE "%XR%"')
total_count = cursor.fetchone()[0]
conn.close()
print(f'XR total: {total_count}, deleted: {deleted_count}')
