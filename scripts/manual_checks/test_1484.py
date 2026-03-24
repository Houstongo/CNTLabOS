import sqlite3
from pathlib import Path

FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR if (FILE_DIR / "backend").exists() else FILE_DIR.parents[1]

conn = sqlite3.connect(str(PROJECT_ROOT / 'database' / 'cnta_experiments.sqlite'))
conn.row_factory = sqlite3.Row
cursor = conn.cursor()
cursor.execute('SELECT file_path, magnification FROM images WHERE id = 1484')
result = cursor.fetchone()
conn.close()
print(dict(result))
