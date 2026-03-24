import sqlite3
from pathlib import Path

FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR if (FILE_DIR / "backend").exists() else FILE_DIR.parents[1]

conn = sqlite3.connect(str(PROJECT_ROOT / 'database' / 'cnta_experiments.sqlite'))
cursor = conn.cursor()

# 检查XR数据的管径分布字段
cursor.execute('SELECT COUNT(*) FROM images WHERE file_path LIKE "%XR%" AND magnification >= 20000')
total_xr = cursor.fetchone()[0]

cursor.execute('SELECT COUNT(*) FROM images WHERE file_path LIKE "%XR%" AND magnification >= 20000 AND diameter_mean IS NOT NULL')
xr_with_dist = cursor.fetchone()[0]

print(f'XR high-mag images: {total_xr}')
print(f'XR with diameter distribution: {xr_with_dist}')

# 检查processed状态
cursor.execute('SELECT processed, COUNT(*) FROM images WHERE file_path LIKE "%XR%" GROUP BY processed')
print('\nXR processed status:')
for row in cursor.fetchall():
    print(f'  processed={row[0]}: {row[1]} images')

# 检查所有字段状态
cursor.execute('PRAGMA table_info(images)')
columns = [row[1] for row in cursor.fetchall()]
dist_cols = ['diameter_mean', 'diameter_std', 'diameter_min', 'diameter_max', 'diameter_p25', 'diameter_p50', 'diameter_p75']
print('\nDiameter distribution fields:')
for col in dist_cols:
    status = 'OK' if col in columns else 'MISSING'
    print(f'  [{status}] {col}')

conn.close()
