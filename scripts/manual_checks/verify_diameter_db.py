import sqlite3
from pathlib import Path

FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR if (FILE_DIR / "backend").exists() else FILE_DIR.parents[1]

conn = sqlite3.connect(str(PROJECT_ROOT / 'database' / 'cnta_experiments.sqlite'))
cursor = conn.cursor()

cursor.execute('SELECT diameter_mean, diameter_std, diameter_min, diameter_max, diameter_p50 FROM images WHERE file_path LIKE "%ZZY%" AND magnification >= 20000 LIMIT 3')
print('管径分布数据样本:')
for row in cursor.fetchall():
    print(f'  mean={row[0]:.2f}, std={row[1]:.2f}, min={row[2]:.2f}, max={row[3]:.2f}, p50={row[4]:.2f}')

conn.close()
