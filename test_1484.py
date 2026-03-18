import sqlite3

conn = sqlite3.connect('database/cnta_experiments.sqlite')
conn.row_factory = sqlite3.Row
cursor = conn.cursor()
cursor.execute('SELECT file_path, magnification FROM images WHERE id = 1484')
result = cursor.fetchone()
conn.close()
print(dict(result))
