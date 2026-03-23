import sqlite3
import os
import sys

# 强制 UTF-8 输出
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

kb_path = 'database/cnta_knowledge_base.sqlite'
conn = sqlite3.connect(kb_path)
cursor = conn.cursor()

# 统计
cursor.execute('SELECT COUNT(*) FROM kb_msfu')
msfu_count = cursor.fetchone()[0]

cursor.execute('SELECT COUNT(DISTINCT doc_id) FROM kb_msfu')
doc_count = cursor.fetchone()[0]

cursor.execute('SELECT AVG(confidence) FROM kb_msfu')
avg_conf = cursor.fetchone()[0] or 0

print('=' * 60)
print('MSFU 提取统计 (规则方法)')
print('=' * 60)
print(f'MSFU 总数: {msfu_count}')
print(f'涉及文档: {doc_count}')
print(f'平均置信度: {avg_conf:.3f}')

# 按关系类型统计
print('\n按关系类型:')
cursor.execute('SELECT relation_type, COUNT(*) as cnt FROM kb_msfu GROUP BY relation_type ORDER BY cnt DESC')
for row in cursor.fetchall():
    print(f'  {row[0]}: {row[1]}')

# 按方向统计
print('\n按方向:')
cursor.execute('SELECT direction, COUNT(*) as cnt FROM kb_msfu GROUP BY direction ORDER BY cnt DESC')
for row in cursor.fetchall():
    print(f'  {row[0]}: {row[1]}')

# 查看示例
print('\n' + '=' * 60)
print('示例 MSFU (Top 5 by confidence)')
print('=' * 60)
cursor.execute('''
    SELECT source_entity, relation_type, target_entity, direction, confidence,
           substr(content, 1, 150) as content
    FROM kb_msfu
    ORDER BY confidence DESC
    LIMIT 5
''')
for i, row in enumerate(cursor.fetchall(), 1):
    src, rel, tgt, dir_, conf, content = row
    print(f'\n[{i}] {src} --[{rel}]--> {tgt}')
    print(f'    方向: {dir_}, 置信度: {conf:.2f}')
    print(f'    内容: {content}...')

conn.close()
