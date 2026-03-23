import sqlite3
import os

kb_path = 'database/cnta_knowledge_base.sqlite'
conn = sqlite3.connect(kb_path)
cursor = conn.cursor()

# 检查 kb_msfu 表是否存在
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='kb_msfu'")
if not cursor.fetchone():
    print('kb_msfu 表不存在')
    exit()

# 统计
cursor.execute('SELECT COUNT(*) FROM kb_msfu')
msfu_count = cursor.fetchone()[0]

cursor.execute('SELECT COUNT(DISTINCT doc_id) FROM kb_msfu')
doc_count = cursor.fetchone()[0]

cursor.execute('SELECT AVG(confidence) FROM kb_msfu')
avg_conf = cursor.fetchone()[0] or 0

print(f'MSFU 总数: {msfu_count}')
print(f'涉及文档: {doc_count}')
print(f'平均置信度: {avg_conf:.3f}')

# 查看几个示例
print()
print('=== 示例 MSFU ===')
cursor.execute('''
    SELECT source_entity, relation_type, target_entity, direction, confidence, content
    FROM kb_msfu
    ORDER BY confidence DESC
    LIMIT 5
''')
for row in cursor.fetchall():
    src, rel, tgt, dir_, conf, content = row
    print(f'{src} --[{rel}]--> {tgt} ({dir_}, conf={conf:.2f})')
    print(f'  内容: {content[:100]}...')
    print()

conn.close()
