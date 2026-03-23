import sqlite3

conn = sqlite3.connect('database/cnta_knowledge_base.sqlite')
c = conn.cursor()

print("=" * 60)
print("MSFU 提取结果对比")
print("=" * 60)

# 总体统计
c.execute('SELECT COUNT(*) FROM kb_msfu')
total = c.fetchone()[0]

c.execute('SELECT AVG(confidence) FROM kb_msfu')
avg_conf = c.fetchone()[0] or 0

print(f"\n总数: {total}, 平均置信度: {avg_conf:.3f}")

# 按提取方法统计
print("\n按提取方法:")
c.execute('SELECT extraction_method, COUNT(*) FROM kb_msfu GROUP BY extraction_method')
for row in c.fetchall():
    print(f"  {row[0]}: {row[1]}")

# 实体质量检查
print("\n实体格式检查:")
c.execute('''
    SELECT COUNT(*) FROM kb_msfu
    WHERE source_entity LIKE '%:%' AND target_entity LIKE '%:%'
''')
valid_format = c.fetchone()[0]
print(f"  格式正确 (category:entity): {valid_format}/{total} ({100*valid_format/total:.1f}%)")

# 查看高质量示例
print("\n" + "=" * 60)
print("高质量 MSFU 示例 (置信度 > 0.8)")
print("=" * 60)
c.execute('''
    SELECT source_entity, relation_type, target_entity, direction, confidence
    FROM kb_msfu
    WHERE confidence > 0.8
    AND source_entity LIKE '%:%'
    AND target_entity LIKE '%:%'
    ORDER BY confidence DESC
    LIMIT 10
''')
for i, row in enumerate(c.fetchall(), 1):
    src, rel, tgt, dir_, conf = row
    print(f"{i}. {src} --[{rel}]--> {tgt}")
    print(f"   方向: {dir_}, 置信度: {conf:.2f}")

conn.close()
