"""
知识库技术加固现状诊断报告
"""
import sqlite3
import os

def diagnose_knowledge_base():
    db_path = 'database/cnta_knowledge_base.sqlite'

    if not os.path.exists(db_path):
        print(f'知识库数据库不存在: {db_path}')
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    print('知识库技术加固现状诊断报告')
    print('=' * 60)

    # 基础统计
    stats = cursor.execute('SELECT COUNT(*) as docs FROM kb_documents').fetchone()
    chunks = cursor.execute('SELECT COUNT(*) as c FROM kb_chunks').fetchone()
    links = cursor.execute('SELECT COUNT(*) as l FROM kb_links').fetchone()
    msfu = cursor.execute('SELECT COUNT(*) as m FROM kb_msfu').fetchone()

    print(f'\n基础统计:')
    print(f'  文档数: {stats[0]}')
    print(f'  文本块数: {chunks[0]}')
    print(f'  关联链接数: {links[0]}')
    print(f'  MSFU断言数: {msfu[0]}')

    # 关系类型分布
    rel_types = cursor.execute('''
        SELECT relation_type, COUNT(*) as c
        FROM kb_links
        GROUP BY relation_type
        ORDER BY c DESC
    ''').fetchall()

    print(f'\n关系类型分布:')
    for row in rel_types:
        print(f'  - {row[0]}: {row[1]}')

    # MSFU数据完整性
    print(f'\nMSFU数据完整性检查:')
    print('-' * 40)

    conditions = cursor.execute('''
        SELECT condition_param, condition_op, COUNT(*)
        FROM kb_msfu
        WHERE condition_param IS NOT NULL
        GROUP BY condition_param, condition_op
        ORDER BY COUNT(*) DESC
        LIMIT 10
    ''').fetchall()

    if conditions:
        print('\n条件约束数据 (Top 10):')
        for row in conditions:
            param = row[0]
            op = row[1]
            count = row[2]
            print(f'  {param} {op}: {count} 条')

    directions = cursor.execute('''
        SELECT direction, COUNT(*)
        FROM kb_msfu
        WHERE direction IS NOT NULL
        GROUP BY direction
        ORDER BY COUNT(*) DESC
    ''').fetchall()

    if directions:
        print('\n影响方向数据:')
        for row in directions:
            print(f'  {row[0]}: {row[1]} 条')

    # 提取方法
    methods = cursor.execute('''
        SELECT extraction_method, COUNT(*)
        FROM kb_msfu
        GROUP BY extraction_method
    ''').fetchall()

    if methods:
        print('\n提取方法分布:')
        for row in methods:
            print(f'  {row[0]}: {row[1]} 条')

    # 置信度分布
    conf_dist = cursor.execute('''
        SELECT
            CASE
                WHEN confidence >= 0.8 THEN 'high'
                WHEN confidence >= 0.6 THEN 'medium'
                WHEN confidence >= 0.4 THEN 'low'
                ELSE 'very_low'
            END as level,
            COUNT(*) as count
        FROM kb_msfu
        GROUP BY level
        ORDER BY level
    ''').fetchall()

    if conf_dist:
        print('\n置信度分布:')
        for row in conf_dist:
            print(f'  {row[0]}: {row[1]} 条')

    # 高质量MSFU示例
    print(f'\n高质量MSFU示例 (confidence > 0.6):')
    print('-' * 40)

    examples = cursor.execute('''
        SELECT source_entity, relation_type, target_entity, direction, condition_param, condition_op, content, confidence
        FROM kb_msfu
        WHERE confidence > 0.6
        ORDER BY RANDOM()
        LIMIT 5
    ''').fetchall()

    if examples:
        for i, row in enumerate(examples, 1):
            print(f'\n示例 {i}:')
            print(f'  实体: {row[0]} -> {row[2]}')
            print(f'  关系: {row[1]}')
            print(f'  方向: {row[3]}')
            if row[4]:
                print(f'  条件: {row[4]} {row[5]} {row[6]}')
            print(f'  置信度: {row[7]:.2f}')
            print(f'  内容: {row[8][:150]}...')

    # 检查最近导入的文档内容质量
    print(f'\n最近导入的文档示例 (Top 3):')
    print('-' * 40)

    recent_docs = cursor.execute('''
        SELECT d.title, c.text
        FROM kb_documents d
        LEFT JOIN kb_chunks c ON c.doc_id = d.id
        ORDER BY d.id DESC
        LIMIT 3
    ''').fetchall()

    if recent_docs:
        for i, row in enumerate(recent_docs, 1):
            print(f'\n文档 {i}: {row[0]}')
            print(f'  内容: {row[1][:200]}...')

    conn.close()
    print('\n' + '=' * 60)

if __name__ == '__main__':
    diagnose_knowledge_base()
