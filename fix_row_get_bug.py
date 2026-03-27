"""
修复 _extract_relations_from_chunk 中的 sqlite3.Row.get() 访问错误

问题根源：
- sqlite3.Row 是命名组（namedtuple），不是字典
- 使用 row.get("xxx") 会导致 AttributeError: 'sqlite3.Row' object has no attribute 'get'
- 导致所有关系字段都返回 None，TCCER 无法进行多跳推理
"""

import sqlite3

# 连接数据库
db_path = r'd:\CNTDATA\CNTA_ML_Project\database\cnta_knowledge_base.sqlite'

# 测试数据
test_rows = {
    'doc_id': 5,
    'text': '测试文本',
    'relation_type': 'process_to_morphology',
    'source_node': '温度',
    'target_node': 'alignment',
    'process_factor': '温度',
    'morphology_factor': 'alignment',
    'confidence': 0.9
}

print('=== 1. 测试 SQLite3.Row.get() ===')

# 模拟 sqlite3.Row 对象的行为
class MockRow:
    def __getitem__(self, key):
        if key in test_rows:
            return test_rows[key]
        raise AttributeError(f"No such attribute: '{key}'")

mock_row = MockRow()

# 测试 1: 模拟 .get() 失败的情况
try:
    result = mock_row.get("source_node")
    print(f'mock_row.get(\"source_node\") 失败: {result}')  # 期望失败
except AttributeError as e:
    print(f'mock_row.get(\"source_node\") 失败: {e}')

# 测试 2: 模拟正确的情况
try:
    result = mock_row.source_node
    print(f'mock_row.source_node 成功: {result}')  # 应该成功
except AttributeError as e:
    print(f'mock_row.source_node 失败: {e}')

print()

print('=== 2. 测试 sqlite3.Row 对象 ===')

# 实际的 sqlite3.Row 对象
conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

# 查询一条测试数据
cursor.execute('SELECT id, doc_id, text FROM kb_chunks WHERE id = 5')
rows = cursor.fetchall()

if rows:
    actual_row = rows[0]
    print(f'实际_row 类型: {type(actual_row)}')  # 应该是 sqlite3.Row

    # 测试访问方式
    print('\\n测试不同的访问方式:')

    # 1. 方括号访问
    try:
        result = actual_row["source_node"]
        print(f'  actual_row[\"source_node\"]: {result}')
    except Exception as e:
        print(f'  actual_row[\"source_node\"]: 失败: {e}')

    # 2. getattr 访问
    try:
        result = getattr(actual_row, "source_node")
        print(f'getattr(actual_row, \"source_node\"): {result}')
    except Exception as e:
        print(f'getattr(actual_row, \"source_node\"): 失败: {e}')

    # 3. 直接属性访问
    try:
        result = actual_row.source_node
        print(f'actual_row.source_node: {result}')
    except Exception as e:
        print(f'actual_row.source_node: 失败: {e}')

    # 4. 字典包装
    try:
        result = dict(actual_row)["source_node"]
        print(f'dict(actual_row)[\"source_node\"]: {result}')
    except Exception as e:
        print(f'dict(actual_row)[\"source_node\"]: 失败: {e}')

    # 5. 使用索引
    try:
        result = actual_row[3]
        print(f'actual_row[3]: {result}')
    except Exception as e:
        print(f'actual_row[3]: 失败: {e}')

    cursor.close()
    conn.close()
"