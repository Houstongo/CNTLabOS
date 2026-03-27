"""测试根据任务类型动态设置 max_depth 的功能"""
import sys
sys.path.insert(0, 'd:\\CNTDATA\\CNTA_ML_Project')

from backend.core.knowledge_base import KnowledgeBaseService

# 初始化知识库
kb = KnowledgeBaseService(
    db_path='d:\\CNTDATA\\CNTA_ML_Project\\database\\cnta_knowledge_base.sqlite'
)

query = "温度对纳米管形貌的影响"

# 测试1: process_analysis 应该使用 H=2
print("\n测试1: process_analysis 任务（预期 H=2）")
print("=" * 60)
result1 = kb.tccer_retrieve(
    query=query,
    task_name="process_analysis",
    top_k=3
)
print(f"max_depth: {result1.get('max_depth')}")
print(f"路径数量: {result1.get('path_count')}")
if result1.get('results'):
    max_actual_depth = max(r.get('depth', 0) for r in result1['results'])
    print(f"实际最大深度: {max_actual_depth}")

# 测试2: morphology_interpretation 应该使用 H=3
print("\n测试2: morphology_interpretation 任务（预期 H=3）")
print("=" * 60)
result2 = kb.tccer_retrieve(
    query=query,
    task_name="morphology_interpretation",
    top_k=3
)
print(f"max_depth: {result2.get('max_depth')}")
print(f"路径数量: {result2.get('path_count')}")
if result2.get('results'):
    max_actual_depth = max(r.get('depth', 0) for r in result2['results'])
    print(f"实际最大深度: {max_actual_depth}")

# 测试3: prediction_explanation 应该使用 H=3
print("\n测试3: prediction_explanation 任务（预期 H=3）")
print("=" * 60)
result3 = kb.tccer_retrieve(
    query=query,
    task_name="prediction_explanation",
    top_k=3
)
print(f"max_depth: {result3.get('max_depth')}")
print(f"路径数量: {result3.get('path_count')}")
if result3.get('results'):
    max_actual_depth = max(r.get('depth', 0) for r in result3['results'])
    print(f"实际最大深度: {max_actual_depth}")

# 测试4: 传入自定义 max_depth 应该覆盖默认值
print("\n测试4: 传入自定义 max_depth=4（应该覆盖默认值）")
print("=" * 60)
result4 = kb.tccer_retrieve(
    query=query,
    task_name="process_analysis",
    top_k=3,
    max_depth=4
)
print(f"max_depth: {result4.get('max_depth')}")

print("\n测试完成！")
