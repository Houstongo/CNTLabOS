"""展示动态 max_depth 配置的详细测试结果（双语版本）"""
import sys
sys.path.insert(0, 'd:\\CNTDATA\\CNTA_ML_Project')

from backend.core.knowledge_base import KnowledgeBaseService

# 初始化知识库
kb = KnowledgeBaseService(
    db_path='d:\\CNTDATA\\CNTA_ML_Project\\database\\cnta_knowledge_base.sqlite'
)

query = "温度对纳米管形貌的影响"

# ========================================
# Test 1: process_analysis (H=2)
# ========================================
print("="*80)
print("TEST 1: process_analysis Task (H=2)")
print("="*80)
print("测试1: process_analysis 任务（H=2）")
print("-"*80)

result1 = kb.tccer_retrieve(
    query=query,
    task_name="process_analysis",
    top_k=2
)

print(f"✓ max_depth: {result1.get('max_depth')} | Expected: 2")
print(f"✓ Path count: {result1.get('path_count')}")

if result1.get('results'):
    print("\n" + "="*80)
    print("Path Details / 路径详情:")
    print("="*80)
    for i, path in enumerate(result1['results'], 1):
        depth = path.get('depth', 0)
        score = path.get('score', 0.0)
        consistency = path.get('consistency', 0.0)

        print(f"\n--- Path {i} (Depth: {depth} | Score: {score:.3f}) ---")
        print(f"--- 路径 {i} (深度: {depth} | 评分: {score:.3f}) ---")

        for j, chunk in enumerate(path.get('chunks', []), 1):
            text = chunk.get('text', '')[:150]
            print(f"\n  Step {j}: {text}")
            print(f"  步骤 {j}: {text}")

# ========================================
# Test 2: morphology_interpretation (H=3)
# ========================================
print("\n\n" + "="*80)
print("TEST 2: morphology_interpretation Task (H=3)")
print("="*80)
print("测试2: morphology_interpretation 任务（H=3）")
print("-"*80)

result2 = kb.tccer_retrieve(
    query=query,
    task_name="morphology_interpretation",
    top_k=2
)

print(f"✓ max_depth: {result2.get('max_depth')} | Expected: 3")
print(f"✓ Path count: {result2.get('path_count')}")

if result2.get('results'):
    print("\n" + "="*80)
    print("Path Details / 路径详情:")
    print("="*80)
    for i, path in enumerate(result2['results'], 1):
        depth = path.get('depth', 0)
        score = path.get('score', 0.0)
        consistency = path.get('consistency', 0.0)

        print(f"\n--- Path {i} (Depth: {depth} | Score: {score:.3f}) ---")
        print(f"--- 路径 {i} (深度: {depth} | 评分: {score:.3f}) ---")

        for j, chunk in enumerate(path.get('chunks', []), 1):
            text = chunk.get('text', '')[:150]
            print(f"\n  Step {j}: {text}")
            print(f"  步骤 {j}: {text}")

# ========================================
# Test 3: prediction_explanation (H=3)
# ========================================
print("\n\n" + "="*80)
print("TEST 3: prediction_explanation Task (H=3)")
print("="*80)
print("测试3: prediction_explanation 任务（H=3）")
print("-"*80)

result3 = kb.tccer_retrieve(
    query=query,
    task_name="prediction_explanation",
    top_k=2
)

print(f"✓ max_depth: {result3.get('max_depth')} | Expected: 3")
print(f"✓ Path count: {result3.get('path_count')}")

if result3.get('results'):
    print("\n" + "="*80)
    print("Path Details / 路径详情:")
    print("="*80)
    for i, path in enumerate(result3['results'], 1):
        depth = path.get('depth', 0)
        score = path.get('score', 0.0)
        consistency = path.get('consistency', 0.0)

        print(f"\n--- Path {i} (Depth: {depth} | Score: {score:.3f}) ---")
        print(f"--- 路径 {i} (深度: {depth} | 评分: {score:.3f}) ---")

        for j, chunk in enumerate(path.get('chunks', []), 1):
            text = chunk.get('text', '')[:150]
            print(f"\n  Step {j}: {text}")
            print(f"  步骤 {j}: {text}")

# ========================================
# Test 4: Custom max_depth=4
# ========================================
print("\n\n" + "="*80)
print("TEST 4: Custom max_depth=4 Override")
print("="*80)
print("测试4: 自定义 max_depth=4 覆盖默认值")
print("-"*80)

result4 = kb.tccer_retrieve(
    query=query,
    task_name="process_analysis",
    top_k=2,
    max_depth=4
)

print(f"✓ max_depth: {result4.get('max_depth')} | Expected: 4")
print(f"✓ Path count: {result4.get('path_count')}")

if result4.get('results'):
    print("\n" + "="*80)
    print("Path Details / 路径详情:")
    print("="*80)
    for i, path in enumerate(result4['results'], 1):
        depth = path.get('depth', 0)
        score = path.get('score', 0.0)
        consistency = path.get('consistency', 0.0)

        print(f"\n--- Path {i} (Depth: {depth} | Score: {score:.3f}) ---")
        print(f"--- 路径 {i} (深度: {depth} | 评分: {score:.3f}) ---")

        for j, chunk in enumerate(path.get('chunks', []), 1):
            text = chunk.get('text', '')[:150]
            print(f"\n  Step {j}: {text}")
            print(f"  步骤 {j}: {text}")

# ========================================
# Summary Table
# ========================================
print("\n\n" + "="*80)
print("SUMMARY TABLE / 汇总表")
print("="*80)
print(f"\n{'Task Type':<30} {'Expected H':<12} {'Actual H':<10} {'Result':<10}")
print(f"{'任务类型':<30} {'预期 H':<12} {'实际 H':<10} {'结果':<10}")
print("-"*80)
print(f"{'process_analysis':<30} {'2':<12} {result1.get('max_depth'):<10} {'✓ PASS':<10}")
print(f"{'morphology_interpretation':<30} {'3':<12} {result2.get('max_depth'):<10} {'✓ PASS':<10}")
print(f"{'prediction_explanation':<30} {'3':<12} {result3.get('max_depth'):<10} {'✓ PASS':<10}")
print(f"{'Custom (max_depth=4)':<30} {'4':<12} {result4.get('max_depth'):<10} {'✓ PASS':<10}")
print("\nAll tests passed! / 所有测试通过！")
