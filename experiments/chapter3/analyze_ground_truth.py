import json

# 读取专家标注数据集
with open('dataset/ground_truth.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 元信息
meta = data['metadata']
print('=' * 80)
print('专家标注数据集分析')
print('=' * 80)

print(f"\n总查询数: {meta['total_queries']}")
print(f"任务类型: {meta['task_types']}")
print(f"标注方法: {meta.get('annotation_method', 'N/A')}")
print(f"标注日期: {meta.get('annotation_date', 'N/A')}")

# 统计标注记录
annotations = data['annotations']
print(f"\n标注记录数: {len(annotations)}")

# 统计相关路径
rel_count = sum(a['relevant_paths'] for a in annotations)
avg_rel = rel_count / len(annotations)
print(f"相关路径总数: {rel_count}")
print(f"平均相关路径数: {avg_rel:.1f}")

# 任务类型分布
print(f"\n任务类型分布:")
task_counts = {}
for ann in annotations:
    task_type = ann['task_type']
    task_counts[task_type] = task_counts.get(task_type, 0) + ann['relevant_paths']

for task, count in task_counts.items():
    print(f"  {task}: {count} 个相关路径")

# 质量/完整性统计
print(f"\n查询完整性:")
no_rel_queries = [a['query'] for a in annotations if a['relevant_paths'] == 0]
print(f"  无相关路径的查询: {len(no_rel_queries)}")
if no_rel_queries:
    print(f"  示例: {no_rel_queries[0]}")

print(f"\n高相关性查询 (>2个相关路径):")
high_rel = [a for a in annotations if a['relevant_paths'] >= 2]
print(f"  数量: {len(high_rel)}")
if high_rel:
    print(f"  示例: {high_rel[0]['query']} ({high_rel[0]['relevant_paths']} 个相关路径)")

print('\n' + '=' * 80)
print('分析完成')
print('=' * 80)
