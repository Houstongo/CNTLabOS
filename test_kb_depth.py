"""
测试KB LINK深度扩展是否正确工作
验证max_depth参数是否生效
"""
import sys
import os

# 添加backend路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

from backend.core.knowledge_base import KnowledgeBaseService


def test_depth_expansion():
    """测试深度扩展功能"""
    print("=" * 60)
    print("测试KB LINK深度扩展功能")
    print("=" * 60)

    # 初始化知识库
    kb_path = "database/cnta_knowledge_base.sqlite"
    if not os.path.exists(kb_path):
        print(f"错误：知识库数据库不存在: {kb_path}")
        return

    kb = KnowledgeBaseService(kb_path)

    # 测试1: 验证max_depth=2能否扩展到深度2
    print("\n测试1: max_depth=2")
    print("-" * 60)
    query = "温度对直径的影响"
    result = kb.tccer_retrieve(
        query=query,
        task_name="morphology_interpretation",
        top_k=3,
        max_depth=2
    )

    print(f"查询: {query}")
    print(f"找到路径数: {result.get('path_count', 0)}")
    print("\n路径详情:")
    for i, path in enumerate(result.get('results', []), 1):
        depth = path.get('depth', 0)
        chunks = path.get('chunks', [])
        relations = path.get('relations', [])
        score = path.get('score', 0)

        print(f"\n路径 {i}:")
        print(f"  深度: {depth}")
        print(f"  评分: {score:.3f}")
        print(f"  chunk数: {len(chunks)}")
        print(f"  关系数: {len(relations)}")

        for j, chunk in enumerate(chunks[:3]):  # 只显示前3个chunk
            chunk_text = chunk.get('text', '')[:100]
            chunk_score = chunk.get('score', 0)
            print(f"    Chunk {j+1}: [score={chunk_score:.3f}] {chunk_text}...")

        if relations:
            print(f"  关系链:")
            for k, rel in enumerate(relations[:5]):  # 只显示前5个关系
                rel_type = rel.get('type', 'unknown')
                proc = rel.get('process_factor', '')
                morph = rel.get('morphology_factor', '')
                direction = rel.get('effect_direction', '')
                print(f"    {k+1}. {rel_type}: {proc} → {morph} [{direction}]")

    # 检查是否成功扩展到深度2
    max_achieved_depth = 0
    for path in result.get('results', []):
        max_achieved_depth = max(max_achieved_depth, path.get('depth', 0))

    if max_achieved_depth >= 2:
        print(f"\n✅ 成功扩展到深度 {max_achieved_depth}")
    else:
        print(f"\n❌ 扩展失败，最大深度只有 {max_achieved_depth}")

    # 测试2: 验证max_depth=3能否扩展到深度3
    print("\n" + "=" * 60)
    print("测试2: max_depth=3")
    print("-" * 60)
    result3 = kb.tccer_retrieve(
        query=query,
        task_name="process_analysis",
        top_k=2,
        max_depth=3
    )

    print(f"查询: {query}")
    print(f"找到路径数: {result3.get('path_count', 0)}")

    # 检查是否成功扩展到深度3
    max_achieved_depth3 = 0
    for path in result3.get('results', []):
        max_achieved_depth3 = max(max_achieved_depth3, path.get('depth', 0))

        depth = path.get('depth', 0)
        print(f"路径深度: {depth}")

    if max_achieved_depth3 >= 3:
        print(f"\n✅ 成功扩展到深度 {max_achieved_depth3}")
    else:
        print(f"\n❌ 扩展失败，最大深度只有 {max_achieved_depth3}")

    # 测试3: 验证三层关系链 Process → Mechanism → Morphology
    print("\n" + "=" * 60)
    print("测试3: 三层关系链构建")
    print("-" * 60)
    query3 = "生长温度通过机理影响取向"
    result_chain = kb.tccer_retrieve(
        query=query3,
        task_name="process_analysis",
        top_k=1,
        max_depth=3
    )

    print(f"查询: {query3}")
    print(f"路径数: {result_chain.get('path_count', 0)}")

    has_process_to_mechanism = False
    has_mechanism_to_morphology = False

    for i, path in enumerate(result_chain.get('results', []), 1):
        print(f"\n路径 {i}:")
        relations = path.get('relations', [])
        print(f"  关系数: {len(relations)}")

        for j, rel in enumerate(relations):
            rel_type = rel.get('type', '')
            print(f"  {j+1}. {rel_type}")

            if rel_type == "process_to_mechanism":
                has_process_to_mechanism = True
            if rel_type == "mechanism_to_morphology":
                has_mechanism_to_morphology = True

    if has_process_to_mechanism and has_mechanism_to_morphology:
        print(f"\n✅ 成功构建三层关系链: Process → Mechanism → Morphology")
    else:
        print(f"\n❌ 三层关系链构建失败")
        print(f"   - process_to_mechanism: {'✅' if has_process_to_mechanism else '❌'}")
        print(f"   - mechanism_to_morphology: {'✅' if has_mechanism_to_morphology else '❌'}")

    # 统计信息
    print("\n" + "=" * 60)
    print("统计摘要")
    print("=" * 60)

    stats = kb.get_stats()
    print(f"文档数: {stats['document_count']}")
    print(f"chunk数: {stats['chunk_count']}")
    print(f"link数: {stats['link_count']}")
    print(f"\n关系类型分布:")
    for rel_type, count in stats['relation_counts'].items():
        print(f"  {rel_type}: {count}")

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == "__main__":
    test_depth_expansion()
