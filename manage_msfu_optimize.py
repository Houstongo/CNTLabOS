"""
MSFU提取优化工具
分析当前MSFU提取效果，并提供优化建议
"""

import sqlite3
import sys
from collections import Counter

sys.path.insert(0, '.')

from backend.core.msfu_extractor import get_msfu_stats


def analyze_current_state(kb_path: str):
    """分析当前MSFU提取状态"""
    print("="*60)
    print("MSFU提取状态分析")
    print("="*60)

    conn = sqlite3.connect(kb_path)

    # 基础统计
    total_docs = conn.execute("SELECT COUNT(*) FROM kb_documents").fetchone()[0]
    total_chunks = conn.execute("SELECT COUNT(*) FROM kb_chunks").fetchone()[0]
    total_msfus = conn.execute("SELECT COUNT(*) FROM kb_msfu").fetchone()[0]
    docs_with_msfu = conn.execute("SELECT COUNT(DISTINCT doc_id) FROM kb_msfu").fetchone()[0]

    print(f"\n基础统计:")
    print(f"  总文献数: {total_docs}")
    print(f"  总文本块数: {total_chunks}")
    print(f"  总MSFU数: {total_msfus}")
    print(f"  有MSFU的文档: {docs_with_msfu}")
    print(f"  文档覆盖率: {docs_with_msfu}/{total_docs} ({docs_with_msfu/total_docs*100:.1f}%)")
    print(f"  文本块覆盖率: {total_msfus}/{total_chunks} ({total_msfus/total_chunks*100:.1f}%)")

    # MSFU质量分析
    msfu_stats = get_msfu_stats(kb_path)
    print(f"\nMSFU质量:")
    print(f"  平均置信度: {msfu_stats['average_confidence']:.3f}")
    print(f"  按关系类型分布: {msfu_stats['by_relation_type']}")
    print(f"  按方向分布: {msfu_stats['by_direction']}")

    # 实体分布分析
    print(f"\n实体类型分布:")
    entity_distribution = conn.execute("""
        SELECT substr(source_entity, 1, instr(source_entity, ':')-1) as category, COUNT(*) as count
        FROM kb_msfu
        GROUP BY category
    """).fetchall()

    for category, count in entity_distribution:
        print(f"  {category}: {count}")

    # 检查问题MSFU
    print("\nPotential Issues:")

    # 1. 重复关系
    duplicates = conn.execute("""
        SELECT source_entity, target_entity, relation_type, COUNT(*) as count
        FROM kb_msfu
        GROUP BY source_entity, target_entity, relation_type
        HAVING count > 1
    """).fetchall()

    if duplicates:
        print(f"  Found {len(duplicates)} duplicate relations:")
        for src, tgt, rel, cnt in duplicates:
            print(f"    {src} -> {tgt} ({rel}) duplicated {cnt} times")
    else:
        print(f"  [OK] No duplicate relations")

    # 2. 低置信度MSFU
    low_confidence = conn.execute("SELECT COUNT(*) FROM kb_msfu WHERE confidence < 0.5").fetchone()[0]
    print(f"  Low confidence MSFUs (<0.5): {low_confidence}")

    # 3. 格式异常
    invalid_format = conn.execute("""
        SELECT COUNT(*) FROM kb_msfu
        WHERE source_entity NOT LIKE '%:%'
           OR target_entity NOT LIKE '%:%'
    """).fetchone()[0]

    if invalid_format > 0:
        print(f"  Invalid format entities: {invalid_format}")
    else:
        print(f"  [OK] Entity format valid")

    # 4. 文本长度异常
    abnormal_length = conn.execute("""
        SELECT COUNT(*) FROM kb_msfu
        WHERE length(content) < 20 OR length(content) > 1000
    """).fetchone()[0]

    if abnormal_length > 0:
        print(f"  Abnormal text length: {abnormal_length}")
    else:
        print(f"  [OK] Text length valid")

    # 文献提取能力分析
    print(f"\n文献提取能力分析:")

    # 计算每篇文献的平均MSFU数
    avg_msfu_per_doc = conn.execute("""
        SELECT AVG(msfu_count) as avg_msfus
        FROM (
            SELECT doc_id, COUNT(*) as msfu_count
            FROM kb_msfu
            GROUP BY doc_id
        )
    """).fetchone()

    if avg_msfu_per_doc and avg_msfu_per_doc[0]:
        print(f"  MSFU per document (avg): {avg_msfu_per_doc[0]:.2f}")
    else:
        print(f"  No data (all documents without MSFU)")

    # 检查无MSFU文献的文本质量
    no_msfu_chunks = conn.execute("""
        SELECT AVG(LENGTH(text)) as avg_length,
               COUNT(*) as count
        FROM kb_chunks c
        WHERE c.doc_id NOT IN (SELECT DISTINCT doc_id FROM kb_msfu)
        GROUP BY doc_id
    """).fetchall()

    if no_msfu_chunks:
        avg_len = sum(c[0] for c in no_msfu_chunks) / len(no_msfu_chunks)
        total_chunks = sum(c[1] for c in no_msfu_chunks)
        print(f"  Documents without MSFU avg text length: {avg_len:.0f} chars")
        print(f"  Documents without MSFU total chunks: {total_chunks}")

    conn.close()


def provide_optimization_suggestions():
    """提供优化建议"""
    print("\n" + "="*60)
    print("优化建议")
    print("="*60)

    print("\n1. 接受当前覆盖率的合理性")
    print("   - 6%的文献覆盖率可能符合实际情况")
    print("   - 理由：大部分文献是数据结果/方法描述，不适合关系提取")
    print("   - 建议：预期有效提取文献20-30篇，MSFU总数50-200条")

    print("\n2. 提高已提取MSFU的质量")
    print("   - 手动检查和修正实体分类")
    print("   - 合并重复关系")
    print("   - 调整置信度评分")

    print("\n3. 改进提取策略")
    print("   - 使用更宽松的关系模式")
    print("   - 降低实体匹配严格度")
    print("   - 增加领域特定模式")

    print("\n4. 分层处理文献")
    print("   - 高质量文献 → 精确提取 + LLM精炼")
    print("   - 中等质量文献 → 增强规则提取")
    print("   - 低质量文献 → 跳过或仅统计")

    print("\n5. 持续监控")
    print("   - 定期检查MSFU质量")
    print("   - 根据提取效果调整参数")
    print("   - 记录优化过程和结果")

    print("\n6. 扩展实体库")
    print("   - 根据已提取的实体补充缺失项")
    print("   - 添加更多同义词和变体")
    print("   - 优化实体匹配模式")

    print("\n7. 考虑混合策略")
    print("   - 规则提取：快速、稳定的基础关系")
    print("   - LLM提取：复杂句式和隐含关系")
    print("   - 手动标注：关键文献的高质量提取")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="MSFU提取优化分析")
    parser.add_argument("--kb-path", default="database/cnta_knowledge_base.sqlite",
                       help="知识库数据库路径")

    args = parser.parse_args()

    analyze_current_state(args.kb_path)
    provide_optimization_suggestions()

    print("\n" + "="*60)
    print("分析完成")
    print("="*60)


if __name__ == "__main__":
    main()
