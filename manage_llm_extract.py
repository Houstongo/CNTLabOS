"""
快速MSFU提取工具 - 支持LLM辅助提取

用法：
    python manage_llm_extract.py --provider deepseek --api-key YOUR_KEY
    python manage_llm_extract.py --provider glm --api-key YOUR_KEY
"""

import os
import sys
import argparse
import sqlite3
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from backend.core.ai_interpreter import AIInterpreter
from backend.core.msfu_extractor import MSFUMetadata, store_msfus_in_db, get_msfu_stats


def extract_with_llm(
    kb_path: str,
    provider: str,
    api_key: str,
    model: str = None,
    doc_id: int = None,
    limit: int = 10
):
    """
    使用LLM辅助提取MSFU

    Args:
        kb_path: 知识库数据库路径
        provider: LLM提供商（glm/deepseek）
        api_key: API密钥
        model: 模型名称（可选）
        doc_id: 文档ID（可选，不指定则批量处理）
        limit: 处理的文档数量限制
    """
    print(f"初始化LLM客户端: {provider}")
    llm_client = AIInterpreter(provider=provider, api_key=api_key, model=model)
    print("LLM客户端初始化成功")

    # 获取待处理的文档
    conn = sqlite3.connect(kb_path)
    conn.row_factory = sqlite3.Row

    if doc_id:
        # 处理指定文档
        cursor = conn.execute(
            "SELECT id, title, source_type FROM kb_documents WHERE id = ?",
            (doc_id,)
        )
        docs = [cursor.fetchone()]
    else:
        # 批量处理未处理的文档
        cursor = conn.execute("""
            SELECT d.id, d.title, d.source_type,
                   COUNT(c.id) as chunk_count,
                   COUNT(m.id) as msfu_count
            FROM kb_documents d
            LEFT JOIN kb_chunks c ON c.doc_id = d.id
            LEFT JOIN kb_msfu m ON m.doc_id = d.id
            GROUP BY d.id
            HAVING msfu_count = 0 OR msfu_count < chunk_count * 0.5
            ORDER BY d.id DESC
            LIMIT ?
        """, (limit,))
        docs = cursor.fetchall()

    conn.close()

    if not docs:
        print("没有找到需要处理的文档")
        return

    print(f"\n找到 {len(docs)} 个文档待处理")

    # 处理每个文档
    for i, doc in enumerate(docs):
        doc_id = doc["id"]
        title = doc["title"]
        source_type = doc["source_type"]

        print(f"\n[{i+1}/{len(docs)}] 处理文档: {title}")

        # 加载文档的所有分块
        conn = sqlite3.connect(kb_path)
        conn.row_factory = sqlite3.Row
        chunks = conn.execute(
            "SELECT id, chunk_index, text FROM kb_chunks WHERE doc_id = ? ORDER BY chunk_index",
            (doc_id,)
        ).fetchall()
        conn.close()

        print(f"  - 加载 {len(chunks)} 个文本块")

        # 批量提取
        total_msfus = 0
        for chunk in chunks:
            chunk_id = chunk["id"]
            chunk_text = chunk["text"]
            chunk_index = chunk["chunk_index"]

            # 创建元数据
            metadata = MSFUMetadata(
                doc_id=str(doc_id),
                chunk_id=str(chunk_id),
                doc_title=title,
                doc_type="pdf",
                page_num=None
            )

            # 调用AIInterpreter的精炼功能
            # 这里需要一个简化的调用方式
            try:
                # 先用规则提取（快速）
                from backend.core.rule_based_extractor_enhanced import EnhancedRuleExtractor
                rule_extractor = EnhancedRuleExtractor(confidence_threshold=0.4)
                rule_msfus = rule_extractor.extract(chunk_text, metadata, title)

                # 如果有规则提取结果，使用LLM精炼
                if rule_msfus:
                    # 转换为字典格式
                    candidate_dicts = [msfu.to_dict() for msfu in rule_msfus]
                    refined_dicts = llm_client.refine_msfu_batch(
                        candidate_dicts,
                        chunk_text
                    )

                    # 转换回MSFU对象
                    refined_msfus = []
                    for item in refined_dicts:
                        if item.get("valid", True):
                            try:
                                # 重新构建MSFU
                                assertion = item.get("assertion", item)
                                if "assertion" not in item:
                                    # 兼容扁平格式
                                    assertion = {
                                        "source_entity": item.get("source_entity", ""),
                                        "relation_type": item.get("relation_type", ""),
                                        "target_entity": item.get("target_entity", ""),
                                        "condition": item.get("condition"),
                                        "direction": item.get("direction", "unknown")
                                    }

                                from backend.core.msfu_extractor import (
                                    Assertion, Evidence, MSFU, Condition, ExtractionMethod
                                )

                                # 解析条件
                                condition_data = assertion.get("condition")
                                condition = Condition.from_dict(condition_data) if condition_data else None

                                msfu_assertion = Assertion(
                                    source_entity=assertion.get("source_entity", ""),
                                    relation_type=assertion.get("relation_type", ""),
                                    target_entity=assertion.get("target_entity", ""),
                                    condition=condition,
                                    direction=assertion.get("direction", "unknown")
                                )

                                msfu_evidence = Evidence(
                                    text_snippet=chunk_text[:200],
                                    doc_title=title,
                                    confidence=item.get("confidence", 0.7),
                                    extraction_method=ExtractionMethod.HYBRID.value,
                                    page_num=metadata.page_num,
                                    chunk_id=int(metadata.chunk_id)
                                )

                                refined_msfu = MSFU(
                                    content=chunk_text[:500],
                                    metadata=metadata,
                                    assertion=msfu_assertion,
                                    evidence=msfu_evidence
                                )
                                refined_msfus.append(refined_msfu)

                            except Exception as e:
                                print(f"    警告：精炼MSFU失败: {e}")
                                continue

                    if refined_msfus:
                        # 存储精炼后的MSFU
                        stored_ids = store_msfus_in_db(refined_msfus, kb_path, doc_id, chunk_id)
                        total_msfus += len(stored_ids)
                        print(f"    Chunk {chunk_index}: 存储 {len(stored_ids)} 个MSFU (LLM精炼)")

                else:
                    print(f"    Chunk {chunk_index}: 无规则提取结果，跳过LLM精炼")

            except Exception as e:
                print(f"    Chunk {chunk_index}: 提取失败: {e}")
                continue

        print(f"  - 文档总计: 提取 {total_msfus} 个MSFU")

    # 显示统计
    print("\n" + "="*60)
    print("提取完成！")
    print("="*60)
    stats = get_msfu_stats(kb_path)
    print(f"知识库MSFU总数: {stats['total_msfus']}")
    print(f"平均置信度: {stats['average_confidence']}")
    print(f"按关系类型统计: {stats['by_relation_type']}")
    print(f"按方向统计: {stats['by_direction']}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="MSFU提取工具 - 支持LLM辅助提取")
    parser.add_argument("--provider", required=True, choices=["glm", "deepseek"],
                       help="LLM提供商")
    parser.add_argument("--api-key", required=True, help="API密钥")
    parser.add_argument("--model", help="模型名称（可选）")
    parser.add_argument("--doc-id", type=int, help="指定文档ID（可选）")
    parser.add_argument("--limit", type=int, default=10, help="处理的文档数量限制（默认10）")
    parser.add_argument("--kb-path", default="database/cnta_knowledge_base.sqlite",
                       help="知识库数据库路径")

    args = parser.parse_args()

    print("="*60)
    print("MSFU提取工具")
    print("="*60)
    print(f"提供商: {args.provider}")
    print(f"模型: {args.model or '默认'}")
    print(f"文档限制: {args.limit}")
    print(f"知识库路径: {args.kb_path}")
    print("="*60)

    try:
        extract_with_llm(
            kb_path=args.kb_path,
            provider=args.provider,
            api_key=args.api_key,
            model=args.model,
            doc_id=args.doc_id,
            limit=args.limit
        )
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
