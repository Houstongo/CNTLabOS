"""测试 MSFU 提取"""
import sys
import os
import sqlite3

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.core.msfu_extractor import MSFUExtractor, MSFUMetadata, store_msfus_in_db, get_msfu_stats

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
kb_path = os.path.join(PROJECT_ROOT, "database", "cnta_knowledge_base.sqlite")

# 连接数据库
conn = sqlite3.connect(kb_path)
cursor = conn.cursor()

# 查找没有 MSFU 的文档
cursor.execute("""
    SELECT d.id, d.title, d.source_type, COUNT(c.id) as chunk_count
    FROM kb_documents d
    LEFT JOIN kb_chunks c ON c.doc_id = d.id
    LEFT JOIN kb_msfu m ON m.doc_id = d.id
    GROUP BY d.id
    HAVING m.id IS NULL
    LIMIT 3
""")
docs = cursor.fetchall()
conn.close()

print(f"找到 {len(docs)} 个待处理文档")

extractor = MSFUExtractor(use_llm_refinement=False)
total_count = 0

for doc_id, title, source_type, chunk_count in docs:
    print(f"\n处理文档: {title[:50]}... (id={doc_id}, chunks={chunk_count})")

    conn = sqlite3.connect(kb_path)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT id, text FROM kb_chunks WHERE doc_id = ? ORDER BY chunk_index LIMIT 5",
        (doc_id,)
    )
    chunks = cursor.fetchall()

    for chunk_id, chunk_text in chunks:
        metadata = MSFUMetadata(
            doc_id=str(doc_id),
            chunk_id=str(chunk_id),
            doc_title=title,
            doc_type=source_type
        )
        msfus = extractor.extract(chunk_text, metadata, title)
        print(f"  Chunk {chunk_id}: 提取 {len(msfus)} 个 MSFU")

        if msfus:
            # 显示第一个 MSFU 的详情
            msfu = msfus[0]
            print(f"    - source: {msfu.assertion.source_entity}")
            print(f"    - relation: {msfu.assertion.relation_type}")
            print(f"    - target: {msfu.assertion.target_entity}")
            print(f"    - direction: {msfu.assertion.direction}")
            print(f"    - confidence: {msfu.evidence.confidence:.2f}")

        stored_ids = store_msfus_in_db(msfus, kb_path, doc_id, chunk_id)
        total_count += len(stored_ids)

    conn.close()

print(f"\n共提取 {total_count} 个 MSFU")

# 显示统计
stats = get_msfu_stats(kb_path)
print(f"\nMSFU 统计:")
print(f"  总数: {stats.get('total_msfus', 0)}")
print(f"  平均置信度: {stats.get('average_confidence', 0):.2f}")
