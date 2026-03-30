import os
import re
import sys
import subprocess
import argparse
import sqlite3

# 设置环境路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)


def safe_console_text(text):
    if text is None:
        return ""
    try:
        text.encode("gbk")
        return text
    except UnicodeEncodeError:
        return text.encode("gbk", errors="replace").decode("gbk")

def run_backend():
    print("正在启动后端 API 服务 (FastAPI)...")
    env = os.environ.copy()
    env["PYTHONPATH"] = PROJECT_ROOT
    # 使用绝对路径确保能找到backend/main.py
    backend_path = os.path.join(PROJECT_ROOT, "backend", "main.py")
    subprocess.run([sys.executable, backend_path], env=env)

def run_frontend(port=8080):
    print(f"正在启动前端 HTTP 服务器 (Port {port})...")
    subprocess.run([sys.executable, "-m", "http.server", str(port)])

def init_db(clear=False):
    print(f"正在{'重新' if clear else ''}初始化数据库结构与数据扫描...")
    try:
        # 确保 src 目录在路径中
        sys.path.append(os.path.join(PROJECT_ROOT, "src", "analysis"))
        from backend.core.populate_db import populate
        populate(clear=clear)
    except Exception as e:
        print(f"错误: 无法加载后端核心模块: {e}")

def analyze_batch(reprocess=False, limit=None, source=None):
    print("正在启动批量特征提取任务（v2.0）...")
    try:
        sys.path.insert(0, PROJECT_ROOT)
        from backend.core.batch_processor import batch_process
        batch_process(reprocess=reprocess, limit=limit, source=source)
    except Exception as e:
        print(f"执行失败: {e}")

def run_data_etl():
    print("正在启动数据集成引擎 (ETL)...")
    try:
        sys.path.append(os.path.join(PROJECT_ROOT, "tools", "data_processing"))
        from data_engine import run_full_etl
        run_full_etl()
    except Exception as e:
        print(f"数据处理失败: {e}")

def update_magnification():
    print("正在同步图片倍率信息至数据库...")
    try:
        sys.path.append(os.path.join(PROJECT_ROOT, "src", "analysis"))
        sys.path.append(os.path.join(PROJECT_ROOT, "tools", "maintenance"))
        from db_mag_updater import update_all_magnifications
        update_all_magnifications()
    except Exception as e:
        print(f"执行失败: {e}")

def kb_bootstrap(source_dir=None, source_type="notes", theme="growth_mechanism", is_core=False):
    print("Initializing knowledge-base backend...")
    try:
        from backend.core.knowledge_rag import RAGRetriever
        from backend.core.knowledge_seed import infer_theme_from_path

        db_path = os.path.join(PROJECT_ROOT, "database", "cnta_experiments.sqlite")
        kb_path = os.path.join(PROJECT_ROOT, "database", "cnta_knowledge_base.sqlite")
        retriever = RAGRetriever(db_path, knowledge_db_path=kb_path)
        stats_before = retriever.get_stats()
        print(
            f"Knowledge base before import: docs={stats_before['document_count']}, "
            f"chunks={stats_before['chunk_count']}"
        )

        if source_dir:
            resolved_theme = theme or infer_theme_from_path(source_dir)
            imported = retriever.knowledge_base.ingest_directory(
                source_dir=source_dir,
                source_type=source_type,
                theme=resolved_theme,
                is_core=is_core,
            )
            print(f"Imported documents: {imported['document_count']}")

        stats_after = retriever.get_stats()
        print(
            f"Knowledge base ready: docs={stats_after['document_count']}, "
            f"chunks={stats_after['chunk_count']}, "
            f"core_docs={stats_after['core_document_count']}"
        )
    except Exception as e:
        print(f"Knowledge-base bootstrap failed: {e}")


def kb_import_core():
    print("Importing core knowledge-base sources...")
    try:
        from backend.core.knowledge_rag import RAGRetriever
        from backend.core.knowledge_seed import DEFAULT_KB_SEED_SOURCES, import_seed_sources

        db_path = os.path.join(PROJECT_ROOT, "database", "cnta_experiments.sqlite")
        kb_path = os.path.join(PROJECT_ROOT, "database", "cnta_knowledge_base.sqlite")
        retriever = RAGRetriever(db_path, knowledge_db_path=kb_path)

        result = import_seed_sources(retriever.knowledge_base, DEFAULT_KB_SEED_SOURCES)
        stats = retriever.get_stats()
        print(
            f"Imported sources: {result['source_count']}, "
            f"documents: {result['document_count']}"
        )
        print(
            f"Knowledge base status: docs={stats['document_count']}, "
            f"chunks={stats['chunk_count']}, core_docs={stats['core_document_count']}"
        )
    except Exception as e:
        print(f"Knowledge-base core import failed: {e}")


def kb_relabel_themes():
    print("Relabeling knowledge-base document themes...")
    try:
        from backend.core.knowledge_rag import RAGRetriever
        from backend.core.knowledge_seed import relabel_document_themes

        db_path = os.path.join(PROJECT_ROOT, "database", "cnta_experiments.sqlite")
        kb_path = os.path.join(PROJECT_ROOT, "database", "cnta_knowledge_base.sqlite")
        retriever = RAGRetriever(db_path, knowledge_db_path=kb_path)
        result = relabel_document_themes(retriever.knowledge_base)
        print(f"Updated documents: {result['updated_count']}")
        stats = retriever.get_stats()
        print(
            f"Knowledge base status: docs={stats['document_count']}, "
            f"chunks={stats['chunk_count']}, core_docs={stats['core_document_count']}"
        )
    except Exception as e:
        print(f"Knowledge-base relabel failed: {e}")


def kb_search(query, task_name=None, top_k=5):
    print("Searching knowledge base...")
    try:
        from backend.core.knowledge_rag import RAGRetriever

        db_path = os.path.join(PROJECT_ROOT, "database", "cnta_experiments.sqlite")
        kb_path = os.path.join(PROJECT_ROOT, "database", "cnta_knowledge_base.sqlite")
        retriever = RAGRetriever(db_path, knowledge_db_path=kb_path)
        results = retriever.retrieve_from_pdf(query=query, top_k=top_k, task_name=task_name)
        print(f"Results: {len(results)}")
        for index, item in enumerate(results, 1):
            print(f"[{index}] {item['title']} | theme={item['theme']} | score={item['score']}")
            print(safe_console_text(item["text"]))
            print("-" * 60)
    except Exception as e:
        print(f"Knowledge-base search failed: {e}")


def msfu_batch(reprocess=False, use_llm=False, provider=None, api_key=None, limit=None, dry_run=False):
    """批量处理文献：MinerU解析 → 重新分块 → MSFU提取"""
    print("=" * 60)
    print("MSFU 批量提取 (MinerU + 规则)")
    print("=" * 60)

    kb_path = os.path.join(PROJECT_ROOT, "database", "cnta_knowledge_base.sqlite")
    conn = sqlite3.connect(kb_path)
    cursor = conn.cursor()

    # 查询所有 paper 和 notes 类型文档（都有 PDF file_path）
    cursor.execute("""
        SELECT d.id, d.title, d.source_type, d.file_path, d.theme, d.language, d.is_core
        FROM kb_documents d
        WHERE d.source_type IN ('paper', 'notes')
          AND d.file_path LIKE '%.pdf'
        ORDER BY d.id
    """)
    docs = cursor.fetchall()
    conn.close()

    if not docs:
        print("没有待处理的文档")
        return

    if limit:
        docs = docs[:limit]
        print(f"限制处理前 {limit} 篇（共 {len(docs)} 篇）")
    else:
        print(f"共 {len(docs)} 篇文献待处理")

    # 初始化 MinerU 解析器
    from backend.core.mineru_extractor import MinerUExtractor, MINERU_OUTPUT_DIR
    from backend.core.knowledge_base import KnowledgeBaseService
    from backend.core.msfu_extractor import MSFUExtractor, MSFUMetadata, store_msfus_in_db

    extractor = MinerUExtractor()

    # 初始化 LLM（可选）
    llm_client = None
    if use_llm and provider and api_key:
        from backend.core.ai_interpreter import AIInterpreter
        llm_client = AIInterpreter(provider=provider, api_key=api_key)
        print(f"LLM 精炼已启用: {provider}")
    elif use_llm:
        print("警告: --use-llm 需要 --provider 和 --api-key，将使用纯规则提取")

    msfu_extractor = MSFUExtractor(llm_client=llm_client, use_llm_refinement=use_llm)

    # 初始化 KnowledgeBase（复用分块逻辑）
    kb_service = KnowledgeBaseService(kb_path)

    total_docs = 0
    total_chunks = 0
    total_msfus = 0
    total_api_calls = 0
    total_cache_hits = 0
    errors = []

    for idx, (doc_id, title, source_type, file_path, theme, language, is_core) in enumerate(docs, 1):
        short_title = title[:60] + "..." if len(title) > 60 else title
        print(f"\n[{idx}/{len(docs)}] {short_title} (id={doc_id})")

        if dry_run:
            has_pdf = os.path.exists(file_path) if file_path else False
            has_cache = _check_mineru_cache(file_path)
            print(f"  PDF: {'存在' if has_pdf else '缺失'}, MinerU缓存: {'有' if has_cache else '无'}")
            continue

        # 1. 获取高质量 Markdown
        markdown = _get_mineru_markdown(file_path, extractor, doc_id, title)
        if not markdown:
            print(f"  跳过: 无法获取 Markdown（PDF 不存在或解析失败）")
            errors.append((doc_id, title, "无 Markdown"))
            continue

        # 2. 清除旧 chunks 和 MSFU
        conn = sqlite3.connect(kb_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM kb_msfu WHERE doc_id = ?", (doc_id,))
        cursor.execute("DELETE FROM kb_links WHERE doc_id = ?", (doc_id,))
        cursor.execute("DELETE FROM kb_chunks WHERE doc_id = ?", (doc_id,))
        conn.commit()
        conn.close()

        # 3. 重新分块并插入
        from backend.core.knowledge_base import KnowledgeBaseService
        chunks = KnowledgeBaseService._split_text(
            re.sub(r"\s+", " ", markdown).strip()
        )

        conn = sqlite3.connect(kb_path)
        cursor = conn.cursor()
        for index, chunk_text in enumerate(chunks):
            keywords = " ".join(kb_service._extract_keywords(chunk_text))
            knowledge_type = kb_service._infer_knowledge_type(theme)
            cursor.execute(
                """
                INSERT INTO kb_chunks (doc_id, chunk_index, text, keywords, knowledge_type)
                VALUES (?, ?, ?, ?, ?)
                """,
                (doc_id, index, chunk_text, keywords, knowledge_type),
            )
        conn.commit()
        conn.close()

        # 4. MSFU 提取
        doc_msfu_count = 0
        conn = sqlite3.connect(kb_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, text FROM kb_chunks WHERE doc_id = ? ORDER BY chunk_index",
            (doc_id,)
        )
        new_chunks = cursor.fetchall()
        conn.close()

        for chunk_id, chunk_text in new_chunks:
            metadata = MSFUMetadata(
                doc_id=str(doc_id),
                chunk_id=str(chunk_id),
                doc_title=title,
                doc_type=source_type
            )
            msfus = msfu_extractor.extract(chunk_text, metadata, title)
            stored_ids = store_msfus_in_db(msfus, kb_path, doc_id, chunk_id)
            doc_msfu_count += len(stored_ids)

        print(f"  Chunks: {len(new_chunks)}, MSFU: {doc_msfu_count}")
        total_docs += 1
        total_chunks += len(new_chunks)
        total_msfus += doc_msfu_count

    # 打印总结
    print("\n" + "=" * 60)
    print("处理完成")
    print(f"  处理文档: {total_docs}/{len(docs)}")
    print(f"  总 Chunks: {total_chunks}")
    print(f"  总 MSFU: {total_msfus}")
    if errors:
        print(f"  错误/跳过: {len(errors)}")
        for doc_id, title, reason in errors:
            print(f"    - [{doc_id}] {reason}: {title[:50]}")

    # 显示最终统计
    try:
        from backend.core.msfu_extractor import get_msfu_stats
        stats = get_msfu_stats(kb_path)
        print(f"\nMSFU 全局统计:")
        print(f"  总数: {stats.get('total_msfus', 0)}")
        print(f"  平均置信度: {stats.get('average_confidence', 0)}")
        print(f"  按关系类型: {stats.get('by_relation_type', {})}")
        print(f"  按方向: {stats.get('by_direction', {})}")
    except Exception as e:
        print(f"获取统计失败: {e}")


def msfu_llm(limit=None, model=None):
    """对 0 MSFU 文档用 LLM 直接提取，然后导出全部 MSFU 供人工核验"""
    import json

    kb_path = os.path.join(PROJECT_ROOT, "database", "cnta_knowledge_base.sqlite")

    # 读取 GLM API Key
    config_path = os.path.join(os.path.expanduser("~"), "magic-pdf.json")
    glm_api_key = ""
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        glm_api_key = config.get("glm-api-key", "")

    if not glm_api_key:
        print("错误: 未找到 GLM API Key")
        print("请在 ~/magic-pdf.json 中添加 \"glm-api-key\": \"your_key\"")
        return

    from backend.core.ai_interpreter import AIInterpreter
    llm = AIInterpreter(provider="glm", api_key=glm_api_key, model=model or "glm-4-flash")
    print(f"LLM 初始化: provider=glm, model={llm.model}")

    # 查找 0 MSFU 的文档
    conn = sqlite3.connect(kb_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT d.id, d.title, d.source_type
        FROM kb_documents d
        LEFT JOIN kb_msfu m ON m.doc_id = d.id
        WHERE d.source_type IN ('paper', 'notes')
        GROUP BY d.id
        HAVING COUNT(m.id) = 0
        ORDER BY d.id
    """)
    docs = cursor.fetchall()
    conn.close()

    if not docs:
        print("所有文档都已有 MSFU，无需 LLM 提取")
        return

    if limit:
        docs = docs[:limit]

    print(f"\n{'='*60}")
    print(f"LLM 直接提取 MSFU: {len(docs)} 篇文档")
    print(f"{'='*60}")

    total_new = 0
    total_chunks = 0

    for idx, (doc_id, title, source_type) in enumerate(docs, 1):
        short_title = title[:55] + "..." if len(title) > 55 else title
        print(f"\n[{idx}/{len(docs)}] {short_title} (id={doc_id})")

        conn = sqlite3.connect(kb_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, text FROM kb_chunks WHERE doc_id = ? ORDER BY chunk_index",
            (doc_id,)
        )
        chunks = cursor.fetchall()
        conn.close()

        doc_msfu_count = 0

        for chunk_id, chunk_text in chunks:
            # 直接用 LLM 提取
            llm_results = llm.extract_msfu_from_text(chunk_text)

            for item in llm_results:
                se = item.get("source_entity", "")
                te = item.get("target_entity", "")
                if not se or not te:
                    continue

                try:
                    with sqlite3.connect(kb_path) as conn2:
                        cursor2 = conn2.cursor()
                        cursor2.execute("""
                            INSERT INTO kb_msfu (
                                chunk_id, doc_id, source_entity, relation_type, target_entity,
                                condition_param, condition_op, condition_value, condition_unit,
                                direction, content, confidence, extraction_method,
                                process_factor, morphology_factor, performance_factor,
                                effect_direction, mechanism_summary, evidence_text,
                                doc_title, page_num
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            chunk_id, doc_id,
                            se, item.get("relation_type", "affects"), te,
                            None, None, None, None,
                            item.get("direction", "unknown"),
                            item.get("content", chunk_text[:500]),
                            item.get("confidence", 0.6),
                            "llm",
                            se.split(":")[1] if ":" in se else None,
                            te.split(":")[1] if ":" in te else None,
                            None,
                            item.get("direction", "unknown"),
                            item.get("content", "")[:220],
                            item.get("content", "")[:320],
                            title, None,
                        ))
                    doc_msfu_count += 1
                except Exception as e:
                    print(f"    存储失败: {e}")

            total_chunks += 1

        print(f"  Chunks: {len(chunks)}, 新 MSFU: {doc_msfu_count}")
        total_new += doc_msfu_count

    print(f"\n{'='*60}")
    print(f"LLM 提取完成")
    print(f"  处理文档: {len(docs)}, 新 MSFU: {total_new}")

    # 自动导出
    print("\n正在导出全部 MSFU 供人工核验...")
    _export_msfu_csv(kb_path)


def _export_msfu_csv(kb_path, output_path=None):
    """导出全部 MSFU 到 CSV 文件供人工核验"""
    import csv
    import json

    if not output_path:
        output_path = os.path.join(PROJECT_ROOT, "database", "msfu_review.csv")

    conn = sqlite3.connect(kb_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT
            m.id, m.doc_id, d.title as doc_title, d.source_type,
            m.source_entity, m.relation_type, m.target_entity,
            m.direction, m.confidence, m.extraction_method,
            m.content, m.mechanism_summary, m.evidence_text,
            m.process_factor, m.morphology_factor, m.performance_factor,
            m.condition_param, m.condition_op, m.condition_value, m.condition_unit
        FROM kb_msfu m
        LEFT JOIN kb_documents d ON d.id = m.doc_id
        ORDER BY m.doc_id, m.id
    """).fetchall()
    conn.close()

    columns = [
        "id", "doc_id", "doc_title", "source_type",
        "source_entity", "relation_type", "target_entity",
        "direction", "confidence", "extraction_method",
        "content", "process_factor", "morphology_factor", "performance_factor",
        "condition_param", "condition_op", "condition_value", "condition_unit",
        "review_status", "review_note",
    ]

    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()

        for row in rows:
            d = dict(row)
            d["review_status"] = ""
            d["review_note"] = ""
            d["doc_title"] = d["doc_title"] or ""
            d["content"] = (d.get("content") or "")[:300]
            d["mechanism_summary"] = (d.get("mechanism_summary") or "")[:200]
            d["evidence_text"] = (d.get("evidence_text") or "")[:200]

            # 只保留 columns 中的字段
            row_out = {k: d.get(k, "") for k in columns}
            writer.writerow(row_out)

    print(f"已导出 {len(rows)} 条 MSFU 到: {output_path}")
    return output_path


def _check_mineru_cache(file_path):
    """检查是否已有 MinerU 缓存的 full.md"""
    if not file_path:
        return False
    import os
    from backend.core.mineru_extractor import MINERU_OUTPUT_DIR
    basename = os.path.splitext(os.path.basename(file_path))[0].replace(" ", "_")[:80]
    md_path = os.path.join(MINERU_OUTPUT_DIR, basename, "full.md")
    return os.path.exists(md_path)


def _get_mineru_markdown(file_path, extractor, doc_id, title):
    """获取 MinerU Markdown，优先使用本地缓存"""
    import os
    from backend.core.mineru_extractor import MINERU_OUTPUT_DIR

    if not file_path or not os.path.exists(file_path):
        return None

    # 检查缓存
    basename = os.path.splitext(os.path.basename(file_path))[0].replace(" ", "_")[:80]
    cache_dir = os.path.join(MINERU_OUTPUT_DIR, basename)
    md_path = os.path.join(cache_dir, "full.md")

    if os.path.exists(md_path):
        with open(md_path, "r", encoding="utf-8") as f:
            markdown = f.read()
        if len(markdown) > 100:
            print(f"  缓存命中: {len(markdown)} 字符")
            return markdown

    # 调用 MinerU API
    print(f"  调用 MinerU API...")
    result = extractor.parse_pdf_document(doc_id, file_path)
    if result and result.markdown and len(result.markdown) > 100:
        return result.markdown

    return None


def msfu_migrate(clear=False):
    """迁移MSFU表结构（添加kb_msfu表）"""
    print("正在迁移MSFU表结构...")
    try:
        from backend.core.knowledge_rag import RAGRetriever

        kb_path = os.path.join(PROJECT_ROOT, "database", "cnta_knowledge_base.sqlite")
        retriever = RAGRetriever(db_path, knowledge_db_path=kb_path)

        # 重新初始化schema会自动创建kb_msfu表
        retriever.knowledge_base.init_schema()

        print("MSFU表结构迁移完成")
        stats = get_msfu_stats_internal(kb_path)
        print(f"MSFU统计: 总数={stats.get('total_msfus', 0)}, 平均置信度={stats.get('average_confidence', 0)}")
    except Exception as e:
        print(f"MSFU迁移失败: {e}")


def msfu_extract(doc_id=None, reextract=False, use_llm=False, provider=None, api_key=None):
    """提取MSFU"""
    print(f"正在提取MSFU (doc_id={doc_id}, reextract={reextract}, use_llm={use_llm})...")
    try:
        from backend.core.msfu_extractor import MSFUExtractor, MSFUMetadata, store_msfus_in_db, get_msfu_stats
        from backend.core.knowledge_rag import RAGRetriever

        db_path = os.path.join(PROJECT_ROOT, "database", "cnta_experiments.sqlite")
        kb_path = os.path.join(PROJECT_ROOT, "database", "cnta_knowledge_base.sqlite")
        retriever = RAGRetriever(db_path, knowledge_db_path=kb_path)

        # 初始化 LLM client（如果启用）
        llm_client = None
        if use_llm and provider and api_key:
            from backend.core.ai_interpreter import AIInterpreter
            llm_client = AIInterpreter(provider=provider, api_key=api_key)
            print(f"已启用 LLM 精炼: provider={provider}")
        elif use_llm:
            print("警告: --use-llm 需要同时指定 --provider 和 --api-key，将使用纯规则提取")

        if doc_id:
            # 处理单个文档
            conn = sqlite3.connect(kb_path)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, title, source_type FROM kb_documents WHERE id = ?",
                (doc_id,)
            )
            doc_row = cursor.fetchone()
            if not doc_row:
                print(f"文档 {doc_id} 不存在")
                return

            doc_id, title, source_type = doc_row

            if reextract:
                cursor.execute("DELETE FROM kb_msfu WHERE doc_id = ?", (doc_id,))

            cursor.execute(
                "SELECT id, text FROM kb_chunks WHERE doc_id = ? ORDER BY chunk_index",
                (doc_id,)
            )
            chunks = cursor.fetchall()
            conn.close()

            extractor = MSFUExtractor(llm_client=llm_client, use_llm_refinement=use_llm)
            total_count = 0

            for chunk_id, chunk_text in chunks:
                metadata = MSFUMetadata(
                    doc_id=str(doc_id),
                    chunk_id=str(chunk_id),
                    doc_title=title,
                    doc_type=source_type
                )
                msfus = extractor.extract(chunk_text, metadata, title)
                stored_ids = store_msfus_in_db(msfus, kb_path, doc_id, chunk_id)
                total_count += len(stored_ids)
                print(f"  Chunk {chunk_id}: 提取 {len(msfus)} 个MSFU")

            print(f"文档 {title} ({doc_id}): 共提取 {total_count} 个MSFU")

        else:
            # 处理所有未提取MSFU的文档
            conn = sqlite3.connect(kb_path)
            cursor = conn.cursor()

            # 找出没有MSFU的文档
            if reextract:
                cursor.execute("""
                    SELECT d.id, d.title, d.source_type, COUNT(c.id) as chunk_count
                    FROM kb_documents d
                    LEFT JOIN kb_chunks c ON c.doc_id = d.id
                    GROUP BY d.id
                """)
            else:
                cursor.execute("""
                    SELECT d.id, d.title, d.source_type, COUNT(c.id) as chunk_count
                    FROM kb_documents d
                    LEFT JOIN kb_chunks c ON c.doc_id = d.id
                    LEFT JOIN kb_msfu m ON m.doc_id = d.id
                    GROUP BY d.id
                    HAVING m.id IS NULL
                """)
            docs = cursor.fetchall()
            conn.close()

            extractor = MSFUExtractor(llm_client=llm_client, use_llm_refinement=use_llm)
            total_count = 0

            for doc_id, title, source_type, chunk_count in docs:
                print(f"处理文档: {title} ({doc_id}), chunks: {chunk_count}")

                conn = sqlite3.connect(kb_path)
                cursor = conn.cursor()

                if reextract:
                    cursor.execute("DELETE FROM kb_msfu WHERE doc_id = ?", (doc_id,))
                    conn.commit()

                cursor.execute(
                    "SELECT id, text FROM kb_chunks WHERE doc_id = ? ORDER BY chunk_index",
                    (doc_id,)
                )
                chunks = cursor.fetchall()
                conn.close()  # 先关闭读取连接

                for chunk_id, chunk_text in chunks:
                    metadata = MSFUMetadata(
                        doc_id=str(doc_id),
                        chunk_id=str(chunk_id),
                        doc_title=title,
                        doc_type=source_type
                    )
                    msfus = extractor.extract(chunk_text, metadata, title)
                    stored_ids = store_msfus_in_db(msfus, kb_path, doc_id, chunk_id)
                    total_count += len(stored_ids)

            print(f"共提取 {total_count} 个MSFU")

        # 显示统计
        stats = get_msfu_stats(kb_path)
        print(f"\nMSFU统计:")
        print(f"  总数: {stats.get('total_msfus', 0)}")
        print(f"  平均置信度: {stats.get('average_confidence', 0)}")
        print(f"  按关系类型: {stats.get('by_relation_type', {})}")
        print(f"  按方向: {stats.get('by_direction', {})}")

    except Exception as e:
        print(f"MSFU提取失败: {e}")


def get_msfu_stats_internal(kb_path):
    """获取MSFU统计（内部函数）"""
    try:
        from backend.core.msfu_extractor import get_msfu_stats
        return get_msfu_stats(kb_path)
    except:
        return {"total_msfus": 0, "average_confidence": 0}


def main():
    parser = argparse.ArgumentParser(description="CNTA Project 统一管理工具")
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # 服务管理
    subparsers.add_parser("run-backend", help="启动后端 FastAPI (8000)")
    subparsers.add_parser("run-frontend", help="启动前端服务器 (8080)")
    
    # 数据管理
    init_parser = subparsers.add_parser("init-db", help="初始化数据库并扫描物理文件")
    init_parser.add_argument("--clear", action="store_true", help="清空现有数据库记录")
    
    subparsers.add_parser("sync-mag", help="从文件名同步倍率信息")
    subparsers.add_parser("data-etl", help="运行数据集成引擎 (解析 PPT/Excel/Word)")
    
    # 核心分析
    analyze_parser = subparsers.add_parser("analyze", help="执行特征批量提取（v2.0）")
    analyze_parser.add_argument("--reprocess", action="store_true", help="重新处理已处理过的图像")
    analyze_parser.add_argument("--limit",     type=int,            help="只处理前 N 张（测试用）")
    analyze_parser.add_argument("--source",    type=str,            help="只处理指定来源 ZZY/XR")

    kb_parser = subparsers.add_parser("kb-bootstrap", help="Initialize the knowledge-base backend")
    kb_parser.add_argument("--source-dir", type=str, help="Directory of text files to import")
    kb_parser.add_argument("--source-type", type=str, default="notes", help="Document source type")
    kb_parser.add_argument("--theme", type=str, default=None, help="Document theme")
    kb_parser.add_argument("--core", action="store_true", help="Mark imported files as core documents")

    subparsers.add_parser("kb-import-core", help="Import default core knowledge-base sources")
    subparsers.add_parser("kb-relabel-themes", help="Relabel existing document themes from path rules")

    kb_search_parser = subparsers.add_parser("kb-search", help="Search the knowledge base")
    kb_search_parser.add_argument("query", type=str, help="Search query")
    kb_search_parser.add_argument("--task-name", type=str, default=None, help="Task-aware retrieval profile")
    kb_search_parser.add_argument("--top-k", type=int, default=5, help="Maximum number of results")

    # MSFU管理
    msfu_migrate_parser = subparsers.add_parser("msfu-migrate", help="迁移MSFU表结构")
    msfu_migrate_parser.add_argument("--clear", action="store_true", help="清空现有MSFU数据")

    msfu_extract_parser = subparsers.add_parser("msfu-extract", help="提取MSFU")
    msfu_extract_parser.add_argument("--doc-id", type=int, default=None, help="只处理指定文档")
    msfu_extract_parser.add_argument("--reextract", action="store_true", help="重新提取已处理文档")
    msfu_extract_parser.add_argument("--use-llm", action="store_true", help="使用LLM精炼")
    msfu_extract_parser.add_argument("--provider", type=str, default=None, help="LLM提供商 (glm/deepseek)")
    msfu_extract_parser.add_argument("--api-key", type=str, default=None, help="API密钥")

    msfu_batch_parser = subparsers.add_parser("msfu-batch", help="批量MinerU解析+分块+MSFU提取")
    msfu_batch_parser.add_argument("--reprocess", action="store_true", help="强制重新调用MinerU API（忽略缓存）")
    msfu_batch_parser.add_argument("--use-llm", action="store_true", help="使用LLM精炼MSFU")
    msfu_batch_parser.add_argument("--provider", type=str, default=None, help="LLM提供商 (glm/deepseek)")
    msfu_batch_parser.add_argument("--api-key", type=str, default=None, help="API密钥")
    msfu_batch_parser.add_argument("--limit", type=int, default=None, help="只处理前N篇（测试用）")
    msfu_batch_parser.add_argument("--dry-run", action="store_true", help="仅检查PDF和缓存状态，不执行")

    msfu_llm_parser = subparsers.add_parser("msfu-llm", help="对0 MSFU文档用LLM提取 + 导出CSV供人工核验")
    msfu_llm_parser.add_argument("--limit", type=int, default=None, help="只处理前N篇（测试用）")
    msfu_llm_parser.add_argument("--model", type=str, default=None, help="LLM模型名 (默认 glm-4-flash)")

    msfu_export_parser = subparsers.add_parser("msfu-export", help="导出全部MSFU为CSV供人工核验")

    args = parser.parse_args()

    if args.command == "run-backend":
        run_backend()
    elif args.command == "run-frontend":
        run_frontend()
    elif args.command == "init-db":
        init_db(clear=args.clear)
    elif args.command == "sync-mag":
        update_magnification()
    elif args.command == "data-etl":
        run_data_etl()
    elif args.command == "kb-bootstrap":
        kb_bootstrap(
            source_dir=getattr(args, "source_dir", None),
            source_type=getattr(args, "source_type", "notes"),
            theme=getattr(args, "theme", None),
            is_core=getattr(args, "core", False),
        )
    elif args.command == "kb-import-core":
        kb_import_core()
    elif args.command == "kb-relabel-themes":
        kb_relabel_themes()
    elif args.command == "kb-search":
        kb_search(
            query=getattr(args, "query"),
            task_name=getattr(args, "task_name", None),
            top_k=getattr(args, "top_k", 5),
        )
    elif args.command == "msfu-migrate":
        msfu_migrate(clear=getattr(args, "clear", False))
    elif args.command == "msfu-extract":
        msfu_extract(
            doc_id=getattr(args, "doc_id", None),
            reextract=getattr(args, "reextract", False),
            use_llm=getattr(args, "use_llm", False),
            provider=getattr(args, "provider", None),
            api_key=getattr(args, "api_key", None),
        )
    elif args.command == "msfu-batch":
        msfu_batch(
            reprocess=getattr(args, "reprocess", False),
            use_llm=getattr(args, "use_llm", False),
            provider=getattr(args, "provider", None),
            api_key=getattr(args, "api_key", None),
            limit=getattr(args, "limit", None),
            dry_run=getattr(args, "dry_run", False),
        )
    elif args.command == "msfu-llm":
        msfu_llm(
            limit=getattr(args, "limit", None),
            model=getattr(args, "model", None),
        )
    elif args.command == "msfu-export":
        kb_path = os.path.join(PROJECT_ROOT, "database", "cnta_knowledge_base.sqlite")
        _export_msfu_csv(kb_path)
    elif args.command == "analyze":
        analyze_batch(
            reprocess=getattr(args, 'reprocess', False),
            limit=getattr(args, 'limit', None),
            source=getattr(args, 'source', None),
        )
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
