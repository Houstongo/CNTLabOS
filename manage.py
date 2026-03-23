import os
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
    subprocess.run([sys.executable, "backend/main.py"], env=env)

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
