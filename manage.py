import os
import sys
import subprocess
import argparse

# 设置环境路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

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

        db_path = os.path.join(PROJECT_ROOT, "database", "cnta_experiments.sqlite")
        kb_path = os.path.join(PROJECT_ROOT, "database", "cnta_knowledge_base.sqlite")
        retriever = RAGRetriever(db_path, knowledge_db_path=kb_path)
        stats_before = retriever.get_stats()
        print(
            f"Knowledge base before import: docs={stats_before['document_count']}, "
            f"chunks={stats_before['chunk_count']}"
        )

        if source_dir:
            imported = retriever.knowledge_base.ingest_directory(
                source_dir=source_dir,
                source_type=source_type,
                theme=theme,
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
    kb_parser.add_argument("--theme", type=str, default="growth_mechanism", help="Document theme")
    kb_parser.add_argument("--core", action="store_true", help="Mark imported files as core documents")

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
            theme=getattr(args, "theme", "growth_mechanism"),
            is_core=getattr(args, "core", False),
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
