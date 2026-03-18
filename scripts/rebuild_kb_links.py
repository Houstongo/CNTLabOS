"""
Rebuild relation links for existing knowledge-base documents/chunks.

Usage:
    python scripts/rebuild_kb_links.py ^
      --kb-db database/cnta_knowledge_base.sqlite
"""

from __future__ import annotations

import argparse
import sqlite3
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild kb_links from existing kb_chunks.")
    parser.add_argument(
        "--kb-db",
        default=r"D:\CNTDATA\CNTA_ML_Project\database\cnta_knowledge_base.sqlite",
        help="Path to knowledge-base sqlite file.",
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Append new links instead of clearing existing kb_links.",
    )
    return parser.parse_args()


def relation_type_counts(kb_db: str) -> dict[str, int]:
    conn = sqlite3.connect(kb_db)
    try:
        rows = conn.execute(
            """
            SELECT COALESCE(relation_type, 'unknown') AS relation_type, COUNT(*) AS n
            FROM kb_links
            GROUP BY COALESCE(relation_type, 'unknown')
            ORDER BY n DESC
            """
        ).fetchall()
        return {str(r[0]): int(r[1]) for r in rows}
    finally:
        conn.close()


def main() -> int:
    args = parse_args()
    sys.path.insert(0, r"D:\CNTDATA\CNTA_ML_Project")
    from backend.core.knowledge_base import KnowledgeBaseService  # local import

    svc = KnowledgeBaseService(args.kb_db)
    before_stats = svc.get_stats()
    result = svc.rebuild_links(clear_existing=not args.keep_existing)
    after_stats = svc.get_stats()
    by_type = relation_type_counts(args.kb_db)

    print(f"KB: {args.kb_db}")
    print(f"Before links: {before_stats['link_count']}")
    print(f"Rebuilt doc_count: {result['doc_count']}")
    print(f"Inserted links: {result['link_count']}")
    print(f"After links: {after_stats['link_count']}")
    print("Relation type counts:")
    for k, v in by_type.items():
        print(f"  - {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

