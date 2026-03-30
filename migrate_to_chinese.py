"""
知识库中英文数据迁移脚本
将kb_links和kb_msfu表中的英文关系类型和实体类型转换为中文
"""
import sqlite3
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.core.knowledge_base import (
    RELATION_TYPE_MAPPING,
    ENTITY_TYPE_MAPPING,
    PROCESS_FACTOR_MAPPING,
    MORPHOLOGY_FACTOR_MAPPING,
    PERFORMANCE_FACTOR_MAPPING,
    MECHANISM_FACTOR_MAPPING,
)

DB_PATH = r"d:\CNTDATA\CNTA_ML_Project\database\cnta_knowledge_base.sqlite"


def translate_entity_node(node: str) -> str:
    """将实体节点从英文转换为中文

    例如: process:temperature -> 工艺:温度
           morphology:alignment -> 形貌:取向度
    """
    if not node or ':' not in node:
        return node

    entity_type, factor_name = node.split(':', 1)
    entity_type = entity_type.strip().lower()

    # 转换实体类型
    entity_type_cn = ENTITY_TYPE_MAPPING.get(entity_type, entity_type)

    # 转换因子名称
    all_factor_mappings = {
        **PROCESS_FACTOR_MAPPING,
        **MORPHOLOGY_FACTOR_MAPPING,
        **PERFORMANCE_FACTOR_MAPPING,
        **MECHANISM_FACTOR_MAPPING,
    }
    factor_cn = all_factor_mappings.get(factor_name.strip(), factor_name)

    return f"{entity_type_cn}:{factor_cn}"


def migrate_kb_links(db_path: str, dry_run: bool = False):
    """迁移kb_links表"""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    print("=== 开始迁移 kb_links 表 ===")

    try:
        # 查询所有记录
        rows = conn.execute("SELECT id, relation_type, source_node, target_node, process_factor, morphology_factor, performance_factor FROM kb_links").fetchall()

        updated_count = 0
        skipped_count = 0

        for row in rows:
            row_id = row['id']
            relation_type = row['relation_type']
            source_node = row['source_node']
            target_node = row['target_node']
            process_factor = row['process_factor']
            morphology_factor = row['morphology_factor']
            performance_factor = row['performance_factor']

            # 转换关系类型
            relation_type_cn = RELATION_TYPE_MAPPING.get(relation_type, relation_type)

            # 转换实体节点
            source_node_cn = translate_entity_node(source_node) if source_node else None
            target_node_cn = translate_entity_node(target_node) if target_node else None

            # 转换因子名称
            all_factor_mappings = {
                **PROCESS_FACTOR_MAPPING,
                **MORPHOLOGY_FACTOR_MAPPING,
                **PERFORMANCE_FACTOR_MAPPING,
                **MECHANISM_FACTOR_MAPPING,
            }

            process_factor_cn = all_factor_mappings.get(process_factor, process_factor) if process_factor else None
            morphology_factor_cn = all_factor_mappings.get(morphology_factor, morphology_factor) if morphology_factor else None
            performance_factor_cn = all_factor_mappings.get(performance_factor, performance_factor) if performance_factor else None

            # 检查是否需要更新
            needs_update = (
                relation_type_cn != relation_type or
                source_node_cn != source_node or
                target_node_cn != target_node or
                process_factor_cn != process_factor or
                morphology_factor_cn != morphology_factor or
                performance_factor_cn != performance_factor
            )

            if not needs_update:
                skipped_count += 1
                continue

            if dry_run:
                print(f"[DRY RUN] 将更新记录 {row_id}:")
                print(f"  relation_type: {relation_type} -> {relation_type_cn}")
                print(f"  source_node: {source_node} -> {source_node_cn}")
                print(f"  target_node: {target_node} -> {target_node_cn}")
            else:
                conn.execute("""
                    UPDATE kb_links
                    SET relation_type = ?,
                        source_node = ?,
                        target_node = ?,
                        process_factor = ?,
                        morphology_factor = ?,
                        performance_factor = ?
                    WHERE id = ?
                """, (relation_type_cn, source_node_cn, target_node_cn,
                       process_factor_cn, morphology_factor_cn, performance_factor_cn, row_id))

            updated_count += 1

        if not dry_run:
            conn.commit()

        print(f"kb_links 迁移完成:")
        print(f"  更新记录数: {updated_count}")
        print(f"  跳过记录数: {skipped_count}")

    finally:
        conn.close()


def migrate_kb_msfu(db_path: str, dry_run: bool = False):
    """迁移kb_msfu表"""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    print("\n=== 开始迁移 kb_msfu 表 ===")

    try:
        # 查询所有记录
        rows = conn.execute("SELECT id, relation_type, source_entity, target_entity FROM kb_msfu").fetchall()

        updated_count = 0
        skipped_count = 0

        for row in rows:
            row_id = row['id']
            relation_type = row['relation_type']
            source_entity = row['source_entity']
            target_entity = row['target_entity']

            # 转换关系类型
            relation_type_cn = RELATION_TYPE_MAPPING.get(relation_type, relation_type)

            # 转换实体
            source_entity_cn = translate_entity_node(source_entity) if source_entity else None
            target_entity_cn = translate_entity_node(target_entity) if target_entity else None

            # 检查是否需要更新
            needs_update = (
                relation_type_cn != relation_type or
                source_entity_cn != source_entity or
                target_entity_cn != target_entity
            )

            if not needs_update:
                skipped_count += 1
                continue

            if dry_run:
                print(f"[DRY RUN] 将更新记录 {row_id}:")
                print(f"  relation_type: {relation_type} -> {relation_type_cn}")
                print(f"  source_entity: {source_entity} -> {source_entity_cn}")
                print(f"  target_entity: {target_entity} -> {target_entity_cn}")
            else:
                conn.execute("""
                    UPDATE kb_msfu
                    SET relation_type = ?,
                        source_entity = ?,
                        target_entity = ?
                    WHERE id = ?
                """, (relation_type_cn, source_entity_cn, target_entity_cn, row_id))

            updated_count += 1

        if not dry_run:
            conn.commit()

        print(f"kb_msfu 迁移完成:")
        print(f"  更新记录数: {updated_count}")
        print(f"  跳过记录数: {skipped_count}")

    finally:
        conn.close()


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='RAG知识库中英文数据迁移工具')
    parser.add_argument('--dry-run', action='store_true', help='预览模式，不实际修改数据库')
    parser.add_argument('--execute', action='store_true', help='执行迁移，实际修改数据库')

    args = parser.parse_args()

    # 如果没有参数，使用交互模式
    if not args.dry_run and not args.execute:
        print("RAG知识库中英文数据迁移工具")
        print("=" * 50)

        dry_run_input = input("\n是否只预览不实际执行？(y/n，默认y): ").strip().lower()
        dry_run = dry_run_input != 'n'

        if dry_run:
            print("\n模式: DRY RUN（预览，不实际修改数据库）")
        else:
            print("\n模式: 执行迁移（将实际修改数据库）")
            confirm = input("确认继续？(yes/no): ").strip().lower()
            if confirm != 'yes':
                print("已取消")
                return
    else:
        dry_run = args.dry_run
        print("RAG知识库中英文数据迁移工具")
        print("=" * 50)

        if dry_run:
            print("\n模式: DRY RUN（预览，不实际修改数据库）")
        elif args.execute:
            print("\n模式: 执行迁移（将实际修改数据库）")
        else:
            print("\n错误：必须指定 --dry-run 或 --execute")
            return

    # 迁移kb_links
    migrate_kb_links(DB_PATH, dry_run=dry_run)

    # 迁移kb_msfu
    migrate_kb_msfu(DB_PATH, dry_run=dry_run)

    if not dry_run:
        print("\n所有迁移完成！")
        print("请重新启动后端服务以应用更改。")
    else:
        print("\n预览完成！")
        print("如要实际执行，请运行: python migrate_to_chinese.py --execute")


if __name__ == "__main__":
    main()
