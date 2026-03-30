"""
因子名称中英文迁移脚本
将kb_links和kb_msfu表中的英文因子名转换为中文
"""
import sqlite3
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.core.knowledge_base import (
    PROCESS_FACTOR_MAPPING,
    MORPHOLOGY_FACTOR_MAPPING,
    PERFORMANCE_FACTOR_MAPPING,
    MECHANISM_FACTOR_MAPPING,
)

DB_PATH = r"d:\CNTDATA\CNTA_ML_Project\database\cnta_knowledge_base.sqlite"


def translate_factor_name(factor_name: str) -> str:
    """将因子名称从英文转换为中文"""
    if not factor_name:
        return factor_name

    # 定义纯中英文字典（不包含正则表达式）
    # 从映射表提取纯字符串键值对
    pure_mapping = {}

    def add_mapping(mapping_dict):
        for key, value in mapping_dict.items():
            # 如果值是列表（正则表达式），跳过
            if isinstance(value, list):
                continue
            pure_mapping[key] = value

    add_mapping(PROCESS_FACTOR_MAPPING)
    add_mapping(MORPHOLOGY_FACTOR_MAPPING)
    add_mapping(PERFORMANCE_FACTOR_MAPPING)
    add_mapping(MECHANISM_FACTOR_MAPPING)

    return pure_mapping.get(factor_name, factor_name)


def migrate_kb_links_factors(db_path: str, dry_run: bool = False):
    """迁移kb_links表的因子字段"""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    print("=== 开始迁移 kb_links 表因子字段 ===")

    try:
        # 查询所有需要更新的记录
        rows = conn.execute("""
            SELECT id, process_factor, morphology_factor, performance_factor
            FROM kb_links
            WHERE process_factor IS NOT NULL
               OR morphology_factor IS NOT NULL
               OR performance_factor IS NOT NULL
        """).fetchall()

        updated_count = 0
        skipped_count = 0

        for row in rows:
            row_id = row['id']
            process_factor = row['process_factor']
            morphology_factor = row['morphology_factor']
            performance_factor = row['performance_factor']

            # 转换因子名称
            process_factor_cn = translate_factor_name(process_factor) if process_factor else None
            morphology_factor_cn = translate_factor_name(morphology_factor) if morphology_factor else None
            performance_factor_cn = translate_factor_name(performance_factor) if performance_factor else None

            # 检查是否需要更新
            needs_update = (
                process_factor_cn != process_factor or
                morphology_factor_cn != morphology_factor or
                performance_factor_cn != performance_factor
            )

            if not needs_update:
                skipped_count += 1
                continue

            if dry_run:
                print(f"[DRY RUN] 将更新记录 {row_id}:")
                if process_factor_cn != process_factor:
                    print(f"  process_factor: {process_factor} -> {process_factor_cn}")
                if morphology_factor_cn != morphology_factor:
                    print(f"  morphology_factor: {morphology_factor} -> {morphology_factor_cn}")
                if performance_factor_cn != performance_factor:
                    print(f"  performance_factor: {performance_factor} -> {performance_factor_cn}")
            else:
                conn.execute("""
                    UPDATE kb_links
                    SET process_factor = ?,
                        morphology_factor = ?,
                        performance_factor = ?
                    WHERE id = ?
                """, (process_factor_cn, morphology_factor_cn, performance_factor_cn, row_id))

            updated_count += 1

        if not dry_run:
            conn.commit()

        print(f"kb_links 因子迁移完成:")
        print(f"  更新记录数: {updated_count}")
        print(f"  跳过记录数: {skipped_count}")

    finally:
        conn.close()


def migrate_kb_msfu_entities(db_path: str, dry_run: bool = False):
    """迁移kb_msfu表的实体字段（解析并转换因子名）"""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    print("\n=== 开始迁移 kb_msfu 表实体字段 ===")

    try:
        # 查询所有记录
        rows = conn.execute("SELECT id, source_entity, target_entity FROM kb_msfu").fetchall()

        updated_count = 0
        skipped_count = 0

        for row in rows:
            row_id = row['id']
            source_entity = row['source_entity']
            target_entity = row['target_entity']

            # 解析并转换实体
            def translate_entity(entity):
                if not entity or ':' not in entity:
                    return entity
                entity_type, factor_name = entity.split(':', 1)
                entity_type = entity_type.strip()
                factor_name_cn = translate_factor_name(factor_name.strip())

                # 转换实体类型（确保中文）
                entity_type_mapping = {"process": "工艺", "morphology": "形貌", "performance": "性能", "mechanism": "机理", "evidence": "证据"}
                entity_type_cn = entity_type_mapping.get(entity_type.lower(), entity_type)

                return f"{entity_type_cn}:{factor_name_cn}"

            source_entity_cn = translate_entity(source_entity)
            target_entity_cn = translate_entity(target_entity)

            # 检查是否需要更新
            needs_update = (
                source_entity_cn != source_entity or
                target_entity_cn != target_entity
            )

            if not needs_update:
                skipped_count += 1
                continue

            if dry_run:
                print(f"[DRY RUN] 将更新记录 {row_id}:")
                if source_entity_cn != source_entity:
                    print(f"  source_entity: {source_entity} -> {source_entity_cn}")
                if target_entity_cn != target_entity:
                    print(f"  target_entity: {target_entity} -> {target_entity_cn}")
            else:
                conn.execute("""
                    UPDATE kb_msfu
                    SET source_entity = ?,
                        target_entity = ?
                    WHERE id = ?
                """, (source_entity_cn, target_entity_cn, row_id))

            updated_count += 1

        if not dry_run:
            conn.commit()

        print(f"kb_msfu 实体迁移完成:")
        print(f"  更新记录数: {updated_count}")
        print(f"  跳过记录数: {skipped_count}")

    finally:
        conn.close()


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='因子名称中英文迁移工具')
    parser.add_argument('--dry-run', action='store_true', help='预览模式，不实际修改数据库')
    parser.add_argument('--execute', action='store_true', help='执行迁移，实际修改数据库')

    args = parser.parse_args()

    # 如果没有参数，使用交互模式
    if not args.dry_run and not args.execute:
        print("因子名称中英文迁移工具")
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
        print("因子名称中英文迁移工具")
        print("=" * 50)

        if dry_run:
            print("\n模式: DRY RUN（预览，不实际修改数据库）")
        elif args.execute:
            print("\n模式: 执行迁移（将实际修改数据库）")
        else:
            print("\n错误：必须指定 --dry-run 或 --execute")
            return

    # 迁移kb_links的因子字段
    migrate_kb_links_factors(DB_PATH, dry_run=dry_run)

    # 迁移kb_msfu的实体字段
    migrate_kb_msfu_entities(DB_PATH, dry_run=dry_run)

    if not dry_run:
        print("\n所有迁移完成！")
        print("请重新启动后端服务以应用更改。")
    else:
        print("\n预览完成！")
        print("如要实际执行，请运行: python migrate_factor_names.py --execute")


if __name__ == "__main__":
    main()
