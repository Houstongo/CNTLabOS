"""
实验 1: 知识库数据统计与 MSFU 构建分析
Experiment 1: Knowledge Base Statistics and MSFU Construction Analysis
"""

import sqlite3
import json
from pathlib import Path
from collections import Counter
from typing import Dict, List


class DataStatisticsCollector:
    """数据统计收集器"""

    def __init__(self, kb_path: str):
        self.kb_path = kb_path
        self.conn = None

    def connect(self):
        """连接数据库"""
        self.conn = sqlite3.connect(self.kb_path)
        self.conn.row_factory = sqlite3.Row

    def close(self):
        """关闭连接"""
        if self.conn:
            self.conn.close()

    def collect_document_stats(self) -> Dict:
        """收集文档统计信息"""
        cursor = self.conn.execute("SELECT COUNT(*) as count FROM kb_documents")
        doc_count = cursor.fetchone()["count"]

        cursor = self.conn.execute("""
            SELECT COUNT(*) as count
            FROM kb_documents
            WHERE source_type = 'paper'
        """)
        pdf_count = cursor.fetchone()["count"]

        cursor = self.conn.execute("""
            SELECT AVG(chunk_count) as avg_chunks
            FROM (
                SELECT d.id, COUNT(c.id) as chunk_count
                FROM kb_documents d
                LEFT JOIN kb_chunks c ON c.doc_id = d.id
                GROUP BY d.id
            )
        """)
        avg_chunks_per_doc = cursor.fetchone()["avg_chunks"]

        return {
            "total_documents": doc_count,
            "pdf_documents": pdf_count,
            "avg_chunks_per_doc": avg_chunks_per_doc or 0
        }

    def collect_chunk_stats(self) -> Dict:
        """收集 Chunk 统计信息"""
        cursor = self.conn.execute("SELECT COUNT(*) as count FROM kb_chunks")
        chunk_count = cursor.fetchone()["count"]

        cursor = self.conn.execute("""
            SELECT AVG(LENGTH(text)) as avg_length,
                   MIN(LENGTH(text)) as min_length,
                   MAX(LENGTH(text)) as max_length
            FROM kb_chunks
        """)
        length_stats = cursor.fetchone()

        cursor = self.conn.execute("""
            SELECT knowledge_type, COUNT(*) as count
            FROM kb_chunks
            GROUP BY knowledge_type
        """)
        knowledge_type_dist = {row["knowledge_type"]: row["count"] for row in cursor.fetchall()}

        return {
            "total_chunks": chunk_count,
            "avg_chunk_length": length_stats["avg_length"] or 0,
            "min_chunk_length": length_stats["min_length"] or 0,
            "max_chunk_length": length_stats["max_length"] or 0,
            "knowledge_type_distribution": knowledge_type_dist
        }

    def collect_link_stats(self) -> Dict:
        """收集 Link 统计信息"""
        cursor = self.conn.execute("SELECT COUNT(*) as count FROM kb_links")
        link_count = cursor.fetchone()["count"]

        cursor = self.conn.execute("""
            SELECT relation_type, COUNT(*) as count
            FROM kb_links
            GROUP BY relation_type
            ORDER BY count DESC
        """)
        relation_type_dist = {row["relation_type"]: row["count"] for row in cursor.fetchall()}

        cursor = self.conn.execute("""
            SELECT AVG(confidence) as avg_confidence,
                   MIN(confidence) as min_confidence,
                   MAX(confidence) as max_confidence
            FROM kb_links
        """)
        confidence_stats = cursor.fetchone()

        # 计算平均每篇文档的 Link 数
        cursor = self.conn.execute("""
            SELECT AVG(link_count) as avg_links_per_doc
            FROM (
                SELECT d.id, COUNT(l.id) as link_count
                FROM kb_documents d
                LEFT JOIN kb_links l ON l.doc_id = d.id
                GROUP BY d.id
            )
        """)
        avg_links_per_doc = cursor.fetchone()["avg_links_per_doc"]

        return {
            "total_links": link_count,
            "relation_type_distribution": relation_type_dist,
            "avg_confidence": confidence_stats["avg_confidence"] or 0,
            "min_confidence": confidence_stats["min_confidence"] or 0,
            "max_confidence": confidence_stats["max_confidence"] or 0,
            "avg_links_per_doc": avg_links_per_doc or 0
        }

    def collect_msfu_field_stats(self) -> Dict:
        """收集 MSFU 字段统计信息"""
        # 统计各字段的完整性
        cursor = self.conn.execute("""
            SELECT
                COUNT(CASE WHEN source_node IS NOT NULL AND source_node != '' THEN 1 END) as source_entity_count,
                COUNT(CASE WHEN target_node IS NOT NULL AND target_node != '' THEN 1 END) as target_entity_count,
                COUNT(CASE WHEN process_factor IS NOT NULL AND process_factor != '' THEN 1 END) as process_factor_count,
                COUNT(CASE WHEN morphology_factor IS NOT NULL AND morphology_factor != '' THEN 1 END) as morphology_factor_count,
                COUNT(CASE WHEN performance_factor IS NOT NULL AND performance_factor != '' THEN 1 END) as performance_factor_count,
                COUNT(CASE WHEN effect_direction IS NOT NULL AND effect_direction != '' THEN 1 END) as direction_count,
                COUNT(CASE WHEN mechanism_summary IS NOT NULL AND mechanism_summary != '' THEN 1 END) as mechanism_count,
                COUNT(CASE WHEN evidence_text IS NOT NULL AND evidence_text != '' THEN 1 END) as evidence_count,
                COUNT(*) as total_count
            FROM kb_links
        """)
        stats = cursor.fetchone()
        total = stats["total_count"]

        return {
            "source_entity_completeness": (stats["source_entity_count"] / total * 100) if total > 0 else 0,
            "target_entity_completeness": (stats["target_entity_count"] / total * 100) if total > 0 else 0,
            "process_factor_completeness": (stats["process_factor_count"] / total * 100) if total > 0 else 0,
            "morphology_factor_completeness": (stats["morphology_factor_count"] / total * 100) if total > 0 else 0,
            "performance_factor_completeness": (stats["performance_factor_count"] / total * 100) if total > 0 else 0,
            "direction_completeness": (stats["direction_count"] / total * 100) if total > 0 else 0,
            "mechanism_completeness": (stats["mechanism_count"] / total * 100) if total > 0 else 0,
            "evidence_completeness": (stats["evidence_count"] / total * 100) if total > 0 else 0,
        }

    def generate_report(self) -> Dict:
        """生成统计报告"""
        self.connect()

        try:
            report = {
                "documents": self.collect_document_stats(),
                "chunks": self.collect_chunk_stats(),
                "links": self.collect_link_stats(),
                "msfu_fields": self.collect_msfu_field_stats()
            }
            return report
        finally:
            self.close()


def print_statistics_report(report: Dict):
    """打印统计报告"""

    print("=" * 80)
    print("第三章实验 1: 知识库数据统计")
    print("=" * 80)

    # 文档统计
    print("\n【文档统计】")
    print(f"  总文档数: {report['documents']['total_documents']}")
    print(f"  PDF 文档数: {report['documents']['pdf_documents']}")
    print(f"  平均每文档 Chunk 数: {report['documents']['avg_chunks_per_doc']:.1f}")

    # Chunk 统计
    print("\n【Chunk 统计】")
    print(f"  总 Chunk 数: {report['chunks']['total_chunks']}")
    print(f"  平均 Chunk 长度: {report['chunks']['avg_chunk_length']:.0f} 字符")
    print(f"  Chunk 长度范围: {report['chunks']['min_chunk_length']} - {report['chunks']['max_chunk_length']}")
    print("\n  知识类型分布:")
    for ktype, count in report['chunks']['knowledge_type_distribution'].items():
        print(f"    {ktype}: {count}")

    # Link 统计
    print("\n【Link 统计】")
    print(f"  总 Link 数: {report['links']['total_links']}")
    print(f"  平均每文档 Link 数: {report['links']['avg_links_per_doc']:.1f}")
    print(f"  平均置信度: {report['links']['avg_confidence']:.3f}")
    print(f"  置信度范围: {report['links']['min_confidence']:.3f} - {report['links']['max_confidence']:.3f}")
    print("\n  关系类型分布:")
    for rtype, count in report['links']['relation_type_distribution'].items():
        percentage = (count / report['links']['total_links'] * 100)
        bar_length = int(percentage / 3)
        bar = "█" * bar_length
        print(f"    {rtype}: {count:3d} ({percentage:5.1f}%) {bar}")

    # MSFU 字段统计
    print("\n【MSFU 字段完整性统计】")
    fields = {
        "source_entity_completeness": "源实体",
        "target_entity_completeness": "目标实体",
        "process_factor_completeness": "工艺因子",
        "morphology_factor_completeness": "形貌因子",
        "performance_factor_completeness": "性能因子",
        "direction_completeness": "作用方向",
        "mechanism_completeness": "机理",
        "evidence_completeness": "证据"
    }
    for field_key, field_name in fields.items():
        value = report['msfu_fields'][field_key]
        print(f"  {field_name}: {value:.1f}%")

    print("\n" + "=" * 80)


def save_report_to_json(report: Dict, output_path: str):
    """保存报告到 JSON 文件"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n报告已保存到: {output_path}")


def main():
    """主函数"""
    # 设置路径
    base_dir = Path(__file__).parent
    kb_path = base_dir.parent.parent / "database" / "cnta_knowledge_base.sqlite"
    output_dir = base_dir / "results"
    output_dir.mkdir(exist_ok=True)

    print(f"知识库路径: {kb_path}")

    if not kb_path.exists():
        print(f"错误：知识库数据库不存在: {kb_path}")
        return

    # 收集统计
    collector = DataStatisticsCollector(str(kb_path))
    report = collector.generate_report()

    # 打印报告
    print_statistics_report(report)

    # 保存报告
    output_path = output_dir / "exp_01_data_statistics.json"
    save_report_to_json(report, str(output_path))


if __name__ == "__main__":
    main()
