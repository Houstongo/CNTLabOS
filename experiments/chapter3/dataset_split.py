"""
数据集划分脚本
Dataset Split Script

按照论文比例划分 MSFU 数据集：
- 训练集：70%
- 验证集：15%
- 测试集：15%

遵循来源隔离和样品隔离原则
"""

import sys
import os
import sqlite3
import json
from pathlib import Path
from typing import Dict, List, Tuple
import random
import io

# 设置 UTF-8 编码输出
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 添加 backend 路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
backend_path = os.path.join(project_root, "backend")
sys.path.insert(0, backend_path)


def get_link_data(kb_path: str) -> List[Dict]:
    """获取所有 Link 数据"""
    conn = sqlite3.connect(kb_path)
    conn.row_factory = sqlite3.Row

    cursor = conn.execute("""
        SELECT
            id,
            doc_id,
            source_node,
            target_node,
            relation_type,
            process_factor,
            morphology_factor,
            performance_factor,
            effect_direction,
            mechanism_summary,
            evidence_text,
            confidence,
            chunk_id
        FROM kb_links
        WHERE confidence > 0.3
    """)

    links = []
    for row in cursor.fetchall():
        links.append(dict(row))

    conn.close()
    return links


def get_document_sources(kb_path: str) -> Dict[int, str]:
    """获取文档来源信息"""
    conn = sqlite3.connect(kb_path)
    conn.row_factory = sqlite3.Row

    cursor = conn.execute("""
        SELECT id, source_type, title, file_path
        FROM kb_documents
    """)

    doc_sources = {}
    for row in cursor.fetchall():
        # 使用 doc_id 作为隔离标识
        doc_sources[row["id"]] = f"doc_{row['id']}"

    conn.close()
    return doc_sources


def create_isolation_groups(links: List[Dict], doc_sources: Dict[int, str]) -> Dict[int, str]:
    """创建隔离组

    同一文档的 links 属于同一组
    """
    groups = {}
    for link in links:
        doc_id = link["doc_id"]
        source = doc_sources.get(doc_id, f"doc_{doc_id}")
        groups[link["id"]] = source
    return groups


def split_by_groups(
    links: List[Dict],
    groups: Dict[int, str],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """按组划分数据集

    确保同一组的 links 不会分散到不同子集
    """
    # 收集所有组
    all_groups = set(groups.values())

    # 随机打乱组
    shuffled_groups = list(all_groups)
    random.shuffle(shuffled_groups)

    # 计算划分点
    n_groups = len(shuffled_groups)
    train_end = int(n_groups * train_ratio)
    val_end = train_end + int(n_groups * val_ratio)

    # 划分组
    train_groups = set(shuffled_groups[:train_end])
    val_groups = set(shuffled_groups[train_end:val_end])
    test_groups = set(shuffled_groups[val_end:])

    # 按组划分 links
    train_links = []
    val_links = []
    test_links = []

    for link in links:
        link_id = link["id"]
        group = groups[link_id]

        if group in train_groups:
            train_links.append(link)
        elif group in val_groups:
            val_links.append(link)
        else:
            test_links.append(link)

    return train_links, val_links, test_links


def save_split_data(
    train_links: List[Dict],
    val_links: List[Dict],
    test_links: List[Dict],
    output_dir: Path
):
    """保存划分后的数据"""
    output_dir.mkdir(exist_ok=True)

    # 保存完整数据（带标注）
    splits = {
        "train": train_links,
        "validation": val_links,
        "test": test_links,
        "statistics": {
            "train_count": len(train_links),
            "val_count": len(val_links),
            "test_count": len(test_links),
            "total_count": len(train_links) + len(val_links) + len(test_links),
            "train_ratio": len(train_links) / (len(train_links) + len(val_links) + len(test_links)),
            "val_ratio": len(val_links) / (len(train_links) + len(val_links) + len(test_links)),
            "test_ratio": len(test_links) / (len(train_links) + len(val_links) + len(test_links))
        }
    }

    output_file = output_dir / "msfu_split.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(splits, f, ensure_ascii=False, indent=2)

    return splits


def create_query_annotation_file(queries: Dict[str, List[str]], output_dir: Path):
    """创建查询标注文件

    按照论文格式，每个查询标注：
    - 查询类型
    - 查询文本
    - 相关 MSFU（占位，需人工标注）
    """
    annotation_data = {
        "metadata": {
            "total_queries": sum(len(qs) for qs in queries.values()),
            "task_types": list(queries.keys()),
            "annotation_status": "pending"
        },
        "queries": []
    }

    for task_type, query_list in queries.items():
        for i, query in enumerate(query_list, 1):
            annotation_data["queries"].append({
                "query_id": f"{task_type}_{i:03d}",
                "task_type": task_type,
                "query_text": query,
                "relevant_msfus": [],  # 待人工标注
                "annotation_notes": ""  # 待人工填写
            })

    output_file = output_dir / "query_annotations.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(annotation_data, f, ensure_ascii=False, indent=2)


def main():
    """主函数"""
    # 设置路径
    project_root = Path(__file__).parent.parent.parent
    kb_path = project_root / "database" / "cnta_knowledge_base.sqlite"
    output_dir = Path(__file__).parent / "dataset"
    output_dir.mkdir(exist_ok=True)

    print("=" * 80)
    print("第三章实验数据集划分")
    print("=" * 80)
    print(f"知识库路径: {kb_path}")
    print(f"输出目录: {output_dir}")

    if not kb_path.exists():
        print(f"错误：知识库数据库不存在: {kb_path}")
        return

    # 读取数据
    print("\n读取数据...")
    links = get_link_data(str(kb_path))
    doc_sources = get_document_sources(str(kb_path))

    print(f"  Link 数量: {len(links)}")
    print(f"  文档数量: {len(doc_sources)}")

    # 创建隔离组
    print("\n创建隔离组...")
    groups = create_isolation_groups(links, doc_sources)
    print(f"  隔离组数量: {len(set(groups.values()))}")

    # 设置随机种子（保证可复现）
    random.seed(42)

    # 划分数据集
    print("\n划分数据集...")
    train_links, val_links, test_links = split_by_groups(links, groups)

    print(f"  训练集: {len(train_links)} 条")
    print(f"  验证集: {len(val_links)} 条")
    print(f"  测试集: {len(test_links)} 条")

    # 保存数据
    print("\n保存数据...")
    splits = save_split_data(train_links, val_links, test_links, output_dir)

    # 打印统计信息
    print("\n划分统计:")
    stats = splits["statistics"]
    print(f"  总数: {stats['total_count']}")
    print(f"  训练集比例: {stats['train_ratio']:.1%}")
    print(f"  验证集比例: {stats['val_ratio']:.1%}")
    print(f"  测试集比例: {stats['test_ratio']:.1%}")

    # 创建查询标注文件
    print("\n创建查询标注文件模板...")

    # 直接定义查询集，避免循环导入
    queries = {
        "process_analysis": [
            "生长温度通过机理影响取向",
            "催化剂失活的影响机制",
            "碳源裂解与生长速率",
            "扩散受限的作用",
            "表面反应的影响",
            "催化剂状态对生长的影响",
            "温度调控的机理",
            "气氛组成的作用机制",
            "生长稳定性的影响因素",
            "生长终止的原因",
            "催化剂颗粒大小对生长的影响",
            "基底温度对形貌的影响机制",
            "气体压强对生长速率的影响",
            "反应时间与生长机理的关系",
            "催化剂形貌对碳管结构的影响",
            "温度梯度对生长模式的影响",
            "载气类型对生长的影响机制",
            "碳源浓度对生长的作用",
            "退火过程的机理分析",
            "催化剂预处理的机理",
            "生长初期与后期的机理差异",
            "多层催化与单层催化的机理对比",
            "添加剂对生长机理的影响",
            "反应器内流动状态对生长的影响",
            "催化剂活性位点的作用机制",
            "温度变化对催化剂相态的影响",
            "气体组成对催化剂表面状态的影响",
            "生长密度对催化剂利用率的影响",
            "生长过程中的传质机理",
            "能量输入对生长反应的影响"
        ],
        "morphology_interpretation": [
            "温度对直径的影响",
            "催化剂厚度对密度的作用",
            "生长时间与取向度的关系",
            "退火温度对形貌的影响",
            "气体流量对曲率的作用",
            "C2H4 流量对生长的影响",
            "H2 流量的作用机制",
            "催化剂活性的影响因素",
            "阵列高度的变化规律",
            "覆盖密度的控制条件",
            "退火时间对直径分布的影响",
            "催化剂层厚度对取向的作用",
            "气体压强对密度的影响",
            "基底位置与形貌的关系",
            "温度对曲率半径的影响",
            "生长时间对高度的作用",
            "催化剂状态对密度的调控",
            "气流速度对形貌的影响",
            "退火气氛对取向的影响",
            "反应温度与密度的关系"
        ],
        "prediction_explanation": [
            "调整 fe_thickness 对直径的影响趋势",
            "改变 growth_temp 对密度的影响",
            "增加 growth_time 对取向的作用",
            "调节 ar_flow 对形貌的影响",
            "提高 h2 流量的效果",
            "降低 c2h4 流量的影响",
            "增加 anneal_temp 的影响",
            "延长 anneal_time 的效果",
            "改变 position 的影响",
            "优化 magnification 的意义",
            "调整 al2o3_power 对形貌的影响",
            "改变 fe_power 对直径的作用",
            "调整 al2o3_thickness 对密度的影响",
            "改变 ar_flow 对取向的影响",
            "调整 h2_flow 对曲率的作用",
            "改变 c2h4_flow 对高度的影响",
            "增加 anneal_temp 对活性的影响",
            "调整 anneal_time 对形貌的作用",
            "改变 growth_temp 对取向的影响",
            "调整 growth_time 对密度的影响"
        ]
    }

    create_query_annotation_file(queries, output_dir)

    print(f"\n  查询总数: {sum(len(qs) for qs in queries.values())}")
    print(f"  工艺分析: {len(queries['process_analysis'])} 个")
    print(f"  形貌解释: {len(queries['morphology_interpretation'])} 个")
    print(f"  预测解释: {len(queries['prediction_explanation'])} 个")

    print("\n" + "=" * 80)
    print("数据集划分完成！")
    print("=" * 80)
    print(f"\n输出文件:")
    print(f"  - {output_dir / 'msfu_split.json'}")
    print(f"  - {output_dir / 'query_annotations.json'}")

    print(f"\n下一步:")
    print(f"  1. 人工标注 query_annotations.json 中的 relevant_msfus 字段")
    print(f"  2. 使用标注结果运行评估实验")


if __name__ == "__main__":
    main()
