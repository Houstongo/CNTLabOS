"""
半自动标注脚本
Semi-Automatic Annotation Script

基于 TCCER 检索结果和相似度生成查询标注
"""

import sys
import os
import sqlite3
import json
from pathlib import Path
from typing import Dict, List, Set
import io

# 设置 UTF-8 编码输出
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 添加 backend 路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
backend_path = os.path.join(project_root, "backend")
sys.path.insert(0, backend_path)

from core.knowledge_base import KnowledgeBaseService
from sentence_transformers import SentenceTransformer
import torch


class AutoAnnotator:
    """半自动标注器"""

    def __init__(self, kb_path: str):
        self.kb_path = kb_path
        self.kb = None
        self.encoder = None

    def init_kb(self):
        """初始化知识库"""
        self.kb = KnowledgeBaseService(self.kb_path)

    def init_encoder(self):
        """初始化编码器"""
        print("初始化编码器...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"使用设备: {device}")

        self.encoder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device=device)

    def get_queries(self) -> Dict[str, List[str]]:
        """获取查询列表"""
        return {
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

    def extract_entities(self, query: str) -> Set[str]:
        """从查询中提取实体关键词"""
        # CNT 领域的实体列表
        entities = set()

        # 工艺因子
        process_factors = [
            '温度', '生长温度', '退火温度', '基底温度',
            '催化剂厚度', '催化剂层厚度', '催化剂颗粒',
            '生长时间', '退火时间', '反应时间',
            '气体流量', '气流速度', 'ar_flow', 'h2_flow', 'c2h4_flow',
            '气体压强', '载气', '气氛组成',
            'fe_thickness', 'growth_temp', 'anneal_temp', 'growth_time',
            'ar_flow', 'h2_flow', 'c2h4_flow', 'al2o3_power', 'fe_power',
            'al2o3_thickness', 'position'
        ]

        # 形貌因子
        morphology_factors = [
            '直径', '密度', '取向', '取向度', '曲率', '曲率半径',
            '高度', '阵列高度', '覆盖密度', '形貌',
            'diameter', 'density', 'alignment', 'curvature', 'height'
        ]

        # 机理词汇
        mechanism_words = [
            '机理', '机制', '影响', '作用', '关系', '调控', '扩散',
            '失活', '裂解', '反应', '传质', '相态', '活性位点',
            'deactivation', 'diffusion', 'mechanism', 'reaction'
        ]

        # 提取实体
        query_lower = query.lower()

        for factor in process_factors + morphology_factors + mechanism_words:
            if factor.lower() in query_lower:
                entities.add(factor)

        # 提取参数名
        for word in query_lower.split():
            if '_' in word and any(keyword in word for keyword in ['flow', 'temp', 'thickness', 'time']):
                entities.add(word)

        return entities

    def calculate_relevance_score(self, query: str, msfu: Dict, path: Dict) -> float:
        """计算相关性分数

        综合多个维度：
        1. TCCER 评分 (score)
        2. 路径深度 (depth)
        3. 实体匹配 (entity_match)
        4. 语义相似度 (semantic_sim)
        """
        tccer_score = path.get('score', 0)

        # 实体匹配分数
        query_entities = self.extract_entities(query)

        msfu_entities = set()
        if msfu.get('process_factor'):
            msfu_entities.add(msfu['process_factor'])
        if msfu.get('morphology_factor'):
            msfu_entities.add(msfu['morphology_factor'])
        if msfu.get('performance_factor'):
            msfu_entities.add(msfu['performance_factor'])

        entity_match = len(query_entities & msfu_entities) / max(len(query_entities), 1)

        # 综合分数
        relevance_score = (
            0.5 * tccer_score +
            0.3 * entity_match +
            0.2 * (1.0 if path.get('depth', 0) >= 2 else 0.5)
        )

        return relevance_score

    def annotate_query(self, query: str, task_type: str, top_k: int = 5) -> Dict:
        """标注单个查询

        返回相关 MSFU 列表和相关性分数
        """
        print(f"\n标注查询: {query}")

        # 使用 TCCER 检索
        result = self.kb.tccer_retrieve(
            query=query,
            task_name=task_type,
            top_k=top_k,
            max_depth=2
        )

        paths = result.get('results', [])

        annotations = []

        for i, path in enumerate(paths, 1):
            relations = path.get('relations', [])
            if not relations:
                continue

            # 取第一个关系作为主要关系
            msfu = {
                'source_node': relations[0].get('source_node', ''),
                'target_node': relations[0].get('target_node', ''),
                'relation_type': relations[0].get('type', ''),
                'process_factor': relations[0].get('process_factor', ''),
                'morphology_factor': relations[0].get('morphology_factor', ''),
                'performance_factor': relations[0].get('performance_factor', ''),
                'effect_direction': relations[0].get('effect_direction', ''),
                'mechanism_summary': relations[0].get('mechanism_summary', ''),
                'confidence': relations[0].get('confidence', 0)
            }

            # 计算相关性分数
            relevance_score = self.calculate_relevance_score(query, msfu, path)

            annotations.append({
                'path_id': i,
                'msfu': msfu,
                'tccer_score': path.get('score', 0),
                'depth': path.get('depth', 0),
                'relevance_score': relevance_score,
                'is_relevant': relevance_score > 0.4  # 阈值
            })

            print(f"  {i}. 关系: {msfu['relation_type']}, 评分: {path.get('score', 0):.3f}, 相关性: {relevance_score:.3f}")

        # 过滤出相关的 MSFU
        relevant_msfus = [
            {
                'msfu_id': ann['path_id'],
                'source_node': ann['msfu']['source_node'],
                'target_node': ann['msfu']['target_node'],
                'relation_type': ann['msfu']['relation_type'],
                'relevance_score': ann['relevance_score']
            }
            for ann in annotations
            if ann['is_relevant']
        ]

        return {
            'query': query,
            'task_type': task_type,
            'total_paths': len(paths),
            'relevant_paths': len([ann for ann in annotations if ann['is_relevant']]),
            'all_annotations': annotations,
            'relevant_msfus': relevant_msfus
        }

    def generate_annotation_dataset(self, output_path: str):
        """生成标注数据集"""
        print("=" * 80)
        print("半自动标注数据集生成")
        print("=" * 80)

        queries = self.get_queries()

        dataset = {
            'metadata': {
                'total_queries': sum(len(qs) for qs in queries.values()),
                'task_types': list(queries.keys()),
                'annotation_method': 'semi-automatic',
                'annotation_date': '2026-03-26',
                'note': '基于 TCCER 检索结果和实体匹配生成，未经人工审核'
            },
            'annotations': []
        }

        query_id = 1

        for task_type, query_list in queries.items():
            print(f"\n处理任务类型: {task_type}")
            print(f"查询数: {len(query_list)}")

            for query in query_list:
                annotation = self.annotate_query(query, task_type)

                annotation['query_id'] = f"{task_type}_{query_id:03d}"
                annotation['note'] = '需人工审核'

                dataset['annotations'].append(annotation)
                query_id += 1

        # 保存标注数据集
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)

        print("\n" + "=" * 80)
        print("标注完成！")
        print("=" * 80)

        # 打印统计信息
        total_queries = len(dataset['annotations'])
        total_relevant = sum(ann['relevant_paths'] for ann in dataset['annotations'])
        avg_relevance = sum(
            ann['relevant_paths'] / ann['total_paths'] if ann['total_paths'] > 0 else 0
            for ann in dataset['annotations']
        ) / total_queries

        print(f"\n统计信息:")
        print(f"  总查询数: {total_queries}")
        print(f"  总相关路径数: {total_relevant}")
        print(f"  平均相关率: {avg_relevance:.1%}")
        print(f"\n标注文件: {output_path}")
        print(f"\n注意事项:")
        print(f"  - 这是半自动标注，需要人工审核")
        print(f"  - 相关性阈值设为 0.4，可根据实际情况调整")
        print(f"  - 建议抽样 10-20 个查询进行人工复核")


def main():
    """主函数"""
    # 设置路径
    project_root = Path(__file__).parent.parent.parent
    kb_path = project_root / "database" / "cnta_knowledge_base.sqlite"
    output_dir = Path(__file__).parent / "dataset"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "auto_annotations.json"

    print(f"知识库路径: {kb_path}")
    print(f"输出路径: {output_path}")

    if not kb_path.exists():
        print(f"错误：知识库数据库不存在: {kb_path}")
        return

    # 初始化标注器
    annotator = AutoAnnotator(str(kb_path))
    annotator.init_kb()
    annotator.init_encoder()

    # 生成标注数据集
    annotator.generate_annotation_dataset(str(output_path))


if __name__ == "__main__":
    main()
