"""
实验 2: 深度扩展功能验证
Experiment 2: Depth Extension Functionality Verification
"""

import sys
import os
import json
from pathlib import Path
from typing import Dict, List

# 添加 backend 路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
backend_path = os.path.join(project_root, "backend")
sys.path.insert(0, backend_path)

from core.knowledge_base import KnowledgeBaseService


class DepthExtensionExperiment:
    """深度扩展实验"""

    def __init__(self, kb_path: str):
        self.kb_path = kb_path
        self.kb = None

    def init_kb(self):
        """初始化知识库"""
        self.kb = KnowledgeBaseService(self.kb_path)

    def experiment_depth_comparison(self) -> Dict:
        """实验 2.1: 深度对比实验

        对比无深度扩展 vs max_depth=2 的检索效果
        """
        print("\n实验 2.1: 深度对比实验")
        print("-" * 60)

        query = "温度对直径的影响"
        task_name = "morphology_interpretation"
        top_k = 5

        results = {}

        # 无深度扩展（原始实现，max_depth=1）
        print(f"\n测试: max_depth=1 (原始)")
        result_depth1 = self.kb.tccer_retrieve(
            query=query,
            task_name=task_name,
            top_k=top_k,
            max_depth=1
        )
        results["max_depth_1"] = self._analyze_result(result_depth1)

        # 有深度扩展
        print(f"\n测试: max_depth=2")
        result_depth2 = self.kb.tccer_retrieve(
            query=query,
            task_name=task_name,
            top_k=top_k,
            max_depth=2
        )
        results["max_depth_2"] = self._analyze_result(result_depth2)

        return {
            "query": query,
            "task": task_name,
            "top_k": top_k,
            "results": results
        }

    def experiment_three_layer_chain(self) -> Dict:
        """实验 2.2: 三层关系链构建实验

        验证能否构建 Process → Mechanism → Morphology 三层关系链
        """
        print("\n实验 2.2: 三层关系链构建实验")
        print("-" * 60)

        query = "生长温度通过机理影响取向"
        task_name = "process_analysis"
        top_k = 3
        max_depth = 3

        result = self.kb.tccer_retrieve(
            query=query,
            task_name=task_name,
            top_k=top_k,
            max_depth=max_depth
        )

        analyzed = self._analyze_result(result)

        # 检查三层关系链
        has_process_to_mechanism = False
        has_mechanism_to_morphology = False

        for path in result.get('results', []):
            for rel in path.get('relations', []):
                rel_type = rel.get('type', '')
                if rel_type == 'process_to_mechanism':
                    has_process_to_mechanism = True
                if rel_type == 'mechanism_to_morphology':
                    has_mechanism_to_morphology = True

        three_layer_success = has_process_to_mechanism and has_mechanism_to_morphology

        return {
            "query": query,
            "task": task_name,
            "top_k": top_k,
            "max_depth": max_depth,
            "results": analyzed,
            "three_layer_chain": {
                "has_process_to_mechanism": has_process_to_mechanism,
                "has_mechanism_to_morphology": has_mechanism_to_morphology,
                "success": three_layer_success
            }
        }

    def experiment_depth_sweep(self) -> Dict:
        """实验 2.3: 深度扫描实验

        测试 max_depth 从 1 到 3 的性能变化
        """
        print("\n实验 2.3: 深度扫描实验")
        print("-" * 60)

        queries = [
            "温度对直径的影响",
            "生长温度对取向的作用",
            "催化剂厚度对密度的影响"
        ]
        task_name = "morphology_interpretation"
        top_k = 5

        depth_results = {}

        for max_depth in [1, 2, 3]:
            print(f"\n测试 max_depth={max_depth}")

            all_paths = []
            all_scores = []

            for query in queries:
                result = self.kb.tccer_retrieve(
                    query=query,
                    task_name=task_name,
                    top_k=top_k,
                    max_depth=max_depth
                )

                for path in result.get('results', []):
                    all_paths.append(path)
                    all_scores.append(path.get('score', 0))

            # 计算统计信息
            if all_paths:
                avg_depth = sum(p['depth'] for p in all_paths) / len(all_paths)
                max_achieved_depth = max(p['depth'] for p in all_paths)
                avg_score = sum(all_scores) / len(all_scores)
            else:
                avg_depth = 0
                max_achieved_depth = 0
                avg_score = 0

            depth_results[f"max_depth_{max_depth}"] = {
                "num_queries": len(queries),
                "num_paths": len(all_paths),
                "avg_depth": avg_depth,
                "max_achieved_depth": max_achieved_depth,
                "avg_score": avg_score
            }

        return {
            "queries": queries,
            "task": task_name,
            "top_k": top_k,
            "results": depth_results
        }

    def _analyze_result(self, result: Dict) -> Dict:
        """分析检索结果"""
        paths = result.get('results', [])
        path_count = len(paths)

        if path_count == 0:
            return {
                "path_count": 0,
                "avg_depth": 0,
                "max_depth": 0,
                "avg_score": 0
            }

        depths = [p.get('depth', 0) for p in paths]
        scores = [p.get('score', 0) for p in paths]

        return {
            "path_count": path_count,
            "avg_depth": sum(depths) / len(depths),
            "max_depth": max(depths),
            "min_depth": min(depths),
            "avg_score": sum(scores) / len(scores),
            "min_score": min(scores),
            "max_score": max(scores)
        }

    def print_result(self, result: Dict):
        """打印结果"""
        print("\n结果:")
        print(f"  查询: {result['query']}")
        print(f"  任务: {result['task']}")
        print(f"  Top-K: {result['top_k']}")

        if 'results' in result:
            for depth_key, data in result['results'].items():
                print(f"\n  {depth_key}:")
                print(f"    路径数: {data['path_count']}")
                print(f"    平均深度: {data['avg_depth']:.1f}")
                print(f"    最大深度: {data['max_depth']}")
                print(f"    平均评分: {data['avg_score']:.3f}")

        if 'three_layer_chain' in result:
            print(f"\n  三层关系链构建:")
            chain = result['three_layer_chain']
            print(f"    process_to_mechanism: {'✅' if chain['has_process_to_mechanism'] else '❌'}")
            print(f"    mechanism_to_morphology: {'✅' if chain['has_mechanism_to_morphology'] else '❌'}")
            print(f"    整体成功: {'✅' if chain['success'] else '❌'}")

        if 'results' in result and 'max_depth_1' in result['results']:
            print("\n  深度扩展效果:")
            depth1 = result['results']['max_depth_1']
            depth2 = result['results']['max_depth_2']
            depth_improvement = depth2['max_depth'] - depth1['max_depth']
            print(f"    深度提升: {depth1['max_depth']} → {depth2['max_depth']} (+{depth_improvement})")

        if 'results' in result and 'max_depth_1' in result['results']:
            print("\n  深度对比:")
            for i in range(1, 4):
                depth_key = f"max_depth_{i}"
                if depth_key in result['results']:
                    data = result['results'][depth_key]
                    print(f"    {depth_key}:")
                    print(f"      路径数: {data['num_paths']}")
                    print(f"      平均深度: {data['avg_depth']:.2f}")
                    print(f"      最大深度: {data['max_achieved_depth']}")

    def save_result(self, result: Dict, output_path: str):
        """保存结果到 JSON 文件"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存到: {output_path}")


def main():
    """主函数"""
    # 设置路径
    base_dir = Path(__file__).parent
    kb_path = base_dir.parent.parent / "database" / "cnta_knowledge_base.sqlite"
    output_dir = base_dir / "results"
    output_dir.mkdir(exist_ok=True)

    print("=" * 80)
    print("第三章实验 2: 深度扩展功能验证")
    print("=" * 80)
    print(f"知识库路径: {kb_path}")

    if not kb_path.exists():
        print(f"错误：知识库数据库不存在: {kb_path}")
        return

    # 初始化知识库
    experiment = DepthExtensionExperiment(str(kb_path))
    experiment.init_kb()

    # 运行实验 2.1: 深度对比
    result1 = experiment.experiment_depth_comparison()
    experiment.print_result(result1)
    output1 = output_dir / "exp_02_1_depth_comparison.json"
    experiment.save_result(result1, str(output1))

    # 运行实验 2.2: 三层关系链
    result2 = experiment.experiment_three_layer_chain()
    experiment.print_result(result2)
    output2 = output_dir / "exp_02_2_three_layer_chain.json"
    experiment.save_result(result2, str(output2))

    # 运行实验 2.3: 深度扫描
    result3 = experiment.experiment_depth_sweep()
    experiment.print_result(result3)
    output3 = output_dir / "exp_02_3_depth_sweep.json"
    experiment.save_result(result3, str(output3))

    print("\n" + "=" * 80)
    print("实验 2 完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
