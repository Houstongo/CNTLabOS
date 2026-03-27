"""
实验 2: 深度扩展功能验证（修复版）
Experiment 2: Depth Extension Functionality Verification (Fixed)
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import Dict

# 设置 UTF-8 编码输出
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

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
        """实验 2.1: 深度对比实验"""
        print("\n实验 2.1: 深度对比实验")
        print("-" * 60)

        query = "温度对直径的影响"
        task_name = "morphology_interpretation"
        top_k = 5

        results = {}

        # max_depth=1
        print(f"\n测试: max_depth=1 (原始)")
        try:
            result1 = self.kb.tccer_retrieve(
                query=query,
                task_name=task_name,
                top_k=top_k,
                max_depth=1
            )
            results["max_depth_1"] = self._analyze_result(result1)
            print(f"  成功: 路径数={results['max_depth_1']['path_count']}, 平均深度={results['max_depth_1']['avg_depth']:.1f}")
        except Exception as e:
            print(f"  失败: {e}")
            results["max_depth_1"] = {"path_count": 0, "avg_depth": 0, "max_depth": 0, "avg_score": 0}

        # max_depth=2
        print(f"\n测试: max_depth=2")
        try:
            result2 = self.kb.tccer_retrieve(
                query=query,
                task_name=task_name,
                top_k=top_k,
                max_depth=2
            )
            results["max_depth_2"] = self._analyze_result(result2)
            print(f"  成功: 路径数={results['max_depth_2']['path_count']}, 平均深度={results['max_depth_2']['avg_depth']:.1f}")
        except Exception as e:
            print(f"  失败: {e}")
            results["max_depth_2"] = {"path_count": 0, "avg_depth": 0, "max_depth": 0, "avg_score": 0}

        # 计算深度提升
        depth_improvement = results["max_depth_2"]["max_depth"] - results["max_depth_1"]["max_depth"]

        return {
            "query": query,
            "task": task_name,
            "top_k": top_k,
            "results": results,
            "depth_improvement": depth_improvement
        }

    def experiment_three_layer_chain(self) -> Dict:
        """实验 2.2: 三层关系链构建实验"""
        print("\n实验 2.2: 三层关系链构建实验")
        print("-" * 60)

        query = "生长温度通过机理影响取向"
        task_name = "process_analysis"
        top_k = 3
        max_depth = 3

        print(f"\n测试查询: {query}")

        try:
            result = self.kb.tccer_retrieve(
                query=query,
                task_name=task_name,
                top_k=top_k,
                max_depth=max_depth
            )

            # 分析三层关系链
            paths = result.get('results', [])
            has_process_to_mechanism = False
            has_mechanism_to_morphology = False

            print(f"\n找到路径数: {len(paths)}")

            for i, path in enumerate(paths, 1):
                relations = path.get('relations', [])
                depth = path.get('depth', 0)
                print(f"\n路径 {i}: 深度={depth}, 关系数={len(relations)}")

                for j, rel in enumerate(relations, 1):
                    rel_type = rel.get('type', '')
                    print(f"  {j}. {rel_type}")
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
                "three_layer_chain": {
                    "has_process_to_mechanism": has_process_to_mechanism,
                    "has_mechanism_to_morphology": has_mechanism_to_morphology,
                    "success": three_layer_success
                }
            }
        except Exception as e:
            print(f"  失败: {e}")
            return {
                "query": query,
                "task": task_name,
                "top_k": top_k,
                "max_depth": max_depth,
                "error": str(e)
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
            "avg_score": sum(scores) / len(scores)
        }


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
    output1 = output_dir / "exp_02_1_depth_comparison.json"
    with open(output1, 'w', encoding='utf-8') as f:
        json.dump(result1, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到: {output1}")

    # 运行实验 2.2: 三层关系链
    print("\n" + "=" * 80)
    result2 = experiment.experiment_three_layer_chain()
    output2 = output_dir / "exp_02_2_three_layer_chain.json"
    with open(output2, 'w', encoding='utf-8') as f:
        json.dump(result2, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到: {output2}")

    # 总结
    print("\n" + "=" * 80)
    print("实验 2 完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
