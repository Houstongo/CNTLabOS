"""
实验 4: 消融实验
Experiment 4: Ablation Study
"""

import sys
import os
import json
from pathlib import Path
from typing import Dict, List, Callable
from copy import deepcopy

# 设置 UTF-8 编码输出
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 添加 backend 路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
backend_path = os.path.join(project_root, "backend")
sys.path.insert(0, backend_path)

from core.knowledge_base import KnowledgeBaseService


class AblationStudyExperiment:
    """消融实验"""

    def __init__(self, kb_path: str):
        self.kb_path = kb_path
        self.kb = None
        self.original_methods = {}

    def init_kb(self):
        """初始化知识库并保存原始方法"""
        self.kb = KnowledgeBaseService(self.kb_path)

        # 保存原始方法，用于恢复
        self.original_methods = {
            '_constrained_path_expansion': self.kb._constrained_path_expansion,
        }

    def restore_original_methods(self):
        """恢复原始方法"""
        for method_name, method in self.original_methods.items():
            if hasattr(self.kb, method_name):
                setattr(self.kb, method_name, method)

    def experiment_baseline(self, queries: List[str]) -> Dict:
        """实验 4.1: 基准实验（完整 TCCER）"""
        print("\n实验 4.1: 基准实验（完整 TCCER）")
        print("-" * 60)

        results = []
        for query in queries:
            result = self.kb.tccer_retrieve(
                query=query,
                task_name="morphology_interpretation",
                top_k=5,
                max_depth=2
            )
            results.append(self._analyze_result(query, result))

        return {
            "experiment": "baseline",
            "description": "完整 TCCER 算法",
            "results": results
        }

    def experiment_without_relation_expansion(self, queries: List[str]) -> Dict:
        """实验 4.2: 去除关系约束扩展

        模拟仅单跳检索的效果
        """
        print("\n实验 4.2: 去除关系约束扩展")
        print("-" * 60)

        # 临时修改路径扩展方法，强制只执行 1 跳
        def _single_hop_expansion(self, initial_chunks, parsed_query, task_profile, max_depth):
            """仅 1 跳的路径扩展"""
            paths = []
            path_visited = set()

            chunk_relations_map = self._get_chunk_relations_map(
                [chunk["chunk_id"] for chunk in initial_chunks]
            )

            for chunk in initial_chunks:
                chunk_id = chunk["chunk_id"]
                chunk_with_relation = dict(chunk)
                chunk_with_relation["relation"] = chunk_relations_map.get(chunk_id, {}).get("relation", {})

                path = {
                    "chunks": [chunk_with_relation],
                    "relations": [chunk_with_relation.get("relation")],
                    "depth": 1,
                }
                paths.append(path)

            # 不执行深度扩展，直接返回深度为 1 的路径
            return paths

        # 替换原始方法
        self.kb._constrained_path_expansion = lambda *args: _single_hop_expansion(self, *args)

        try:
            results = []
            for query in queries:
                result = self.kb.tccer_retrieve(
                    query=query,
                    task_name="morphology_interpretation",
                    top_k=5,
                    max_depth=2  # 传入 max_depth=2 但实际只返回深度 1
                )
                results.append(self._analyze_result(query, result))

            return {
                "experiment": "without_relation_expansion",
                "description": "去除关系约束扩展（仅单跳）",
                "results": results
            }
        finally:
            self.restore_original_methods()

    def experiment_without_condition_check(self, queries: List[str]) -> Dict:
        """实验 4.3: 去除条件一致性检查

        模拟不考虑条件约束的检索效果
        """
        print("\n实验 4.3: 去除条件一致性检查")
        print("-" * 60)

        # 修改关系扩展方法，跳过条件一致性检查
        def _no_condition_check_expansion(self, initial_chunks, parsed_query, task_profile, max_depth):
            """无条件检查的路径扩展"""
            paths = []
            path_visited = set()

            chunk_relations_map = self._get_chunk_relations_map(
                [chunk["chunk_id"] for chunk in initial_chunks]
            )

            for chunk in initial_chunks:
                chunk_id = chunk["chunk_id"]
                chunk_with_relation = dict(chunk)
                chunk_with_relation["relation"] = chunk_relations_map.get(chunk_id, {}).get("relation", {})

                path = {
                    "chunks": [chunk_with_relation],
                    "relations": [chunk_with_relation.get("relation")],
                    "depth": 1,
                }
                paths.append(path)

                path_visited = {chunk_id}

                # 深度扩展（跳过条件一致性）
                for depth_step in range(1, max_depth):
                    last_chunk = path["chunks"][-1]

                    # 获取相关 chunks，不检查条件一致性
                    next_chunks = self._get_related_chunks(
                        last_chunk["chunk_id"],
                        parsed_query,
                        task_profile,
                        max_hops=1
                    )

                    if not next_chunks:
                        break

                    # 找到第一个未访问的 chunk
                    found_next = False
                    for next_chunk in next_chunks:
                        next_chunk_id = next_chunk["chunk_id"]
                        if next_chunk_id not in path_visited:
                            relations_map = self._get_chunk_relations_map([next_chunk_id])
                            if next_chunk_id in relations_map:
                                next_chunk["relation"] = relations_map[next_chunk_id]["relation"]

                            path["chunks"].append(next_chunk)
                            path["relations"].append(next_chunk.get("relation", {}))
                            path["depth"] += 1
                            path_visited.add(next_chunk_id)
                            found_next = True
                            break

                    if not found_next:
                        break

            return paths

        # 替换原始方法
        self.kb._constrained_path_expansion = lambda *args: _no_condition_check_expansion(self, *args)

        try:
            results = []
            for query in queries:
                result = self.kb.tccer_retrieve(
                    query=query,
                    task_name="morphology_interpretation",
                    top_k=5,
                    max_depth=2
                )
                results.append(self._analyze_result(query, result))

            return {
                "experiment": "without_condition_check",
                "description": "去除条件一致性检查",
                "results": results
            }
        finally:
            self.restore_original_methods()

    def experiment_without_direction_check(self, queries: List[str]) -> Dict:
        """实验 4.4: 去除方向一致性检查

        模拟不考虑方向约束的检索效果
        """
        print("\n实验 4.4: 去除方向一致性检查")
        print("-" * 60)

        # 修改关系扩展方法，跳过方向一致性检查
        def _no_direction_check_expansion(self, initial_chunks, parsed_query, task_profile, max_depth):
            """无方向检查的路径扩展"""
            paths = []
            path_visited = set()

            chunk_relations_map = self._get_chunk_relations_map(
                [chunk["chunk_id"] for chunk in initial_chunks]
            )

            for chunk in initial_chunks:
                chunk_id = chunk["chunk_id"]
                chunk_with_relation = dict(chunk)
                chunk_with_relation["relation"] = chunk_relations_map.get(chunk_id, {}).get("relation", {})

                path = {
                    "chunks": [chunk_with_relation],
                    "relations": [chunk_with_relation.get("relation")],
                    "depth": 1,
                }
                paths.append(path)

                path_visited = {chunk_id}

                # 深度扩展（跳过方向一致性）
                for depth_step in range(1, max_depth):
                    last_chunk = path["chunks"][-1]

                    next_chunks = self._get_related_chunks(
                        last_chunk["chunk_id"],
                        parsed_query,
                        task_profile,
                        max_hops=1
                    )

                    if not next_chunks:
                        break

                    # 找到第一个未访问的 chunk（不检查方向）
                    found_next = False
                    for next_chunk in next_chunks:
                        next_chunk_id = next_chunk["chunk_id"]
                        if next_chunk_id not in path_visited:
                            relations_map = self._get_chunk_relations_map([next_chunk_id])
                            if next_chunk_id in relations_map:
                                next_chunk["relation"] = relations_map[next_chunk_id]["relation"]

                            path["chunks"].append(next_chunk)
                            path["relations"].append(next_chunk.get("relation", {}))
                            path["depth"] += 1
                            path_visited.add(next_chunk_id)
                            found_next = True
                            break

                    if not found_next:
                        break

            return paths

        # 替换原始方法
        self.kb._constrained_path_expansion = lambda *args: _no_direction_check_expansion(self, *args)

        try:
            results = []
            for query in queries:
                result = self.kb.tccer_retrieve(
                    query=query,
                    task_name="morphology_interpretation",
                    top_k=5,
                    max_depth=2
                )
                results.append(self._analyze_result(query, result))

            return {
                "experiment": "without_direction_check",
                "description": "去除方向一致性检查",
                "results": results
            }
        finally:
            self.restore_original_methods()

    def _get_chunk_relations_map(self, chunk_ids):
        """辅助方法：获取 chunks 关系映射"""
        return self.kb._get_chunk_relations_map(chunk_ids)

    def _get_related_chunks(self, chunk_id, parsed_query, task_profile, max_hops):
        """辅助方法：获取相关 chunks"""
        return self.kb._get_related_chunks(chunk_id, parsed_query, task_profile, max_hops)

    def _analyze_result(self, query: str, result: Dict) -> Dict:
        """分析检索结果"""
        paths = result.get('results', [])
        path_count = len(paths)

        if path_count == 0:
            return {
                "query": query,
                "path_count": 0,
                "avg_depth": 0,
                "max_depth": 0,
                "avg_score": 0
            }

        depths = [p.get('depth', 0) for p in paths]
        scores = [p.get('score', 0) for p in paths]

        return {
            "query": query,
            "path_count": path_count,
            "avg_depth": sum(depths) / len(depths),
            "max_depth": max(depths),
            "avg_score": sum(scores) / len(scores)
        }

    def print_result(self, result: Dict):
        """打印结果"""
        print(f"\n实验: {result['experiment']}")
        print(f"描述: {result['description']}")

        # 计算整体指标
        all_results = result['results']
        if all_results:
            avg_depth = sum(r['avg_depth'] for r in all_results) / len(all_results)
            max_depth = max(r['max_depth'] for r in all_results)
            avg_score = sum(r['avg_score'] for r in all_results) / len(all_results)

            print(f"\n整体指标:")
            print(f"  平均深度: {avg_depth:.2f}")
            print(f"  最大深度: {max_depth}")
            print(f"  平均评分: {avg_score:.3f}")

            # 显示每个查询的结果
            print(f"\n各查询结果:")
            for r in all_results[:3]:  # 只显示前 3 个
                print(f"  {r['query'][:30]}...: 深度={r['avg_depth']:.1f}, 评分={r['avg_score']:.3f}")

    def calculate_comparison(self, baseline: Dict, variants: List[Dict]) -> Dict:
        """计算对比结果"""
        comparison = []

        baseline_avg = {
            "avg_depth": sum(r['avg_depth'] for r in baseline['results']) / len(baseline['results']),
            "avg_score": sum(r['avg_score'] for r in baseline['results']) / len(baseline['results']),
        }

        for variant in variants:
            variant_avg = {
                "avg_depth": sum(r['avg_depth'] for r in variant['results']) / len(variant['results']),
                "avg_score": sum(r['avg_score'] for r in variant['results']) / len(variant['results']),
            }

            comparison.append({
                "experiment": variant['experiment'],
                "description": variant['description'],
                "depth_change": variant_avg['avg_depth'] - baseline_avg['avg_depth'],
                "depth_change_pct": ((variant_avg['avg_depth'] - baseline_avg['avg_depth']) / baseline_avg['avg_depth'] * 100) if baseline_avg['avg_depth'] > 0 else 0,
                "score_change": variant_avg['avg_score'] - baseline_avg['avg_score'],
                "score_change_pct": ((variant_avg['avg_score'] - baseline_avg['avg_score']) / baseline_avg['avg_score'] * 100) if baseline_avg['avg_score'] > 0 else 0,
            })

        return {
            "baseline": baseline_avg,
            "variants": comparison
        }

    def print_comparison(self, comparison: Dict):
        """打印对比结果"""
        print("\n" + "=" * 80)
        print("消融实验对比")
        print("=" * 80)

        print("\n基准指标:")
        baseline = comparison['baseline']
        print(f"  平均深度: {baseline['avg_depth']:.2f}")
        print(f"  平均评分: {baseline['avg_score']:.3f}")

        print("\n变体指标变化:")
        print(f"{'实验':<35} {'深度变化':<15} {'深度变化%':<12} {'评分变化':<15} {'评分变化%':<12}")
        print("-" * 80)

        for variant in comparison['variants']:
            exp = variant['experiment']
            desc = variant['description'][:30]
            depth_change = variant['depth_change']
            depth_pct = variant['depth_change_pct']
            score_change = variant['score_change']
            score_pct = variant['score_change_pct']

            depth_sign = "+" if depth_change > 0 else ""
            score_sign = "+" if score_change > 0 else ""

            print(f"{exp:<35} {depth_sign}{depth_change:<14.1f} {depth_sign}{depth_pct:<11.1f}% {score_sign}{score_change:<14.3f} {score_sign}{score_pct:<11.1f}%")

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
    print("第三章实验 4: 消融实验")
    print("=" * 80)
    print(f"知识库路径: {kb_path}")

    if not kb_path.exists():
        print(f"错误：知识库数据库不存在: {kb_path}")
        return

    # 测试查询
    queries = [
        "温度对直径的影响",
        "催化剂厚度对密度的作用",
        "生长时间与取向度的关系",
        "退火温度对形貌的影响",
        "气体流量对曲率的作用"
    ]

    # 初始化实验
    experiment = AblationStudyExperiment(str(kb_path))
    experiment.init_kb()

    try:
        # 运行基准实验
        baseline = experiment.experiment_baseline(queries)
        experiment.print_result(baseline)

        # 运行消融实验
        variants = []
        variants.append(experiment.experiment_without_relation_expansion(queries))
        experiment.print_result(variants[-1])

        variants.append(experiment.experiment_without_condition_check(queries))
        experiment.print_result(variants[-1])

        variants.append(experiment.experiment_without_direction_check(queries))
        experiment.print_result(variants[-1])

        # 计算对比结果
        comparison = experiment.calculate_comparison(baseline, variants)
        experiment.print_comparison(comparison)

        # 保存结果
        all_results = {
            "baseline": baseline,
            "variants": variants,
            "comparison": comparison
        }
        output_path = output_dir / "exp_04_ablation_study.json"
        experiment.save_result(all_results, str(output_path))

    finally:
        print("\n" + "=" * 80)
        print("实验 4 完成！")
        print("=" * 80)


if __name__ == "__main__":
    main()
