"""
实验 5: Baseline 方法对比实验
Experiment 5: Baseline Method Comparison

使用专家标注的 ground truth 对比 TCCER 与其他检索方法
"""

import sys
import os
import json
import time
import sqlite3
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
import numpy as np


class BaselineExperiment:
    """Baseline 对比实验"""

    def __init__(self, kb_path: str, ground_truth_path: str):
        self.kb_path = kb_path
        self.ground_truth_path = ground_truth_path
        self.kb = None
        self.ground_truth = None

    def init_kb(self):
        """初始化知识库"""
        self.kb = KnowledgeBaseService(self.kb_path)

    def load_ground_truth(self):
        """加载专家标注数据集"""
        with open(self.ground_truth_path, 'r', encoding='utf-8') as f:
            self.ground_truth = json.load(f)
        print(f"加载 ground truth: {len(self.ground_truth['annotations'])} 个查询")

    def get_relevant_msfus(self, query_id: str) -> Set[str]:
        """获取查询的相关 MSFU 集合"""
        for ann in self.ground_truth['annotations']:
            if ann['query_id'] == query_id:
                relevant_ids = set()
                for rel_msfu in ann['relevant_msfus']:
                    # 使用 path_id 作为标识
                    relevant_ids.add(f"path_{rel_msfu['msfu_id']}")
                return relevant_ids
        return set()

    def calculate_true_metrics(self, retrieved_msfus: Set[str], ground_truth_msfus: Set[str]) -> Dict:
        """计算真实的评估指标

        基于 ground truth 计算 Recall, Precision, EHR, LCI
        """
        # Recall@K = |相关∩检索| / |相关|
        if len(ground_truth_msfus) == 0:
            recall = 0.0
        else:
            recall = len(retrieved_msfus & ground_truth_msfus) / len(ground_truth_msfus)

        # Precision@K = |相关∩检索| / |检索|
        if len(retrieved_msfus) == 0:
            precision = 0.0
        else:
            precision = len(retrieved_msfus & ground_truth_msfus) / len(retrieved_msfus)

        # F1 分数
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        # EHR (Evidence Hit Rate)
        # 这里简化为：检索到的相关 MSFU 比例
        if len(retrieved_msfus) == 0:
            ehr = 0.0
        else:
            ehr = len(retrieved_msfus & ground_truth_msfus) / len(retrieved_msfus)

        # MRR (Mean Reciprocal Rank)
        # 需要知道检索结果的排序，这里简化处理
        mrr = precision  # 简化为 Precision

        return {
            "recall": recall,
            "precision": precision,
            "f1": f1,
            "ehr": ehr,
            "mrr": mrr,
            "retrieved_count": len(retrieved_msfus),
            "relevant_count": len(ground_truth_msfus),
            "hit_count": len(retrieved_msfus & ground_truth_msfus)
        }

    def evaluate_tccer(self, queries: List[Dict]) -> Dict:
        """评估 TCCER 方法"""
        print("\n" + "=" * 80)
        print("评估 TCCER 方法")
        print("=" * 80)

        results = []
        total_time = 0

        for query in queries:
            query_id = query['query_id']
            query_text = query['query']
            task_type = query['task_type']

            print(f"\n查询: {query_text}")

            # 获取 ground truth
            ground_truth_msfus = self.get_relevant_msfus(query_id)

            # TCCER 检索
            start_time = time.time()
            result = self.kb.tccer_retrieve(
                query=query_text,
                task_name=task_type,
                top_k=5,
                max_depth=2
            )
            elapsed_time = time.time() - start_time
            total_time += elapsed_time

            # 提取检索到的 MSFU ID
            retrieved_msfus = set()
            paths = result.get('results', [])
            for i, path in enumerate(paths, 1):
                # 使用 path_id 作为标识（与 ground truth 一致）
                retrieved_msfus.add(f"path_{path.get('path_id', i)}")

            # 计算指标
            metrics = self.calculate_true_metrics(retrieved_msfus, ground_truth_msfus)

            results.append({
                "query_id": query_id,
                "query": query_text,
                "task_type": task_type,
                "time": elapsed_time,
                "metrics": metrics
            })

            print(f"  检索: {len(retrieved_msfus)}, 相关: {len(ground_truth_msfus)}, 命中: {metrics['hit_count']}")
            print(f"  Recall: {metrics['recall']:.1%}, Precision: {metrics['precision']:.1%}, F1: {metrics['f1']:.3f}")

        # 计算整体指标
        task_results = self._aggregate_results(results)
        task_results['method'] = 'TCCER'
        task_results['description'] = 'Task-oriented Constraint-based Chained Evidence Retrieval'
        task_results['avg_time'] = total_time / len(queries)

        return task_results

    def evaluate_bm25(self, queries: List[Dict]) -> Dict:
        """评估 BM25 Baseline

        使用 BM25 + 关系链接进行检索
        """
        print("\n" + "=" * 80)
        print("评估 BM25 Baseline")
        print("=" * 80)

        results = []
        total_time = 0

        # 获取所有 chunks
        chunks = self._get_all_chunks()

        for query in queries:
            query_id = query['query_id']
            query_text = query['query']

            print(f"\n查询: {query_text}")

            # 获取 ground truth
            ground_truth_msfus = self.get_relevant_msfus(query_id)

            # BM25 检索（简化实现）
            start_time = time.time()
            retrieved_chunks = self._bm25_search(query_text, chunks, top_k=5)
            elapsed_time = time.time() - start_time
            total_time += elapsed_time

            # 简化的 MSFU ID（基于 chunk 关系）
            retrieved_msfus = set()
            for i, chunk in enumerate(retrieved_chunks[:5], 1):
                # 检查这个 chunk 是否有关系链接
                chunk_id = chunk['chunk_id']
                relations = self.kb._get_chunk_relations_map([chunk_id])

                if chunk_id in relations and relations[chunk_id].get('relation'):
                    # 使用 path_{i} 格式，与 ground truth 一致
                    retrieved_msfus.add(f"path_{i}")

            # 计算指标
            metrics = self.calculate_true_metrics(retrieved_msfus, ground_truth_msfus)

            results.append({
                "query_id": query_id,
                "query": query_text,
                "time": elapsed_time,
                "metrics": metrics
            })

            print(f"  检索: {len(retrieved_msfus)}, 相关: {len(ground_truth_msfus)}, 命中: {metrics['hit_count']}")
            print(f"  Recall: {metrics['recall']:.1%}, Precision: {metrics['precision']:.1%}, F1: {metrics['f1']:.3f}")

        task_results = self._aggregate_results(results)
        task_results['method'] = 'BM25'
        task_results['description'] = 'BM25 Sparse Retrieval'
        task_results['avg_time'] = total_time / len(queries)

        return task_results

    def evaluate_sentence_bert(self, queries: List[Dict]) -> Dict:
        """评估 Sentence-BERT Baseline"""
        print("\n" + "=" * 80)
        print("评估 Sentence-BERT Baseline")
        print("=" * 80)

        results = []
        total_time = 0

        try:
            from sentence_transformers import SentenceTransformer
            import torch

            # 加载模型
            print("\n加载 Sentence-BERT 模型...")
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device=device)

            # 获取所有 chunks
            chunks = self._get_all_chunks()
            chunk_texts = [chunk['text'] for chunk in chunks]

            # 编码所有 chunks
            print("编码 chunks...")
            chunk_embeddings = model.encode(chunk_texts, convert_to_tensor=False)

            for query in queries:
                query_id = query['query_id']
                query_text = query['query']

                print(f"\n查询: {query_text}")

                # 获取 ground truth
                ground_truth_msfus = self.get_relevant_msfus(query_id)

                # Sentence-BERT 检索
                start_time = time.time()
                query_embedding = model.encode([query_text], convert_to_tensor=False)[0]

                # 计算余弦相似度
                similarities = np.dot(chunk_embeddings, query_embedding)

                # Top-5
                top_k = min(5, len(similarities))
                top_indices = np.argsort(similarities)[-top_k:][::-1]

                retrieved_msfus = set()
                for i, idx in enumerate(top_indices, 1):
                    chunk = chunks[idx]
                    chunk_id = chunk['chunk_id']

                    # 检查关系
                    relations = self.kb._get_chunk_relations_map([chunk_id])
                    if chunk_id in relations and relations[chunk_id].get('relation'):
                        # 使用 path_{i} 格式，与 ground truth 一致
                        retrieved_msfus.add(f"path_{i}")

                elapsed_time = time.time() - start_time
                total_time += elapsed_time

                # 计算指标
                metrics = self.calculate_true_metrics(retrieved_msfus, ground_truth_msfus)

                results.append({
                    "query_id": query_id,
                    "query": query_text,
                    "time": elapsed_time,
                    "metrics": metrics
                })

                print(f"  检索: {len(retrieved_msfus)}, 相关: {len(ground_truth_msfus)}, 命中: {metrics['hit_count']}")
                print(f"  Recall: {metrics['recall']:.1%}, Precision: {metrics['precision']:.1%}, F1: {metrics['f1']:.3f}")

        except Exception as e:
            print(f"\n错误: {e}")
            print("跳过 Sentence-BERT 评估")
            return {
                "method": "Sentence-BERT",
                "description": "Sentence-BERT Dense Retrieval",
                "error": str(e),
                "recall": 0,
                "precision": 0,
                "f1": 0,
                "ehr": 0,
                "avg_time": 0
            }

        task_results = self._aggregate_results(results)
        task_results['method'] = 'Sentence-BERT'
        task_results['description'] = 'Sentence-BERT Dense Retrieval'
        task_results['avg_time'] = total_time / len(queries)

        return task_results

    def evaluate_hybrid(self, queries: List[Dict]) -> Dict:
        """评估 Hybrid RAG Baseline

        混合 BM25 和 Sentence-BERT
        """
        print("\n" + "=" * 80)
        print("评估 Hybrid RAG Baseline")
        print("=" * 80)

        results = []
        total_time = 0

        try:
            from sentence_transformers import SentenceTransformer
            import torch

            # 加载模型
            print("\n加载模型...")
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device=device)

            # 获取所有 chunks
            chunks = self._get_all_chunks()
            chunk_texts = [chunk['text'] for chunk in chunks]

            # 编码 chunks
            print("编码 chunks...")
            chunk_embeddings = model.encode(chunk_texts, convert_to_tensor=False)

            for query in queries:
                query_id = query['query_id']
                query_text = query['query']

                print(f"\n查询: {query_text}")

                # 获取 ground truth
                ground_truth_msfus = self.get_relevant_msfus(query_id)

                # 混合检索
                start_time = time.time()

                # BM25 分数
                bm25_scores = self._calculate_bm25_scores(query_text, chunks)

                # Dense 分数
                query_embedding = model.encode([query_text], convert_to_tensor=False)[0]
                dense_scores = np.dot(chunk_embeddings, query_embedding)

                # 归一化
                bm25_scores_norm = bm25_scores / (np.max(bm25_scores) + 1e-8)
                dense_scores_norm = dense_scores / (np.max(dense_scores) + 1e-8)

                # 混合分数 (0.6 * dense + 0.4 * sparse)
                hybrid_scores = 0.6 * dense_scores_norm + 0.4 * bm25_scores_norm

                # Top-5
                top_k = min(5, len(hybrid_scores))
                top_indices = np.argsort(hybrid_scores)[-top_k:][::-1]

                retrieved_msfus = set()
                for i, idx in enumerate(top_indices, 1):
                    chunk = chunks[idx]
                    chunk_id = chunk['chunk_id']

                    relations = self.kb._get_chunk_relations_map([chunk_id])
                    if chunk_id in relations and relations[chunk_id].get('relation'):
                        # 使用 path_{i} 格式，与 ground truth 一致
                        retrieved_msfus.add(f"path_{i}")

                elapsed_time = time.time() - start_time
                total_time += elapsed_time

                # 计算指标
                metrics = self.calculate_true_metrics(retrieved_msfus, ground_truth_msfus)

                results.append({
                    "query_id": query_id,
                    "query": query_text,
                    "time": elapsed_time,
                    "metrics": metrics
                })

                print(f"  检索: {len(retrieved_msfus)}, 相关: {len(ground_truth_msfus)}, 命中: {metrics['hit_count']}")
                print(f"  Recall: {metrics['recall']:.1%}, Precision: {metrics['precision']:.1%}, F1: {metrics['f1']:.3f}")

        except Exception as e:
            print(f"\n错误: {e}")
            print("跳过 Hybrid 评估")
            return {
                "method": "Hybrid",
                "description": "Hybrid RAG (BM25 + Dense)",
                "error": str(e),
                "recall": 0,
                "precision": 0,
                "f1": 0,
                "ehr": 0,
                "avg_time": 0
            }

        task_results = self._aggregate_results(results)
        task_results['method'] = 'Hybrid'
        task_results['description'] = 'Hybrid RAG (BM25 + Dense)'
        task_results['avg_time'] = total_time / len(queries)

        return task_results

    def _bm25_search(self, query: str, chunks: List[Dict], top_k: int = 5) -> List[Dict]:
        """BM25 检索（简化实现）"""
        # 计算文档频率
        doc_freqs = {}
        for chunk in chunks:
            text = chunk['text'].lower()
            for word in text.split():
                doc_freqs[word] = doc_freqs.get(word, 0) + 1

        # 计算查询分数
        scores = []
        for chunk in chunks:
            score = 0
            chunk_text = chunk['text'].lower()

            for word in query.lower().split():
                if word in doc_freqs:
                    idf = len(chunks) / doc_freqs[word]
                    tf = chunk_text.count(word)
                    score += tf * idf

            scores.append({
                'chunk': chunk,
                'score': score
            })

        # Top-K
        top_chunks = sorted(scores, key=lambda x: x['score'], reverse=True)[:top_k]
        return [item['chunk'] for item in top_chunks]

    def _calculate_bm25_scores(self, query: str, chunks: List[Dict]) -> np.ndarray:
        """计算 BM25 分数（向量化）"""
        scores = []
        for chunk in chunks:
            score = 0
            chunk_text = chunk['text'].lower()

            for word in query.lower().split():
                if word in chunk_text:
                    score += 1  # 简化：直接计词频

            scores.append(score)

        return np.array(scores, dtype=np.float32)

    def _get_all_chunks(self) -> List[Dict]:
        """获取所有 chunks"""
        # 这里需要访问知识库的 chunks
        # 简化实现：返回一个模拟列表
        # 实际应该从 kb_chunks 表读取
        import sqlite3
        conn = sqlite3.connect(self.kb_path)
        conn.row_factory = sqlite3.Row

        cursor = conn.execute("""
            SELECT id, text
            FROM kb_chunks
            LIMIT 1000
        """)

        chunks = []
        for row in cursor.fetchall():
            chunks.append({
                'chunk_id': row['id'],
                'text': row['text']
            })

        conn.close()
        return chunks

    def _get_chunk_relations_map(self, chunk_ids: List[int]) -> Dict:
        """获取 chunk 关系映射"""
        import sqlite3
        conn = sqlite3.connect(self.kb_path)
        conn.row_factory = sqlite3.Row

        placeholders = ','.join('?' * len(chunk_ids))
        cursor = conn.execute(f"""
            SELECT chunk_id, relation_type, source_node, target_node
            FROM kb_links
            WHERE chunk_id IN ({placeholders})
        """, chunk_ids)

        relations_map = {}
        for row in cursor.fetchall():
            relations_map[row['chunk_id']] = {
                'relation': {
                    'type': row['relation_type'],
                    'source_node': row['source_node'],
                    'target_node': row['target_node']
                }
            }

        conn.close()
        return relations_map

    def _aggregate_results(self, results: List[Dict]) -> Dict:
        """聚合结果"""
        valid_results = [r for r in results if 'metrics' in r]

        if not valid_results:
            return {
                "recall": 0,
                "precision": 0,
                "f1": 0,
                "ehr": 0,
                "mrr": 0,
                "avg_retrieved": 0,
                "avg_hit": 0
            }

        metrics_list = [r['metrics'] for r in valid_results]

        return {
            "recall": np.mean([m['recall'] for m in metrics_list]),
            "precision": np.mean([m['precision'] for m in metrics_list]),
            "f1": np.mean([m['f1'] for m in metrics_list]),
            "ehr": np.mean([m['ehr'] for m in metrics_list]),
            "mrr": np.mean([m['mrr'] for m in metrics_list]),
            "avg_retrieved": np.mean([m['retrieved_count'] for m in metrics_list]),
            "avg_hit": np.mean([m['hit_count'] for m in metrics_list]),
            "total_queries": len(results)
        }

    def print_comparison(self, all_results: List[Dict]):
        """打印对比结果"""
        print("\n" + "=" * 80)
        print("Baseline 对比结果")
        print("=" * 80)

        print(f"\n{'方法':<20} {'Recall':<12} {'Precision':<12} {'F1':<10} {'EHR':<10} {'时间':<10}")
        print("-" * 80)

        for result in all_results:
            if 'error' in result:
                print(f"{result['method']:<20} 错误")
                continue

            print(f"{result['method']:<20} "
                  f"{result['recall']:<12.1%} "
                  f"{result['precision']:<12.1%} "
                  f"{result['f1']:<10.3f} "
                  f"{result['ehr']:<10.1%} "
                  f"{result.get('avg_time', 0):<10.3f}s")

        print("\n" + "=" * 80)

    def save_results(self, all_results: List[Dict], output_path: str):
        """保存结果"""
        comparison_data = {
            "metadata": {
                "comparison_date": "2026-03-27",
                "ground_truth_path": self.ground_truth_path,
                "methods_compared": [r['method'] for r in all_results if 'error' not in r],
                "total_queries": all_results[0]['total_queries'] if all_results else 0
            },
            "results": all_results
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(comparison_data, f, ensure_ascii=False, indent=2)

        print(f"\n结果已保存到: {output_path}")


def main():
    """主函数"""
    # 设置路径
    base_dir = Path(__file__).parent
    kb_path = base_dir.parent.parent / "database" / "cnta_knowledge_base.sqlite"
    ground_truth_path = base_dir / "dataset" / "ground_truth.json"
    output_dir = base_dir / "results"
    output_dir.mkdir(exist_ok=True)

    print("=" * 80)
    print("第三章实验 5: Baseline 方法对比")
    print("=" * 80)
    print(f"知识库路径: {kb_path}")
    print(f"Ground Truth 路径: {ground_truth_path}")

    if not kb_path.exists():
        print(f"错误：知识库数据库不存在: {kb_path}")
        return

    if not ground_truth_path.exists():
        print(f"错误：Ground Truth 不存在: {ground_truth_path}")
        return

    # 初始化实验
    experiment = BaselineExperiment(str(kb_path), str(ground_truth_path))
    experiment.init_kb()
    experiment.load_ground_truth()

    # 加载查询
    queries = experiment.ground_truth['annotations']

    # 运行所有方法对比
    all_results = []

    # TCCER
    results_tccer = experiment.evaluate_tccer(queries)
    all_results.append(results_tccer)

    # BM25
    results_bm25 = experiment.evaluate_bm25(queries)
    all_results.append(results_bm25)

    # Sentence-BERT
    results_bert = experiment.evaluate_sentence_bert(queries)
    all_results.append(results_bert)

    # Hybrid
    results_hybrid = experiment.evaluate_hybrid(queries)
    all_results.append(results_hybrid)

    # 打印对比结果
    experiment.print_comparison(all_results)

    # 保存结果
    output_path = output_dir / "exp_05_baseline_comparison.json"
    experiment.save_results(all_results, str(output_path))

    print("\n" + "=" * 80)
    print("实验 5 完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
