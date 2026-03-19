import unittest
from unittest.mock import patch

import numpy as np

from scripts.evaluate_retrieval_models import (
    Chunk,
    EvalItem,
    evaluate_sentence_transformer,
    evaluate_service_search,
    ndcg_at_k,
)


class RetrievalEvaluationTests(unittest.TestCase):
    def test_evaluate_sentence_transformer_uses_local_files_only(self):
        chunks = [
            Chunk(
                chunk_id=1,
                doc_id=1,
                title="Alignment note",
                theme="process_morphology",
                text="Alignment improves conductivity in CNT arrays.",
                keywords="alignment conductivity",
            )
        ]
        eval_items = [
            EvalItem(
                qid="q1",
                query="alignment conductivity",
                relevant_chunk_ids={1},
                relevant_doc_ids=set(),
            )
        ]

        class FakeModel:
            def encode(self, texts, convert_to_numpy=True, show_progress_bar=False):
                return np.asarray([[1.0, 0.0] for _ in texts], dtype=float)

        with patch("sentence_transformers.SentenceTransformer", return_value=FakeModel()) as mock_cls:
            ranked = evaluate_sentence_transformer("sentence-transformers/all-MiniLM-L6-v2", eval_items, chunks)

        self.assertEqual(ranked["q1"], [0])
        _, kwargs = mock_cls.call_args
        self.assertTrue(kwargs.get("local_files_only"))

    def test_ndcg_is_one_for_perfect_ranking(self):
        chunks = [
            Chunk(1, 1, "Doc A", "theme", "text a", ""),
            Chunk(2, 2, "Doc B", "theme", "text b", ""),
            Chunk(3, 3, "Doc C", "theme", "text c", ""),
        ]
        item = EvalItem(
            qid="q1",
            query="test",
            relevant_chunk_ids=set(),
            relevant_doc_ids={1, 2},
        )

        score = ndcg_at_k([0, 1, 2], chunks, item, k=3)

        self.assertAlmostEqual(score, 1.0, places=6)

    def test_evaluate_service_search_maps_chunk_ids_for_hybrid_modes(self):
        chunks = [
            Chunk(10, 1, "Doc A", "theme", "text a", ""),
            Chunk(20, 2, "Doc B", "theme", "text b", ""),
            Chunk(30, 3, "Doc C", "theme", "text c", ""),
        ]
        eval_items = [
            EvalItem(
                qid="q1",
                query="alignment conductivity",
                relevant_chunk_ids={20},
                relevant_doc_ids=set(),
            )
        ]

        class FakeService:
            def __init__(self):
                self._relation_constraint_scores = lambda query, rows: {"kept": 1}
                self.calls = []
                self.relation_scores_seen = []

            def search(self, query, task_name=None, top_k=5):
                self.calls.append((query, task_name, top_k))
                self.relation_scores_seen.append(self._relation_constraint_scores(query, []))
                return [
                    {"chunk_id": 20, "doc_id": 2},
                    {"chunk_id": 10, "doc_id": 1},
                ]

        fake_service = FakeService()

        with patch("scripts.evaluate_retrieval_models.KnowledgeBaseService", return_value=fake_service):
            ranked = evaluate_service_search(
                kb_db="dummy.sqlite",
                eval_items=eval_items,
                chunks=chunks,
                top_k=5,
                disable_relation_reranker=True,
            )

        self.assertEqual(ranked["q1"], [1, 0])
        self.assertEqual(fake_service.calls[0], ("alignment conductivity", None, 5))
        self.assertEqual(fake_service.relation_scores_seen, [{}])
        self.assertEqual(fake_service._relation_constraint_scores("x", []), {"kept": 1})


if __name__ == "__main__":
    unittest.main()
