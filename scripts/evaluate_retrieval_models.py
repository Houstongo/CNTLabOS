"""
Retrieval model comparison for CNTA RAG knowledge base.

Usage:
    python scripts/evaluate_retrieval_models.py ^
      --kb-db database/cnta_knowledge_base.sqlite ^
      --eval-set data/eval/retrieval_eval_set.json ^
      --k 5 10 ^
      --models bm25 sentence-transformers/all-mpnet-base-v2

Notes:
- `bm25` is always available.
- Any other model name is treated as a sentence-transformers model id.
- sentence-transformers dependency is optional; missing models are skipped gracefully.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sqlite3
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import json


STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "into",
    "that",
    "this",
    "of",
    "to",
    "in",
    "on",
    "is",
    "are",
    "by",
    "or",
    "a",
    "an",
    "的",
    "和",
    "在",
    "是",
    "与",
    "及",
    "并",
    "对",
    "中",
}


@dataclass
class Chunk:
    chunk_id: int
    doc_id: int
    title: str
    theme: Optional[str]
    text: str
    keywords: str


@dataclass
class EvalItem:
    qid: str
    query: str
    relevant_chunk_ids: Set[int]
    relevant_doc_ids: Set[int]


def tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[a-zA-Z]+|[\u4e00-\u9fff]+", (text or "").lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def load_chunks(kb_db: str) -> List[Chunk]:
    conn = sqlite3.connect(kb_db)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT c.id AS chunk_id, c.doc_id, d.title, d.theme, c.text, c.keywords
            FROM kb_chunks c
            JOIN kb_documents d ON d.id = c.doc_id
            """
        ).fetchall()
    finally:
        conn.close()

    return [
        Chunk(
            chunk_id=int(r["chunk_id"]),
            doc_id=int(r["doc_id"]),
            title=r["title"],
            theme=r["theme"],
            text=r["text"] or "",
            keywords=r["keywords"] or "",
        )
        for r in rows
    ]


def load_eval_set(path: str) -> List[EvalItem]:
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)

    items = payload.get("items", payload if isinstance(payload, list) else [])
    result: List[EvalItem] = []
    for i, item in enumerate(items):
        qid = str(item.get("id") or f"q{i+1}")
        query = str(item.get("query") or "").strip()
        if not query:
            continue

        relevant_chunk_ids = {int(x) for x in item.get("relevant_chunk_ids", [])}
        relevant_doc_ids = {int(x) for x in item.get("relevant_doc_ids", [])}
        result.append(
            EvalItem(
                qid=qid,
                query=query,
                relevant_chunk_ids=relevant_chunk_ids,
                relevant_doc_ids=relevant_doc_ids,
            )
        )
    return result


def bm25_scores(query: str, chunks: Sequence[Chunk], k1: float = 1.5, b: float = 0.75) -> np.ndarray:
    q_terms = set(tokenize(query))
    if not q_terms:
        return np.zeros(len(chunks), dtype=float)

    tokenized = [tokenize(c.text) for c in chunks]
    token_sets = [set(ts) | set(c.keywords.split()) for ts, c in zip(tokenized, chunks)]
    doc_lens = np.array([max(1, len(ts)) for ts in tokenized], dtype=float)
    avgdl = float(np.mean(doc_lens)) if len(doc_lens) else 1.0
    n_docs = len(chunks)

    scores = np.zeros(n_docs, dtype=float)
    for term in q_terms:
        tf = np.array([ts.count(term) for ts in tokenized], dtype=float)
        df = sum(1 for s in token_sets if term in s)
        if df == 0:
            continue
        idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
        denom = tf + k1 * (1 - b + b * (doc_lens / avgdl))
        term_score = idf * ((tf * (k1 + 1)) / np.where(denom == 0, 1, denom))
        scores += term_score
    return scores


def cosine_similarity_matrix(query_emb: np.ndarray, doc_embs: np.ndarray) -> np.ndarray:
    q = query_emb / np.maximum(np.linalg.norm(query_emb, axis=1, keepdims=True), 1e-12)
    d = doc_embs / np.maximum(np.linalg.norm(doc_embs, axis=1, keepdims=True), 1e-12)
    return q @ d.T


def rank_from_scores(scores: np.ndarray) -> List[int]:
    return np.argsort(-scores).tolist()


def is_relevant(chunk: Chunk, eval_item: EvalItem) -> bool:
    return (chunk.chunk_id in eval_item.relevant_chunk_ids) or (chunk.doc_id in eval_item.relevant_doc_ids)


def recall_at_k(ranked_idx: Sequence[int], chunks: Sequence[Chunk], eval_item: EvalItem, k: int) -> float:
    rel_total = sum(1 for c in chunks if is_relevant(c, eval_item))
    if rel_total == 0:
        return 0.0
    topk = ranked_idx[:k]
    rel_hit = sum(1 for i in topk if is_relevant(chunks[i], eval_item))
    return rel_hit / rel_total


def mrr(ranked_idx: Sequence[int], chunks: Sequence[Chunk], eval_item: EvalItem, k: int) -> float:
    for rank, idx in enumerate(ranked_idx[:k], start=1):
        if is_relevant(chunks[idx], eval_item):
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_idx: Sequence[int], chunks: Sequence[Chunk], eval_item: EvalItem, k: int) -> float:
    dcg = 0.0
    for rank, idx in enumerate(ranked_idx[:k], start=1):
        rel = 1.0 if is_relevant(chunks[idx], eval_item) else 0.0
        if rel > 0:
            dcg += rel / math.log2(rank + 1)

    rel_total = sum(1 for c in chunks if is_relevant(c, eval_item))
    ideal_rel = min(k, rel_total)
    if ideal_rel == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(i + 1) for i in range(2, ideal_rel + 2))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_ranking(
    ranked_indices_by_query: Dict[str, List[int]],
    eval_items: Sequence[EvalItem],
    chunks: Sequence[Chunk],
    ks: Sequence[int],
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for k in ks:
        recalls = []
        mrrs = []
        ndcgs = []
        for item in eval_items:
            ranked = ranked_indices_by_query[item.qid]
            recalls.append(recall_at_k(ranked, chunks, item, k))
            mrrs.append(mrr(ranked, chunks, item, k))
            ndcgs.append(ndcg_at_k(ranked, chunks, item, k))
        metrics[f"Recall@{k}"] = float(np.mean(recalls)) if recalls else 0.0
        metrics[f"MRR@{k}"] = float(np.mean(mrrs)) if mrrs else 0.0
        metrics[f"nDCG@{k}"] = float(np.mean(ndcgs)) if ndcgs else 0.0
    return metrics


def evaluate_bm25(eval_items: Sequence[EvalItem], chunks: Sequence[Chunk]) -> Dict[str, List[int]]:
    ranked: Dict[str, List[int]] = {}
    for item in eval_items:
        scores = bm25_scores(item.query, chunks)
        ranked[item.qid] = rank_from_scores(scores)
    return ranked


def evaluate_sentence_transformer(
    model_name: str, eval_items: Sequence[EvalItem], chunks: Sequence[Chunk]
) -> Dict[str, List[int]]:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is not installed. Install it to compare embedding models."
        ) from exc

    model = SentenceTransformer(model_name)
    doc_texts = [c.text for c in chunks]
    query_texts = [i.query for i in eval_items]

    doc_emb = np.asarray(model.encode(doc_texts, convert_to_numpy=True, show_progress_bar=False))
    query_emb = np.asarray(model.encode(query_texts, convert_to_numpy=True, show_progress_bar=False))

    sim = cosine_similarity_matrix(query_emb, doc_emb)

    ranked: Dict[str, List[int]] = {}
    for row_idx, item in enumerate(eval_items):
        ranked[item.qid] = rank_from_scores(sim[row_idx])
    return ranked


def save_results_csv(path: str, rows: List[Dict[str, object]], metric_headers: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=["model", *metric_headers, "status", "note"])
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare retrieval models on CNTA KB.")
    parser.add_argument(
        "--kb-db",
        default=r"D:\CNTDATA\CNTA_ML_Project\database\cnta_knowledge_base.sqlite",
        help="Path to knowledge-base sqlite file.",
    )
    parser.add_argument(
        "--eval-set",
        default=r"D:\CNTDATA\CNTA_ML_Project\data\eval\retrieval_eval_set.json",
        help="Path to evaluation set json.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["bm25", "sentence-transformers/all-mpnet-base-v2"],
        help="Model names. Use `bm25` for lexical baseline; others are sentence-transformers model ids.",
    )
    parser.add_argument("--k", nargs="+", type=int, default=[5, 10], help="k values for Recall/MRR/nDCG.")
    parser.add_argument(
        "--output-csv",
        default=r"D:\CNTDATA\CNTA_ML_Project\data\eval\retrieval_model_comparison.csv",
        help="Output csv path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    chunks = load_chunks(args.kb_db)
    if not chunks:
        raise RuntimeError(f"No chunks found in KB: {args.kb_db}")

    eval_items = load_eval_set(args.eval_set)
    if not eval_items:
        raise RuntimeError(f"No eval items found: {args.eval_set}")

    ks = sorted(set(args.k))
    metric_headers = [f"{m}@{k}" for k in ks for m in ("Recall", "MRR", "nDCG")]
    results_rows: List[Dict[str, object]] = []

    print(f"Loaded chunks: {len(chunks)}")
    print(f"Loaded eval queries: {len(eval_items)}")
    print(f"K values: {ks}")

    for model_name in args.models:
        row: Dict[str, object] = {"model": model_name, "status": "ok", "note": ""}
        try:
            if model_name.lower() == "bm25":
                ranked = evaluate_bm25(eval_items, chunks)
            else:
                ranked = evaluate_sentence_transformer(model_name, eval_items, chunks)

            metrics = evaluate_ranking(ranked, eval_items, chunks, ks)
            for k in ks:
                row[f"Recall@{k}"] = round(metrics[f"Recall@{k}"], 4)
                row[f"MRR@{k}"] = round(metrics[f"MRR@{k}"], 4)
                row[f"nDCG@{k}"] = round(metrics[f"nDCG@{k}"], 4)
            print(f"[OK] {model_name} -> " + ", ".join(f"{k}={row[k]}" for k in row if "@" in k))
        except Exception as exc:  # keep comparison resilient
            row["status"] = "error"
            row["note"] = str(exc)
            for k in ks:
                row[f"Recall@{k}"] = ""
                row[f"MRR@{k}"] = ""
                row[f"nDCG@{k}"] = ""
            print(f"[SKIP] {model_name}: {exc}")
        results_rows.append(row)

    save_results_csv(args.output_csv, results_rows, metric_headers)
    print(f"Saved comparison to: {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

