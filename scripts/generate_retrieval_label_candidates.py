"""
Generate top-N retrieval candidates for manual labeling.

Usage:
    python scripts/generate_retrieval_label_candidates.py ^
      --kb-db database/cnta_knowledge_base.sqlite ^
      --eval-set data/eval/retrieval_eval_set.json ^
      --model bm25 ^
      --top-n 20 ^
      --output-json data/eval/retrieval_label_candidates.json

Notes:
- `bm25` is always available.
- Any non-bm25 value is treated as a sentence-transformers model id.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np


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


def load_eval_items(path: str) -> List[EvalItem]:
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    items = payload.get("items", payload if isinstance(payload, list) else [])
    output: List[EvalItem] = []
    for i, item in enumerate(items):
        qid = str(item.get("id") or f"q{i+1}")
        query = str(item.get("query") or "").strip()
        if query:
            output.append(EvalItem(qid=qid, query=query))
    return output


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


def embedding_scores(model_name: str, query_texts: List[str], doc_texts: List[str]) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is not installed. Install it before using embedding models."
        ) from exc

    model = SentenceTransformer(model_name)
    query_emb = np.asarray(model.encode(query_texts, convert_to_numpy=True, show_progress_bar=False))
    doc_emb = np.asarray(model.encode(doc_texts, convert_to_numpy=True, show_progress_bar=False))

    query_emb = query_emb / np.maximum(np.linalg.norm(query_emb, axis=1, keepdims=True), 1e-12)
    doc_emb = doc_emb / np.maximum(np.linalg.norm(doc_emb, axis=1, keepdims=True), 1e-12)
    return query_emb @ doc_emb.T


def top_indices(scores: np.ndarray, n: int) -> List[int]:
    return np.argsort(-scores)[:n].tolist()


def build_candidates(
    eval_items: Sequence[EvalItem],
    chunks: Sequence[Chunk],
    model: str,
    top_n: int,
) -> Dict[str, object]:
    items_out = []

    if model.lower() == "bm25":
        for item in eval_items:
            scores = bm25_scores(item.query, chunks)
            idxs = top_indices(scores, top_n)
            candidates = []
            for rank, idx in enumerate(idxs, start=1):
                chunk = chunks[idx]
                candidates.append(
                    {
                        "rank": rank,
                        "score": float(scores[idx]),
                        "chunk_id": chunk.chunk_id,
                        "doc_id": chunk.doc_id,
                        "title": chunk.title,
                        "theme": chunk.theme,
                        "snippet": chunk.text[:260],
                        "is_relevant": None,
                    }
                )
            items_out.append({"id": item.qid, "query": item.query, "candidates": candidates})
    else:
        sim = embedding_scores(model, [item.query for item in eval_items], [c.text for c in chunks])
        for row_idx, item in enumerate(eval_items):
            scores = sim[row_idx]
            idxs = top_indices(scores, top_n)
            candidates = []
            for rank, idx in enumerate(idxs, start=1):
                chunk = chunks[idx]
                candidates.append(
                    {
                        "rank": rank,
                        "score": float(scores[idx]),
                        "chunk_id": chunk.chunk_id,
                        "doc_id": chunk.doc_id,
                        "title": chunk.title,
                        "theme": chunk.theme,
                        "snippet": chunk.text[:260],
                        "is_relevant": None,
                    }
                )
            items_out.append({"id": item.qid, "query": item.query, "candidates": candidates})

    return {
        "model": model,
        "top_n": top_n,
        "items": items_out,
        "labeling_guide": "Set is_relevant=true/false for each candidate. Then extract relevant chunk/doc ids into retrieval_eval_set.json.",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate retrieval candidates for manual labeling.")
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
        "--model",
        default="bm25",
        help="Model to generate candidates. `bm25` or sentence-transformers model id.",
    )
    parser.add_argument("--top-n", type=int, default=20, help="Candidates per query.")
    parser.add_argument(
        "--output-json",
        default=r"D:\CNTDATA\CNTA_ML_Project\data\eval\retrieval_label_candidates.json",
        help="Output json path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    chunks = load_chunks(args.kb_db)
    if not chunks:
        raise RuntimeError(f"No chunks found in KB: {args.kb_db}")
    eval_items = load_eval_items(args.eval_set)
    if not eval_items:
        raise RuntimeError(f"No valid query items found in eval set: {args.eval_set}")

    payload = build_candidates(eval_items, chunks, args.model, args.top_n)

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    print(f"Model: {args.model}")
    print(f"Queries: {len(eval_items)}, top_n: {args.top_n}")
    print(f"Saved: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

