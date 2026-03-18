"""
Sync manual labels from retrieval candidates back to eval set.

Usage:
    python scripts/sync_retrieval_labels.py ^
      --eval-set data/eval/retrieval_eval_set.json ^
      --candidates data/eval/retrieval_label_candidates.json ^
      --output data/eval/retrieval_eval_set.json

Notes:
- Reads `is_relevant=true` candidates.
- Writes deduplicated `relevant_chunk_ids` and `relevant_doc_ids` for each query id.
- Existing labels are preserved and merged by default.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Set, Tuple


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8-sig") as fh:
        return json.load(fh)


def parse_label_candidates(payload: Dict) -> Dict[str, Tuple[Set[int], Set[int]]]:
    result: Dict[str, Tuple[Set[int], Set[int]]] = {}
    for item in payload.get("items", []):
        qid = str(item.get("id", "")).strip()
        if not qid:
            continue
        chunk_ids: Set[int] = set()
        doc_ids: Set[int] = set()
        for cand in item.get("candidates", []):
            if cand.get("is_relevant") is True:
                if cand.get("chunk_id") is not None:
                    chunk_ids.add(int(cand["chunk_id"]))
                if cand.get("doc_id") is not None:
                    doc_ids.add(int(cand["doc_id"]))
        result[qid] = (chunk_ids, doc_ids)
    return result


def merge_labels(eval_payload: Dict, labels: Dict[str, Tuple[Set[int], Set[int]]], merge: bool) -> Dict:
    items = eval_payload.get("items", eval_payload if isinstance(eval_payload, list) else [])
    if not isinstance(items, list):
        raise ValueError("Invalid eval set format: items must be a list.")

    for i, item in enumerate(items):
        qid = str(item.get("id") or f"q{i+1}")
        add_chunk_ids, add_doc_ids = labels.get(qid, (set(), set()))

        if merge:
            existing_chunk_ids = {int(x) for x in item.get("relevant_chunk_ids", [])}
            existing_doc_ids = {int(x) for x in item.get("relevant_doc_ids", [])}
        else:
            existing_chunk_ids = set()
            existing_doc_ids = set()

        merged_chunk_ids = sorted(existing_chunk_ids | add_chunk_ids)
        merged_doc_ids = sorted(existing_doc_ids | add_doc_ids)

        item["id"] = qid
        item["relevant_chunk_ids"] = merged_chunk_ids
        item["relevant_doc_ids"] = merged_doc_ids

    return {"items": items}


def summarize(items: List[Dict]) -> Tuple[int, int, int]:
    q_count = len(items)
    q_labeled = 0
    pair_count = 0
    for item in items:
        chunk_n = len(item.get("relevant_chunk_ids", []))
        doc_n = len(item.get("relevant_doc_ids", []))
        if chunk_n or doc_n:
            q_labeled += 1
        pair_count += chunk_n + doc_n
    return q_count, q_labeled, pair_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync retrieval labels into eval set.")
    parser.add_argument(
        "--eval-set",
        default=r"D:\CNTDATA\CNTA_ML_Project\data\eval\retrieval_eval_set.json",
        help="Path to eval set json.",
    )
    parser.add_argument(
        "--candidates",
        default=r"D:\CNTDATA\CNTA_ML_Project\data\eval\retrieval_label_candidates.json",
        help="Path to labeled candidates json.",
    )
    parser.add_argument(
        "--output",
        default=r"D:\CNTDATA\CNTA_ML_Project\data\eval\retrieval_eval_set.json",
        help="Output eval set path.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace existing labels instead of merging.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    eval_payload = load_json(args.eval_set)
    cand_payload = load_json(args.candidates)

    labels = parse_label_candidates(cand_payload)
    merged = merge_labels(eval_payload, labels, merge=not args.replace)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(merged, fh, ensure_ascii=False, indent=2)

    q_count, q_labeled, pair_count = summarize(merged["items"])
    print(f"Saved: {args.output}")
    print(f"Queries: {q_count}")
    print(f"Labeled queries: {q_labeled}")
    print(f"Total relevance ids: {pair_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
