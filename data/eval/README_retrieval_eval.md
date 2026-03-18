# Retrieval Evaluation Quick Guide

## 1) Prepare eval set
1. Copy `retrieval_eval_set.example.json` to `retrieval_eval_set.json`.
2. Add 50-100 query items.
3. For each query, fill at least one of:
- `relevant_chunk_ids`
- `relevant_doc_ids`

## 2) Get candidate ids for labeling
You can use existing API:
- `POST /api/rag/search` with a query to see candidate passages.
- Record returned `doc_id` / `chunk_id` as ground-truth labels.

Or generate candidates in batch:
```powershell
python scripts/generate_retrieval_label_candidates.py `
  --kb-db database/cnta_knowledge_base.sqlite `
  --eval-set data/eval/retrieval_eval_set.json `
  --model bm25 `
  --top-n 20 `
  --output-json data/eval/retrieval_label_candidates.json
```

Then edit `is_relevant` in each candidate (`true/false`) and sync selected ids into `retrieval_eval_set.json`.
```powershell
python scripts/sync_retrieval_labels.py `
  --eval-set data/eval/retrieval_eval_set.json `
  --candidates data/eval/retrieval_label_candidates.json `
  --output data/eval/retrieval_eval_set.json
```

## 3) Run comparison
```powershell
python scripts/evaluate_retrieval_models.py `
  --kb-db database/cnta_knowledge_base.sqlite `
  --eval-set data/eval/retrieval_eval_set.json `
  --k 5 10 `
  --models bm25 sentence-transformers/all-mpnet-base-v2
```

## 4) Output
Results are saved to:
- `data/eval/retrieval_model_comparison.csv`

Metrics:
- `Recall@k`
- `MRR@k`
- `nDCG@k`

## 5) Notes
- `bm25` always runs.
- sentence-transformers models require `sentence-transformers` package and local model availability.
- If a model fails to load, it is marked as `status=error` and others continue.
