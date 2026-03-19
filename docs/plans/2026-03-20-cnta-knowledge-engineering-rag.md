# CNTA Knowledge Engineering RAG Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Upgrade the current CNTA RAG module into a knowledge-engineering-oriented scientific retrieval system with embedding recall, light reranking, stronger process-morphology-performance knowledge units, and reproducible evaluation outputs.

**Architecture:** Keep the existing SQLite knowledge-base and FastAPI API surface. Extend the knowledge schema and retriever so the main retrieval path becomes embedding-first, relation-enhanced, and reranked. Reuse the current evaluation scripts and label sets, but formalize them into a repeatable experimental workflow.

**Tech Stack:** Python, sqlite3, FastAPI, sentence-transformers, optional cross-encoder reranking, unittest

---

### Task 1: Extend Knowledge-Unit Schema For Performance-Centric Relations

**Files:**
- Modify: `D:\CNTDATA\CNTA_ML_Project\backend\core\knowledge_base.py`
- Test: `D:\CNTDATA\CNTA_ML_Project\tests\test_knowledge_base_service.py`

**Step 1: Write the failing test**

Add tests covering:
- performance entity normalization for `conductivity`, `tensile_strength`, and `modulus`
- extraction of `process_to_performance` and `morphology_to_performance` relations
- relation records storing `source_node`, `target_node`, `relation_type`, and `performance_factor`

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_knowledge_base_service -v`

Expected: FAIL because current extraction and normalization coverage is still limited.

**Step 3: Write minimal implementation**

Update `knowledge_base.py` to:
- expand performance term normalization
- improve relation extraction templates around conductivity, strength, and modulus
- keep compatibility with existing `kb_links` records

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_knowledge_base_service -v`

Expected: PASS

### Task 2: Add Embedding Storage And Embedding-First Recall

**Files:**
- Modify: `D:\CNTDATA\CNTA_ML_Project\backend\core\knowledge_base.py`
- Modify: `D:\CNTDATA\CNTA_ML_Project\backend\core\knowledge_rag.py`
- Test: `D:\CNTDATA\CNTA_ML_Project\tests\test_knowledge_base_service.py`

**Step 1: Write the failing test**

Add tests covering:
- embedding vectors can be stored or regenerated for chunks
- `retrieve_from_pdf()` can use embedding-first recall
- fallback to lexical retrieval still works when embedding backend is unavailable

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_knowledge_base_service -v`

Expected: FAIL because there is no embedding-backed recall path yet.

**Step 3: Write minimal implementation**

Implement:
- an embedding table or serialized embedding field for chunks
- a chunk embedding build/update method
- an embedding recall method that returns top-N candidates
- fallback behavior to current lexical search

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_knowledge_base_service -v`

Expected: PASS

### Task 3: Add Relation Enhancement And Lightweight Reranking

**Files:**
- Modify: `D:\CNTDATA\CNTA_ML_Project\backend\core\knowledge_base.py`
- Modify: `D:\CNTDATA\CNTA_ML_Project\backend\core\knowledge_rag.py`
- Modify: `D:\CNTDATA\CNTA_ML_Project\backend\main.py`
- Test: `D:\CNTDATA\CNTA_ML_Project\tests\test_knowledge_base_service.py`

**Step 1: Write the failing test**

Add tests covering:
- top-20 embedding candidates can be relation-enhanced and reranked to top-5
- task-aware retrieval boosts candidates matching entity and relation expectations
- `/api/rag/search` still returns the same response shape

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_knowledge_base_service -v`

Expected: FAIL because reranking and relation-aware reordering are not implemented.

**Step 3: Write minimal implementation**

Implement:
- query intent parsing for process, morphology, performance, and mechanism hints
- relation-aware score boosting
- a lightweight reranker layer for top-N candidates
- integration of reranked results into the current API response

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_knowledge_base_service -v`

Expected: PASS

### Task 4: Rebuild Existing KB Links And Performance Knowledge Units

**Files:**
- Modify: `D:\CNTDATA\CNTA_ML_Project\scripts\rebuild_kb_links.py`
- Optionally modify: `D:\CNTDATA\CNTA_ML_Project\backend\core\knowledge_seed.py`
- Test: `D:\CNTDATA\CNTA_ML_Project\database\cnta_knowledge_base.sqlite`

**Step 1: Add or review the script behavior**

Ensure the rebuild script:
- clears and rebuilds `kb_links`
- prints counts by relation type
- can be rerun safely on the current knowledge database

**Step 2: Run rebuild on the current database**

Run: `python scripts/rebuild_kb_links.py --kb-db database/cnta_knowledge_base.sqlite`

Expected: relation totals increase or become more balanced, especially for performance-related links.

**Step 3: Verify database stats**

Check:
- total `kb_links`
- relation type distribution
- whether conductivity, tensile strength, and modulus relations exist

**Step 4: Document observed before/after counts**

Save the numbers for later thesis tables and system summaries.

### Task 5: Formalize Evaluation Workflow For Publication Figures

**Files:**
- Modify: `D:\CNTDATA\CNTA_ML_Project\scripts\evaluate_retrieval_models.py`
- Modify: `D:\CNTDATA\CNTA_ML_Project\data\eval\README_retrieval_eval.md`
- Modify: `D:\CNTDATA\CNTA_ML_Project\data\eval\retrieval_eval_set.json`
- Modify: `D:\CNTDATA\CNTA_ML_Project\data\eval\retrieval_eval_set_20.json`
- Output: `D:\CNTDATA\CNTA_ML_Project\data\eval\retrieval_model_comparison_extended_online.csv`

**Step 1: Review and clean the current eval set**

Ensure query labels cover:
- process analysis
- morphology interpretation
- performance analysis

**Step 2: Add missing queries**

Expand to a publication-friendly evaluation set, ideally 20-50 labeled queries.

**Step 3: Run baseline and upgraded retrieval**

Run:
- lexical baseline
- embedding retrieval
- embedding + relation enhancement + reranking

**Step 4: Export thesis-ready tables**

Produce CSV outputs that compare:
- Recall@5 and Recall@10
- MRR@5 and MRR@10
- nDCG@5 and nDCG@10

### Task 6: Surface The Upgraded Retrieval In The Existing UI

**Files:**
- Modify: `D:\CNTDATA\CNTA_ML_Project\backend\main.py`
- Modify: `D:\CNTDATA\CNTA_ML_Project\index.html`

**Step 1: Keep current endpoints stable**

Preserve:
- `GET /api/rag/stats`
- `POST /api/rag/search`
- `POST /api/rag/links`
- `GET /api/rag/documents`

**Step 2: Add retrieval metadata to responses**

Expose:
- retrieval mode
- rerank status
- relation match hints

**Step 3: Show upgraded result hints in the front end**

Display lightweight signals such as:
- semantic recall
- relation enhanced
- top reranked result

**Step 4: Verify end-to-end behavior**

Check that uploaded literature, search, relation graph, and result lists still work in the web UI.

### Task 7: Run End-To-End Verification

**Files:**
- Test: `D:\CNTDATA\CNTA_ML_Project\tests\test_knowledge_base_service.py`
- Test: `D:\CNTDATA\CNTA_ML_Project\database\cnta_knowledge_base.sqlite`
- Test: `D:\CNTDATA\CNTA_ML_Project\data\eval\retrieval_model_comparison*.csv`

**Step 1: Run knowledge-base tests**

Run: `python -m unittest tests.test_knowledge_base_service -v`

Expected: PASS

**Step 2: Rebuild links on the real KB**

Run: `python scripts/rebuild_kb_links.py --kb-db database/cnta_knowledge_base.sqlite`

Expected: PASS with printed relation counts

**Step 3: Run retrieval evaluation**

Run the updated evaluation script against the labeled query set.

Expected: upgraded retrieval outperforms or at least complements lexical baseline on the task set.

**Step 4: Verify the live system**

Open the web app and confirm:
- literature management still works
- search returns evidence passages
- relation chain summaries still render
- upgraded retrieval metadata appears correctly

Plan complete and saved to `docs/plans/2026-03-20-cnta-knowledge-engineering-rag.md`. Two execution options:

1. Subagent-Driven (this session) - I dispatch fresh subagent per task, review between tasks, fast iteration
2. Parallel Session (separate) - Open new session with executing-plans, batch execution with checkpoints

Which approach?
