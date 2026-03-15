# Knowledge Base Backend Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a runnable backend knowledge-base foundation that matches the paper structure and remains compatible with the current RAG entry points.

**Architecture:** Add a dedicated knowledge-base service around SQLite-backed document, chunk, relation, and task-profile tables. Keep the existing `RAGRetriever` public surface, but route its document-management and retrieval work through the new service so the current FastAPI endpoints continue to function.

**Tech Stack:** Python, sqlite3, FastAPI, unittest, pdfplumber

---

### Task 1: Add Failing Tests For Knowledge-Base Schema And Ingestion

**Files:**
- Create: `D:\CNTDATA\CNTA_ML_Project\tests\test_knowledge_base_service.py`
- Test: `D:\CNTDATA\CNTA_ML_Project\backend\core\knowledge_base.py`

**Step 1: Write the failing test**

Cover:
- schema initialization creates the expected tables
- text ingestion creates one document and multiple chunks
- task-aware retrieval returns matching chunks

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_knowledge_base_service -v`
Expected: FAIL because the service module does not exist yet

**Step 3: Write minimal implementation**

Create the service with just enough behavior to satisfy the tests.

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_knowledge_base_service -v`
Expected: PASS

### Task 2: Implement Knowledge-Base Service And Compatibility Layer

**Files:**
- Create: `D:\CNTDATA\CNTA_ML_Project\backend\core\knowledge_base.py`
- Modify: `D:\CNTDATA\CNTA_ML_Project\backend\core\rag_retriever.py`

**Step 1: Write the failing test**

Add coverage for:
- listing and deleting documents through the retriever
- `retrieve_all()` returning both similar experiments and knowledge passages

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_knowledge_base_service -v`
Expected: FAIL because the retriever still uses the legacy schema only

**Step 3: Write minimal implementation**

Make `RAGRetriever` delegate knowledge-base document/chunk work to the new service while preserving the current API shape.

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_knowledge_base_service -v`
Expected: PASS

### Task 3: Add CLI And API Support For The New Foundation

**Files:**
- Modify: `D:\CNTDATA\CNTA_ML_Project\manage.py`
- Modify: `D:\CNTDATA\CNTA_ML_Project\backend\main.py`

**Step 1: Write the failing test**

Add coverage for:
- knowledge-base stats/query methods used by API handlers
- manage command wiring for bootstrap/import

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_knowledge_base_service -v`
Expected: FAIL because the new operations are not exposed yet

**Step 3: Write minimal implementation**

Add:
- a `kb-bootstrap` style manage command
- API-compatible helpers for stats and task-aware search

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_knowledge_base_service -v`
Expected: PASS

### Task 4: Verify The Runnable Flow

**Files:**
- Test: `D:\CNTDATA\CNTA_ML_Project\tests\test_knowledge_base_service.py`

**Step 1: Run the full test file**

Run: `python -m unittest tests.test_knowledge_base_service -v`

**Step 2: Run a basic bootstrap smoke check**

Run: `python manage.py kb-bootstrap`
Expected: prints a knowledge-base initialization summary without errors

