# RAG Page Subpages Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Restructure the RAG knowledge page into compact subpages so overview, graph, performance, and document management are separated and readable without page-level scrolling on desktop.

**Architecture:** Keep the existing single-page HTML application and current `/api/rag/*` endpoints. Replace the current two-tab RAG layout with a shared compact toolbar plus four dedicated subpages that render from the same `loadRagLinks()` data payload. Use internal scroll areas for dense lists instead of scrolling the whole page.

**Tech Stack:** Static HTML, Tailwind utility classes already present in `index.html`, inline JavaScript rendering functions, Node contract tests.

---

### Task 1: Lock the new navigation contract

**Files:**
- Create: `D:\CNTDATA\CNTA_ML_Project\tests\test_rag_workspace_contract.mjs`
- Modify: `D:\CNTDATA\CNTA_ML_Project\index.html`

**Step 1: Write the failing test**
- Assert the RAG page exposes four tab buttons:
  - `rag-tab-overview-btn`
  - `rag-tab-graph-btn`
  - `rag-tab-performance-btn`
  - `rag-tab-manage-btn`
- Assert the page exposes four subpage containers:
  - `rag-subpage-overview`
  - `rag-subpage-graph`
  - `rag-subpage-performance`
  - `rag-subpage-manage`

**Step 2: Run test to verify it fails**
- Run: `node tests/test_rag_workspace_contract.mjs`

**Step 3: Implement the minimal markup**
- Add the new tab buttons and container shells in `index.html`.

**Step 4: Run test to verify it passes**
- Run: `node tests/test_rag_workspace_contract.mjs`

### Task 2: Split the current mixed success page into focused subpages

**Files:**
- Modify: `D:\CNTDATA\CNTA_ML_Project\index.html`

**Step 1: Keep one shared compact toolbar**
- Title + tabs row
- Query input + refresh action row

**Step 2: Build subpages**
- `overview`: KPI cards, relation summary, compact chain lists
- `graph`: full-width graph panel + compact graph stats
- `performance`: performance summary + performance list
- `manage`: existing upload and document list

**Step 3: Reduce page-level scrolling**
- Make `#rag-page` and the active subpage fill the viewport area
- Push scrolling into the chain/performance/document list panels

### Task 3: Update rendering logic

**Files:**
- Modify: `D:\CNTDATA\CNTA_ML_Project\index.html`

**Step 1: Expand `switchRagSubPage(page)`**
- Support `overview / graph / performance / manage`
- Toggle active button state and the four containers

**Step 2: Update `loadRagLinks()`**
- Render shared query results into:
  - overview summary targets
  - graph chart/stat targets
  - performance targets

**Step 3: Keep existing data sources**
- Reuse `loadRagSuccessStats()`
- Reuse `/api/rag/links`
- Do not add backend changes

### Task 4: Verify layout and regressions

**Files:**
- Modify: `D:\CNTDATA\CNTA_ML_Project\tests\test_rag_performance_view_contract.mjs` only if needed

**Step 1: Run contract tests**
- `node tests/test_rag_layout_contract.mjs`
- `node tests/test_rag_workspace_contract.mjs`
- `node tests/test_rag_performance_view_contract.mjs`

**Step 2: Run existing Python regression tests**
- `python -m unittest tests.test_knowledge_base_service tests.test_retrieval_evaluation -v`

**Step 3: Browser verify**
- Open the app
- Switch to `文献知识库`
- Confirm:
  - four tabs render
  - graph gets a dedicated page
  - performance gets a dedicated page
  - no large left blank area

