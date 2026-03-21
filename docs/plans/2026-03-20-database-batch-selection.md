# Database Batch Selection Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add multi-select batch actions to the main database list so users can batch-run image characterization and logical deletion.

**Architecture:** Keep the change scoped to the main database page in `index.html`, add batch endpoints in `backend/main.py`, and reuse the existing single-image analysis and logical delete semantics. Persist selection only in frontend runtime state and reconcile it after every reload.

**Tech Stack:** FastAPI, SQLite, vanilla JavaScript, inline HTML, unittest, node:test

---

### Task 1: Document and anchor the UI structure

**Files:**
- Modify: `D:\CNTDATA\CNTA_ML_Project\index.html`
- Test: `D:\CNTDATA\CNTA_ML_Project\tests\test_index_page_structure.py`
- Test: `D:\CNTDATA\CNTA_ML_Project\tests\test_database_batch_ui_contract.mjs`

**Step 1: Write the failing tests**

- Assert the main data page exposes:
  - a batch action bar mount
  - a select-all checkbox
  - batch analyze and batch delete buttons

**Step 2: Run tests to verify they fail**

Run:

```bash
python -m unittest tests.test_index_page_structure
node --test tests/test_database_batch_ui_contract.mjs
```

Expected: failures for the missing batch UI hooks.

**Step 3: Write minimal implementation**

- Add the batch toolbar container near the main data page header.
- Add the select-all checkbox to the table header.

**Step 4: Run tests to verify they pass**

Run the same commands and confirm they pass.

### Task 2: Add failing backend tests for batch endpoints

**Files:**
- Create: `D:\CNTDATA\CNTA_ML_Project\tests\test_batch_image_actions.py`
- Modify: `D:\CNTDATA\CNTA_ML_Project\backend\main.py`

**Step 1: Write the failing tests**

- Batch analyze succeeds for multiple active records.
- Batch analyze skips logically deleted records.
- Batch logical delete marks only requested active records.
- Empty batch payload returns 400.

**Step 2: Run tests to verify they fail**

Run:

```bash
python -m unittest tests.test_batch_image_actions -v
```

Expected: failures because the endpoints do not exist yet.

**Step 3: Write minimal implementation**

- Add request model for image ID lists.
- Extract single-image analysis into a reusable helper.
- Add batch analyze endpoint and batch logical delete endpoint.

**Step 4: Run tests to verify they pass**

Run the same unittest command and confirm green output.

### Task 3: Wire batch selection state into the main data page

**Files:**
- Modify: `D:\CNTDATA\CNTA_ML_Project\index.html`

**Step 1: Write the failing tests**

- Extend the contract tests only if needed for new stateful hooks.

**Step 2: Run test to verify it fails**

Run the relevant node or unittest command if a new assertion is added.

**Step 3: Write minimal implementation**

- Add a `selectedDataIds` state holder.
- Add helpers for row checkbox toggle, select-all, clearing, and state reconciliation after reload.
- Ensure checkbox clicks do not open the details drawer.

**Step 4: Run tests to verify it passes**

Run the relevant frontend contract tests again.

### Task 4: Wire the batch action buttons

**Files:**
- Modify: `D:\CNTDATA\CNTA_ML_Project\index.html`

**Step 1: Write the failing tests**

- Reuse the existing structure tests unless a new contract assertion is needed.

**Step 2: Run test to verify it fails**

- Only if a new test was added.

**Step 3: Write minimal implementation**

- Connect batch analyze to `POST /api/images/batch/analyze`.
- Connect batch logical delete to `PUT /api/images/batch/delete`.
- Refresh the table after success and show a concise summary.
- Hide batch logical delete while viewing deleted records.

**Step 4: Run tests to verify it passes**

- Re-run the frontend contract tests.

### Task 5: Final verification

**Files:**
- Modify: `D:\CNTDATA\CNTA_ML_Project\backend\main.py`
- Modify: `D:\CNTDATA\CNTA_ML_Project\index.html`
- Test: `D:\CNTDATA\CNTA_ML_Project\tests\test_batch_image_actions.py`
- Test: `D:\CNTDATA\CNTA_ML_Project\tests\test_index_page_structure.py`
- Test: `D:\CNTDATA\CNTA_ML_Project\tests\test_database_batch_ui_contract.mjs`

**Step 1: Run targeted verification**

```bash
python -m unittest tests.test_batch_image_actions tests.test_index_page_structure -v
node --test tests/test_database_batch_ui_contract.mjs
```

Expected: all tests pass.

**Step 2: Manual sanity check**

- Confirm row checkbox clicks do not open details.
- Confirm select-all reflects current page only.
- Confirm batch analyze and batch delete refresh the list and clear stale selections.
