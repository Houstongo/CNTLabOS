# Soft Delete Trash Interface Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Separate logical deletion, physical deletion, and trash display behavior across the API and frontend.

**Architecture:** Keep `is_deleted` as the canonical logical deletion flag, make normal views consume only active rows, add an explicit trash view for deleted rows, and restrict physical deletion to items that are already in the trash. Expose the mode through list APIs so the frontend no longer infers trash state with ad-hoc filtering.

**Tech Stack:** FastAPI, SQLite, inline frontend JavaScript in `index.html`, Python `unittest`, Node `node:test`

---

### Task 1: Lock backend deletion semantics with tests

**Files:**
- Modify: `D:\CNTDATA\CNTA_ML_Project\tests\test_xr_api_listing.py`

**Step 1: Write the failing tests**

- Add a test that requests deleted records from `get_image_list(...)` and expects logically deleted rows to be returned.
- Add a test that requests all records from `get_image_list(...)` and expects both active and deleted rows.
- Add a test that calls `delete_image(...)` on an active row and expects a conflict instead of physical deletion.

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_xr_api_listing`
Expected: FAIL because the list API only returns active rows and the delete endpoint still hard-deletes active rows.

**Step 3: Write minimal implementation**

- Extend the list API to accept a deletion view selector.
- Restrict physical deletion to rows already marked with `is_deleted = 1`.

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_xr_api_listing`
Expected: PASS

### Task 2: Implement backend list-mode and trash-safe delete behavior

**Files:**
- Modify: `D:\CNTDATA\CNTA_ML_Project\backend\main.py`

**Step 1: Add visibility helpers**

- Add one helper that builds the SQL clause for `active`, `deleted`, and `all`.
- Keep the existing active-only helper behavior for endpoints that must ignore deleted rows.

**Step 2: Update image listing**

- Extend `get_image_list(...)` with a `deletion_view` query parameter.
- Preserve compatibility with existing callers by defaulting to `active`.
- Return `is_deleted` in the payload so frontend state can render the correct action set.

**Step 3: Update summary payload**

- Add deleted-count summary fields for the frontend trash badge and dashboard context.
- Keep the default total focused on active data.

**Step 4: Guard physical delete**

- Change `DELETE /api/images/{id}` so it only permanently deletes rows already in the trash.
- Return a clear error when a user tries to hard-delete an active record.

**Step 5: Run backend tests**

Run: `python -m unittest tests.test_xr_api_listing tests.test_batch_processor`
Expected: PASS

### Task 3: Lock the frontend trash affordances with lightweight contract tests

**Files:**
- Modify: `D:\CNTDATA\CNTA_ML_Project\tests\test_data_cleaning_ui_contract.mjs`
- Modify: `D:\CNTDATA\CNTA_ML_Project\tests\test_index_page_structure.py`

**Step 1: Write the failing tests**

- Assert the page contains a trash-view toggle/control.
- Assert the detail panel contains a dedicated actions mount instead of a single unconditional hard-delete button.

**Step 2: Run test to verify it fails**

Run: `node --test tests/test_data_cleaning_ui_contract.mjs && python -m unittest tests.test_index_page_structure`
Expected: FAIL because the current page still exposes the old permanent delete flow.

**Step 3: Implement minimal markup changes**

- Add the trash toggle/badge container.
- Replace the hard-coded detail delete button with a dynamic action container.

**Step 4: Run tests to verify they pass**

Run: `node --test tests/test_data_cleaning_ui_contract.mjs && python -m unittest tests.test_index_page_structure`
Expected: PASS

### Task 4: Implement frontend trash view and dual delete actions

**Files:**
- Modify: `D:\CNTDATA\CNTA_ML_Project\index.html`

**Step 1: Add frontend state for deletion view**

- Extend `cleanState` with `view`, deleted counters, and selection reset behavior.

**Step 2: Update list loading**

- Send `deletion_view=active|deleted` to `/api/images`.
- Remove the client-side “always hide deleted” filter.

**Step 3: Implement action rendering**

- Active view: show “移入回收站”.
- Trash view: show “恢复” and “彻底删除”.
- General detail modal: same rule, no hard-delete button in active mode.

**Step 4: Implement trash switching**

- Replace the alert-based deleted batch listing with a true recycle-bin view.
- Show counts so users know whether they are looking at active data or trash.

**Step 5: Run UI contract tests**

Run: `node --test tests/test_data_cleaning_ui_contract.mjs`
Expected: PASS

### Task 5: Verify end-to-end behavior

**Files:**
- Modify: `D:\CNTDATA\CNTA_ML_Project\backend\main.py`
- Modify: `D:\CNTDATA\CNTA_ML_Project\index.html`
- Modify: `D:\CNTDATA\CNTA_ML_Project\tests\test_xr_api_listing.py`
- Modify: `D:\CNTDATA\CNTA_ML_Project\tests\test_data_cleaning_ui_contract.mjs`
- Modify: `D:\CNTDATA\CNTA_ML_Project\tests\test_index_page_structure.py`

**Step 1: Run focused verification**

Run: `python -m unittest tests.test_xr_api_listing tests.test_batch_processor tests.test_index_page_structure`
Expected: PASS

**Step 2: Run frontend contract verification**

Run: `node --test tests/test_data_cleaning_ui_contract.mjs`
Expected: PASS

**Step 3: Sanity-check runtime behavior**

- Open the clean page.
- Confirm active view hides deleted items.
- Confirm trash view shows deleted items.
- Confirm hard delete is only exposed in trash view.

