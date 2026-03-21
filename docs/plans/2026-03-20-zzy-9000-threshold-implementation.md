# ZZY 9000 Threshold Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Raise the ZZY import magnification threshold to 9000 and soft-delete active ZZY rows below that threshold.

**Architecture:** Keep the existing ZZY filename parsing and `mid*` position matching, but change the eligibility gate from `>= 5000` to `>= 9000`. Use a one-time database update to mark existing active ZZY rows below 9000 as logically deleted without touching XR rows or already deleted ZZY rows.

**Tech Stack:** Python, SQLite, inline import scripts, Python `unittest`

---

### Task 1: Lock the new threshold with tests

**Files:**
- Modify: `D:\CNTDATA\CNTA_ML_Project\tests\test_zzy_import_rules.py`

**Step 1: Write the failing tests**

- Change the include-case to `mid 9000-1`.
- Add or update a case showing `mid 5000-1` is now excluded.

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_zzy_import_rules`
Expected: FAIL because the parser still accepts `>= 5000`.

**Step 3: Write minimal implementation**

- Update the ZZY inclusion rule to require `magnification >= 9000`.

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_zzy_import_rules`
Expected: PASS

### Task 2: Update parser and import gate

**Files:**
- Modify: `D:\CNTDATA\CNTA_ML_Project\backend\core\data_manager.py`
- Modify: `D:\CNTDATA\CNTA_ML_Project\backend\core\populate_db.py`

**Step 1: Update the threshold constant or comparison**

- Change the ZZY gate from `>= 5000` to `>= 9000`.

**Step 2: Keep the existing mid-position rule**

- Do not widen the accepted positions.
- Keep `mid*` matching behavior unchanged.

**Step 3: Run focused tests**

Run: `python -m unittest tests.test_zzy_import_rules`
Expected: PASS

### Task 3: Soft-delete existing active ZZY rows below 9000

**Files:**
- Modify data in: `D:\CNTDATA\CNTA_ML_Project\database\cnta_experiments.sqlite`

**Step 1: Inspect the current scope**

Run a count query for active `ZZY` rows where `magnification < 9000`.

**Step 2: Apply the logical delete**

- Update only rows where:
  - `source = 'ZZY'`
  - `COALESCE(is_deleted, 0) = 0`
  - `magnification < 9000`

**Step 3: Verify the result**

Run count queries to confirm:
- the targeted rows are now logically deleted
- no active ZZY rows remain below 9000
- total ZZY row count is unchanged

### Task 4: Final verification

**Files:**
- Modify: `D:\CNTDATA\CNTA_ML_Project\backend\core\data_manager.py`
- Modify: `D:\CNTDATA\CNTA_ML_Project\backend\core\populate_db.py`
- Modify: `D:\CNTDATA\CNTA_ML_Project\tests\test_zzy_import_rules.py`

**Step 1: Run tests**

Run: `python -m unittest tests.test_zzy_import_rules tests.test_xr_api_listing tests.test_batch_processor`
Expected: PASS

**Step 2: Verify database counts**

Run SQLite queries to confirm the new active/deleted ZZY split after the migration.
