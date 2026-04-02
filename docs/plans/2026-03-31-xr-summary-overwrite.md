# XR Summary Overwrite Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Overwrite legacy XR morphology fields in `images` from the latest XR summary report and add explicit columns for the new L2/junction feature set.

**Architecture:** Add a maintenance script that reads the verified XR `summary.csv`, ensures destination columns exist, and updates only XR rows by `image_id`. Run the update against a temporary SQLite copy and copy the result back to avoid environment-specific SQLite I/O issues.

**Tech Stack:** Python, sqlite3, csv, unittest

---

### Task 1: Add regression test for XR summary-to-images mapping

**Files:**
- Create: `D:\CNTDATA\CNTA_ML_Project\tests\test_sync_xr_summary_to_images.py`
- Create: `D:\CNTDATA\CNTA_ML_Project\tools\maintenance\sync_xr_summary_to_images.py`

**Step 1: Write the failing test**

Cover:
- missing columns are added
- XR row is updated from summary by `image_id`
- legacy overwrite fields use the intended new columns
- non-XR rows are untouched

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_sync_xr_summary_to_images -v`

Expected: FAIL because the sync script does not exist yet.

**Step 3: Write minimal implementation**

Implement:
- column ensure helper
- summary row parsing helper
- row update helper

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_sync_xr_summary_to_images -v`

Expected: PASS

### Task 2: Implement real XR sync script

**Files:**
- Modify: `D:\CNTDATA\CNTA_ML_Project\tools\maintenance\sync_xr_summary_to_images.py`

**Step 1: Add CLI arguments**

Support:
- `--db-path`
- `--summary-csv`

**Step 2: Add temp-copy write strategy**

Copy db to temp dir, update there, copy back.

**Step 3: Add summary filtering and reporting**

Print:
- updated row count
- skipped row count
- missing image ids

**Step 4: Run script on real XR summary**

Run against:
- `database/cnta_experiments.sqlite`
- `reports/slice_standard_batch_20260331_005741/summary.csv`

Expected: XR rows updated successfully.

### Task 3: Verify DB values and report results

**Files:**
- Test: `D:\CNTDATA\CNTA_ML_Project\database\cnta_experiments.sqlite`

**Step 1: Query updated columns**

Check:
- row counts
- non-null counts
- sample rows

**Step 2: Compare updated values against summary**

Spot-check several XR `image_id`.

**Step 3: Report final mapping**

List:
- overwritten legacy fields
- newly added fields
- counts updated
