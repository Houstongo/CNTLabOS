# Soft Delete Default Filtering Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make XR batch processing and default image reads ignore logically deleted records by default.

**Architecture:** Keep logical deletion in the database via `images.is_deleted` and enforce `COALESCE(is_deleted, 0) = 0` at read-time. Update both the XR listing/query path in the FastAPI backend and the batch processor selection query so deleted rows are hidden and never processed.

**Tech Stack:** Python, FastAPI, SQLite, unittest

---

### Task 1: Cover XR API filtering

**Files:**
- Modify: `D:\CNTDATA\CNTA_ML_Project\tests\test_xr_api_listing.py`
- Modify: `D:\CNTDATA\CNTA_ML_Project\backend\main.py`

**Step 1: Write the failing test**
- Add a case where an XR image row has `is_deleted = 1` and verify `/api/images?source=XR` does not return it.

**Step 2: Run test to verify it fails**
- Run: `python -m unittest tests.test_xr_api_listing`

**Step 3: Write minimal implementation**
- Add `COALESCE(i.is_deleted, 0) = 0` to the XR-specific listing query path.

**Step 4: Run test to verify it passes**
- Run: `python -m unittest tests.test_xr_api_listing`

### Task 2: Cover batch processor filtering

**Files:**
- Add: `D:\CNTDATA\CNTA_ML_Project\tests\test_batch_processor.py`
- Modify: `D:\CNTDATA\CNTA_ML_Project\backend\core\batch_processor.py`

**Step 1: Write the failing test**
- Create a small temporary SQLite database with one active XR row and one logically deleted XR row, then verify only the active row is processed.

**Step 2: Run test to verify it fails**
- Run: `python -m unittest tests.test_batch_processor`

**Step 3: Write minimal implementation**
- Add `COALESCE(is_deleted, 0) = 0` and `COALESCE(processed, 0) = 0` to the batch selection query.

**Step 4: Run test to verify it passes**
- Run: `python -m unittest tests.test_batch_processor`

### Task 3: Tighten other default reads

**Files:**
- Modify: `D:\CNTDATA\CNTA_ML_Project\backend\main.py`

**Step 1: Write the failing test**
- Extend existing API tests or add a focused test to ensure image-id based reads reject logically deleted rows by default.

**Step 2: Run test to verify it fails**
- Run the targeted unittest module.

**Step 3: Write minimal implementation**
- Add `COALESCE(is_deleted, 0) = 0` to default single-image read endpoints and summary/simple-model queries where appropriate.

**Step 4: Run test to verify it passes**
- Run the targeted unittest module.
