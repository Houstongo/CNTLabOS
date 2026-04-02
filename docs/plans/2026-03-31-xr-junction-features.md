# XR Junction Features Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add junction topology features to the XR report pipeline and backfill the current XR report for modeling analysis.

**Architecture:** Compute junction features directly from the already-generated skeleton in the XR report analysis stage. Keep the existing geometry metrics unchanged and add a separate backfill script that updates cached report records and rewrites summary files.

**Tech Stack:** Python, OpenCV, NumPy, existing XR batch report pipeline

---

### Task 1: Document the metric definitions

**Files:**
- Create: `D:\CNTDATA\CNTA_ML_Project\docs\plans\2026-03-31-xr-junction-features-design.md`
- Create: `D:\CNTDATA\CNTA_ML_Project\docs\plans\2026-03-31-xr-junction-features.md`

**Step 1: Record the chosen fields**

Write down:
- `junction_count`
- `junction_ratio`

**Step 2: Record what is intentionally unchanged**

Note that curvature and waviness formulas are not changed.

### Task 2: Add junction metrics to the XR report generator

**Files:**
- Modify: `D:\CNTDATA\CNTA_ML_Project\tools\generate_xr_slice_standard_batch.py`

**Step 1: Add a helper that counts junction connected components**

Use the skeleton neighbor-count rule and connected-component counting.

**Step 2: Add a normalized junction density metric**

Use the already-removed junction pixels and compute `junction_ratio = junction_count / skeleton_length_px`.

**Step 3: Persist the new fields**

Write the new fields into:
- `features.json`
- flattened summary records

### Task 3: Backfill the current XR report

**Files:**
- Create: `D:\CNTDATA\CNTA_ML_Project\tools\enrich_xr_report_with_junction_metrics.py`

**Step 1: Iterate over cached feature files**

Load each report item, read image and mask, recompute junction metrics, update the record, and rewrite the file.

**Step 2: Rewrite summary files**

Reuse the existing summary writer so the report-level CSV/JSON picks up the new columns.

### Task 4: Verify with a single item and the full report

**Files:**
- Output: updated `features.json`
- Output: updated `summary.csv`

**Step 1: Run a single-item XR generation check**

Confirm the newly generated record contains junction fields.

**Step 2: Run backfill on the current report**

Confirm the target report now includes junction metrics across the batch.

**Step 3: Spot-check values**

Inspect one or two samples to confirm the metrics are finite and plausible.
