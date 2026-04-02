# XR Review Report Timeout Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add 100-second timeout skipping and expanded V3-oriented review metrics to the XR visual-review report pipeline.

**Architecture:** Keep `tools/generate_xr_slice_standard_batch.py` as the single review-only entrypoint. Run each image in a spawned subprocess with a hard timeout so segmentation and feature analysis can be interrupted safely, then store success/timeout/error status in the item JSON plus the aggregate summaries.

**Tech Stack:** Python, multiprocessing, NumPy, OpenCV, PyTorch, matplotlib, existing `FeatureExtractor`

---

### Task 1: Add timeout-aware XR item processing

**Files:**
- Modify: `D:\CNTDATA\CNTA_ML_Project\tools\generate_xr_slice_standard_batch.py`

**Step 1: Add CLI timeout control**

- Add a `--timeout-s` argument with a default value of `100.0`.

**Step 2: Move single-image work into a worker-safe function**

- Isolate the current per-image logic so it can run in a subprocess and return a JSON-serializable payload.

**Step 3: Add subprocess timeout supervision**

- Reuse the same timeout pattern already used in `backend/core/batch_processor.py`.
- Return a structured timeout record instead of raising and aborting the whole batch.

### Task 2: Expand XR review summary fields

**Files:**
- Modify: `D:\CNTDATA\CNTA_ML_Project\tools\generate_xr_slice_standard_batch.py`

**Step 1: Add the requested V3 curvature aggregations**

- Include `p70`, `p75`, `mean`, and `trimmed_mean`
- Include both `sqrt_length` and `length` weighted variants

**Step 2: Keep other review metrics in the payload**

- Preserve density, alignment, diameter, waviness, and tortuosity

**Step 3: Flatten the new fields into CSV-safe columns**

- Ensure review summaries can be compared quickly in spreadsheet form

### Task 3: Make summary writing status-aware

**Files:**
- Modify: `D:\CNTDATA\CNTA_ML_Project\tools\generate_xr_slice_standard_batch.py`

**Step 1: Include success/timeout/error status in item records**

- Record elapsed seconds and the last completed stage when available

**Step 2: Preserve timed-out items in aggregate summaries**

- They should appear in `summary.json` and `summary.csv` instead of disappearing silently

### Task 4: Verify the new XR review flow

**Files:**
- Verify via script execution

**Step 1: Run a small XR batch**

```powershell
python D:\CNTDATA\CNTA_ML_Project\tools\generate_xr_slice_standard_batch.py --limit 1 --timeout-s 100
```

**Step 2: Confirm outputs**

- item JSON exists
- panel PNG exists for successful items
- summary files contain the new metrics and status fields

**Step 3: Sanity-check timeout behavior**

- Confirm the script can represent timeout skips without aborting the full batch
