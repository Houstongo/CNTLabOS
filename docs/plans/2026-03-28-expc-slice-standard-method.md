# ExpC Slice Standard Method Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Define the single-image Exp C mask study as a standard method using `L1-L4`, adaptive curvature histograms in `um^-1`, and dual curvature aggregations with and without `sqrt(length)` weighting.

**Architecture:** Reuse the already generated Exp C mask from the desktop report output, keep all computation local to the existing single-panel study script, and avoid rerunning model inference. Extend the script to (1) drop `L0`, (2) compute both `sqrt(length)` and linear-length curvature summaries from the same pruned branch sets, and (3) regenerate one panel plus JSON summary from cached data.

**Tech Stack:** Python, NumPy, OpenCV, Matplotlib, existing `FeatureExtractor`

---

### Task 1: Update threshold study script to standard-method settings

**Files:**
- Modify: `D:\CNTDATA\CNTA_ML_Project\tools\generate_expc_slice_v3_threshold_single_panel.py`

**Step 1: Change thresholds from `L0-L4` to `L1-L4`**

Keep only:
- `L1 = 3.0`
- `L2 = 5.0`
- `L3 = 7.0`
- `L4 = 9.0`

**Step 2: Add two aggregate curvature outputs per threshold**

For each pruned branch set, compute:
- `curvature_nm_v3_sqrt_length`
- `curvature_nm_v3_length`

using the same branch-level `p75` values, with weights:
- `sqrt(path_length_px)`
- `path_length_px`

**Step 3: Change curvature histogram display to `um^-1`**

Convert point curvatures from `nm^-1` to `um^-1` for plotting only, and set x-axis upper bound using a high percentile so distributions do not collapse to the left.

### Task 2: Update output text and JSON summary

**Files:**
- Modify: `D:\CNTDATA\CNTA_ML_Project\tools\generate_expc_slice_v3_threshold_single_panel.py`

**Step 1: Add standard-method description to the panel header**

Describe the method as:
- `CNTSegNet-SLICE`
- `V3 branch pruning`
- `L1-L4`
- `dual aggregation: sqrt(length) and length`

**Step 2: Add both curvature summaries to each threshold text block**

Show:
- `sqrt_len`
- `linear_len`
- branch count
- waviness/tortuosity

**Step 3: Add the same fields to `summary.json`**

Persist:
- threshold factor
- branch count
- `curvature_nm_v3_sqrt_length`
- `curvature_nm_v3_length`
- distribution point counts

### Task 3: Run one minimal test from existing cached Exp C mask

**Files:**
- Reuse: `D:\CNTDATA\CNTA_ML_Project\reports\desktop_expc_baseline_v2v3_report_20260328_020832\items\text10_No28_200w_15_0nm_50w_0_5nm_600_300_150_600_750_15min_180min_mid_100000-1`

**Step 1: Run the updated script once**

Use the existing `features.json` and `expc_mask.png` from the item directory.

**Step 2: Verify output artifacts**

Check:
- panel PNG exists
- `summary.json` exists
- `summary.json` includes `L1-L4` plus both curvature aggregations

**Step 3: Report the output path and key values**

Summarize the new standard-method panel path and the four threshold rows.
