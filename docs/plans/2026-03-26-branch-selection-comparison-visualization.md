# Branch Selection Comparison Visualization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a standalone script that renders 9 comparison panels for `coredata/u`, showing original ROI, V2 branches, V3 branches, fast-mode branches, and feature metrics.

**Architecture:** Reuse the current threshold-mask and skeleton pipeline once per image, derive accurate and fast branch sets from `FeatureExtractor`, compute feature summaries from the same skeleton, and render a research panel with mask-based branch overlays.

**Tech Stack:** Python, NumPy, OpenCV, matplotlib, existing `FeatureExtractor`

---

### Task 1: Create the comparison script

**Files:**
- Create: `D:\CNTDATA\CNTA_ML_Project\tools\generate_branch_selection_compare_panels.py`

**Step 1: Enumerate all 9 images in `coredata/u`**

- Use both `50000x` and `100000x` folders.

**Step 2: Compute common image preparation once**

- ROI
- preprocess
- threshold mask
- skeleton

**Step 3: Compute branch selections**

- accurate V2 branch set
- accurate V3 branch set
- fast V2 branch set
- fast V3 branch set

**Step 4: Compute feature summaries**

- density
- diameter
- legacy curvature
- V2 curvature
- V3 curvature
- waviness/tortuosity

**Step 5: Render a panel per image**

- original ROI
- V2 accurate branch overlay
- V3 accurate branch overlay
- fast branch overlay
- text table

### Task 2: Verify outputs

**Files:**
- Output: new report directory under `D:\CNTDATA\CNTA_ML_Project\reports`

**Step 1: Run the script in `LAB_AGENT`**

- Generate all 9 panels.

**Step 2: Verify summary files**

- Confirm `summary.csv` and `summary.json` are present.
