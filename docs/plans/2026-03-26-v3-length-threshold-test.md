# V3 Length Threshold Test Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a small research script that compares 5 V3 branch-length thresholds and renders skeleton panels for 6 sample SEM images.

**Architecture:** Reuse the current threshold-mask and skeleton pipeline once per image, derive V3-style ordered branches at five different length thresholds, compute per-threshold metrics, and render an 8-panel comparison image.

**Tech Stack:** Python, NumPy, OpenCV, matplotlib, existing `FeatureExtractor`

---

### Task 1: Create the standalone threshold-comparison script

**Files:**
- Create: `D:\CNTDATA\CNTA_ML_Project\tools\generate_v3_length_threshold_panels.py`

**Step 1: Select 6 sample images**

- 3 from `50000x`
- 3 from `100000x`

**Step 2: Compute shared image preparation once**

- ROI
- preprocess
- threshold mask
- skeleton

**Step 3: Derive V3 branch sets for `L0..L4`**

- Use the current V3 `min_points`
- Change only `min_length_factor`

**Step 4: Compute per-threshold metrics**

- branch count
- `curvature_nm_v3`
- `waviness_ratio_v2`
- `tortuosity_v2`

**Step 5: Render one panel per image**

- original
- raw skeleton
- `L0..L4`
- metrics table

### Task 2: Verify

**Files:**
- Output: new report directory under `D:\CNTDATA\CNTA_ML_Project\reports`

**Step 1: Run the script in `LAB_AGENT`**

- Generate all 6 panels.

**Step 2: Verify summary outputs**

- Confirm `summary.json` and image panels exist.
