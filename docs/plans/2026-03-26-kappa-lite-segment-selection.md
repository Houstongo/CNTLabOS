# Kappa-Lite Segment Selection Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a research-only module that selects the top 10 measurable centerline segments from a real `100000x` CNT image and reports Fiji/Kappa-inspired per-segment geometry metrics.

**Architecture:** Reuse the current `FeatureExtractor` preprocessing and V2 ordered-branch extraction, add lightweight candidate scoring and non-overlap selection, then compute per-segment `L/D` and curvature metrics and render a validation panel.

**Tech Stack:** Python, NumPy, OpenCV, scikit-image, matplotlib, existing `FeatureExtractor`

---

### Task 1: Add lightweight branch scoring helpers

**Files:**
- Modify: `D:\CNTDATA\CNTA_ML_Project\src\analysis\feature_extractor.py`

**Step 1: Add candidate-level utility helpers**

- Add helpers for:
  - border distance
  - nearest-junction distance
  - width statistics from distance transform
  - simple non-overlap comparison between candidate segments

**Step 2: Add a branch-scoring helper**

- Score each ordered branch from:
  - length
  - junction distance
  - width consistency

**Step 3: Add a top-segment selection helper**

- Select the highest-scoring non-overlapping branches.
- Default to `top_k = 10`.

### Task 2: Add per-segment metric extraction

**Files:**
- Modify: `D:\CNTDATA\CNTA_ML_Project\src\analysis\feature_extractor.py`

**Step 1: Add per-segment geometry helper**

- For a selected segment, compute:
  - path length
  - span
  - `L/D`
  - mean curvature
  - p90 curvature
  - mean width
  - width CV

**Step 2: Add a research-only API**

- Add a method such as `extract_kappa_lite_segments(...)` that returns:
  - selected segment records
  - minimal context needed for visualization

### Task 3: Create a validation script

**Files:**
- Create: `D:\CNTDATA\CNTA_ML_Project\tools\generate_kappa_lite_segment_panel.py`

**Step 1: Load a real `100000x` image**

- Start with one selected validation image.

**Step 2: Run the new research API**

- Extract the top 10 segments and metrics.

**Step 3: Render a validation panel**

- Save a panel with:
  - original ROI
  - mask
  - ordered-branch reference
  - selected top 10 segments with labels
  - per-segment table

**Step 4: Save machine-readable outputs**

- Save per-segment results to JSON and CSV.

### Task 4: Add focused tests

**Files:**
- Modify: `D:\CNTDATA\CNTA_ML_Project\tests\test_feature_extractor_curvature_v2.py`
- Or create: `D:\CNTDATA\CNTA_ML_Project\tests\test_feature_extractor_kappa_lite.py`

**Step 1: Add candidate-scoring tests**

- Verify longer, more stable branches score higher than short noisy ones.

**Step 2: Add non-overlap selection tests**

- Verify overlapping candidates do not all survive into the final top-K set.

**Step 3: Add geometry sanity tests**

- Verify straight synthetic lines have `L/D` near `1`.
- Verify wavy synthetic lines have `L/D > 1`.

### Task 5: Verify

**Files:**
- Test: `D:\CNTDATA\CNTA_ML_Project\tests\...`

**Step 1: Run targeted unit tests**

Run the relevant `unittest` module in `LAB_AGENT`.

**Step 2: Run the validation script**

- Confirm the panel, CSV, and JSON are produced.

**Step 3: Review visual plausibility**

- Confirm the selected top 10 segments look like human-plausible measurement targets.
