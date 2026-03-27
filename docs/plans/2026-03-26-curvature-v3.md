# Curvature V3 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Correct legacy curvature scaling and add a comparison-oriented V3 curvature metric plus a 5-image comparison report.

**Architecture:** Keep the current `FeatureExtractor` pipeline intact, fix only the legacy unit conversion, and add a new V3 curvature aggregation path that reuses ordered V2 branches but changes branch inclusion and summary weighting. Validate with focused unit tests and a small offline comparison script.

**Tech Stack:** Python, NumPy, OpenCV, existing `FeatureExtractor`, unittest

---

### Task 1: Add curvature V3 implementation to `FeatureExtractor`

**Files:**
- Modify: `D:\CNTDATA\CNTA_ML_Project\src\analysis\feature_extractor.py`

**Step 1: Add a reusable local-curvature helper**

- Factor the repeated ordered-branch three-point curvature loop into a helper that returns point-wise `px^-1` curvature values for sampled ordered coords.

**Step 2: Correct legacy calibrated conversion**

- In `calculate_curvature(...)`, replace fixed `1/15` conversion with calibrated `self.px_per_um / 1000.0`.

**Step 3: Add `calculate_curvature_v3(...)`**

- Reuse ordered branches.
- Relax branch inclusion threshold.
- Use branch `p75`.
- Use `sqrt(path_length_px)` weights.

**Step 4: Expose V3 in `extract_all(...)`**

- Compute `curvature_v3` and `curvature_nm_v3`.
- Include them in progress payload and returned result dict.

### Task 2: Add tests for calibrated legacy curvature and V3

**Files:**
- Modify: `D:\CNTDATA\CNTA_ML_Project\tests\test_feature_extractor_curvature_v2.py`

**Step 1: Add a legacy calibration test**

- Verify larger calibrated `px_per_um` produces larger corrected legacy `curvature_nm` on the same skeleton.

**Step 2: Add a straight-line V3 test**

- Verify V3 stays near zero and labels straight skeletons as `Straight`.

**Step 3: Add a wavy-line V3 sensitivity test**

- Verify V3 is positive and is not less sensitive than V2 on the same wavy skeleton.

### Task 3: Add a small 5-image comparison report

**Files:**
- Create: `D:\CNTDATA\CNTA_ML_Project\tools\generate_curvature_v3_quick_compare.py`

**Step 1: Select 5 images from the existing rough buckets**

- Pick a mixed sample across `50000x` and `100000x`.

**Step 2: Run `FeatureExtractor.extract_all(...)`**

- Collect `density`, `waviness_ratio_v2`, `curvature_nm`, `curvature_nm_v2`, `curvature_nm_v3`.

**Step 3: Write report outputs**

- Save one simple panel per image.
- Save `summary.csv` and `summary.json`.

### Task 4: Verify

**Files:**
- Test: `D:\CNTDATA\CNTA_ML_Project\tests\test_feature_extractor_curvature_v2.py`

**Step 1: Run targeted tests**

Run: `conda run -n LAB_AGENT python -m pytest D:\CNTDATA\CNTA_ML_Project\tests\test_feature_extractor_curvature_v2.py -q`

**Step 2: Run the 5-image report**

Run: `conda run -n LAB_AGENT python D:\CNTDATA\CNTA_ML_Project\tools\generate_curvature_v3_quick_compare.py`

**Step 3: Inspect outputs**

- Confirm report directory exists.
- Confirm all 5 image panels, `summary.csv`, and `summary.json` are generated.
