# Waviness Metric Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a true skeleton-centerline waviness metric based on wave height to wavelength ratio, and expose it through feature extraction and batch processing without conflating it with curvature.

**Architecture:** Extend `FeatureExtractor` with a branch-level waviness analyzer that projects each connected skeleton component onto its principal axis, measures detrended lateral oscillation, and aggregates per-wave `H/L` ratios with length weighting. Keep existing `curvature` output intact, fix the uninitialized `tortuosity` path, and let batch processing persist waviness fields only when the database schema supports them.

**Tech Stack:** Python, NumPy, OpenCV, SQLite, unittest

---

### Task 1: Add failing extractor tests for waviness behavior

**Files:**
- Create: `D:\CNTDATA\CNTA_ML_Project\tests\test_feature_extractor_waviness.py`

**Step 1: Write the failing test**

```python
def test_calculate_waviness_detects_wave_ratio():
    extractor = FeatureExtractor(magnification=50000)
    extractor.px_per_um = 100.0
    metrics = extractor.calculate_waviness(skeleton)
    assert metrics["waviness_ratio"] > 0.0
```

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_feature_extractor_waviness -v`
Expected: FAIL because `calculate_waviness` does not exist yet.

**Step 3: Write minimal implementation**

Implement branch-level waviness extraction in `src/analysis/feature_extractor.py`.

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_feature_extractor_waviness -v`
Expected: PASS

### Task 2: Add failing batch persistence test

**Files:**
- Modify: `D:\CNTDATA\CNTA_ML_Project\tests\test_batch_processor.py`
- Modify: `D:\CNTDATA\CNTA_ML_Project\backend\core\batch_processor.py`

**Step 1: Write the failing test**

```python
def test_batch_process_updates_waviness_columns_when_present():
    ...
    assert row["waviness_ratio"] == 0.25
```

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_batch_processor -v`
Expected: FAIL because batch updates do not persist the new fields yet.

**Step 3: Write minimal implementation**

Make batch updates schema-aware and persist `tortuosity` plus waviness columns only when they exist.

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_batch_processor -v`
Expected: PASS

### Task 3: Integrate waviness into extractor outputs

**Files:**
- Modify: `D:\CNTDATA\CNTA_ML_Project\src\analysis\feature_extractor.py`

**Step 1: Write the failing test**

Use the extractor tests from Task 1 to assert the final output dict includes waviness fields.

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_feature_extractor_waviness -v`
Expected: FAIL until `extract_all()` returns waviness metadata.

**Step 3: Write minimal implementation**

Populate `waviness_ratio`, `waviness_height_nm`, `waviness_wavelength_nm`, and `waviness_branches` in `extract_all()`.

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_feature_extractor_waviness -v`
Expected: PASS

### Task 4: Verify the full regression slice

**Files:**
- Test: `D:\CNTDATA\CNTA_ML_Project\tests\test_feature_extractor_waviness.py`
- Test: `D:\CNTDATA\CNTA_ML_Project\tests\test_batch_processor.py`

**Step 1: Run targeted regression**

Run: `python -m unittest tests.test_feature_extractor_waviness tests.test_batch_processor -v`
Expected: PASS

**Step 2: Run adjacent regressions**

Run: `python -m unittest tests.test_xr_api_listing tests.test_visualize_preprocess tests.test_analyze_endpoint -v`
Expected: PASS
