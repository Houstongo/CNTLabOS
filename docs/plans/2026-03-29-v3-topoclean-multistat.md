# V3 Topology-Clean Multi-Stat Curvature Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make `extract_all()` use a V3-only topology-clean branch path and output cached multi-stat curvature metrics for both `length` and `sqrt(length)` weighting.

**Architecture:** Reuse the existing topology-clean branch collector, add branch-curvature stat caching inside `FeatureExtractor`, and aggregate all requested V3 metrics from a single branch-stat bundle. Keep the old V2 helpers callable, but remove them from the default `extract_all()` hot path.

**Tech Stack:** Python, NumPy, OpenCV, existing `FeatureExtractor` branch analysis code, `unittest`

---

### Task 1: Add V3 multi-stat branch helpers

**Files:**
- Modify: `D:\CNTDATA\CNTA_ML_Project\src\analysis\feature_extractor.py`
- Test: `D:\CNTDATA\CNTA_ML_Project\tests\test_feature_extractor_curvature_v2.py`

**Step 1: Write/update tests for cached V3 stats**

Add tests that assert:
- `calculate_curvature_v3_bundle()` returns all requested keys
- `P75 + sqrt(length)` remains the primary V3 value
- `trimmed mean` is finite and non-negative

**Step 2: Implement branch stat caching**

Add helpers to:
- cache per-branch curvature distributions/statistics
- aggregate one named stat with one weight mode
- build the full V3 bundle from cached branch stats

**Step 3: Run focused tests**

Run: `python -m unittest tests.test_feature_extractor_curvature_v2 -v`

**Step 4: Commit**

Use a single commit after curvature helpers and tests are green.

### Task 2: Move extract_all() to V3-only hot path

**Files:**
- Modify: `D:\CNTDATA\CNTA_ML_Project\src\analysis\feature_extractor.py`
- Test: `D:\CNTDATA\CNTA_ML_Project\tests\test_feature_extractor_waviness.py`

**Step 1: Update extract_all()**

Change `extract_all()` to:
- prepare only V3 topology-clean branches
- compute the V3 bundle once
- stop calling `calculate_curvature_v2()` and `calculate_waviness_v2()`
- emit compatibility aliases for `curvature_v2` / `curvature_nm_v2`
- emit `None` for `waviness_ratio_v2` and `tortuosity_v2`

**Step 2: Update extract_all() tests**

Adjust the extract-all smoke test to assert the new V3 multi-stat fields are present and numeric.

**Step 3: Run focused tests**

Run: `python -m unittest tests.test_feature_extractor_curvature_v2 tests.test_feature_extractor_waviness -v`

**Step 4: Commit**

Use a second commit after `extract_all()` and tests are green.

### Task 3: Verify the new path on a real sample

**Files:**
- Modify: none required
- Verify: `D:\CNTDATA\CNTA_ML_Project\src\analysis\feature_extractor.py`

**Step 1: Run a real-sample smoke script**

Run a one-off script against the `No28 ... 50000-1` sample to print:
- `curvature_nm_v3`
- all new V3 multi-stat outputs
- branch cleanup counts

**Step 2: Confirm no regressions in runtime shape**

Ensure the staged path still finishes in seconds, not minutes.

**Step 3: Report residual compatibility risks**

Call out that scripts reading `waviness_ratio_v2` / `tortuosity_v2` from `extract_all()` now see `None`.
