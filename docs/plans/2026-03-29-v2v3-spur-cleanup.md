# V2 V3 Spur Cleanup Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Insert spur-cleaned skeleton preprocessing before V2/V3 branch-based feature extraction while preserving the current density, diameter, and alignment paths.

**Architecture:** Add reusable skeleton cleanup helpers to `FeatureExtractor`, run them after raw skeleton generation, and route only V2/V3 branch-based metrics through the cleaned skeleton. Keep the public output backward compatible by preserving existing V2/V3 field names and adding cleanup metadata as extra fields.

**Tech Stack:** Python, NumPy, OpenCV, scikit-image, existing `FeatureExtractor`, pytest

---

### Task 1: Review Existing V2 V3 Test Coverage

**Files:**
- Review: `D:\CNTDATA\CNTA_ML_Project\tests\test_feature_extractor_curvature_v2.py`
- Review: `D:\CNTDATA\CNTA_ML_Project\tests\test_feature_extractor_waviness.py`
- Review: `D:\CNTDATA\CNTA_ML_Project\src\analysis\feature_extractor.py`

**Step 1: Inspect current tests**

Check which assertions already cover branch preparation, curvature aggregation, and waviness behavior.

**Step 2: Identify the smallest synthetic case**

Pick or create a skeleton fixture with:

- one long trunk
- one short spur
- enough points to survive current V2/V3 branch filters before cleanup distortion

**Step 3: Define success criteria**

Expected:

- spur pixels removed by cleanup
- fewer false branches after cleanup
- V2/V3 outputs still compute successfully

### Task 2: Add Failing Tests For Skeleton Cleanup

**Files:**
- Modify: `D:\CNTDATA\CNTA_ML_Project\tests\test_feature_extractor_curvature_v2.py`
- Modify: `D:\CNTDATA\CNTA_ML_Project\tests\test_feature_extractor_waviness.py`

**Step 1: Add cleanup helper test**

Write a test for `_clean_branch_skeleton()` using a synthetic spur skeleton.

**Step 2: Add branch-preparation test**

Assert that cleaned branch preparation yields no more branches than the raw branch preparation on the same synthetic case.

**Step 3: Add output metadata test**

Assert that `extract_all()` returns cleanup metadata fields when an external binary mask is used.

**Step 4: Run focused tests to confirm failure**

Run:

```bash
pytest tests/test_feature_extractor_curvature_v2.py tests/test_feature_extractor_waviness.py -q
```

Expected: FAIL because cleanup helpers and metadata do not exist yet.

### Task 3: Implement Reusable Skeleton Cleanup Helpers

**Files:**
- Modify: `D:\CNTDATA\CNTA_ML_Project\src\analysis\feature_extractor.py`

**Step 1: Move demo cleanup logic into `FeatureExtractor`**

Add internal helpers for:

- neighbor iteration
- endpoint-to-junction spur tracing
- spur path length
- short isolated component removal
- terminal spur pruning

**Step 2: Add one orchestration helper**

Implement `_clean_branch_skeleton()` to:

- accept a raw skeleton
- run isolated cleanup first
- run terminal spur cleanup second
- return cleaned skeleton plus cleanup metadata

**Step 3: Keep defaults conservative**

Use the same threshold style as the demo:

- isolated min length based on `expected_tube_px * 2.0`
- isolated min points based on `expected_tube_px * 1.2`
- spur length limit based on `expected_tube_px * 3.0`

### Task 4: Wire Cleaned Skeleton Into V2 V3 Features

**Files:**
- Modify: `D:\CNTDATA\CNTA_ML_Project\src\analysis\feature_extractor.py`

**Step 1: Run cleanup after raw skeleton generation**

In `extract_all()`, build `cleaned_branch_skeleton` immediately after the raw skeleton is computed.

**Step 2: Route V2 V3 branch prep through cleaned skeleton**

Update `_prepare_curvature_branch_sets(...)` call to use the cleaned skeleton.

**Step 3: Route V2/V3 branch-based metrics through cleaned skeleton**

Use the cleaned branch sets for:

- `calculate_curvature_v2`
- `calculate_curvature_v3`
- `calculate_waviness_v2`

**Step 4: Keep other paths unchanged**

Do not change:

- `density`
- `diameter`
- `alignment`
- legacy `calculate_curvature`
- legacy `calculate_waviness`

**Step 5: Add cleanup metadata to result**

Append the cleanup counters and thresholds to the `extract_all()` result dict.

### Task 5: Verify With Focused Tests And One Smoke Run

**Files:**
- Test via script execution only

**Step 1: Run focused pytest checks**

Run:

```bash
pytest tests/test_feature_extractor_curvature_v2.py tests/test_feature_extractor_waviness.py -q
```

Expected: PASS

**Step 2: Run one smoke comparison**

Use a single representative image or synthetic mask to confirm:

- cleanup metadata is populated
- V2/V3 values are still produced
- no exceptions in `extract_all(external_binary_mask=...)`

**Step 3: Inspect result drift**

Record whether:

- branch count drops
- V2/V3 outputs remain finite
- cleanup counters look plausible

### Task 6: Summarize Rollout Boundaries

**Files:**
- No code changes required unless verification reveals issues

**Step 1: Document what changed**

State clearly that only V2/V3 branch-based metrics now use cleaned skeleton topology.

**Step 2: Document what did not change**

State clearly that:

- density
- diameter
- alignment
- legacy curvature
- legacy waviness

still use the previous path.
