# Curvature V2 and Waviness V2 Comparison Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add corrected `v2` curvature and waviness metrics to the real `FeatureExtractor`, then generate an offline batch comparison between yesterday's frozen `WCNTSegNET` results and the approved paper-repro `cldice` checkpoint without changing production database fields.

**Architecture:** Keep all legacy feature keys unchanged, add a new ordered-centerline `v2` branch extraction path inside `FeatureExtractor`, and build a dedicated comparison tool that reads the March 24 baseline summary while running the new checkpoint on the same images. The report should make historical baseline values and new `v2` outputs visible side by side.

**Tech Stack:** Python, NumPy, OpenCV, scikit-image, matplotlib, existing batch/report tooling, SQLite-backed summary inputs

---

### Task 1: Add focused tests for ordered centerline geometry

**Files:**
- Modify: `D:\CNTDATA\CNTA_ML_Project\tests\test_feature_extractor_waviness.py`
- Create: `D:\CNTDATA\CNTA_ML_Project\tests\test_feature_extractor_curvature_v2.py`

**Step 1: Add synthetic helpers for ordered centerlines**

- Reuse the existing wave-skeleton style from `test_feature_extractor_waviness.py`
- Add branchy skeleton fixtures and straight-line fixtures for curvature-specific checks

**Step 2: Add failing curvature v2 tests**

- Assert `curvature_nm_v2` is near zero on a straight centerline
- Assert `curvature_nm_v2` is positive on a sinusoidal centerline
- Assert calibrated conversion depends on `px_per_um`, not a hard-coded `1 / 15`

**Step 3: Extend waviness tests for v2**

- Assert `waviness_ratio_v2` stays zero-like on a straight line
- Assert `waviness_ratio_v2` is positive on a wave
- Assert branchy skeletons do not crash and return bounded values

**Step 4: Run targeted tests and confirm they fail for the right reason**

Run:

```bash
python -m pytest D:\CNTDATA\CNTA_ML_Project\tests\test_feature_extractor_curvature_v2.py -q
python -m pytest D:\CNTDATA\CNTA_ML_Project\tests\test_feature_extractor_waviness.py -q
```

Expected: at least the new `v2` assertions fail before implementation

### Task 2: Implement ordered branch extraction and curvature/waviness v2

**Files:**
- Modify: `D:\CNTDATA\CNTA_ML_Project\src\analysis\feature_extractor.py`

**Step 1: Add ordered branch helpers**

- Add internal helpers for:
  - neighbor counting
  - endpoint / junction detection
  - spur pruning
  - branch splitting after junction removal
  - ordered skeleton tracing
  - coordinate smoothing

**Step 2: Add `v2` metric calculators**

- Implement:
  - `calculate_curvature_v2(...)`
  - `calculate_waviness_v2(...)`
- Make both consume the ordered branch data instead of raw connected-component coordinates

**Step 3: Use calibrated unit conversion**

- Convert pixel geometry to `nm` using `self.px_per_um / 1000.0`
- Remove any `v2` dependence on the hard-coded `1 / 15` conversion

**Step 4: Extend `extract_all()`**

- Preserve all existing keys
- Add new keys:
  - `curvature_v2`
  - `curvature_nm_v2`
  - `tortuosity_v2`
  - `waviness_ratio_v2`
  - `waviness_height_nm_v2`
  - `waviness_wavelength_nm_v2`
  - `waviness_branches_v2`

**Step 5: Run targeted tests and confirm they pass**

Run:

```bash
python -m pytest D:\CNTDATA\CNTA_ML_Project\tests\test_feature_extractor_curvature_v2.py -q
python -m pytest D:\CNTDATA\CNTA_ML_Project\tests\test_feature_extractor_waviness.py -q
```

Expected: both test files pass

### Task 3: Keep batch compatibility intact

**Files:**
- Modify: `D:\CNTDATA\CNTA_ML_Project\tests\test_batch_processor.py`
- Review: `D:\CNTDATA\CNTA_ML_Project\backend\core\batch_processor.py`
- Review: `D:\CNTDATA\CNTA_ML_Project\backend\main.py`

**Step 1: Check consumers of `extract_all()`**

- Confirm existing batch and API paths still read legacy keys
- Do not change database write targets in this task

**Step 2: Add or adjust compatibility tests if needed**

- Ensure legacy fields remain populated exactly as before for mocked extractors
- Ensure extra `v2` keys do not break batch processing

**Step 3: Run compatibility tests**

Run:

```bash
python -m pytest D:\CNTDATA\CNTA_ML_Project\tests\test_batch_processor.py -q
python -m pytest D:\CNTDATA\CNTA_ML_Project\tests\test_analyze_endpoint.py -q
```

Expected: existing batch/API tests still pass

### Task 4: Build the offline comparison report tool

**Files:**
- Create: `D:\CNTDATA\CNTA_ML_Project\tools\generate_curvature_v2_comparison_report.py`

**Step 1: Read yesterday's frozen baseline**

- Load:
  - `D:\CNTDATA\CNTA_ML_Project\reports\zzy_wcntsegnet_full_batch_20260324\summary.json`
  - `D:\CNTDATA\CNTA_ML_Project\reports\zzy_wcntsegnet_full_batch_20260324\batch_features.csv`

**Step 2: Run the approved new checkpoint**

- Use:
  - config family under `experiments\cnt_paper_repro`
  - checkpoint `cnt_paper_repro_100000x_center768_cldice_seed42\best_model.pth`
- Predict masks for the same images listed in the baseline summary

**Step 3: Compute comparison metrics**

- For each image collect:
  - frozen baseline curvature / waviness values from yesterday
  - new algorithm legacy values if helpful
  - new algorithm `v2` values
  - deltas versus baseline

**Step 4: Render per-image panels**

- Show:
  - original image
  - yesterday baseline mask if available
  - new algorithm mask
  - comparison table with curvature and waviness values

**Step 5: Write batch artifacts**

- Save:
  - `summary.json`
  - `comparison.csv`
  - comparison image panels

**Step 6: Verify syntax**

Run:

```bash
python -m py_compile D:\CNTDATA\CNTA_ML_Project\tools\generate_curvature_v2_comparison_report.py
```

Expected: no errors

### Task 5: Run an end-to-end sample comparison

**Files:**
- No code changes required

**Step 1: Run the tool on a small subset**

Run:

```bash
python D:\CNTDATA\CNTA_ML_Project\tools\generate_curvature_v2_comparison_report.py --limit 3
```

Expected: 3 comparison folders or panels plus summary files are generated

**Step 2: Inspect outputs**

- Confirm each row includes both historical baseline fields and `v2` fields
- Confirm images render without missing-mask crashes
- Confirm curvature and waviness values are present for the 50k/100k images

**Step 3: Run the full comparison batch**

Run:

```bash
python D:\CNTDATA\CNTA_ML_Project\tools\generate_curvature_v2_comparison_report.py
```

Expected: full report generated for the March 24 baseline image set

### Task 6: Final verification

**Files:**
- No code changes required

**Step 1: Run all targeted validation**

Run:

```bash
python -m pytest D:\CNTDATA\CNTA_ML_Project\tests\test_feature_extractor_curvature_v2.py -q
python -m pytest D:\CNTDATA\CNTA_ML_Project\tests\test_feature_extractor_waviness.py -q
python -m pytest D:\CNTDATA\CNTA_ML_Project\tests\test_batch_processor.py -q
python -m pytest D:\CNTDATA\CNTA_ML_Project\tests\test_analyze_endpoint.py -q
python -m py_compile D:\CNTDATA\CNTA_ML_Project\src\analysis\feature_extractor.py
python -m py_compile D:\CNTDATA\CNTA_ML_Project\tools\generate_curvature_v2_comparison_report.py
```

Expected: tests pass and both Python files compile cleanly

**Step 2: Summarize artifacts for review**

- Record the output directory of the comparison report
- Record a few representative image ids with notable curvature/waviness shifts
- Do not write anything to the database in this step
