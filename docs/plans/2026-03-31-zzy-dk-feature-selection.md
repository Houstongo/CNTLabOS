# ZZY d*k Feature Selection Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add the `d*k` bend proxy to the ZZY >10000X feature engineering pipeline and emit a formal pre-model feature selection recommendation.

**Architecture:** Extend the existing `tools/analyze_zzy_gt10000_feature_engineering.py` script so engineered features are created in one place during row assembly, then reuse those values across quality summaries, grouped means, correlation tables, and a new feature-selection artifact. Keep the output format report-centric so future reruns on the same database/summary inputs stay reproducible.

**Tech Stack:** Python 3, sqlite3, csv/json, project-local report generation

---

### Task 1: Add the d*k engineered feature

**Files:**
- Modify: `D:\CNTDATA\CNTA_ML_Project\tools\analyze_zzy_gt10000_feature_engineering.py`

**Step 1: Identify the row-building stage**

Find the point where derived ZZY features such as `branches_per_100um` and `junction_to_branch_ratio` are injected into `features`.

**Step 2: Add the minimal implementation**

Compute a new bend proxy using:

```python
features["dk_bend_index"] = safe_product(
    features.get("diameter"),
    features.get("curvature_nm_v3_trimmed_mean_sqrt_length"),
)
```

Also add a half-scale variant if needed for downstream reporting:

```python
features["surface_strain_proxy"] = safe_product(
    features.get("diameter"),
    features.get("curvature_nm_v3_trimmed_mean_sqrt_length"),
    scale=0.5,
)
```

**Step 3: Verify usage wiring**

Ensure the new fields can flow through `flatten_row()` into `engineered_dataset_active.csv`.

### Task 2: Update report tables and correlations

**Files:**
- Modify: `D:\CNTDATA\CNTA_ML_Project\tools\analyze_zzy_gt10000_feature_engineering.py`

**Step 1: Extend summary feature lists**

Add `dk_bend_index` to the relevant lists used by:
- feature coverage
- grouped means
- meta correlations
- top correlation pairs

**Step 2: Extend markdown rendering**

Add explicit text that documents the d*k definition and surface-strain interpretation.

**Step 3: Add a dedicated correlation summary**

Render the main d*k correlations against:
- `alignment`
- `tortuosity_v2`
- `waviness_ratio_v2`
- `junction_ratio`
- `magnification_bucket`
- `gas_level`

### Task 3: Emit formal pre-model feature-selection artifacts

**Files:**
- Modify: `D:\CNTDATA\CNTA_ML_Project\tools\analyze_zzy_gt10000_feature_engineering.py`

**Step 1: Add a recommendation builder**

Create a helper that emits rows with:
- `feature`
- `role`
- `recommendation`
- `reason`
- `preprocess`

Use the current project logic:
- keep independent morphology features
- keep confounders as control variables
- drop redundant, diagnostic, or leakage-prone columns

**Step 2: Write recommendation outputs**

Emit:
- `feature_selection_recommendations.csv`
- `model_dataset_core.csv`

**Step 3: Document the final keep/drop logic**

Render the core recommendation into `report.md`.

### Task 4: Rerun and verify

**Files:**
- Modify: none
- Test: generated reports under `D:\CNTDATA\CNTA_ML_Project\reports`

**Step 1: Run the analysis script**

Run:

```bash
python D:\CNTDATA\CNTA_ML_Project\tools\analyze_zzy_gt10000_feature_engineering.py
```

Expected:
- new timestamped report directory is created
- CSV/Markdown outputs include `dk_bend_index`

**Step 2: Inspect generated outputs**

Check that:
- engineered dataset contains `dk_bend_index`
- recommendation CSV exists
- report text includes d*k interpretation and feature-selection guidance

**Step 3: Share the output paths and the recommended model columns**

Summarize:
- recommended retained columns
- control variables
- suggested dropped columns
- whether d*k should replace or complement curvature
