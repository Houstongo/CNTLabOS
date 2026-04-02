# XR d*k Correlation Refresh Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Refresh the XR correlation analysis and include a new `d*k` bend proxy derived from XR diameter and curvature summary features.

**Architecture:** Reuse the existing `build_xr_modeling_section44_artifacts.py` pipeline so all correlation heatmaps, scatter plots, and summary files stay under the current XR modeling artifact directory. Inject `d*k` during dataframe assembly, then extend the saved artifacts and markdown summary to include it.

**Tech Stack:** Python 3, pandas, matplotlib, sqlite3, scikit-learn

---

### Task 1: Add d*k to the XR analysis dataframe

**Files:**
- Modify: `D:\CNTDATA\CNTA_ML_Project\tools\build_xr_modeling_section44_artifacts.py`

**Step 1: Define the d*k formula**

Use:

```python
df["dk_bend_index"] = (
    pd.to_numeric(df["diameter_mean_nm"], errors="coerce")
    * pd.to_numeric(df["l2_curvature_trimmed_mean_sqrt_length_nm"], errors="coerce")
)
```

**Step 2: Add the feature to the XR output feature list**

Ensure `d*k` participates in the same correlation and baseline tables as the existing XR features.

### Task 2: Extend the XR correlation artifacts

**Files:**
- Modify: `D:\CNTDATA\CNTA_ML_Project\tools\build_xr_modeling_section44_artifacts.py`

**Step 1: Update saved heatmaps**

Make sure the saved subset correlation matrices include `dk_bend_index`.

**Step 2: Update scatter plots**

Include at least one temperature-side and one catalyst-side d*k scatter plot so the new feature is visible outside the heatmap.

**Step 3: Add a text summary**

Write a markdown or JSON summary that explicitly reports the strongest XR process-to-feature correlations involving `d*k`.

### Task 3: Rerun and verify

**Files:**
- Modify: none

**Step 1: Run the updated XR artifact script**

Run:

```bash
python D:\CNTDATA\CNTA_ML_Project\tools\build_xr_modeling_section44_artifacts.py
```

Expected:
- correlation CSVs are regenerated
- heatmap image is regenerated
- summary file mentions `dk_bend_index`

**Step 2: Inspect outputs**

Check:
- `corr_full.csv`
- `corr_800C.csv`
- `corr_1.0g.csv`
- summary markdown/json

**Step 3: Report the refreshed XR findings**

Summarize:
- the exact d*k definition used
- the strongest XR correlations involving d*k
- where the refreshed artifacts were written
