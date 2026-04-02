# ZZY Visual Report Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Generate a static visual analysis report for the cleaned ZZY >10000X dataset, including correlation matrices, d*k plots, and grouped distribution charts.

**Architecture:** Read the latest engineered ZZY dataset CSV, derive a compact plotting dataframe, then render a fixed bundle of matplotlib figures into a timestamped report directory with a markdown index. Keep the pipeline file-based and reproducible so the report can be regenerated after feature updates.

**Tech Stack:** Python 3, pandas, matplotlib, project-local CSV/Markdown outputs

---

### Task 1: Build the plotting script

**Files:**
- Create: `D:\CNTDATA\CNTA_ML_Project\tools\generate_zzy_gt10000_visual_report.py`

**Step 1: Load the current engineered dataset**

Read:

```text
D:\CNTDATA\CNTA_ML_Project\reports\zzy_feature_engineering_gt10000_20260331_205709\engineered_dataset_active.csv
```

Fallback to the most recent `zzy_feature_engineering_gt10000_*` directory if needed.

**Step 2: Apply a consistent plotting theme**

Use one shared theme helper so all plots share the same palette, fonts, and axis styling.

**Step 3: Add fixed plot functions**

Render:
- full core-feature correlation matrix
- `50k` correlation matrix
- `100k` correlation matrix
- gas-level distribution panel
- d*k relationship panel
- feature distribution histogram panel

### Task 2: Add markdown report assembly

**Files:**
- Create: `D:\CNTDATA\CNTA_ML_Project\tools\generate_zzy_gt10000_visual_report.py`

**Step 1: Save figures into a timestamped report directory**

Write PNG files under:

```text
D:\CNTDATA\CNTA_ML_Project\reports\zzy_visual_report_<timestamp>
```

**Step 2: Emit a report index**

Write `report.md` with:
- dataset scope
- key observations
- figure list with image embeds and captions

### Task 3: Run and verify

**Files:**
- Modify: none

**Step 1: Run the script**

Run:

```bash
python D:\CNTDATA\CNTA_ML_Project\tools\generate_zzy_gt10000_visual_report.py
```

Expected:
- timestamped report directory exists
- all planned PNG figures are created

**Step 2: Inspect outputs**

Verify:
- correlation matrices are non-empty
- `report.md` references the generated files
- figure count matches the planned bundle

**Step 3: Share output paths**

Summarize where to find:
- visual report markdown
- correlation matrix figures
- d*k dedicated figures
