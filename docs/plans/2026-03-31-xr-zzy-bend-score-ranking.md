# XR ZZY Bend Score Ranking Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a unified bend-score ranking table and visual report across XR and ZZY datasets.

**Architecture:** Load the latest XR batch summary and the latest ZZY engineered dataset, map both datasets onto a shared set of bend-related features, normalize each feature within its own dataset, and aggregate those normalized signals into a comparable bend score. Write the combined ranking, per-dataset rankings, and a static visual report bundle into a new report directory.

**Tech Stack:** Python 3, pandas, numpy, matplotlib

---

### Task 1: Build unified feature mapping

**Files:**
- Create: `D:\CNTDATA\CNTA_ML_Project\tools\generate_xr_zzy_bend_score_report.py`

**Step 1: Load the latest XR and ZZY artifacts**

Read:
- latest XR `summary.csv`
- latest ZZY `engineered_dataset_active.csv`

**Step 2: Map both datasets onto shared bend features**

Create unified columns:
- `dk_bend_index`
- `curvature_proxy`
- `waviness_proxy`
- `tortuosity_proxy`
- `junction_ratio`
- `alignment`

### Task 2: Define the bend score

**Files:**
- Create: `D:\CNTDATA\CNTA_ML_Project\tools\generate_xr_zzy_bend_score_report.py`

**Step 1: Normalize within each dataset**

Use within-dataset percentile or rank normalization so XR and ZZY remain comparable despite scale differences.

**Step 2: Aggregate the score**

Use a mean score of:
- high `d*k`
- high curvature
- high waviness
- high tortuosity
- high junction ratio
- low alignment

**Step 3: Preserve components**

Write the component sub-scores alongside the final bend score.

### Task 3: Emit ranking outputs and visual report

**Files:**
- Create: `D:\CNTDATA\CNTA_ML_Project\tools\generate_xr_zzy_bend_score_report.py`

**Step 1: Write ranking tables**

Emit:
- combined ranking CSV
- XR ranking CSV
- ZZY ranking CSV
- top-N summary CSV

**Step 2: Write figures**

Emit:
- overall bend-score distribution by dataset
- bend score vs d*k scatter
- top-ranked samples bar chart
- component contribution heatmap

**Step 3: Write markdown report**

Document:
- exact score definition
- top XR samples
- top ZZY samples
- top combined samples
- cautions about relative vs absolute comparability

### Task 4: Run and verify

**Files:**
- Modify: none

**Step 1: Run the report generator**

Run:

```bash
python D:\CNTDATA\CNTA_ML_Project\tools\generate_xr_zzy_bend_score_report.py
```

Expected:
- timestamped report directory exists
- ranking CSVs and PNGs exist

**Step 2: Inspect outputs**

Check:
- top-ranked rows are present for both datasets
- report explains the scoring method
- figures are non-empty

**Step 3: Share the ranking and visual report paths**

Summarize:
- top XR bend samples
- top ZZY bend samples
- how the unified score should be interpreted
