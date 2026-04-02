# ZZY 50000X Bend and Process Correlation Report Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Generate a dedicated correlation report for the `ZZY` `50000X` subset, focused on bend-related morphology features and growth-process variables.

**Architecture:** Reuse the cleaned ZZY engineered dataset and join process metadata from the database, then filter to `magnification=50000`. Produce a report bundle with correlation CSVs, heatmaps, process-feature panels, and a markdown summary that explicitly separates variable process factors from constant factors.

**Tech Stack:** Python 3, pandas, sqlite3, matplotlib

---

### Task 1: Build the 50000X analysis script

**Files:**
- Create: `D:\CNTDATA\CNTA_ML_Project\tools\generate_zzy_50000_bend_process_report.py`

**Step 1: Load the latest engineered ZZY dataset**

Read the latest `engineered_dataset_active.csv` and filter to `magnification == 50000`.

**Step 2: Join process variables from the database**

Join:
- `fe_thickness`
- `fe_power`
- `al2o3_thickness`
- `al2o3_power`
- `c2h4_flow`
- `ar_flow`
- `h2_flow`
- constant process fields for reporting only

**Step 3: Define bend-related feature set**

Use:
- `dk_bend_index`
- `curvature_nm_v3_trimmed_mean_sqrt_length`
- `curvature_nm_v3`
- `tortuosity_v2`
- `waviness_ratio_v2`
- `alignment`
- `junction_ratio`
- `junctions_per_100um`
- `density`
- `diameter`

### Task 2: Generate correlation artifacts

**Files:**
- Create: `D:\CNTDATA\CNTA_ML_Project\tools\generate_zzy_50000_bend_process_report.py`

**Step 1: Save matrices**

Emit:
- bend-feature correlation matrix
- process-to-bend correlation matrix
- matching CSV tables

**Step 2: Save process relationship panels**

Emit:
- `Fe thickness` panel
- `Fe power` panel
- `gas level` panel
- catalyst recipe map

### Task 3: Write markdown summary

**Files:**
- Create: `D:\CNTDATA\CNTA_ML_Project\tools\generate_zzy_50000_bend_process_report.py`

**Step 1: Document sample scope**

State row count and the process variables that are constant in this subset.

**Step 2: Document strongest correlations**

List the strongest bend-feature pair correlations and strongest process-to-bend correlations.

### Task 4: Run and verify

**Files:**
- Modify: none

**Step 1: Run the script**

Run:

```bash
python D:\CNTDATA\CNTA_ML_Project\tools\generate_zzy_50000_bend_process_report.py
```

Expected:
- timestamped report directory exists
- PNG and CSV outputs exist

**Step 2: Inspect outputs**

Check:
- matrices include only 50000X rows
- markdown report states the 50000X scope
- strongest-correlation section is populated
