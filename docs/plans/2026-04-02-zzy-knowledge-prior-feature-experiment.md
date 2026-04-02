# ZZY Knowledge Prior Feature Experiment Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a knowledge-prior feature experiment for ZZY 50000x modeling and compare it against the current process-only baseline on curvature, waviness_ratio, tortuosity, and alignment.

**Architecture:** Reuse the current filtered modeling table as the base dataset, derive lightweight knowledge priors from the existing database and knowledge-RAG modules, and append those priors as additional model inputs. Run the experiment in a separate report directory so the current main model remains unchanged.

**Tech Stack:** Python, pandas, scikit-learn, SQLite, existing `knowledge_rag.py` / `knowledge_driven_predictor.py`

---

### Task 1: Define the prior feature set

**Files:**
- Create: `D:\CNTDATA\CNTA_ML_Project\tools\experiment_zzy_knowledge_prior_features.py`
- Reference: `D:\CNTDATA\CNTA_ML_Project\backend\core\knowledge_rag.py`
- Reference: `D:\CNTDATA\CNTA_ML_Project\backend\core\knowledge_driven_predictor.py`

**Step 1: Choose a minimal prior set**

Use lightweight, deterministic priors:
- similar experiment count
- similar experiment target mean / std
- nearest-similarity weighted baseline
- knowledge link count
- relation-chain count

**Step 2: Keep the priors target-specific**

Each target gets its own prior summary so we can compare whether knowledge helps curvature, waviness_ratio, tortuosity, or alignment differently.

### Task 2: Run the feature-augmented experiment

**Files:**
- Create: `D:\CNTDATA\CNTA_ML_Project\tools\experiment_zzy_knowledge_prior_features.py`
- Output: `D:\CNTDATA\CNTA_ML_Project\reports\zzy_50000_fe_time_model_20260402\knowledge_prior_feature_experiment`

**Step 1: Load the current filtered modeling table**

Use:
- `D:\CNTDATA\CNTA_ML_Project\reports\zzy_50000_fe_time_model_20260402\zzy_50000_fe_time_modeling_table.csv`

**Step 2: Add current best process features**

Reuse:
- `fe_power`
- `fe_thickness`
- `anneal_time`
- `fe_deposition_index`
- `power_bin`
- interaction features already proven useful

**Step 3: Append knowledge priors**

Append the lightweight prior columns per target and evaluate with grouped CV.

### Task 3: Compare against the current baseline

**Files:**
- Output: `D:\CNTDATA\CNTA_ML_Project\reports\zzy_50000_fe_time_model_20260402\knowledge_prior_feature_experiment\vs_current.csv`

**Step 1: Save full results**

Write:
- `results.csv`
- `best_results.csv`
- `summary.json`

**Step 2: Save direct comparison**

Write a comparison table vs:
- `D:\CNTDATA\CNTA_ML_Project\reports\zzy_50000_fe_time_model_20260402\best_results_by_target.csv`

### Task 4: Verify and summarize

**Files:**
- Output: `D:\CNTDATA\CNTA_ML_Project\reports\zzy_50000_fe_time_model_20260402\knowledge_prior_feature_experiment\summary.json`

**Step 1: Run the script**

Run:
`python D:\CNTDATA\CNTA_ML_Project\tools\experiment_zzy_knowledge_prior_features.py`

**Step 2: Inspect best results**

Confirm which of:
- curvature
- waviness_ratio
- tortuosity
- alignment

receive measurable improvement from knowledge priors.
