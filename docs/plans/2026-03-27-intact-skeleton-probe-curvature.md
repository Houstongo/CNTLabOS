# Intact Skeleton Probe Curvature Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Export smoothed centerlines and Fiji-like curvature measurements from the intact skeleton probe without changing path selection behavior.

**Architecture:** Keep the current probe as the selection engine, then add a post-selection geometry stage that resamples paths, generates multiple smoothing profiles, computes curvature series/statistics, and writes binary-background visualizations plus CSV/JSON artifacts.

**Tech Stack:** Python, NumPy, OpenCV, Matplotlib, existing `FeatureExtractor` helpers

---

### Task 1: Add profile smoothing and curvature helpers

**Files:**
- Modify: `D:\CNTDATA\CNTA_ML_Project\tools\generate_intact_skeleton_probe.py`

**Steps**
1. Add shared arc-length resampling helpers.
2. Add three smoothing profiles plus raw export.
3. Add Fiji-like local-tangent curvature computation and profile summary helpers.

### Task 2: Add visualization/export outputs

**Files:**
- Modify: `D:\CNTDATA\CNTA_ML_Project\tools\generate_intact_skeleton_probe.py`

**Steps**
1. Switch path overlay backgrounds to binary segmentation canvases.
2. Write per-path curvature CSV and comparison plots.
3. Write per-run summary CSV/JSON with profile curvature statistics.

### Task 3: Validate on the requested 4-image batch

**Files:**
- Modify: `D:\CNTDATA\CNTA_ML_Project\tools\generate_intact_skeleton_probe.py`
- Output: `D:\CNTDATA\CNTA_ML_Project\reports\...`

**Steps**
1. Run a syntax check.
2. Run one image as a smoke test.
3. Re-run the 4-image batch and inspect generated artifacts.
