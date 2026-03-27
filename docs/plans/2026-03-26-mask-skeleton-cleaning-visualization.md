# Mask Skeleton Cleaning Visualization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a standalone script that generates 6 research panels showing mask, skeleton, and simple “large-and-slender” cleaning behavior.

**Architecture:** Use `FeatureExtractor` for ROI/preprocess/threshold, derive connected-object metrics with NumPy/OpenCV, and render panels with matplotlib in the same visual style as the existing V2 research panels.

**Tech Stack:** Python, NumPy, OpenCV, matplotlib, existing `FeatureExtractor`

---

### Task 1: Create the standalone visualization script

**Files:**
- Create: `D:\CNTDATA\CNTA_ML_Project\tools\generate_mask_skeleton_cleaning_panels.py`

**Step 1: Load 6 images from `coredata/u`**

- Select a balanced subset from `50000x` and `100000x`.

**Step 2: Generate ROI, mask, and skeleton**

- Reuse the current preprocessing and threshold-based mask pipeline.

**Step 3: Compute lightweight connected-object metrics**

- For each connected object, compute:
  - area
  - elongation
  - skeleton length

**Step 4: Render a 5-panel visualization**

- Save one panel per image.

### Task 2: Verify outputs

**Files:**
- Output: a new report directory under `D:\CNTDATA\CNTA_ML_Project\reports`

**Step 1: Run the script in `LAB_AGENT`**

- Generate 6 panels plus summary JSON.

**Step 2: Visually inspect generated outputs**

- Confirm that kept objects look “large and slender.”
