# Coredata Five-Profile Batch Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a resumable batch script that processes the original `coredata` SEM images and outputs five non-visual feature profiles per image.

**Architecture:** Enumerate the 82 source images once, compute shared mask/skeleton preparation once per image, derive the five profile outputs from shared intermediates, persist one JSON per image, and merge summaries only after per-image outputs exist.

**Tech Stack:** Python, NumPy, OpenCV, existing `FeatureExtractor`, JSON/CSV

---

### Task 1: Create the manifest and output layout

**Files:**
- Create: `D:\CNTDATA\CNTA_ML_Project\tools\run_coredata_five_profile_batch.py`

**Step 1: Enumerate source images**

- Exclude generated folders such as:
  - `rough_curvature_buckets_visual`
  - `_review_sheets`
  - previous report outputs

**Step 2: Write manifest**

- Save a frozen `manifest.csv` in the batch output directory.

### Task 2: Add shared per-image preparation

**Files:**
- Modify/Create in the batch script above

**Step 1: Read image and compute shared intermediates once**

- ROI
- preprocess
- threshold mask
- skeleton
- base components

**Step 2: Reuse shared intermediates across profiles**

- legacy
- v2_accurate
- v2_fast
- v3_accurate
- v3_fast

### Task 3: Persist resumable per-image outputs

**Files:**
- Output under `items/<image_slug>/features.json`

**Step 1: Write one complete JSON per image**

- Include all five profiles and shared metadata.

**Step 2: Resume logic**

- Skip images whose `features.json` already exists and is valid.

### Task 4: Generate merged summaries

**Files:**
- Output: `summary.csv`
- Output: `summary.json`

**Step 1: Merge all per-image feature JSON files**

- Only after image processing is complete.

### Task 5: Verify

**Files:**
- Output directory for the batch run

**Step 1: Run the script in `LAB_AGENT`**

- Confirm it starts and writes the manifest.

**Step 2: Confirm resumability**

- Re-run after partial completion and verify existing items are skipped.

**Step 3: Confirm summary outputs**

- Ensure `summary.csv` and `summary.json` are present and include all processed images.
