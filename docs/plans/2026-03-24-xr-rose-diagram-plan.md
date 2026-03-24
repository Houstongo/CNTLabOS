# XR Rose Diagram Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Generate PPT-ready rose diagrams for representative XR SEM images to visualize CNT orientation distribution.

**Architecture:** Reuse the current orientation-analysis logic based on structure tensor and orientation histogram. Select one low-, one medium-, and one high-alignment XR image from the database, then generate three single-image reports and one comparison panel.

**Tech Stack:** Python, sqlite3, OpenCV, NumPy, pandas, matplotlib, Pillow

---

### Task 1: Select representative XR images

**Files:**
- Read: `D:\CNTDATA\CNTA_ML_Project\database\cnta_experiments.sqlite`
- Create: `D:\CNTDATA\CNTA_ML_Project\scripts\generate_xr_rose_diagrams.py`

**Step 1: Query XR images sorted by alignment**

Run a query that fetches `file_path`, `sample_id`, `actual_temp`, `flow_rate`, `catalyst_concentration`, and `alignment` for XR records.

**Step 2: Pick one low, one medium, and one high representative**

Use sorted rows to select representative images near the minimum, median, and maximum alignment.

### Task 2: Implement rose diagram generation

**Files:**
- Create: `D:\CNTDATA\CNTA_ML_Project\scripts\generate_xr_rose_diagrams.py`

**Step 1: Recreate the current orientation-analysis pipeline**

Use CLAHE, light Gaussian smoothing, Scharr gradients, Gaussian-smoothed structure tensor, and orientation extraction.

**Step 2: Convert the orientation field into an axial histogram**

Fold angles into the `0-180 deg` domain and mirror the histogram to create a polar rose diagram.

**Step 3: Build PPT-ready figures**

Generate single-image figures with original SEM image + rose diagram, then a combined comparison panel.

### Task 3: Verify outputs

**Files:**
- Write: `D:\CNTDATA\CNTA_ML_Project\reports\xr_rose_diagrams_20260324\*`

**Step 1: Run the script**

Run: `python D:\CNTDATA\CNTA_ML_Project\scripts\generate_xr_rose_diagrams.py`

**Step 2: Confirm outputs exist**

Check that all PNG files and the README are present in the report directory.

**Step 3: Visually inspect the results**

Open the generated comparison panel and at least one single-image output to confirm the rose diagrams differ meaningfully between low, medium, and high alignment.
