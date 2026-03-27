# V2 Feature Visualization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a standalone tool that renders annotated V2 feature-calculation panels for 3 representative `100000x` images using the latest paper-repro mask and the real `FeatureExtractor` pipeline.

**Architecture:** Reuse the same paper-repro inference path already used by the comparison report, then reconstruct the real `FeatureExtractor` intermediate states in a visualization-specific script. The script should save one large panel per image plus a JSON summary for manual review.

**Tech Stack:** Python, NumPy, OpenCV, matplotlib, torch, existing `FeatureExtractor`, paper-repro checkpoint tooling

---

### Task 1: Define the image selection and panel contents

**Files:**
- Modify: `D:\CNTDATA\CNTA_ML_Project\docs\plans\2026-03-25-v2-feature-visualization-design.md`
- Create: `D:\CNTDATA\CNTA_ML_Project\docs\plans\2026-03-25-v2-feature-visualization.md`

**Step 1: Confirm the first batch image source**

- Use the existing comparison CSV and select 3 representative `100000x` images
- Prefer one image each from `No28`, `No41`, and `No42` when available

**Step 2: Freeze the panel layout**

- Include mask, skeleton, branch split, ordered centerline, diameter, curvature, waviness, and summary sections

### Task 2: Implement the standalone V2 visualization tool

**Files:**
- Create: `D:\CNTDATA\CNTA_ML_Project\tools\generate_v2_feature_visualization_panels.py`

**Step 1: Reuse the paper-repro mask inference path**

- Load the same config/checkpoint as the comparison report
- Predict the ROI mask on each selected image

**Step 2: Reconstruct real extractor intermediates**

- Compute ROI, calibration, preprocessing, mask, skeleton, base components, ordered V2 branches, and raw geometry values
- Preserve both:
  - raw pre-guard values
  - final `extract_all()` values

**Step 3: Add visualization helpers**

- Draw junction removal / branch split
- Draw ordered + smoothed centerlines
- Draw distance transform + diameter circles
- Draw curvature sampling points on smoothed branches
- Draw waviness axis and representative oscillation markers

**Step 4: Render one large annotated panel per image**

- Use a readable multi-row layout
- Add short captions below each step
- Add a summary panel with both raw and final values

### Task 3: Run and verify the first 3 panels

**Files:**
- No additional code changes required

**Step 1: Run the tool on 3 `100000x` images**

Run:

```bash
python D:\CNTDATA\CNTA_ML_Project\tools\generate_v2_feature_visualization_panels.py --limit 3 --magnification 100000
```

Expected:

- 3 image panels written successfully
- a summary manifest is generated

**Step 2: Verify contents**

- Confirm each panel contains the expected steps
- Confirm the summary box explains whether the current magnification is above or below the reporting guard
- Confirm diameter / curvature / waviness raw values are visible together with the final reported values from `extract_all()`

### Task 4: Final validation

**Files:**
- No additional code changes required

**Step 1: Run syntax validation**

Run:

```bash
python -m py_compile D:\CNTDATA\CNTA_ML_Project\tools\generate_v2_feature_visualization_panels.py
```

Expected: no errors

**Step 2: Record output path**

- Save the output directory path for manual review
