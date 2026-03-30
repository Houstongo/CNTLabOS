# No28 Morphology Step 6 Fix Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rebuild the No28 morphology cropped outputs so step 6 shows the cleaned-minus-junction skeleton over prediction and every cropped panel is `768x768`.

**Architecture:** Add one small repair script that operates on the existing `v4` report artifacts instead of trying to resurrect the original one-off generator. The script will recover the cleaned skeleton from step 5, subtract junction pixels, redraw step 6, regenerate the cropped panel set at a fixed size, and update the crop summary metadata.

**Tech Stack:** Python, NumPy, OpenCV, JSON, existing report artifacts

---

### Task 1: Add a reproducible repair script

**Files:**
- Create: `D:\CNTDATA\CNTA_ML_Project\tools\repair_no28_morphology_step6.py`

**Step 1: Load the source report assets**

- Read from `D:\CNTDATA\CNTA_ML_Project\reports\no28_morphology_feature_steps_v4`
- Use:
  - `02_model_prediction_mask.png`
  - `05_cleaned_skeleton_on_prediction.png`
  - existing `summary.json`

**Step 2: Recover the cleaned skeleton from step 5**

- Detect the colored skeleton overlay pixels from the step 5 image.
- Convert the overlay back into a binary cleaned skeleton mask.
- Keep the logic deterministic and local to this script.

**Step 3: Remove junction pixels for step 6**

- Compute 8-neighbor counts on the recovered cleaned skeleton.
- Mark skeleton pixels with degree `>= 3` as junction pixels.
- Produce a cleaned-minus-junction skeleton mask.

**Step 4: Re-render the cropped step 6 image**

- Draw the prediction mask as the base.
- Overlay the cleaned-minus-junction skeleton so it is clearly visible.
- Save the regenerated `06_prediction_plus_minus_junctions.png` into the crop directory.

### Task 2: Rebuild the cropped panel set at 768x768

**Files:**
- Modify via script output: `D:\CNTDATA\CNTA_ML_Project\reports\no28_morphology_feature_steps_v4_crop\*.png`
- Modify via script output: `D:\CNTDATA\CNTA_ML_Project\reports\no28_morphology_feature_steps_v4_crop\summary.json`

**Step 1: Choose a deterministic crop box**

- Build a centered `768x768` crop box from the existing `780x780` crop metadata.
- Keep the crop box inside the image bounds.

**Step 2: Re-crop all six step images**

- Recreate:
  - `01_original_sem.png`
  - `02_model_prediction_mask.png`
  - `03_raw_skeleton_on_prediction.png`
  - `04_removed_vs_kept_on_black.png`
  - `05_cleaned_skeleton_on_prediction.png`
  - regenerated `06_prediction_plus_minus_junctions.png`

**Step 3: Rewrite crop metadata**

- Update `crop_box_xyxy`
- Update cropped file paths
- Record `size: [768, 768]`

### Task 3: Verify the regenerated outputs

**Files:**
- Verify: `D:\CNTDATA\CNTA_ML_Project\reports\no28_morphology_feature_steps_v4_crop\06_prediction_plus_minus_junctions.png`
- Verify: `D:\CNTDATA\CNTA_ML_Project\reports\no28_morphology_feature_steps_v4_crop\summary.json`

**Step 1: Run the repair script**

Run:

```powershell
python D:\CNTDATA\CNTA_ML_Project\tools\repair_no28_morphology_step6.py
```

Expected:
- script completes without errors
- crop directory files are rewritten

**Step 2: Verify image dimensions**

- Confirm all cropped panel PNGs are `768x768`

**Step 3: Verify visual intent**

- Inspect step 6 and confirm the cleaned-minus-junction skeleton is visibly present over prediction

**Step 4: Verify metadata**

- Confirm `summary.json` crop box and size match the regenerated crop outputs
