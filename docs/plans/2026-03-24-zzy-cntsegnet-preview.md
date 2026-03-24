# ZZY CNTSegNet Preview Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a database-driven preview and batch report flow for `No28`, `No41`, and `No42` that exports original images, CNTSegNet masks, and five-feature combination figures.

**Architecture:** The implementation will add a dedicated script under `tools/` that queries the SQLite `images` table, selects preview rows for the requested samples, runs CNTSegNet segmentation using the existing batch processor utilities, computes features through `FeatureExtractor` with the external segmentation mask, and writes per-image visual outputs plus a JSON summary. The same script will support both preview mode and full-batch mode.

**Tech Stack:** Python, SQLite, OpenCV, NumPy, Matplotlib, existing CNTSegNet and feature extraction modules

---

### Task 1: Create the DB-driven preview script

**Files:**
- Create: `D:\CNTDATA\CNTA_ML_Project\tools\generate_zzy_cntsegnet_preview.py`
- Modify: `D:\CNTDATA\CNTA_ML_Project\docs\plans\2026-03-24-zzy-cntsegnet-preview-design.md`
- Test: manual CLI run

**Step 1: Write the script skeleton**

- Add CLI args for:
  - `--samples`
  - `--preview`
  - `--output-dir`
  - `--device`

**Step 2: Add database selection helpers**

- Query `images` rows for `No28`, `No41`, and `No42`.
- In preview mode, select one preferred `50000x` row per sample.

**Step 3: Add CNTSegNet + feature extraction helpers**

- Reuse `_get_cntsegnet_segmenter()` from `backend/core/batch_processor.py`.
- Reuse `FeatureExtractor.extract_all(..., external_binary_mask=mask)`.

**Step 4: Add figure generation**

- Save:
  - `original.png`
  - `cntsegnet_mask.png`
  - `feature_combo.png`
- Include the five requested metrics in the combo figure.

**Step 5: Add summary export**

- Write batch-level `summary.json` with selection info, output paths, and extracted features.

### Task 2: Run the 3-image preview

**Files:**
- Test: generated report directory under `D:\CNTDATA\CNTA_ML_Project\reports\`

**Step 1: Execute preview mode**

Run:

```bash
python tools/generate_zzy_cntsegnet_preview.py --samples No28 No41 No42 --preview --output-dir reports/zzy_cntsegnet_preview
```

Expected:

- 3 image folders created
- each folder contains the original image, mask image, and combo figure
- a `summary.json` file exists at the batch root

**Step 2: Inspect the summary**

- Confirm the selected image IDs and file paths
- Confirm density, alignment, diameter, curvature, and waviness fields exist

### Task 3: Scale to the full batch after preview approval

**Files:**
- Test: generated full-batch directory under `D:\CNTDATA\CNTA_ML_Project\reports\`

**Step 1: Execute full mode**

Run:

```bash
python tools/generate_zzy_cntsegnet_preview.py --samples No28 No41 No42 --output-dir reports/zzy_cntsegnet_full_batch
```

Expected:

- all 89 images are processed unless individual rows fail
- failures are captured in `summary.json` without stopping the batch

**Step 2: Spot-check outputs**

- Open representative `feature_combo.png` files from each sample
- Confirm the mask figure matches the original image orientation and crop

### Task 4: Verify and summarize

**Files:**
- Test: output report directories

**Step 1: Verify preview artifacts exist**

- Check `summary.json`
- Check 3 preview folders and expected PNG outputs

**Step 2: Verify full batch only after approval**

- Count generated image folders against selected DB rows

**Step 3: Report back with exact output locations**

- Include the preview output directory
- Include the exact sample/image IDs used for preview
