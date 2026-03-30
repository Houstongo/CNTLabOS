# XR CNTSegNet-SLICE Standard Batch Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Generate resumable visualization panels and structured feature exports for all non-deleted XR images using the CNTSegNet-SLICE standard method.

**Architecture:** Reuse the existing Exp C segmentation checkpoint as `CNTSegNet-SLICE`, query active XR rows directly from the `images` table, and process each image into a per-item output bundle. Each bundle will contain the binary mask, L1-L4 branch overlays, curvature and diameter distributions, and a JSON feature record with orientation plus multiple diameter statistics. A batch-level manifest and summary CSV/JSON will allow resume and later database import decisions.

**Tech Stack:** Python, SQLite, OpenCV, NumPy, Matplotlib, PyTorch, existing `FeatureExtractor`

---

### Task 1: Define XR query and output schema

**Files:**
- Modify: `D:\CNTDATA\CNTA_ML_Project\tools\generate_xr_feature_visual_report.py`
- Create: `D:\CNTDATA\CNTA_ML_Project\tools\generate_xr_slice_standard_batch.py`

**Step 1: Confirm active XR query**

- Use `images` table only.
- Query rows with:
  - `source = 'XR'`
  - `COALESCE(is_deleted, 0) = 0`
- Order by `id DESC`.

**Step 2: Define per-image outputs**

- `items/<image_slug>/mask.png`
- `items/<image_slug>/panel.png`
- `items/<image_slug>/features.json`

**Step 3: Define batch outputs**

- `manifest.csv`
- `summary.csv`
- `summary.json`
- `all_panels/`

### Task 2: Reuse CNTSegNet-SLICE segmentation and feature helpers

**Files:**
- Modify: `D:\CNTDATA\CNTA_ML_Project\tools\generate_expc_slice_v3_threshold_single_panel.py`
- Create: `D:\CNTDATA\CNTA_ML_Project\tools\generate_xr_slice_standard_batch.py`

**Step 1: Reuse Exp C checkpoint as CNTSegNet-SLICE**

- Load Exp C checkpoint once.
- Use mini-batched patch inference.
- Save pure binary black-background/white-mask output.

**Step 2: Reuse V3 L1-L4 threshold loop**

- Thresholds:
  - `L1 = 3.0`
  - `L2 = 5.0`
  - `L3 = 7.0`
  - `L4 = 9.0`

**Step 3: Reuse branch aggregation variants**

- Per threshold, compute:
  - `median`
  - `p75`
  - `mean`
  - `trimmed_mean`
- For each aggregation, compute:
  - `sqrt_length`
  - `length`

### Task 3: Add orientation and multiple diameter stats

**Files:**
- Modify: `D:\CNTDATA\CNTA_ML_Project\tools\generate_xr_slice_standard_batch.py`

**Step 1: Add alignment/orientation fields**

- `alignment`
- `alignment_raw`
- `mean_phi_deg`
- `mean_phi_raw_deg`
- `hof_method`

**Step 2: Add multiple diameter stats**

- Skeleton-based:
  - `diameter_nm`
  - `diameter_p30_nm`
- Distribution-based from branch/mask samples:
  - `diameter_mean_nm`
  - `diameter_std_nm`
  - `diameter_min_nm`
  - `diameter_p25_nm`
  - `diameter_p50_nm`
  - `diameter_p75_nm`
  - `diameter_max_nm`

**Step 3: Add per-threshold feature block**

- `branch_count`
- `curvature_label`
- `waviness_ratio_v2`
- `tortuosity_v2`
- `curvature_point_count`
- `diameter_point_count`
- all eight curvature aggregate outputs

### Task 4: Build a resume-safe XR panel renderer

**Files:**
- Create: `D:\CNTDATA\CNTA_ML_Project\tools\generate_xr_slice_standard_batch.py`

**Step 1: Skip completed items**

- If both `features.json` and `panel.png` already exist, skip item.

**Step 2: Render per-image panel**

- Recommended layout:
  - original ROI
  - CNTSegNet-SLICE mask
  - L1 overlay + curvature hist + diameter hist + metrics
  - L2 overlay + curvature hist + diameter hist + metrics
  - L3 overlay + curvature hist + diameter hist + metrics
  - L4 overlay + curvature hist + diameter hist + metrics

**Step 3: Flatten all panels**

- Copy or export one flat `all_panels/` directory for quick browsing.

### Task 5: Verify on a small XR subset

**Files:**
- Test via script execution only

**Step 1: Run with `--limit 2`**

- Confirm mask rendering
- Confirm per-threshold metrics
- Confirm summary rows

**Step 2: Adjust labels/formatting if needed**

- Ensure panel text stays readable.
- Ensure curvature histogram uses `um^-1`.

### Task 6: Run full XR batch

**Files:**
- Run: `D:\CNTDATA\CNTA_ML_Project\tools\generate_xr_slice_standard_batch.py`

**Step 1: Start full run**

- Process all active XR images.

**Step 2: Record output directory**

- Report:
  - total image count
  - output root
  - `summary.csv`
  - `summary.json`
  - `all_panels/`

**Step 3: Only after verification, discuss DB write-back**

- Keep this phase export-only.
- Do not write XR results back into the database yet.
