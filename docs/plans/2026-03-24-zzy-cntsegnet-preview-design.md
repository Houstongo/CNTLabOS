# ZZY CNTSegNet Preview Design

**Goal:** For the newly imported `No28`, `No41`, and `No42` ZZY SEM database records, generate a preview-first workflow that outputs per-image CNTSegNet segmentation visuals and a five-feature summary figure before scaling to the full batch.

**Context**

- The database of record is `D:\CNTDATA\CNTA_ML_Project\database\cnta_experiments.sqlite`.
- The relevant rows live in the `images` table and are identified by `sample_id LIKE 'No28%'`, `No41%`, and `No42%`.
- Existing code already supports:
  - CNTSegNet inference through `backend/core/batch_processor.py`
  - feature extraction through `src/analysis/feature_extractor.py`
  - step-by-step CNTSegNet visualization through `backend/core/cntsegnet_visualizer.py`

**Approved Scope**

- First run a preview on 3 images:
  - one `50000x` image from `No28`
  - one `50000x` image from `No41`
  - one `50000x` image from `No42`
- Each preview image should output:
  - original SEM image
  - CNTSegNet mask / overlay result
  - a combined feature visualization containing:
    - alignment
    - density
    - diameter
    - mean curvature
    - waviness
- After preview approval, reuse the same script to process the full batch:
  - `No28`: 17 images
  - `No41`: 60 images
  - `No42`: 12 images

**Design**

- Build a dedicated script instead of extending `test_today_batch.py`.
- Query the database directly so the selection is based on database records rather than filesystem timestamps.
- Reuse `FeatureExtractor.extract_all(..., external_binary_mask=mask)` so the reported density/alignment/diameter/curvature/waviness are all computed from the CNTSegNet segmentation result.
- Reuse the CNTSegNet tiling/inference path from `backend/core/batch_processor.py` to avoid divergence from the existing batch processing backend.
- Generate one per-image summary figure with a compact layout:
  - panel 1: original image
  - panel 2: CNTSegNet mask overlay
  - panel 3: metric card / bar panel with the five requested metrics
- Save both raw assets and the combined figure to a dedicated report directory under `reports/`.

**Error Handling**

- Skip missing files but record them in a JSON summary.
- Skip unreadable images but keep the rest of the batch running.
- If a selected sample has no `50000x` record, fall back to the latest available magnification and record the fallback in the summary.

**Verification**

- Run the new script in preview mode for the 3 selected images.
- Verify that the output directory contains:
  - `original.png`
  - `cntsegnet_mask.png`
  - `feature_combo.png`
  - batch-level `summary.json`
- Verify that each preview summary includes non-empty values for density, alignment, diameter, curvature, and waviness where the extractor supports them.
