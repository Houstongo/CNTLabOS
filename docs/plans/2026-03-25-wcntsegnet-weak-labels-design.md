# WCNTSegNET Weak Labels Design

**Goal:** Generate reproducible `WCNTSegNET` weak-label assets for the fixed `zzy_mid_100000_train50_test50_v2` experiment dataset across `train`, `test`, and `reserve`.

**Context**

- The fixed dataset root is [zzy_mid_100000_train50_test50_v2](D:\CNTDATA\CNTA_ML_Project\experiments\cnt_loss_compare\datasets\zzy_mid_100000_train50_test50_v2).
- Each split already has a curated `images/` directory and matching CSV manifests under `manifests/`.
- The current canonical traditional segmentation path is `WCNTSegNET`, which is implemented through the existing `FeatureExtractor` preprocessing and density-mask path.
- For the planned loss-comparison experiments, these labels are weak supervision assets, not ground truth. They must be reproducible and tied to the fixed image split.

**Approved Scope**

- Generate `WCNTSegNET` masks for:
  - `train`
  - `test`
  - `reserve`
- Keep the existing `images/` folders unchanged.
- Add side-by-side weak-label assets inside the same dataset root:
  - `train/masks_wcntsegnet/`
  - `test/masks_wcntsegnet/`
  - `reserve/masks_wcntsegnet/`
- Export dataset-level metadata for later training, evaluation, and visualization reuse:
  - `labels_manifest.csv`
  - `label_stats.csv`
  - preview overlays for quick inspection

**Design**

- Add a dedicated script under `tools/` rather than overloading the original dataset-preparation script.
- Reuse the current `WCNTSegNET` path by calling `FeatureExtractor.extract_roi()`, `FeatureExtractor.preprocess()`, and `FeatureExtractor.calculate_density()` to obtain the binary weak mask.
- Process each split from the curated dataset directory, not from the raw `ZZY` tree or a new database query. This keeps the weak labels locked to the approved split.
- Preserve one-to-one naming with the image files:
  - image: `train/images/01484_....png`
  - mask: `train/masks_wcntsegnet/01484_....png`
- Export a single combined manifest that records:
  - split
  - image filename
  - mask filename
  - image path
  - mask path
  - magnification and process metadata from the split CSV
  - simple weak-label statistics such as foreground ratio and connected-component count
- Export a separate stats table for fast filtering and hard-case selection.
- Generate a small preview set of overlays per split so the labels can be reviewed visually without opening hundreds of files.

**Error Handling**

- Skip unreadable images but record them in the manifest with `status=failed`.
- Continue processing the rest of the split even if individual files fail.
- Refuse to run if a requested split directory exists but lacks `images/`.
- Refuse to overwrite manifests partially; write the full outputs only after each run completes.

**Verification**

- Run the script against `zzy_mid_100000_train50_test50_v2`.
- Confirm that all three splits receive `masks_wcntsegnet/`.
- Confirm that mask counts match image counts for `train`, `test`, and `reserve`.
- Confirm that `labels_manifest.csv` and `label_stats.csv` exist and include all processed files.
- Spot-check overlay previews to make sure the generated masks match the current `WCNTSegNET` segmentation behavior.
