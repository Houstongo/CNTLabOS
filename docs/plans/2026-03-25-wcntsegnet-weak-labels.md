# WCNTSegNET Weak Labels Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a reusable export flow that generates `WCNTSegNET` weak masks and metadata for the fixed `zzy_mid_100000_train50_test50_v2` loss-comparison dataset.

**Architecture:** The implementation will add one dataset-local export script under `tools/` that reads the curated split manifests, generates `WCNTSegNET` masks through the existing `FeatureExtractor` binary-mask path, writes masks into split-local `masks_wcntsegnet/` folders, and exports dataset-level manifest/stat tables plus overlay previews. Lightweight tests will target the pure helper functions so the export format stays stable without needing to run the full dataset every time.

**Tech Stack:** Python, CSV, OpenCV, NumPy, SQLite-derived dataset manifests, existing `FeatureExtractor`

---

### Task 1: Add the weak-label export script

**Files:**
- Create: `D:\CNTDATA\CNTA_ML_Project\tools\generate_wcntsegnet_weak_labels.py`
- Modify: `D:\CNTDATA\CNTA_ML_Project\docs\plans\2026-03-25-wcntsegnet-weak-labels-design.md`
- Test: `D:\CNTDATA\CNTA_ML_Project\tests\test_generate_wcntsegnet_weak_labels.py`

**Step 1: Write the failing test**

- Add tests for:
  - split manifest loading
  - output path derivation for masks and overlays
  - dataset-level row assembly for the exported manifest

**Step 2: Run test to verify it fails**

Run:

```bash
pytest D:\CNTDATA\CNTA_ML_Project\tests\test_generate_wcntsegnet_weak_labels.py -v
```

Expected: FAIL because the new export helpers do not exist yet.

**Step 3: Write minimal implementation**

- Add CLI args for:
  - `--dataset-root`
  - `--splits`
  - `--preview-per-split`
- Add helper functions to:
  - load split manifests
  - build `WCNTSegNET` masks from dataset images
  - save masks
  - save overlay previews
  - assemble rows for `labels_manifest.csv` and `label_stats.csv`

**Step 4: Run test to verify it passes**

Run:

```bash
pytest D:\CNTDATA\CNTA_ML_Project\tests\test_generate_wcntsegnet_weak_labels.py -v
```

Expected: PASS

### Task 2: Generate weak labels for the fixed `100000x v2` dataset

**Files:**
- Test: `D:\CNTDATA\CNTA_ML_Project\experiments\cnt_loss_compare\datasets\zzy_mid_100000_train50_test50_v2`

**Step 1: Execute the export script**

Run:

```bash
python D:\CNTDATA\CNTA_ML_Project\tools\generate_wcntsegnet_weak_labels.py --dataset-root D:\CNTDATA\CNTA_ML_Project\experiments\cnt_loss_compare\datasets\zzy_mid_100000_train50_test50_v2
```

Expected:

- masks generated for `train`, `test`, and `reserve`
- dataset-level manifest and stats CSV files written
- per-split preview overlays created

**Step 2: Verify counts**

- Confirm:
  - `train` image count equals `train/masks_wcntsegnet` count
  - `test` image count equals `test/masks_wcntsegnet` count
  - `reserve` image count equals `reserve/masks_wcntsegnet` count

### Task 3: Verify the exported metadata and previews

**Files:**
- Test: exported files under the dataset root

**Step 1: Check dataset-level tables**

- Inspect:
  - `labels_manifest.csv`
  - `label_stats.csv`

Expected:

- every processed file has a row
- split, image, mask, and weak-label statistics are populated

**Step 2: Spot-check previews**

- Open a few preview overlays from `train`, `test`, and `reserve`

Expected:

- mask aligns with the source image crop
- labels look consistent with current `WCNTSegNET` behavior

### Task 4: Report back and lock this dataset version

**Files:**
- Test: final dataset root

**Step 1: Summarize exact output locations**

- Include:
  - dataset root
  - mask directories
  - manifest/stat CSV files
  - preview directory

**Step 2: Note the next handoff**

- State that this dataset is now ready for:
  - weak-supervision training
  - hard-case review
  - later loss-comparison experiments
