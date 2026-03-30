# Topology-Clean V3 Visual Demo Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Upgrade the existing `generate_slice_spur_prune_demo.py` script so each sample exports an overview panel, per-step PNGs, and a metrics JSON using the integrated Topology-Clean V3 feature path.

**Architecture:** Reuse the existing text10 sample selection and CNTSegNet-SLICE mask inference, then export two artifact layers from the same run: a richer overview panel for fast inspection and a `steps/` folder for single-image review. Metrics come from `FeatureExtractor.extract_all(..., external_binary_mask=mask)` so the visual demo stays aligned with the current Topology-Clean V3 feature pipeline.

**Tech Stack:** Python, OpenCV, Matplotlib, existing `FeatureExtractor`, existing CNTSegNet-SLICE demo script

---

### Task 1: Expand the demo script outputs

**Files:**
- Modify: `D:\CNTDATA\CNTA_ML_Project\tools\generate_slice_spur_prune_demo.py`

**Step 1: Add step image helpers**

Implement helpers that save:
- `01_roi.png`
- `02_mask.png`
- `03_raw_skeleton.png`
- `04_removed_short.png`
- `05_removed_spur.png`
- `06_cleaned_skeleton.png`
- `07_v3_metrics_overlay.png`

**Step 2: Add a metrics card to the overview panel**

Update the panel renderer to show:
- image pipeline views
- a text block with density, alignment, diameter, V3 multi-stat curvature, waviness, tortuosity, and cleanup counts

**Step 3: Save a `metrics.json` per sample**

Use the current integrated feature path so the visual demo and feature outputs stay consistent.

### Task 2: Verify the upgraded demo end-to-end

**Files:**
- Verify: `D:\CNTDATA\CNTA_ML_Project\tools\generate_slice_spur_prune_demo.py`

**Step 1: Compile the script**

Run a Python compile check on the script.

**Step 2: Generate a new text10 output directory**

Run the script with a new output directory name and confirm:
- overview panel exists
- `steps/` single images exist
- `metrics.json` exists

**Step 3: Inspect summary output**

Confirm root `summary.json` includes the new artifact paths.
