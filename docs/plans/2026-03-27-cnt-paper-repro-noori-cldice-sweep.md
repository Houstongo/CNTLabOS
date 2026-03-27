# CNT Paper Repro No-Orientation clDice Sweep Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add four new `cnt_paper_repro` experiment configs for a no-orientation `clDice` sweep plus a `PureDice-9ep` control, then train them and generate six-way desktop-image comparison panels.

**Architecture:** Keep the existing `cnt_paper_repro` training code unchanged and express the new study entirely through config files plus a small reusable comparison-panel script. Reuse the existing baseline and original `Exp C` checkpoints, add four new runs, and render the eight desktop images into compact gap-separated grids.

**Tech Stack:** Python, PyTorch, OpenCV, YAML, existing `cnt_paper_repro` assets

---

### Task 1: Add the four new configs

**Files:**
- Create: `D:\CNTDATA\CNTA_ML_Project\experiments\cnt_paper_repro\configs\paper_100000x_dice9.yaml`
- Create: `D:\CNTDATA\CNTA_ML_Project\experiments\cnt_paper_repro\configs\paper_100000x_cldice_noori_005.yaml`
- Create: `D:\CNTDATA\CNTA_ML_Project\experiments\cnt_paper_repro\configs\paper_100000x_cldice_noori_01.yaml`
- Create: `D:\CNTDATA\CNTA_ML_Project\experiments\cnt_paper_repro\configs\paper_100000x_cldice_noori_02.yaml`

**Step 1: Create `PureDice-9ep`**

Use a single `Dice` phase with `9` epochs.

**Step 2: Create three no-orientation Exp C variants**

Keep the first six Dice epochs, then set:
- `orientation_weight: 0.0`
- `lambda_cl: 0.05 / 0.1 / 0.2`

### Task 2: Add a reusable desktop comparison tool

**Files:**
- Create: `D:\CNTDATA\CNTA_ML_Project\tools\generate_cnt_paper_desktop_compare.py`

**Step 1: Load an arbitrary set of named checkpoints**

The tool should render one panel per input image from desktop folders.

**Step 2: Use compact gap-separated grid layout**

Render:
- `Original`
- `Baseline`
- `Original ExpC`
- `PureDice-9ep`
- `ExpC-noOri-cl0.05`
- `ExpC-noOri-cl0.1`
- `ExpC-noOri-cl0.2`

with white gaps between tiles.

### Task 3: Train the four new configs

**Files:**
- Use: `D:\CNTDATA\CNTA_ML_Project\experiments\cnt_paper_repro\train.py`

**Step 1: Run `PureDice-9ep`**

Write the new run directory and summary.

**Step 2: Run the three no-orientation Exp C variants**

Write three more run directories and summaries.

### Task 4: Generate six-way desktop comparison panels

**Files:**
- Use: `C:\Users\clearlove\Desktop\text`
- Use: `C:\Users\clearlove\Desktop\text10`

**Step 1: Reuse the previous eight source images**

Generate one grid PNG per image.

**Step 2: Put all eight PNGs in one folder**

Also write a short summary file listing the models and source folders.

### Task 5: Verify and summarize

**Files:**
- Inspect: each new run `summary.json`
- Inspect: comparison output directory

**Step 1: Compare metrics**

Summarize how the four new runs compare against baseline and original `Exp C`.

**Step 2: Compare visual behavior**

Call out whether stronger `clDice` mainly improves connectivity or starts over-thickening masks.
