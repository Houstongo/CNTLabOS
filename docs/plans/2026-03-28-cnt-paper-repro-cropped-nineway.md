# CNT Paper Repro Cropped Nine-Way Desktop Compare Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Update the desktop comparison generator so it renders center-detail `3 x 3` panels with weak labels, `ExpD`, and the resumed `Exp C` checkpoint for every image currently under the desktop `text` and `text10` folders.

**Architecture:** Reuse the existing comparison script as the single entry point, extend it with a manifest-backed weak-label lookup and a fixed center detail crop, then rerun it against the live desktop folders so newly added images are included automatically.

**Tech Stack:** Python, PyTorch, OpenCV, CSV manifests, existing `cnt_paper_repro` checkpoints

---

### Task 1: Extend the comparison generator

**Files:**
- Modify: `D:\CNTDATA\CNTA_ML_Project\tools\generate_cnt_paper_desktop_compare.py`

**Step 1: Add weak-label lookup**

Load both the `50000x` and `100000x` manifest families and resolve desktop image filenames to their weak-label masks.

**Step 2: Add center detail crop**

Crop the aligned `768 x 768` patch down to a fixed `384 x 384` center ROI for the SEM image, weak label, and every model output.

**Step 3: Keep a fixed nine-tile order**

Render:
- `Original`
- `WeakLabel`
- `Baseline`
- `Original-ExpC`
- `ExpC+3ep`
- `ExpD`
- `PureDice-9ep`
- `ExpC-noOri-0.1`
- `ExpC-noOri-0.2`

### Task 2: Generate the updated desktop outputs

**Files:**
- Use: `C:\Users\clearlove\Desktop\text`
- Use: `C:\Users\clearlove\Desktop\text10`

**Step 1: Run the generator on both folders**

Point it at the existing checkpoints and write all output PNGs into one fresh report folder.

**Step 2: Include newly added images automatically**

Do not hardcode filenames; let the script iterate the current directory contents.

### Task 3: Verify and summarize

**Files:**
- Inspect: output `summary.json`
- Inspect: output PNG count

**Step 1: Confirm the expected layout metadata**

Check `detail_size`, `cols`, gutters, and model order in the summary file.

**Step 2: Confirm the image count**

Verify the output PNG count matches the combined number of images currently present in `text` and `text10`.
