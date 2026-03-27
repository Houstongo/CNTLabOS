# Exp C Resume And Comparison Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add checkpoint resume support for `cnt_paper_repro` so yesterday's `Exp C` run can continue for a few extra final-phase epochs, then export fixed mask-comparison panels for baseline, original `Exp C`, resumed `Exp C`, and `WCNTSegNet`.

**Architecture:** Extend the existing staged training entrypoint rather than creating a parallel fine-tuning script. Resume state will come from `last_model.pth`, continuation will append only to the final phase, and comparison visualization will reuse the existing patch-manifest workflow while loading multiple checkpoints in one pass.

**Tech Stack:** Python, PyTorch, OpenCV, NumPy, YAML, existing `cnt_paper_repro` experiment assets

---

### Task 1: Add resume-aware training state restoration

**Files:**
- Modify: `D:\CNTDATA\CNTA_ML_Project\experiments\cnt_paper_repro\train.py`

**Step 1: Inspect current checkpoint contents**

Confirm that the saved checkpoint already contains:
- `epoch`
- `phase_name`
- `best_metric`
- `history`
- `model_state_dict`
- `optimizer_state_dict`

**Step 2: Add CLI arguments**

Add:
- `--resume`
- `--extra-final-phase-epochs`

**Step 3: Restore state on resume**

Load checkpoint state and restore:
- model weights
- optimizer state
- history
- best metric
- global epoch index
- last completed phase

**Step 4: Restrict resumed continuation**

When resuming, continue only the final configured phase and append the requested extra epochs.

### Task 2: Persist explicit resume metadata

**Files:**
- Modify: `D:\CNTDATA\CNTA_ML_Project\experiments\cnt_paper_repro\train.py`

**Step 1: Extend summary payload**

Write:
- `resumed_from`
- `resume_epoch`
- `extra_final_phase_epochs`
- `original_history_length`

**Step 2: Preserve old behavior**

For fresh training runs, keep these fields null or zero so old workflows do not break.

### Task 3: Build multi-model comparison visualization

**Files:**
- Modify: `D:\CNTDATA\CNTA_ML_Project\experiments\cnt_paper_repro\visualize.py`

**Step 1: Add comparison-mode CLI**

Support:
- baseline checkpoint
- original `Exp C` checkpoint
- resumed `Exp C` checkpoint
- manifest
- max items

**Step 2: Read WCNTSegNet masks from the manifest**

Use `patch_mask_path` directly so weak masks come from the same patch rows used for prediction comparison.

**Step 3: Export fixed panels**

Generate panels with columns:
- original patch
- `WCNTSegNet`
- baseline prediction
- original `Exp C`
- resumed `Exp C`

### Task 4: Run resumed Exp C training

**Files:**
- Use: `D:\CNTDATA\CNTA_ML_Project\experiments\cnt_paper_repro\configs\paper_100000x_cldice.yaml`
- Use: `D:\CNTDATA\CNTA_ML_Project\experiments\cnt_paper_repro\runs\cnt_paper_repro_100000x_center768_cldice_seed42\last_model.pth`

**Step 1: Resume for a small continuation**

Run with:
- `--resume ...last_model.pth`
- `--extra-final-phase-epochs 3`

**Step 2: Review metrics**

Check whether best validation Dice improves beyond yesterday's `0.902896249294281`.

### Task 5: Export comparison masks

**Files:**
- Use: `D:\CNTDATA\CNTA_ML_Project\experiments\cnt_paper_repro\datasets\zzy_mid_100000_patch768_center_paper_v1\manifests\test_patch_manifest.csv`

**Step 1: Use the first 12 test patches**

Generate comparison panels for a fixed subset.

**Step 2: Keep outputs beside the resumed run**

Store the new comparison directory in the resumed `Exp C` run root.

### Task 6: Verify and summarize

**Files:**
- Inspect: `history.csv`
- Inspect: `summary.json`
- Inspect: exported panel directory

**Step 1: Confirm appended epochs exist**

Check that global epoch count increased beyond 9.

**Step 2: Confirm panels exist**

Check that 12 comparison images were written.

**Step 3: Summarize effect**

Compare:
- baseline test Dice
- original `Exp C` test Dice
- resumed `Exp C` test Dice

Then note whether the extra epochs improved the model or mostly plateaued.
