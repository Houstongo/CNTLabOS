# CNT Paper Repro Task-Adapted Backbone Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a parallel task-adapted `ResNet34UNet` backbone to `cnt_paper_repro`, wire it into the existing staged paper-reproduction training flow, and prepare a clean comparison against the original baseline and original `Exp C`.

**Architecture:** Preserve the original `ResNet34UNet` class and add a second parallel model class that reduces early downsampling, strengthens shallow high-resolution reuse, and upgrades the segmentation head. Add config-driven model selection in the paper-repro training entry point, then create task-adapted baseline and task-adapted `Exp C` configs plus a smoke config.

**Tech Stack:** Python, PyTorch, torchvision ResNet34, YAML configs, existing `cnt_paper_repro` staged training pipeline

---

### Task 1: Add config-driven model construction to `cnt_paper_repro`

**Files:**
- Modify: `D:\CNTDATA\CNTA_ML_Project\experiments\cnt_paper_repro\model.py`
- Modify: `D:\CNTDATA\CNTA_ML_Project\experiments\cnt_paper_repro\train.py`

**Step 1: Preserve the original model class**

Keep `ResNet34UNet` behavior unchanged so all historical runs remain reproducible.

**Step 2: Add a parallel task-adapted model class**

Create a second model class, for example `ResNet34UNetTaskAdapted`, in the same file.

**Step 3: Add a config-driven model builder**

Let training instantiate either:

- `ResNet34UNet`
- `ResNet34UNetTaskAdapted`

based on `model.name`.

### Task 2: Implement the task-adapted `ResNet34UNet`

**Files:**
- Modify: `D:\CNTDATA\CNTA_ML_Project\experiments\cnt_paper_repro\model.py`

**Step 1: Reduce early downsampling**

Adjust the early encoder path so the model keeps more spatial detail before deep feature extraction.

**Step 2: Strengthen shallow high-resolution reuse**

Add one more lightweight high-resolution refinement path near the output while keeping the existing shallow skip behavior.

**Step 3: Upgrade the segmentation head**

Replace the direct `1 x 1` output projection in the task-adapted class with:

- `3 x 3 conv`
- normalization and activation
- `1 x 1 conv`

### Task 3: Add task-adapted configs

**Files:**
- Create: `D:\CNTDATA\CNTA_ML_Project\experiments\cnt_paper_repro\configs\paper_100000x_taskadapted.yaml`
- Create: `D:\CNTDATA\CNTA_ML_Project\experiments\cnt_paper_repro\configs\paper_100000x_cldice_taskadapted.yaml`
- Create: `D:\CNTDATA\CNTA_ML_Project\experiments\cnt_paper_repro\configs\paper_100000x_taskadapted_smoke.yaml`

**Step 1: Mirror the baseline training schedule**

Create a task-adapted baseline config that matches `paper_100000x.yaml` except for `model.name` and `experiment_name`.

**Step 2: Mirror the `Exp C` training schedule**

Create a task-adapted `Exp C` config that matches `paper_100000x_cldice.yaml` except for `model.name` and `experiment_name`.

**Step 3: Add a smoke config**

Mirror `paper_100000x_smoke.yaml` so the task-adapted backbone can be validated quickly before long runs.

### Task 4: Verify framework compatibility

**Files:**
- Use: `D:\CNTDATA\CNTA_ML_Project\experiments\cnt_paper_repro\train.py`
- Use: `D:\CNTDATA\CNTA_ML_Project\experiments\cnt_paper_repro\configs\paper_100000x_taskadapted_smoke.yaml`
- Use: `D:\CNTDATA\CNTA_ML_Project\experiments\cnt_paper_repro\configs\paper_100000x_smoke.yaml`

**Step 1: Run syntax and import validation**

Check that the modified model and training files load correctly.

**Step 2: Run task-adapted smoke training**

Run the new smoke config to confirm:

- forward pass
- staged loss computation
- checkpoint writing
- summary writing

**Step 3: Recheck the original smoke path**

Run the original smoke config again to ensure the old paper-repro path still works unchanged.

### Task 5: Prepare the comparison matrix

**Files:**
- Inspect: run folders under `D:\CNTDATA\CNTA_ML_Project\experiments\cnt_paper_repro\runs`

**Step 1: Keep names explicit**

Use run names that clearly distinguish:

- original vs task-adapted
- baseline vs `Exp C`

**Step 2: Keep first comparison minimal**

Prepare to compare:

- original baseline
- task-adapted baseline
- original `Exp C`
- task-adapted `Exp C`

Plan complete and saved to `docs/plans/2026-03-28-cnt-paper-repro-task-adapted-backbone.md`. Two execution options:

**1. 当前会话继续执行** - 我直接按这个计划改 `cnt_paper_repro` 的 backbone 和配置，再做 smoke test

**2. 单独会话执行** - 新开一个会话，按这个计划逐步实现
