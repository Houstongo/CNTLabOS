# CNT Loss Compare Task-Adapted Backbone Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a parallel task-adapted CNT segmentation backbone with grayscale input support, reduced early downsampling, stronger shallow skip reuse, and a `3 x 3 + 1 x 1` segmentation head, then wire it into three new loss-comparison experiments.

**Architecture:** Keep the legacy `CNTSegNet` path unchanged. Extend the experiment framework so configs can choose either the legacy backbone or a new task-adapted backbone. Make the dataset loader configurable for legacy pseudo-RGB input versus direct single-channel grayscale input, then add three configs that reuse the new backbone under orientation, clDice, and orientation-plus-clDice losses.

**Tech Stack:** Python, PyTorch, torchvision ResNet50, YAML configs, existing `cnt_loss_compare` training pipeline

---

### Task 1: Add config-driven backbone selection

**Files:**
- Modify: `D:\CNTDATA\CNTA_ML_Project\experiments\cnt_loss_compare\train.py`
- Modify: `D:\CNTDATA\CNTA_ML_Project\experiments\cnt_loss_compare\config.py`

**Step 1: Define the config contract**

Add a `model.name` branch that supports both:

- `cntsegnet`
- `cntsegnet_task_adapted`

Keep the current default behavior unchanged for existing configs.

**Step 2: Route model construction by config**

Update training so it instantiates the correct backbone class based on `model.name`.

**Step 3: Keep legacy runs stable**

Make sure every existing config that says `cntsegnet` still constructs the original model without any structural change.

### Task 2: Add grayscale-aware dataset mode

**Files:**
- Modify: `D:\CNTDATA\CNTA_ML_Project\experiments\cnt_loss_compare\data.py`
- Modify: `D:\CNTDATA\CNTA_ML_Project\experiments\cnt_loss_compare\train.py`

**Step 1: Add input-mode configuration**

Let the dataset support:

- `rgb_replicated`
- `grayscale_single_channel`

**Step 2: Preserve normalization behavior**

Keep the current ImageNet-style normalization for legacy pseudo-RGB mode.

For grayscale mode, add a simple single-channel normalization path that is consistent across train, val, and test.

**Step 3: Thread the option through training**

Read the selected input mode from config and pass it into the dataset loader.

### Task 3: Implement the task-adapted backbone

**Files:**
- Modify: `D:\CNTDATA\CNTA_ML_Project\experiments\cnt_loss_compare\backbone.py`

**Step 1: Add a new model class**

Create a parallel backbone class, for example `CNTSegNetTaskAdapted`, rather than mutating the original `CNTSegNet`.

**Step 2: Add a single-channel stem**

Replace the legacy pseudo-RGB entry assumption with a direct grayscale input stem.

**Step 3: Reduce early downsampling**

Adjust the earliest encoder path so thin CNT features retain more spatial detail before deep encoding.

**Step 4: Strengthen shallow skip reuse**

Expose a shallow feature map and feed it back into the decoder path.

**Step 5: Upgrade the segmentation head**

Replace the plain `1 x 1` output head with:

- `3 x 3 conv`
- normalization and activation
- `1 x 1 conv`

### Task 4: Add the new loss variants

**Files:**
- Create: `D:\CNTDATA\CNTA_ML_Project\experiments\cnt_loss_compare\configs\exp_task_adapted_orientation.yaml`
- Create: `D:\CNTDATA\CNTA_ML_Project\experiments\cnt_loss_compare\configs\exp_task_adapted_cldice.yaml`
- Create: `D:\CNTDATA\CNTA_ML_Project\experiments\cnt_loss_compare\configs\exp_task_adapted_orientation_cldice.yaml`

**Step 1: Add orientation-only structure config**

Create a config for:

- new task-adapted backbone
- grayscale input mode
- `Dice + BCE + orientation`

**Step 2: Add clDice structure config**

Create a config for:

- new task-adapted backbone
- grayscale input mode
- `Dice + BCE + clDice`

**Step 3: Add combined structure config**

Create a config for:

- new task-adapted backbone
- grayscale input mode
- `Dice + BCE + orientation + clDice`

### Task 5: Verify framework compatibility

**Files:**
- Use: `D:\CNTDATA\CNTA_ML_Project\experiments\cnt_loss_compare\train.py`
- Use: `D:\CNTDATA\CNTA_ML_Project\experiments\cnt_loss_compare\evaluate.py`

**Step 1: Run import-level validation**

Verify the new backbone and dataset path load correctly without breaking the existing training entry point.

**Step 2: Run smoke training checks**

Run at least one short training smoke test for the new backbone path to confirm:

- loader shape correctness
- forward pass correctness
- loss computation compatibility
- checkpoint writing still works

**Step 3: Recheck legacy compatibility**

Run a short legacy-path smoke test with the old `cntsegnet` config to confirm the old path still works unchanged.

### Task 6: Document outputs for comparison

**Files:**
- Inspect: new run folders under `D:\CNTDATA\CNTA_ML_Project\experiments\cnt_loss_compare\runs`

**Step 1: Confirm experiment naming**

Make sure the run folder names clearly distinguish:

- legacy backbone vs task-adapted backbone
- the three loss variants

**Step 2: Summarize comparable metrics**

Prepare to compare:

- `dice`
- `iou`
- `cldice`
- selected visual masks on the existing desktop comparison workflow if needed

Plan complete and saved to `docs/plans/2026-03-28-cnt-loss-compare-task-adapted-backbone.md`. Two execution options:

**1. 当前会话继续执行** - 我直接按这个计划改代码、做 smoke test、再把结果汇总给你

**2. 单独会话执行** - 新开一个会话，按这个计划逐步实现
