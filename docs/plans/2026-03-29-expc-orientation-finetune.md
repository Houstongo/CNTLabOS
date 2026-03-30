# Exp C Orientation Finetune Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 从原始标准 `Exp C` 权重出发，额外训练 `3` 轮 `0.6*Dice + 3e-7*Orientation`，并在 `test10` 上生成与原始 `Exp C` 的全图对比结果。

**Architecture:** 复用 `experiments/cnt_paper_repro` 的既有训练与可视化链路，仅新增一个 orientation-finetune 配置与一个独立 run。训练从 `pre_resume_last_model.pth` 恢复，保持原始 `Exp C` run 完全不被覆盖；可视化沿用桌面 `test10` 全图、底部信息条自动裁切的口径。

**Tech Stack:** PyTorch, YAML config, PowerShell, OpenCV, existing `cnt_paper_repro` training pipeline.

---

### Task 1: Add Orientation-Finetune Config

**Files:**
- Create: `D:\CNTDATA\CNTA_ML_Project\experiments\cnt_paper_repro\configs\paper_100000x_cldice_orifinetune.yaml`

**Step 1: Create a dedicated config**

- Set `experiment_name` to `cnt_paper_repro_100000x_center768_cldice_orifinetune`
- Keep dataset / model / batch / threshold aligned with standard `Exp C`
- Define phases:
  - `dice`, `epochs: 6`
  - `dice_orientation`, `epochs: 3`, `dice_weight: 0.6`, `orientation_weight: 3e-7`, `lambda_cl: 0.0`

**Step 2: Verify config loads**

Run:

```powershell
C:\Users\clearlove\.conda\envs\lab_agent\python.exe D:\CNTDATA\CNTA_ML_Project\experiments\cnt_paper_repro\train.py --config D:\CNTDATA\CNTA_ML_Project\experiments\cnt_paper_repro\configs\paper_100000x_cldice_orifinetune.yaml --help
```

Expected:
- command prints usage instead of crashing on config path issues

### Task 2: Verify Resume Source

**Files:**
- Read: `D:\CNTDATA\CNTA_ML_Project\experiments\cnt_paper_repro\runs\cnt_paper_repro_100000x_center768_cldice_seed42\pre_resume_last_model.pth`
- Read: `D:\CNTDATA\CNTA_ML_Project\experiments\cnt_paper_repro\runs\cnt_paper_repro_100000x_center768_cldice_seed42\pre_resume_best_model.pth`

**Step 1: Inspect checkpoint metadata**

Run a short Python snippet to confirm:
- source checkpoint exists
- `epoch == 9`
- phase name is the original final `Exp C` phase

**Step 2: Record comparison checkpoint**

- Use `pre_resume_best_model.pth` as the original `Exp C` prediction checkpoint

### Task 3: Run 3-Epoch Orientation Finetune

**Files:**
- Use config: `D:\CNTDATA\CNTA_ML_Project\experiments\cnt_paper_repro\configs\paper_100000x_cldice_orifinetune.yaml`
- Resume from: `D:\CNTDATA\CNTA_ML_Project\experiments\cnt_paper_repro\runs\cnt_paper_repro_100000x_center768_cldice_seed42\pre_resume_last_model.pth`

**Step 1: Launch GPU finetune**

Run:

```powershell
C:\Users\clearlove\.conda\envs\lab_agent\python.exe D:\CNTDATA\CNTA_ML_Project\experiments\cnt_paper_repro\train.py --config D:\CNTDATA\CNTA_ML_Project\experiments\cnt_paper_repro\configs\paper_100000x_cldice_orifinetune.yaml --resume D:\CNTDATA\CNTA_ML_Project\experiments\cnt_paper_repro\runs\cnt_paper_repro_100000x_center768_cldice_seed42\pre_resume_last_model.pth
```

Expected:
- new run directory appears
- history extends for 3 additional epochs
- original `Exp C` run is not overwritten

**Step 2: Verify completion**

Check:
- `summary.json`
- `history.csv`
- `best_model.pth`

### Task 4: Generate test10 Comparison Visuals

**Files:**
- Use desktop folder: `C:\Users\clearlove\Desktop\text10`
- Output: `D:\CNTDATA\CNTA_ML_Project\experiments\cnt_paper_repro\reports\desktop_text10_expc_vs_orifinetune_bottomcrop_20260329`

**Step 1: Reuse bottom-strip cropping rule**

- Automatically crop the bottom black information strip before inference

**Step 2: Predict with two checkpoints**

- Original `Exp C`: `pre_resume_best_model.pth`
- New finetuned model: `best_model.pth` from the new run

**Step 3: Export artifacts**

- `expc_orig_masks`
- `exp_orientation_finetune_masks`
- `compare_panels`

Panel layout:
- `Original`
- `ExpC-Orig`
- `ExpC-OriFinetune`
- optional difference tile if useful

### Task 5: Summarize Outcome

**Files:**
- Read generated `summary.json`
- Read report folder outputs

**Step 1: Compare metrics**

- original `Exp C` vs new `OriFinetune`
- note `best val dice` and `test dice`

**Step 2: Compare qualitative behavior**

- comment on whether masks become thinner / more conservative / less overfilled on `test10`

**Step 3: Report final paths**

- run directory
- report directory
- representative panel examples
