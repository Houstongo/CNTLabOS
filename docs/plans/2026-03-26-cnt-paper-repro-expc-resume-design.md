# Exp C Resume And Comparison Design

**Date:** 2026-03-26

**Goal**

Extend the `cnt_paper_repro` experiment line so `Exp C` (`cnt_paper_repro_100000x_center768_cldice`) can continue training from yesterday's checkpoint, then generate fixed qualitative comparison panels across:

- original grayscale patch
- `WCNTSegNet` weak mask
- paper baseline `cnt_paper_repro_100000x_center768`
- original `Exp C` model
- resumed `Exp C` model

---

## 1. Scope

This design is intentionally narrow.

We are not redesigning the paper-reproduction pipeline. We are adding:

- checkpoint resume support for staged training
- a controlled way to append extra epochs to the last stage of `Exp C`
- a reproducible comparison export on the fixed `test_patch_manifest`

The first target is yesterday's `Exp C` run:

- run root: `D:\CNTDATA\CNTA_ML_Project\experiments\cnt_paper_repro\runs\cnt_paper_repro_100000x_center768_cldice_seed42`

---

## 2. Resume Strategy

The current training entrypoint always starts from epoch 1 and rebuilds history from scratch. That is not enough for "continue yesterday's training".

The new behavior should support:

- loading `last_model.pth`
- restoring model weights
- restoring optimizer state
- restoring history
- restoring global epoch index
- restoring which phase was active

The resumed run should stay in the same run directory so the outputs remain associated with the original `Exp C` experiment.

To keep the logic simple and auditable:

- resumed training will append only to the configured final phase
- the user chooses a small `extra_epochs` value such as `3` or `5`
- the earlier completed phases are not re-run

---

## 3. Config And CLI Rules

The least disruptive interface is:

- keep the existing config file format working unchanged
- add optional CLI flags to control resume behavior

Recommended CLI additions:

- `--resume <checkpoint>`
- `--extra-final-phase-epochs <int>`

Behavior rules:

- no `--resume`: current behavior stays unchanged
- with `--resume`: restore checkpoint state and continue from the saved last epoch
- with `--extra-final-phase-epochs`: extend only the final training phase by that amount during resumed training

This avoids baking one-off continuation values into the canonical YAML config.

---

## 4. Output And Metadata

Resumed training should keep writing:

- `best_model.pth`
- `last_model.pth`
- `history.csv`
- `summary.json`

The summary should also record:

- `resumed_from`
- `resume_epoch`
- `extra_final_phase_epochs`
- `original_history_length`
- final best epoch after continuation

This makes the continuation explicit when we review the run later.

---

## 5. Comparison Visualization

The repository already has a single-checkpoint visualization helper that exports:

- original patch
- weak mask
- probability map
- binary prediction

For this task, we need side-by-side model comparison instead of one-model panels.

The comparison export should use the fixed:

- manifest: `D:\CNTDATA\CNTA_ML_Project\experiments\cnt_paper_repro\datasets\zzy_mid_100000_patch768_center_paper_v1\manifests\test_patch_manifest.csv`
- item count: first `12` rows

Each panel should include, in order:

- original patch
- `WCNTSegNet` mask
- baseline binary prediction
- original `Exp C` binary prediction
- resumed `Exp C` binary prediction

Optional but still useful:

- add a text header row naming each column

This comparison is intentionally binary-mask-first, because the user asked for mask comparison rather than probability heatmaps.

---

## 6. Verification

Before calling the work complete, verify:

- resumed training actually advances beyond global epoch 9
- `history.csv` gains the appended epochs
- `summary.json` shows resume metadata
- the resumed checkpoint can be visualized normally
- a comparison directory exists with 12 panels

We should also compare baseline, original `Exp C`, and resumed `Exp C` summary metrics to see whether the extra epochs helped or plateaued.

---

## 7. Non-Goals

This task does not attempt to:

- add generic resume support to `cnt_loss_compare`
- redesign phase scheduling across arbitrary intermediate phases
- introduce a new loss term
- change the dataset split or patch extraction strategy
- replace the current single-model visualization output
