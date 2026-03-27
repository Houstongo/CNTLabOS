# CNT Paper Repro No-Orientation clDice Sweep Design

**Date:** 2026-03-27

**Goal**

Run a tightly controlled follow-up experiment under `cnt_paper_repro` that compares:

- `Baseline`
- original `Exp C`
- `PureDice-9ep`
- `ExpC-noOri-cl0.05`
- `ExpC-noOri-cl0.1`
- `ExpC-noOri-cl0.2`

and export six-way visual comparisons on the same eight desktop images used in the previous check.

---

## 1. Why This Sweep

The current `Exp C` mixes three ingredients in its second stage:

- `Dice`
- `orientation`
- `clDice`

But the configured orientation weight is extremely small (`1e-7`), so it is unclear whether:

- the visible behavior comes mainly from `clDice`
- orientation is helping at all
- the current `clDice` weight is too weak, balanced, or too strong

To answer that cleanly, we should remove orientation entirely from the new `Exp C` variants and scan only `clDice`.

---

## 2. Fixed Factors

To keep this comparison interpretable, the following stay fixed:

- backbone: `ResNet34UNet`
- dataset: `zzy_mid_100000_patch768_center_paper_v1`
- patch size: `768`
- patch mode: `center`
- optimizer: `Adam`
- learning rate: `5e-4`
- batch size: `2`
- checkpoint selection metric: validation `dice`
- inference threshold: `0.7`
- seed: `42`

Only the loss composition changes.

---

## 3. Six Comparison Variants

### Baseline

Existing paper-style baseline:

- phase 1: `Dice` for `6` epochs
- phase 2: `0.6 * Dice + 1e-7 * Orientation` for `3` epochs

This remains the paper-style reference.

### Original Exp C

Existing original version:

- phase 1: `Dice` for `6` epochs
- phase 2: `0.6 * Dice + 1e-7 * Orientation + 0.1 * clDice` for `3` epochs

This remains the legacy `Exp C` reference.

### PureDice-9ep

New control:

- single-stage `Dice` only for `9` epochs

This answers whether the gains attributed to stage-2 structure terms could instead come from simply training longer with Dice.

### ExpC-noOri-cl0.05

New no-orientation variant:

- phase 1: `Dice` for `6` epochs
- phase 2: `0.6 * Dice + 0.05 * clDice` for `3` epochs

### ExpC-noOri-cl0.1

New no-orientation variant:

- phase 1: `Dice` for `6` epochs
- phase 2: `0.6 * Dice + 0.1 * clDice` for `3` epochs

### ExpC-noOri-cl0.2

New no-orientation variant:

- phase 1: `Dice` for `6` epochs
- phase 2: `0.6 * Dice + 0.2 * clDice` for `3` epochs

---

## 4. Expected Reading Of Results

This sweep is useful because each branch answers a different question:

- `PureDice-9ep` tests whether stage-2 gains are really structural or just more optimization time
- `cl0.05` tests whether weak structural pressure is enough
- `cl0.1` tests the current practical balance without orientation
- `cl0.2` tests whether stronger structural pressure causes over-thick or over-connected masks

The likely trade-off is:

- lower `clDice` weight: cleaner, thinner masks but weaker connectivity
- higher `clDice` weight: stronger continuity but more risk of thickening and false bridges

---

## 5. Visualization Output

The previous one-row layout was hard to compare, so this sweep should export:

- one image per source SEM input
- all outputs in one folder
- a `2 x 3` grid
- explicit white gaps between tiles

Each grid should show:

- `Original`
- `Baseline`
- `Original ExpC`
- `PureDice-9ep`
- `ExpC-noOri-cl0.05`
- `ExpC-noOri-cl0.1`
- `ExpC-noOri-cl0.2`

Because that is seven items, the practical layout should become:

- a `3 x 3` padded grid with one empty slot

This is clearer than stretching into a long strip.

---

## 6. Verification

Before calling the sweep complete, verify:

- all four new runs produce `summary.json`
- each run writes `best_model.pth`
- the six comparison families are rendered for the same eight desktop images
- the final output folder contains exactly eight comparison PNGs

---

## 7. Non-Goals

This task does not attempt to:

- retune learning rate or optimizer
- change dataset split
- change backbone
- prove scientific superiority from the desktop images alone
- replace the existing baseline or legacy `Exp C`
