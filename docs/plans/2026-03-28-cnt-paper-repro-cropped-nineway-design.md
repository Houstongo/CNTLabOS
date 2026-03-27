# CNT Paper Repro Cropped Nine-Way Desktop Compare Design

**Date:** 2026-03-28

**Goal**

Make the desktop comparison output easier to judge by switching from full `768 x 768` panels to a tighter center detail crop, and expand the comparison set to a fixed `3 x 3` layout:

- `Original`
- `WeakLabel`
- `Baseline`
- `Original ExpC`
- `ExpC +3ep`
- `ExpD`
- `PureDice-9ep`
- `ExpC-noOri-0.1`
- `ExpC-noOri-0.2`

This replaces the previous `ExpC-noOri-0.05` tile with the resumed `Exp C` checkpoint and keeps the output compact enough for visual inspection.

---

## 1. Visual Direction

The main problem with the current panels is not missing models, but too much spatial context. CNT detail differences are small, so the viewer has to scan too much irrelevant area.

The new output should therefore:

- crop a fixed center ROI from each already-aligned `768 x 768` patch
- keep all nine tiles in one `3 x 3` grid
- preserve explicit white gutters so each mask stays visually separated
- continue writing one PNG per source SEM image into a single folder

The chosen default detail crop is `384 x 384`, which is large enough to retain topology while reducing clutter.

---

## 2. Weak Label Strategy

The desktop source images come from both `50000x` and `100000x` sets, so weak-label lookup cannot depend on a single manifest.

The comparison tool should build a lookup from:

- `zzy_mid_100000_patch768_center_paper_v1` manifests for `100000x` images
- `zzy_mid_50000_train34_test76_paper_stage_v1` manifests for `50000x` images

For consistency with model predictions, the tool should:

1. resolve the original weak-label full-image mask
2. center-crop or pad it to the same `768 x 768` patch size
3. apply the same center detail crop used for the SEM image and model masks

That keeps the weak-label tile aligned with every model tile.

---

## 3. Non-Goals

This update does not:

- retrain any model
- change backbone or inference threshold
- introduce manual ROI selection per image
- replace the earlier full-field comparison outputs

It only adds a more legible cropped comparison view for the current desktop review workflow.
