# CNT Paper Repro Task-Adapted Backbone Design

**Date:** 2026-03-28

**Goal**

Add a new structure-enhanced backbone variant to the `cnt_paper_repro` pipeline while keeping the original `ResNet34UNet` unchanged for direct comparison. The new backbone should stay within the `Exp C` model family and therefore reuse the existing `cnt_paper_repro` staged loss framework instead of importing the `cnt_loss_compare` loss grid.

The new backbone should combine the following structural adjustments on top of the current `ResNet34UNet` design:

- reduce early downsampling
- strengthen shallow high-resolution feature reuse
- replace the plain `1 x 1` output head with `3 x 3 + 1 x 1`

The new structure should be compared in the smallest clean matrix:

- original baseline
- task-adapted baseline
- original `Exp C`
- task-adapted `Exp C`

---

## 1. Correct Scope

The correct target for this work is the `cnt_paper_repro` module, not `cnt_loss_compare`.

That matters because `Exp C` already uses:

- a `ResNet34UNet` backbone
- direct single-channel input adaptation
- a staged training schedule built around `Dice`, `orientation`, and optional `clDice`

Therefore the backbone change should be implemented inside the paper-reproduction model family and evaluated under the existing paper-repro training protocol.

---

## 2. What Already Exists In `Exp C`

The current `ResNet34UNet` already satisfies two of the previously discussed ideas:

- single-channel SEM input is already handled by replacing the encoder stem convolution
- shallow skip reuse already exists because the decoder fuses the final shallow feature map `x0`

This means the task-adapted version should not pretend to add those two features from scratch. Instead, it should focus on the remaining true upgrade points:

- reduce information loss in the earliest stage
- make shallow high-resolution information more influential
- refine the segmentation head before the final logit projection

---

## 3. Recommended Backbone Changes

### 3.1 Reduced Early Downsampling

The current `ResNet34UNet` still uses the standard ResNet-style early downsampling path:

- stem convolution with stride `2`
- followed by max pooling before `layer1`

For thin CNT structures, this can remove useful local detail too early. The task-adapted version should therefore preserve more spatial detail in the encoder front-end by reducing the effective early downsampling strength.

The intended effect is not to redesign the encoder, but to retain more fine linear structure before deeper semantic compression.

### 3.2 Stronger Shallow High-Resolution Reuse

The current decoder already uses shallow skip input at `dec1`, but the final reconstruction still relies on a relatively light last-stage fusion. The task-adapted version should strengthen this path so shallow structure cues contribute more strongly near the output.

The cleanest way to do that is:

- keep the existing shallow skip path
- add one more light refinement stage near the output
- preserve the original feature resolution as long as possible

This improves continuity and thin-line localization without making the model family unrecognizable.

### 3.3 Segmentation Head Upgrade

The current backbone ends with a plain `1 x 1` convolution. That is simple and stable, but it cannot model any local spatial refinement at the output stage.

The task-adapted version should replace this with:

- `3 x 3` convolution
- normalization and activation
- `1 x 1` projection

This adds a controlled amount of local context before the final logit map while keeping the head lightweight.

---

## 4. Parallel Model Strategy

The original `ResNet34UNet` must remain untouched so historical runs stay reproducible.

The recommended implementation is:

- keep `ResNet34UNet` unchanged
- add a second parallel model class, such as `ResNet34UNetTaskAdapted`
- route model construction from config

This avoids mixing old and new behavior in one class and keeps comparison clean in both experiments and thesis writing.

---

## 5. Experimental Matrix

The first comparison should stay intentionally small:

1. original baseline
2. task-adapted baseline
3. original `Exp C`
4. task-adapted `Exp C`

This is the best first comparison because:

- it isolates backbone change
- it preserves the original paper-repro loss families
- it avoids conflating structure changes with the no-orientation and ridge branches

`noOri` and `Exp D` style variants can be added later if the task-adapted backbone proves useful.

---

## 6. Training Protocol

For this phase, the new backbone should reuse the current `cnt_paper_repro` training schedules rather than inventing a new one.

That means:

- task-adapted baseline should mirror `paper_100000x.yaml`
- task-adapted `Exp C` should mirror `paper_100000x_cldice.yaml`
- smoke validation should mirror `paper_100000x_smoke.yaml`

This keeps the comparison fair: same manifests, same staged losses, same inference threshold, different backbone only.

---

## 7. Non-Goals

This update does not:

- replace the original `ResNet34UNet`
- merge `cnt_loss_compare` and `cnt_paper_repro`
- introduce `BCE` into the paper-repro training line
- expand immediately into `noOri` or `ridge` variants

The goal is a controlled backbone comparison inside the existing `Exp C` family.
