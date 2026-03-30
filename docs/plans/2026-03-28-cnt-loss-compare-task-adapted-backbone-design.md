# CNT Loss Compare Task-Adapted Backbone Design

**Date:** 2026-03-28

**Goal**

Keep the current `CNTSegNet` baseline untouched, and add a new task-adapted backbone variant for weak-label CNT segmentation experiments. The new backbone should combine four structural changes in one version:

- grayscale-aware single-channel stem
- reduced early downsampling
- stronger shallow skip reuse
- `3 x 3 + 1 x 1` segmentation head

The new structure should then be evaluated under three loss settings:

- `Dice + BCE + orientation`
- `Dice + BCE + clDice`
- `Dice + BCE + orientation + clDice`

---

## 1. Why A New Parallel Backbone

The existing `cnt_loss_compare` experiments already provide a stable reference point built around the current `CNTSegNet` backbone and several loss combinations. Those historical results should remain reproducible.

For that reason, the safest design is to introduce a new model class in parallel rather than mutating the existing `CNTSegNet` implementation in place. This keeps all prior runs, configs, and checkpoints interpretable while allowing the new structure to be compared fairly against the existing baseline.

The new backbone is intended as a task-adapted variant rather than a fully new segmentation family. Its purpose is to better match the current data regime:

- SEM grayscale input
- thin, elongated CNT structures
- weak binary labels rather than precise manual masks

---

## 2. Recommended Structural Changes

### 2.1 Grayscale-Aware Stem

The current dataset loader converts grayscale SEM crops into pseudo-RGB tensors to fit the default `ResNet50` input convention. For the new model, the input stem should instead consume a single-channel tensor directly.

This change is expected to:

- better match the actual SEM data distribution
- avoid redundant channel replication
- make the first convolution more task-specific

If pretrained weights are later enabled, the stem should still support single-channel initialization by averaging the RGB kernel weights into one channel.

### 2.2 Reduced Early Downsampling

The standard `ResNet50` stem is aggressive for thin CNT structures because the initial large-kernel stride and max pooling can erase fine linear detail before the decoder ever sees it.

The new design should therefore reduce information loss in the earliest stage by:

- lowering the first downsampling strength
- weakening or removing the first max pooling stage

The exact goal is not to redesign the whole encoder, but to preserve more high-resolution structure in the early feature hierarchy.

### 2.3 Stronger Shallow Skip Reuse

The baseline decoder already uses skip connections from deeper encoder stages, but CNT continuity and thin-line localization depend heavily on shallow features.

The new model should add or strengthen shallow skip reuse so that the decoder receives:

- stronger local edge cues
- finer line continuity cues
- more precise thin-structure localization

This can be implemented by reusing a stem-level feature map or an equivalently shallow feature stage before heavy downsampling.

### 2.4 Segmentation Head Upgrade

The baseline model ends with a plain `1 x 1` convolution. That is parameter-efficient, but it only mixes channels and does not refine local neighborhoods.

The new model should replace this with a lightweight refinement head:

- `3 x 3 conv`
- normalization and activation
- `1 x 1 conv`

This keeps the head simple while adding a small amount of local spatial refinement before the final logit map.

---

## 3. Loss Matrix

The backbone change should be isolated from the loss study by running the same new structure under three explicit loss variants:

1. `Dice + BCE + orientation`
2. `Dice + BCE + clDice`
3. `Dice + BCE + orientation + clDice`

This keeps the experiment interpretable:

- structure-only change relative to the original `CNTSegNet` family
- no ambiguity about which improvement comes from loss and which comes from backbone
- direct comparison against existing `cnt_loss_compare` runs

The existing baseline configs should remain unchanged, and the new configs should be added as separate experiment names.

---

## 4. Data And Interface Changes

Because the new backbone should accept grayscale tensors directly, the data pipeline must support both of these modes:

- legacy pseudo-RGB input for the original `CNTSegNet`
- direct single-channel input for the new task-adapted backbone

This should be implemented as a config-driven switch rather than as two separate dataset implementations. That keeps the data path aligned and reduces accidental divergence across experiments.

The training entry point should also be extended so config files can select between:

- the legacy backbone
- the new task-adapted backbone

without changing the overall experiment workflow.

---

## 5. Naming

To avoid confusion with the historical baseline, the new model should use an explicit parallel name such as:

- `cntsegnet_task_adapted`
- or `cntsegnet_v2`

The experiment names should make both the backbone and the loss family visible, for example:

- `exp_task_adapted_orientation`
- `exp_task_adapted_cldice`
- `exp_task_adapted_orientation_cldice`

This makes later paper figures and result tables easier to read.

---

## 6. Non-Goals

This update does not:

- replace the original `CNTSegNet`
- introduce transformer backbones or major encoder family changes
- change dataset manifests or regenerate labels
- redesign the loss-comparison framework outside the new config additions

The goal is a controlled structural upgrade, not a full experiment-platform rewrite.
