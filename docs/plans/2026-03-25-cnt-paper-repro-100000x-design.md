# 100000x CNTSegNet Paper Reproduction Design

**Date:** 2026-03-25

**Goal**

Build a dedicated paper-reproduction pipeline for CNT-SEM segmentation on the curated `100000x` dataset, using a `ResNet34-U-Net` backbone, `768x768` patches, adaptive-threshold weak labels, FFT-based orientation supervision, and the staged training schedule described in the CNTSegNet paper.

---

## 1. Scope

This line is intentionally separate from the existing loss-comparison framework under `experiments/cnt_loss_compare`.

The purpose of this line is narrow:

- answer whether the original CNTSegNet training recipe can be reproduced on the current `100000x` CNT-SEM dataset
- keep the backbone, loss, patching, and thresholding close to the paper description
- avoid mixing custom clDice or auxiliary structural losses into the reproduction result

This line is not intended to replace the later custom-method comparison. It exists to establish a clean paper-style baseline first.

---

## 2. Fixed Source Dataset

The image source of truth is the already curated dataset:

- [zzy_mid_100000_train50_test50_v2](D:\CNTDATA\CNTA_ML_Project\experiments\cnt_loss_compare\datasets\zzy_mid_100000_train50_test50_v2)

This source is reused because:

- the sample quality has already been manually checked
- weak labels already exist
- image-level train/test boundaries are already fixed
- it avoids re-curating the same 100000x images again

The reproduction pipeline will not rewrite the source dataset. It will derive patch assets from it into a separate paper-reproduction directory.

---

## 3. Derived Dataset Layout

The paper-reproduction assets will live under:

- `D:\CNTDATA\CNTA_ML_Project\experiments\cnt_paper_repro`

Planned structure:

- `datasets/`
- `datasets/zzy_mid_100000_patch768_paper_v1/`
- `datasets/zzy_mid_100000_patch768_paper_v1/train/images`
- `datasets/zzy_mid_100000_patch768_paper_v1/train/masks_wcntsegnet`
- `datasets/zzy_mid_100000_patch768_paper_v1/test/images`
- `datasets/zzy_mid_100000_patch768_paper_v1/test/masks_wcntsegnet`
- `datasets/zzy_mid_100000_patch768_paper_v1/reserve/images`
- `datasets/zzy_mid_100000_patch768_paper_v1/reserve/masks_wcntsegnet`
- `datasets/zzy_mid_100000_patch768_paper_v1/patches/...`
- `runs/`
- `configs/`

Patch extraction rules:

- use the full image and its aligned `masks_wcntsegnet`
- remove the bottom SEM information bar by using the existing ROI convention
- extract fixed `768x768` patches
- keep image patch and weak-mask patch perfectly aligned
- generate explicit patch manifests for train, test, and reserve

The first version should prefer deterministic patch extraction over complex hard-mining logic.

---

## 4. Model Architecture

The current project backbone in [backbone.py](D:\CNTDATA\CNTA_ML_Project\experiments\cnt_loss_compare\backbone.py) is not paper-faithful. It uses a `ResNet50` encoder with ASPP and a custom decoder.

For the paper-reproduction line, the backbone will be replaced with:

- `ResNet34-U-Net`
- single-channel input
- single-channel sigmoid-style output logits

The implementation should be self-contained inside `experiments/cnt_paper_repro`, rather than mutating the current loss-comparison backbone.

Key requirements:

- no ASPP
- no custom attention modules unless required by a standard U-Net decoder design
- encoder from `torchvision.models.resnet34`
- decoder kept simple and auditable

---

## 5. Weak Label Policy

For this phase, weak labels will continue to come from the currently accepted `WCNTSegNET` masks already generated on the fixed `100000x` source dataset.

This is a practical decision:

- the user has already judged current weak-label quality as good enough to proceed
- the immediate task is to reproduce the paper-style training pipeline, not to reopen weak-label curation

This means the reproduction is still a paper-style training protocol on the current weak-label source, not a byte-for-byte recreation of the original paper dataset.

---

## 6. Orientation Supervision

The orientation loss must move closer to the paper description than the current Sobel-histogram approximation.

Planned implementation:

- compute the orientation histogram from the input image using `FFT + radial sum`
- compute the orientation histogram from the predicted class likelihood map, not the binarized mask
- use `360` bins by default
- normalize the histograms before computing the loss
- use MSE in the form described by the paper

This module must be isolated and configurable, because its numerical scale determines whether the `1e-7` weighting behaves as intended.

---

## 7. Training Schedule

The training schedule should follow the paper-styled two-phase routine:

- Phase 1: `Dice`, `Adam`, `lr=5e-4`, `6 epochs`
- Phase 2: `0.6 * Dice + 1e-7 * Orientation-MSE`, `Adam`, `lr=5e-4`, `3 epochs`

The best checkpoint rule should be simple and explicit:

- save checkpoints every epoch
- choose best checkpoint by validation Dice first
- also record validation orientation loss and other auxiliary metrics for inspection

This choice keeps model selection stable even when the orientation loss scale is still being verified.

---

## 8. Inference and Reporting

Inference should default to the paper-reported threshold:

- prediction binarization threshold: `0.7`

The reproduction line should export:

- raw likelihood maps
- binary masks at `0.7`
- per-image test predictions
- side-by-side visualizations
- run summaries and CSV tables

Because weak labels are not ground truth, weak-label Dice and IoU should be treated as fitting diagnostics, not final scientific proof. Visual inspection remains a primary output for this phase.

---

## 9. Verification Strategy

Before claiming the pipeline is ready, we need a smoke test that proves:

- patch extraction is deterministic and aligned
- model forward pass works on `768x768` single-channel patches
- FFT-based orientation loss is finite on real patches
- staged training can run at least one short cycle on GPU in `lab_agent`
- evaluation and visualization can restore outputs from saved checkpoints

---

## 10. Non-Goals

The following are explicitly out of scope for this phase:

- clDice experiments
- auxiliary structural losses
- mixed-magnification training
- replacing the current production segmentation backend
- rewriting the existing `cnt_loss_compare` pipeline

Those belong to later custom-comparison work after the paper-style baseline is stable.
