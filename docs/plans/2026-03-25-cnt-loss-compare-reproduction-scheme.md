# CNT Loss-Comparison Reproduction Scheme

**Date:** 2026-03-25

**Goal**

Build a controlled and reproducible experiment protocol for CNT-SEM segmentation that compares four loss designs under the same backbone, same fixed dataset, same training setup, and same evaluation protocol.

---

## 1. Research Question

We want to test whether orientation-guided weak-supervision loss is less effective than topology/structure-preserving loss on CNT-SEM images with:

- local crossings
- direction heterogeneity
- fragmented thin structures
- noisy and structurally unstable regions

The comparison target is not a new backbone.  
The comparison target is only the loss design.

---

## 2. Fixed Backbone and Fairness Constraints

To keep the comparison fair, the following items must remain fixed across experiments:

- backbone: `CNTSegNet` in [cntsegnet.py](D:\CNTDATA\VLMSAM\cntsegnet.py)
- encoder-decoder structure: unchanged
- input dataset: fixed curated split
- weak labels: fixed `WCNTSegNET` masks
- image preprocessing and normalization: fixed
- optimizer: fixed
- scheduler: fixed
- batch size: fixed
- epochs: fixed
- random seed: fixed
- checkpoint selection rule: fixed
- evaluation metrics: fixed

Only the loss composition and loss weights are allowed to change.

---

## 3. Phase Structure

### Phase 1: Main Reproduction

Use the fixed `100000x` dataset as the main controlled benchmark.

Dataset root:
[zzy_mid_100000_train50_test50_v2](D:\CNTDATA\CNTA_ML_Project\experiments\cnt_loss_compare\datasets\zzy_mid_100000_train50_test50_v2)

Current fixed assets:

- images: `102`
- weak labels: `102`
- weak-label success: `102/102`

Current split before model training:

- `train`: `50`
- `test`: `50`
- `reserve`: `2`

### Phase 2: Secondary Reproduction

After the `100000x` experiments are stable, repeat the same protocol on `50000x`.

Purpose:

- check whether conclusions remain consistent at another scale
- rule out the possibility that the observed effect is specific to `100000x`

### Phase 3: Optional External Check

`XR 20000x` can be used later as an external or domain-shift check, not as the first controlled benchmark.

Reason:

- acquisition style differs from `ZZY`
- primary magnification does not match the `ZZY` main benchmark
- it is better suited for external robustness checking than the first fair loss ablation

---

## 4. Training Split Strategy

For actual model training, the current `50 train / 50 test / 2 reserve` split should be converted into:

- `train`: `40`
- `val`: `10`
- `test`: `50`
- `reserve`: `2`

### Why

- `val` is needed to choose the best checkpoint fairly
- `test` should remain untouched until final reporting
- `reserve` remains outside the optimization loop for qualitative checks and hard-case review

### Split Rule

`val` should be created only from the current `train` split.

This keeps:

- the current test benchmark stable
- previous data curation work intact
- later comparison between experiments simple and auditable

### Reproducibility Rule

The `train -> train/val` subdivision must be:

- deterministic
- fixed by seed
- saved to disk as explicit manifests

Recommended first split:

- stratify as much as possible by current process-condition grouping and sample distribution
- fixed random seed: `42`

---

## 5. Weak Supervision Assets

The weak-label source of truth for Phase 1 is:

- images under `train/test/reserve/images`
- weak masks under `train/test/reserve/masks_wcntsegnet`
- metadata in [labels_manifest.csv](D:\CNTDATA\CNTA_ML_Project\experiments\cnt_loss_compare\datasets\zzy_mid_100000_train50_test50_v2\labels_manifest.csv)
- stats in [label_stats.csv](D:\CNTDATA\CNTA_ML_Project\experiments\cnt_loss_compare\datasets\zzy_mid_100000_train50_test50_v2\label_stats.csv)

Important label convention:

- masks are stored as full-image masks
- the bottom SEM metadata bar is zeroed out
- this avoids size mismatch with the original image and keeps the dataset compatible with later loaders

---

## 6. Four Experiments

### Exp-A: CNTSegNet Baseline

Target idea:

- weak mask supervision
- orientation-guided loss

Recommended loss form:

- `Dice + BCE + Orientation`

Notes:

- if the original CNTSegNet paper formulation cannot be matched exactly from the current repository, this experiment should be implemented as the closest practical approximation under the same backbone and weak labels
- the approximation must be documented clearly in config and summary

### Exp-B: No-Orientation Baseline

Target idea:

- remove orientation prior
- keep simple segmentation supervision only

Loss:

- `Dice + BCE`

Purpose:

- isolate whether orientation guidance itself helps or hurts on this dataset

### Exp-C: Ours-v1

Target idea:

- add connectivity/elongated-structure preservation

Loss:

- `Dice + BCE + clDice`

Purpose:

- compare global orientation regularization against topology-aware regularization

### Exp-D: Ours-v2

Target idea:

- add a second structural constraint on top of clDice

Loss:

- `Dice + BCE + clDice + AuxStructural`

Auxiliary structural loss first implementation:

- skeleton loss preferred if implementation is stable
- boundary loss acceptable as the first extensible placeholder

The important point is not to maximize novelty in one step.  
The important point is to keep the interface extensible and the comparison clean.

---

## 7. Recommended Training Protocol

### Backbone

- model: `CNTSegNet`
- file source: [cntsegnet.py](D:\CNTDATA\VLMSAM\cntsegnet.py)
- no backbone modification

### Input Size

Recommended first protocol:

- resize/pad to `512 x 512`

Reason:

- matches the current operational CNTSegNet training pattern in `VLMSAM`
- keeps GPU cost controlled
- makes the four-loss comparison easier to execute quickly

### Initialization

Use one fixed initialization rule for all experiments:

- same pretrained encoder behavior
- same seed
- same parameter initialization path

### Optimizer

Recommended first protocol:

- `AdamW`
- fixed learning rate for all experiments

### Scheduler

Recommended first protocol:

- cosine annealing

### Epochs

Recommended first protocol:

- keep one fixed epoch budget across all four experiments

Suggested starting point:

- `80` to `120` epochs

If runtime is tight, start from `80`.

### Batch Size

Use the same batch size for all four experiments.

Suggested starting point:

- `4`

or

- the largest stable size the current GPU can hold without changing across experiments

### Seeds

Two-stage recommendation:

Stage 1:

- run one fixed seed first to validate the pipeline

Stage 2:

- repeat with `3` seeds if time allows for more robust conclusions

Recommended seed set:

- `42`
- `52`
- `62`

If only one seed is feasible initially, use `42`.

---

## 8. Checkpoint Rule

Best model must be selected only by validation performance.

Recommended rule:

- choose best checkpoint by `val clDice`

Secondary record:

- `val Dice`
- `val IoU`

Reason:

- this project focuses on thin-structure continuity
- `clDice` is more aligned with the scientific question than pixel accuracy alone

If `clDice` is not stable in the earliest implementation, use:

- primary: `val Dice`
- secondary: `val clDice`

but document the temporary fallback explicitly.

---

## 9. Evaluation Protocol

### Pixel-Level Metrics

Report on the fixed `test` split:

- Dice
- IoU
- Precision
- Recall

### Structure-Level Metrics

Report on the fixed `test` split:

- clDice
- skeleton precision
- skeleton recall

Optional if stable:

- fragment count
- connected-component error
- breakage / discontinuity indicator

### Reporting Rule

For the paper-ready table, each experiment should output:

- one row per experiment
- one column per metric
- checkpoint path
- selected epoch

If multi-seed runs are added later, report:

- mean ± std

---

## 10. Hard Subset Strategy

The hard subset is important for this project because the main hypothesis is local-structure failure rather than average-case segmentation only.

### First Version

Create a hard-subset annotation file after the main pipeline is runnable.

Recommended first implementation:

- select from the current `test` split
- mark samples manually or semi-manually using:
  - crossing density
  - local direction variation
  - branch/junction count
  - fragmentation tendency

### Output Form

Recommended format:

- CSV file keyed by `image_id` or `image_filename`
- column: `difficulty`
- values: `easy`, `hard`

This keeps later evaluation scripts simple.

---

## 11. Output Layout

Recommended experiment root:

`D:\CNTDATA\CNTA_ML_Project\experiments\cnt_loss_compare\runs\`

Recommended structure:

- `configs/`
- `splits/`
- `runs/exp_cntsegnet_seed42/`
- `runs/exp_no_orientation_seed42/`
- `runs/exp_ours_cldice_seed42/`
- `runs/exp_ours_full_seed42/`
- `results/tables/`
- `results/figures/`
- `results/visualizations/`

Each run should contain:

- config snapshot
- train log
- val log
- best checkpoint
- last checkpoint
- per-epoch metrics CSV
- final test metrics JSON/CSV

---

## 12. Recommended Execution Order

### Step 1

Freeze the final Phase 1 training split:

- derive `val` from current `train`
- save explicit manifests

### Step 2

Build one unified training entry that:

- loads the fixed split
- uses one fixed backbone
- selects losses by config

### Step 3

Implement the four loss configurations:

- Exp-A
- Exp-B
- Exp-C
- Exp-D

### Step 4

Run one seed on `100000x`.

### Step 5

Generate:

- quantitative table
- qualitative comparison figure
- hard-case preview set

### Step 6

After pipeline stability is confirmed, reproduce on `50000x`.

---

## 13. Immediate Recommendation

The next concrete implementation step should be:

1. derive a fixed `40 train / 10 val / 50 test / 2 reserve` split from the current `100000x v2` dataset
2. build a unified training/evaluation framework inside `CNTA_ML_Project/experiments/cnt_loss_compare/`
3. keep `CNTSegNet` as the only backbone
4. make the loss configurable so all four experiments differ only in the loss block

This is the cleanest path to a publishable comparison.

---

## 14. What Counts as a Successful Reproduction

A successful Phase 1 reproduction means:

- the same dataset and weak labels can be rerun later without ambiguity
- all four experiments share one training framework
- all four experiments produce comparable logs and metrics
- best checkpoints are chosen by a fixed validation rule
- the final comparison can directly answer:
  - whether orientation-guided loss helps on this dataset
  - whether clDice/structural losses better preserve continuity in crossing and unstable local structures
