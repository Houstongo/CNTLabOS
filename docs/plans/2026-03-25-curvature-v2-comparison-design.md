# Curvature V2 and Waviness V2 Comparison Design

## Goal

Add a more trustworthy centerline-based `v2` metric path for curvature and waviness, then use it to compare the approved new paper-repro checkpoint against the frozen March 24, 2026 `WCNTSegNET` batch results before any database write-back.

## Approved Decision

Use a dual-track strategy.

- Keep the current metrics unchanged for compatibility:
  - `curvature`
  - `curvature_nm`
  - `tortuosity`
  - `waviness_ratio`
  - `waviness_height_nm`
  - `waviness_wavelength_nm`
  - `waviness_branches`
- Add a new `v2` metric family for the corrected centerline pipeline:
  - `curvature_v2`
  - `curvature_nm_v2`
  - `tortuosity_v2`
  - `waviness_ratio_v2`
  - `waviness_height_nm_v2`
  - `waviness_wavelength_nm_v2`
  - `waviness_branches_v2`

This lets us compare new results against yesterday's batch outputs without destroying historical comparability.

## Current Problems

### Existing FeatureExtractor curvature path

- `FeatureExtractor._collect_components()` returns raw connected-component coordinates in pixel-scan order, not true path order.
- `FeatureExtractor.calculate_curvature()` uses those unordered coordinates directly in a three-point curvature calculation.
- Curvature unit conversion is currently hard-coded with `1 / 15` instead of using the calibrated `px_per_um`.
- The current skeleton path is vulnerable to branch junctions, short spurs, and zig-zag skeleton noise.

### Comparison workflow gap

- The existing batch comparison artifacts under `reports/zzy_wcntsegnet_full_batch_20260324/` preserve yesterday's metric values and image-level output folders.
- Today's approved new algorithm is the paper-repro `cldice` checkpoint:
  - `D:\CNTDATA\CNTA_ML_Project\experiments\cnt_paper_repro\runs\cnt_paper_repro_100000x_center768_cldice_seed42\best_model.pth`
- There is no current tool that compares this checkpoint against the frozen March 24 `WCNTSegNET` batch baseline while showing corrected curvature and waviness values.

## Design Decision

Implement the corrected centerline logic inside `FeatureExtractor`, but expose it as additive `v2` outputs only. Build a separate offline comparison report for decision-making. Do not change the formal database schema or overwrite existing `images` feature columns in this step.

## Scope

### In scope

- Add ordered-branch extraction helpers to `FeatureExtractor`
- Add spur pruning / branch splitting / centerline smoothing for the `v2` path
- Add calibrated `v2` curvature and waviness calculations
- Return both legacy metrics and `v2` metrics from `extract_all()`
- Add an offline comparison tool that:
  - reads yesterday's frozen `WCNTSegNET` batch summary
  - runs the approved paper-repro checkpoint on the same image set
  - computes `v2` curvature and waviness for the new algorithm output
  - renders per-image comparison panels
  - writes CSV and JSON summaries for review

### Out of scope

- Changing the meaning of existing database columns
- Adding database columns in `images`
- Rewriting old historical batch results
- Making `v2` metrics the default in frontend or backend API responses for production users

## Architecture

### 1. Ordered centerline extraction for v2

The new `v2` path should build branch-wise ordered centerlines from the skeleton instead of treating each connected component as an unordered point cloud.

Proposed flow:

1. Compute 8-neighbor counts on the skeleton.
2. Identify:
   - endpoints: neighbor count `== 1`
   - junctions: neighbor count `>= 3`
3. Remove junction pixels temporarily to split the skeleton into branch candidates.
4. Label the branch candidates.
5. Drop very short branches as spurs.
6. Reconnect branch endpoints to nearby junction context only for ordering purposes when needed.
7. Trace each candidate branch into an ordered point path.
8. Smooth ordered coordinates before downstream geometry.

This can reuse the proven endpoint tracing idea already present in the visualizer code, but the implementation must live in `src/analysis/feature_extractor.py` so the real analysis pipeline benefits from it.

### 2. Curvature v2

`curvature_nm_v2` should be computed from ordered, smoothed centerline points using calibrated units.

Design details:

- Sample points along the ordered path with a spacing tied to the branch scale.
- Use the three-point circumcircle formula on ordered path triples.
- Convert from pixel curvature to `nm^-1` using:
  - `px_per_nm = px_per_um / 1000.0`
- Aggregate branch curvature robustly:
  - per branch: median curvature
  - whole image: length-weighted median or weighted average over valid branches
- Keep `curvature_v2` as a label derived from the numeric `curvature_nm_v2`, but treat the numeric value as the primary comparison output.

### 3. Waviness v2

`waviness_ratio_v2` and `tortuosity_v2` should use the same ordered, smoothed branch paths.

Design details:

- Project ordered path coordinates onto the branch main axis
- Detrend lateral displacement
- Smooth the signal
- Detect alternating peaks and troughs
- Compute:
  - `waviness_ratio_v2`
  - `waviness_height_nm_v2`
  - `waviness_wavelength_nm_v2`
  - `waviness_branches_v2`
  - `tortuosity_v2`

The existing waviness implementation already does much of the signal-side work; the main correction is that it must consume ordered branch paths from the new branch extractor.

## Comparison Workflow

### Baseline data source

Freeze yesterday's batch data as the baseline:

- `D:\CNTDATA\CNTA_ML_Project\reports\zzy_wcntsegnet_full_batch_20260324\summary.json`
- `D:\CNTDATA\CNTA_ML_Project\reports\zzy_wcntsegnet_full_batch_20260324\batch_features.csv`

Use the stored baseline values exactly as the historical reference. Do not recompute them in this step.

### New algorithm data source

Run the approved paper-repro checkpoint on the same images:

- config family: `experiments/cnt_paper_repro`
- chosen checkpoint: `cnt_paper_repro_100000x_center768_cldice_seed42/best_model.pth`

### Offline report outputs

For each image, generate:

- original image panel
- yesterday `WCNTSegNET` mask panel if available from the saved output folder
- new algorithm mask panel
- comparison table showing:
  - yesterday baseline curvature / waviness values
  - new algorithm legacy values if useful
  - new algorithm `v2` values
  - deltas versus yesterday baseline

For the batch, generate:

- `summary.json`
- `comparison.csv`
- one image panel per image

## Error Handling

- If a saved baseline mask is missing, fall back to using only the baseline numeric values and current image.
- If a new algorithm inference fails for one image, mark that row as failed and continue.
- If the ordered branch extraction yields no valid branches, return zero-like `v2` values and annotate the row as `no_valid_centerline`.
- Keep legacy metrics available even when `v2` fails, so the comparison report still renders.

## Risks

### Risk: v2 values differ sharply from historical values

This is expected. The goal is to measure whether the new definition is more trustworthy, not to preserve numeric equality.

### Risk: branch splitting becomes too aggressive

If junction removal is too strong, long CNT paths can fragment and depress curvature / waviness estimates.

### Risk: smoothing suppresses real bending

If the smoothing window is too large, genuine local curvature gets flattened.

## Mitigations

- Keep legacy metrics unchanged
- Add focused synthetic tests for ordered centerlines, branchy skeletons, and straight-line edge cases
- Make spur pruning and smoothing scale-aware using `expected_tube_px`
- Keep the offline report explicit about which fields are historical baseline and which are `v2`

## Verification Plan

- Unit-test ordered branch extraction on synthetic straight, sinusoidal, and branchy skeletons
- Verify `curvature_nm_v2` uses calibrated `px_per_um`, not hard-coded constants
- Verify `waviness_ratio_v2` stays near zero for straight centerlines
- Verify the comparison tool can reproduce a full report on the same image ids present in the March 24 baseline summary
- Manually inspect a small sample of comparison panels before deciding on any database migration

## Expected Outcome

At the end of this change we will have:

- a safer `v2` metric path inside the real `FeatureExtractor`
- frozen historical comparability through the old fields
- a batch of side-by-side result images and CSV summaries comparing:
  - yesterday's `WCNTSegNET` batch data
  - today's approved new checkpoint
  - corrected curvature and waviness `v2` outputs

This is the right stopping point before any decision to add database columns or promote `v2` metrics into production analysis.
