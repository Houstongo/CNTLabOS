# Curvature V3 Design

## Goal

Correct the legacy `curvature_nm` physical scaling and add a higher-separation `curvature_nm_v3` that stays on the ordered-centerline pipeline while preserving the existing `curvature_nm_v2` output.

## Context

- Legacy `curvature_nm` currently uses a fixed `px_per_nm = 1/15`, which is inconsistent with the calibrated `px_per_um` already computed for each image.
- `curvature_nm_v2` is physically more plausible, but it is intentionally conservative because it cuts junctions, smooths paths, uses branch medians, and then length-weights the global aggregate.
- The current user goal is to preserve a robust baseline while adding a more discriminative curvature value for comparing images.

## Approaches Considered

### 1. Replace `curvature_nm_v2` directly

- Pros: one output to maintain
- Cons: breaks comparability with current V2 outputs and makes it harder to separate “robust” vs “discriminative” behavior

### 2. Add `curvature_nm_v2_relaxed` only

- Pros: minimal change and already explored offline
- Cons: names the behavior as a relaxed copy of V2 instead of a distinct comparison-oriented metric

### 3. Keep legacy + V2, add a new V3 metric

- Pros: preserves backward comparison, keeps V2 as the robust metric, adds a comparison-oriented metric
- Cons: one more field to document and compare

## Recommended Design

Use approach 3.

### Legacy curvature correction

- Keep the legacy skeleton/component traversal behavior unchanged.
- Replace the hardcoded `px_per_nm = 1/15` with calibrated conversion:
  - `px_per_nm = self.px_per_um / 1000.0`
- This keeps the legacy geometry logic stable while fixing the physical unit conversion.

### Curvature V3

- Reuse the V2 ordered-branch pipeline:
  - skeleton
  - junction cut
  - ordered branch tracing
  - path smoothing
- Make V3 more discriminative by changing only the aggregation behavior:
  - use a lower branch inclusion threshold than V2
  - compute point-wise curvature on sampled ordered points
  - summarize each branch with `p75` instead of `median`
  - aggregate branches with `sqrt(path_length_px)` weights instead of pure path-length weights

### Output fields

- Preserve:
  - `curvature`
  - `curvature_nm`
  - `curvature_v2`
  - `curvature_nm_v2`
- Add:
  - `curvature_v3`
  - `curvature_nm_v3`

### Validation

- Update unit tests to verify:
  - legacy curvature scales with calibrated `px_per_um`
  - V3 remains near-zero for a straight skeleton
  - V3 is positive for a wavy skeleton
  - V3 is at least as sensitive as V2 on a wavy skeleton

### Sample comparison

- Generate a 5-image report that shows:
  - image preview
  - legacy calibrated curvature
  - V2 curvature
  - V3 curvature
  - density
  - waviness ratio

