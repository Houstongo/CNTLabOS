# V3 Topology-Clean Multi-Stat Curvature Design

**Goal:** Replace the current `extract_all()` branch-metric path with a V3-only topology-clean workflow that computes multiple curvature aggregations in one pass.

**Scope**
- Keep `CNTSegNet-SLICE -> skeleton -> topology-clean -> V3 branches` as the primary feature path.
- Add branch-curvature caching so `P50 / P75 / mean / trimmed mean` are computed once per branch.
- Aggregate each stat with both `length` and `sqrt(length)` weights.
- Keep direct `calculate_curvature_v2()` / `calculate_waviness_v2()` helpers available for compatibility, but stop using them inside `extract_all()`.

**Primary Output**
- Main label stays V3-style and uses `P75 + sqrt(length)` aggregation.
- New metrics:
  - `curvature_nm_v3_p50_length`
  - `curvature_nm_v3_p50_sqrt_length`
  - `curvature_nm_v3_p75_length`
  - `curvature_nm_v3_p75_sqrt_length`
  - `curvature_nm_v3_mean_length`
  - `curvature_nm_v3_mean_sqrt_length`
  - `curvature_nm_v3_trimmed_mean_length`
  - `curvature_nm_v3_trimmed_mean_sqrt_length`

**Compatibility**
- Preserve `curvature_v3` and `curvature_nm_v3` as aliases of `P75 + sqrt(length)`.
- Preserve `curvature_v2` / `curvature_nm_v2` in `extract_all()` as compatibility aliases derived from the new bundle:
  - label from `P50 + length`
  - value from `curvature_nm_v3_p50_length`
- Do not remove the old `calculate_curvature_v2()` / `calculate_waviness_v2()` methods in this change.
- Stop computing `waviness_v2` inside `extract_all()` to keep the default path fast; emit `None` for `waviness_ratio_v2` and `tortuosity_v2`.

**Performance Design**
- One branch collection pass.
- One per-branch curvature sampling pass.
- All branch stats cached on branch dicts.
- Final output metrics are reduced from cached stats only; adding `length` and `sqrt(length)` versions should add negligible overhead compared with branch tracing.

**Risks**
- Some offline tools may still expect `waviness_ratio_v2` / `tortuosity_v2` to be numeric when reading `extract_all()` output.
- Existing V2 comparison scripts remain available but will no longer match the default `extract_all()` path.
