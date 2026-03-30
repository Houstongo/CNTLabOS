# V2 V3 Spur Cleanup Integration Design

**Goal:** Insert skeleton cleanup before V2/V3 branch-based feature extraction so short isolated fragments and terminal spurs do not create false junctions and unstable branch splits.

**Scope:** Only branch-based features change in the first pass. `density`, `diameter`, and `alignment` stay on the current raw-mask/raw-skeleton path.

**Design Summary**

Current `FeatureExtractor.extract_all()` builds one skeleton from the binary mask and then derives:

- legacy curvature from the raw skeleton
- V2 branch sets
- V3 branch sets
- V2 waviness/tortuosity

The new design keeps that overall flow, but inserts a reusable cleanup stage immediately after raw skeleton generation:

1. Build `raw_skeleton` from the binary mask.
2. Remove short isolated skeleton components.
3. Remove terminal spurs using endpoint-to-junction tracing with a length limit tied to `expected_tube_px`.
4. Use the cleaned skeleton only for V2/V3 branch preparation and branch-based metrics.
5. Keep legacy curvature, density, diameter, and alignment on the existing path for compatibility.

**Why This Split**

- V2/V3 are sensitive to false junctions because they explicitly split the skeleton into ordered branches.
- Short spurs can turn a smooth trunk into multiple branches before length filtering even starts.
- Density and diameter are primarily mask-driven, so coupling them to spur cleanup would add risk without much value.
- Keeping alignment unchanged in the first pass reduces behavioral drift and keeps the rollout easier to verify.

**New Internal API**

Add reusable helpers inside `FeatureExtractor`:

- `_remove_short_isolated_skeleton_components(...)`
- `_prune_terminal_spurs(...)`
- `_clean_branch_skeleton(...)`

These helpers should return both the cleaned skeleton and cleanup metadata so later tools can surface:

- removed short component count
- removed short pixel count
- removed spur count
- removed spur pixel count
- cleanup thresholds used

**Feature Output Changes**

Add non-breaking extra fields to `extract_all()` output:

- `branch_cleanup_enabled`
- `removed_short_component_count`
- `removed_short_pixel_count`
- `removed_spur_count`
- `removed_spur_pixel_count`
- `spur_length_limit_px`

Existing V2/V3 field names stay the same, but their values will now come from cleaned branch topology.

**Verification Strategy**

- Unit-level: create a synthetic skeleton with one main trunk and one short spur, then confirm cleanup removes the spur and reduces branch fragmentation.
- Focused regression: run existing curvature/waviness tests.
- Smoke comparison: run one small image through `extract_all(external_binary_mask=...)` and compare branch cleanup counters plus V2/V3 outputs.

**Risks**

- Over-pruning could erase real short CNT ends near dense junction regions.
- Existing thresholds may need tuning by magnification.
- Any direct consumer that expects V2/V3 outputs to match old numbers exactly will see drift.

**Initial Risk Controls**

- Keep cleanup limited to branch-based metrics only.
- Use conservative defaults based on the existing spur demo.
- Expose cleanup metadata in the result for fast debugging.
