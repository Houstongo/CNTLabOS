# Kappa-Lite Segment Selection Design

## Goal

Build a research-only, lightweight module that automatically selects the top 10 most measurable CNT centerline segments from a real `100000x` SEM image, so we can evaluate Fiji/Kappa-like curvature measurement without solving full crossing reconnection first.

## Scope

This module is intentionally limited to:

- real `100000x` images
- existing binary mask / skeleton pipeline
- no junction reconnection
- no spline optimization in v1
- no database integration
- no batch-processor integration

It is a research and validation tool only.

## Why This Version

The main risk in a Fiji/Kappa-inspired workflow is not the curvature formula itself. The main risk is choosing the wrong line to measure.

This design therefore prioritizes:

- selecting clean, human-plausible segments
- avoiding junction ambiguity
- keeping computation light enough for repeated validation
- making the selected segments visually inspectable

## Proposed Approaches

### 1. Full-image automatic reconnection first

- Pros: closer to eventual single-CNT geometry analysis
- Cons: highest ambiguity, hardest to debug, slowest to validate

### 2. Manual segment picking first

- Pros: closest to Fiji/Kappa manual use, simplest to trust
- Cons: not automatic, not scalable, weak for algorithm validation

### 3. Automatic top-segment selection from existing ordered branches

- Pros: reuses the current V2 centerline pipeline, avoids junction ambiguity, computationally lighter, easy to visualize
- Cons: measures clean fragments rather than full CNTs

## Recommendation

Use approach 3 for v1-lite.

## Data Flow

1. Start from an existing SEM image.
2. Run the current `FeatureExtractor` preprocessing and binary foreground extraction.
3. Skeletonize the mask.
4. Use the current V2 ordered-branch extraction, which already:
   - cuts junctions
   - orders centerline points
   - smooths centerline paths
5. Treat each ordered branch as a candidate measurable segment.
6. Filter out poor candidates.
7. Score remaining candidates.
8. Select the top 10 non-redundant segments.
9. Compute per-segment geometry metrics.
10. Render a validation panel with the selected segments overlaid.

## Candidate Filtering Rules

Each candidate segment must satisfy:

- minimum point count
- minimum path length
- not too close to the image border
- not too close to a junction endpoint
- valid width sampling along the segment

These rules are meant to approximate “a human would be willing to trace this segment.”

## Segment Score v1

Each candidate gets a lightweight score built from three terms:

- `length_score`
  - favors longer segments
  - saturates to avoid over-rewarding one very long branch

- `junction_distance_score`
  - favors segments whose closest point stays farther from junction regions
  - reinforces the “avoid crossings” strategy

- `width_consistency_score`
  - favors segments with more stable local width from the distance transform
  - acts as a proxy for centerline confidence and local measurement quality

These three scores are combined into one `segment_score`.

## Selection Rules

From the scored candidates:

- sort by `segment_score`
- greedily keep the best candidates
- suppress candidates that overlap too much with already selected segments
- stop after selecting `Top 10`

This keeps the output diverse and avoids the top 10 collapsing onto one local region.

## Per-Segment Outputs

For each selected segment, output:

- `segment_id`
- `score`
- `path_length_px`
- `path_length_nm`
- `span_px`
- `span_nm`
- `ld_ratio`
- `mean_curvature_nm`
- `p90_curvature_nm`
- `mean_width_nm`
- `width_cv`

## Visual Output

Generate one research panel per image containing:

- original ROI
- binary mask
- skeleton / ordered-branch reference
- selected top 10 segments with IDs
- per-segment metrics table

This panel is the main validation output.

## Known Limitations

- selected segments are not full CNT instances
- crossing regions are intentionally excluded rather than resolved
- centerline smoothing still affects curvature magnitude
- results should be interpreted as “clear-segment geometry,” not whole-image average curvature

## Success Criteria

The v1-lite module is successful if:

- the selected top 10 segments look human-plausible
- the selected segments avoid ambiguous crossings
- the per-segment `L/D` and curvature values are stable enough for comparison
- the outputs help decide whether a fuller Kappa-like workflow is worth building

