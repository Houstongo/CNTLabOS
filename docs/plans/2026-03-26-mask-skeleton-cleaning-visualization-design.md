# Mask Skeleton Cleaning Visualization Design

## Goal

Create a lightweight research visualization over 6 images from `D:\CNTDATA\coredata\u` that shows how a simple “keep large and slender objects, drop small fragmented noise” cleaning rule affects mask- and skeleton-based geometry preparation.

## Scope

- Reuse the existing `FeatureExtractor` preprocessing and threshold-based mask generation.
- Reuse the existing mask-based visualization style.
- Do not modify the main batch-analysis pipeline.
- Do not run the segmentation model.
- Generate visualization only.

## Recommended Approach

Use a standalone script that:

1. reads 6 images from `coredata/u`
2. extracts ROI and threshold mask with current `FeatureExtractor`
3. skeletonizes the mask with the current analysis pipeline
4. scores connected mask objects using lightweight geometry features:
   - area
   - elongation
   - skeleton length
5. keeps “large and slender” objects and removes “small and fragmented” objects
6. renders a five-panel report per image

## Panel Layout

1. original ROI
2. binary mask
3. skeleton
4. connected-object view before cleaning
5. cleaned keep/drop visualization

## Selection Rule v1

For each connected mask object, compute:

- `area_px`
- `elongation`
- `skeleton_length_px`

Keep an object if it is sufficiently:

- large
- elongated
- long in skeleton length

This is a visualization-first heuristic, not a final production metric.

