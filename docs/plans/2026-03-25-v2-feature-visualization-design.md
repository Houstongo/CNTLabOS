# V2 Feature Visualization Design

## Goal

Build a visual, step-by-step explanation panel for the latest V2 feature pipeline so we can manually judge whether the mask, diameter, curvature, and waviness workflow looks trustworthy on real SEM images.

## Approved Scope

- Use the latest paper-repro `cldice` checkpoint to generate the mask
- Visualize the latest real analysis path, not the old `AlgorithmVisualizer`
- Include diameter visualization in the same panel
- Start with 3 `100000x` images
- Add concise captions below each step

## Design

Create a standalone reporting tool that reconstructs the real feature-extraction flow for one image at a time and saves a large annotated panel.

Each panel should contain:

1. ROI original image
2. Paper-repro probability map
3. Final binary mask
4. Skeleton overlay
5. Junction removal / branch split visualization
6. Ordered and smoothed centerline visualization
7. Distance transform with representative diameter circles
8. Curvature sampling visualization on smoothed centerlines
9. Waviness/tortuosity visualization on a representative branch
10. Final summary box with:
   - raw pre-guard geometry values
   - final system-reported values
   - explanation of the low-magnification guard for `10000x`

## Why This Design

- It reflects the latest actual pipeline instead of a demo-only visualizer
- It makes low-magnification `0` / `N/A` outputs explainable
- It lets us inspect not just the final number, but whether the geometry being measured is sensible

## Data Source

- Input images: 3 representative `100000x` images
- Mask source: `cnt_paper_repro_100000x_center768_cldice_seed42/best_model.pth`
- Geometry source: `FeatureExtractor` with real V2 helpers

## Expected Outcome

At the end we should have one folder containing 3 annotated V2 process panels and a summary manifest, enough for manual review of whether the latest pipeline is scientifically believable.
