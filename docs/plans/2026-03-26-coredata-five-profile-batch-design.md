# Coredata Five-Profile Batch Design

## Goal

Process the original SEM images under `D:\CNTDATA\coredata` and produce resumable, non-visual feature outputs for five analysis profiles:

- legacy
- v2_accurate
- v2_fast
- v3_accurate
- v3_fast

The first phase should optimize for speed and reliability by skipping visualization entirely.

## Confirmed Scope

After excluding generated previews, review sheets, and copied browsing folders, the current original-image set contains `82` SEM images.

## Why This Is a Long Task

The expensive part is not disk traversal but repeated geometry work:

- ROI extraction
- preprocessing
- threshold mask generation
- skeleton extraction
- branch extraction
- curvature/waviness computation

If each profile recomputes these stages independently, the total run becomes unnecessarily slow.

## Recommended Strategy

### 1. Shared per-image preparation

For each image, compute the following exactly once:

- ROI
- preprocessed grayscale
- threshold mask
- skeleton
- base connected components

All five profiles then reuse this shared preparation.

### 2. Reuse branch collections

Where possible, derive stricter branch sets from looser ones instead of rescanning the skeleton repeatedly.

Examples:

- accurate V3 branch set can act as the broad candidate pool
- accurate V2 can reuse stricter filtering on the same skeleton
- fast V2/V3 can reuse the same branch-selection logic with capped branch count and per-branch downsampling

### 3. Resumable outputs

Each image should have its own machine-readable output file. This prevents long-task restarts from discarding finished work.

### 4. No visualization in phase 1

The first batch should output only:

- per-image JSON
- summary CSV
- summary JSON

Visualization becomes a second-stage task run only on selected images.

## Output Structure

Recommended output layout:

- `manifest.csv`
- `items/<image_slug>/features.json`
- `summary.csv`
- `summary.json`

This layout is simple to resume and easy to inspect.

## Breakpoints

### Breakpoint 1: Manifest

Freeze the exact list of 82 source images before any computation begins.

### Breakpoint 2: Per-image features

Each image writes a complete `features.json`. Resume skips images with completed outputs.

### Breakpoint 3: Summary merge

Only after all per-image results exist, generate `summary.csv` and `summary.json`.

## Expected Runtime

With the current CPU-heavy implementation and no visualization:

- expected total runtime: roughly `2.5 to 4.5 hours`

This estimate assumes:

- shared preprocessing per image
- profile reuse
- no rendering

## Success Criteria

The phase-1 batch is successful if:

- all 82 source images are processed
- all 5 profiles are present per image
- interrupted runs can resume cleanly
- summary outputs are complete and consistent

