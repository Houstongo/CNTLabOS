# Branch Selection Comparison Visualization Design

## Goal

Generate comparison panels for all 9 images in `D:\CNTDATA\coredata\u` that show:

- original ROI
- V2 accurate selected branches
- V3 accurate selected branches
- fast-mode selected branches
- a feature summary table

## Scope

- research visualization only
- standalone script
- reuse the current `FeatureExtractor`
- no database changes
- no API changes

## Recommended Approach

Use the current threshold-based mask and skeleton pipeline once per image, then derive three branch views:

1. `V2 accurate`
   - current V2 branch-selection rules
2. `V3 accurate`
   - current V3 branch-selection rules
3. `fast`
   - fast-mode branch selection, shown as a combined overlay for V2-fast and V3-fast

The metrics table should include:

- density
- diameter
- legacy curvature
- V2 curvature
- V3 curvature
- waviness/tortuosity
- accurate and fast branch counts

## Why This Version

This gives a direct visual answer to:

- how many branches are being selected
- how V2 and V3 differ spatially
- what fast mode keeps versus drops
- how those branch-selection differences relate to final feature values

