# V3 Length Threshold Test Design

## Goal

Run a small research comparison that tests whether filtering out short skeleton branches can make `V3` cleaner without destroying curve-separation ability.

## Test Shape

- Sample set: 6 images
  - `50000x`: 3 images
  - `100000x`: 3 images
- Source: use the current `D:\CNTDATA\coredata\u` sample set for speed and consistency

## Thresholds

Use 5 branch-length thresholds on top of the current V3 branch logic:

- `L0 = 1.0`
- `L1 = 3.0`
- `L2 = 5.0`
- `L3 = 7.0`
- `L4 = 9.0`

Each value is interpreted as `min_length_factor` relative to `expected_tube_px`.

## Outputs Per Image

- original ROI
- raw skeleton
- V3 branch skeleton for `L0`
- V3 branch skeleton for `L1`
- V3 branch skeleton for `L2`
- V3 branch skeleton for `L3`
- V3 branch skeleton for `L4`
- metrics table

## Metrics Per Threshold

- `branch_count`
- `curvature_nm_v3`
- `waviness_ratio_v2`
- `tortuosity_v2`

## Why This Test

This directly answers two questions:

1. does removing short skeleton branches make the selected geometry cleaner?
2. at which threshold does `V3` begin to lose too much discriminatory information?

