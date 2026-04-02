# XR Review Report Timeout Design

**Goal:** Extend the XR review report pipeline so it produces review-only visual reports from the current `CNTSegNet-clDice -> external_binary_mask -> FeatureExtractor.extract_all(...) -> topology cleanup -> V3-oriented feature analysis` workflow, while skipping any single image that exceeds 100 seconds.

**User Requirements:**
- Process the `XR` image group for manual review before deciding whether to write anything into the database.
- Keep the workflow review-only: generate reports and summaries, do not perform database writes.
- Include the structural fields needed for intake decisions:
  - V3 curvature outputs and multiple summary statistics
  - other bending metrics
  - alignment
  - diameter
  - density
- Add a hard per-image timeout of 100 seconds and record timed-out items as skipped.

**Chosen Approach:**
- Reuse `tools/generate_xr_slice_standard_batch.py` as the report entrypoint because it is already XR-focused and report-oriented.
- Add a subprocess-based timeout wrapper around per-image processing so a slow segmentation or analysis call can be terminated cleanly.
- Expand the summary payload so V3 curvature statistics and the related bending/structure features are visible both in per-item JSON and in the flattened CSV summary.

**Fields To Expose In Review Output:**
- density
- alignment / alignment_raw / mean_phi_deg
- diameter_nm plus diameter summary statistics
- per-threshold curvature fields:
  - p75 weighted by sqrt(length)
  - p75 weighted by length
  - p70 weighted by sqrt(length)
  - p70 weighted by length
  - mean weighted by sqrt(length)
  - mean weighted by length
  - trimmed mean weighted by sqrt(length)
  - trimmed mean weighted by length
- waviness and tortuosity
- timeout / error / success processing status

**Verification:**
- Script accepts a timeout argument defaulting to 100 seconds.
- A timed-out image does not stop the batch.
- Summary files contain success and timeout/error records.
- The flattened CSV includes the new curvature statistics and other requested structural fields.
