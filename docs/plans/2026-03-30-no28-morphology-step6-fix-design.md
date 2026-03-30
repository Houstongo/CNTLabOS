# No28 Morphology Step 6 Fix Design

**Goal:** Regenerate the No28 morphology feature step outputs so that step 6 visibly overlays the cropped cleaned-minus-junction skeleton on top of the prediction mask, and the cropped outputs are standardized to `768x768`.

**Current Problem:**
- `reports/no28_morphology_feature_steps_v4_crop/06_prediction_plus_minus_junctions.png` does not visibly retain the cropped cleaned skeleton after junction removal.
- Existing cropped report variants use inconsistent sizes (`780x780` in `v4_crop`, `680x680` in `v5_crop`) instead of the desired fixed patch size.
- The original one-off generation path is not fully preserved in the repo, so fixing only the final PNG would not be reproducible.

**Chosen Approach:**
- Build a small reproducible repair script that starts from the existing report artifacts:
  - `02_model_prediction_mask.png` as the prediction base
  - `05_cleaned_skeleton_on_prediction.png` as the visible cleaned skeleton source
- Recover the cleaned skeleton from the step 5 overlay, detect junction pixels on that recovered skeleton, remove junction pixels, and redraw the result over the prediction mask for step 6.
- Rebuild the cropped report set as centered `768x768` patches using a deterministic crop box, and rewrite `summary.json` to reflect the new crop size.

**Why This Approach:**
- It matches the approved visual intent exactly: prediction base plus cleaned-minus-junction skeleton.
- It avoids guessing or partially reimplementing the older full pipeline.
- It creates a reusable repair path so the outputs can be regenerated later.

**Output Scope:**
- Update `reports/no28_morphology_feature_steps_v4_crop/*.png`
- Update `reports/no28_morphology_feature_steps_v4_crop/summary.json`
- Leave the uncropped `reports/no28_morphology_feature_steps_v4` images unchanged unless verification shows a hard dependency.

**Verification:**
- Confirm the regenerated cropped files are all `768x768`.
- Visually inspect step 6 to confirm the cleaned-minus-junction skeleton is present on top of prediction.
- Ensure the updated crop metadata in `summary.json` matches the regenerated images.
