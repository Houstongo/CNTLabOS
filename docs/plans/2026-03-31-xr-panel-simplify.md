# XR Panel Simplify Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the crowded XR standard review panel with a simplified layout that shows only Original ROI, Mask, L2 Overlay, and Global/L2 summary text.

**Architecture:** Keep the analysis pipeline unchanged and only change the presentation layer in the report renderer. Reuse the existing `thresholds["L2"]` data as the single visualization target and limit the summary builder to `Global` and `L2`.

**Tech Stack:** Python, matplotlib, OpenCV, existing XR batch report pipeline

---

### Task 1: Document the simplified panel contract

**Files:**
- Create: `D:\CNTDATA\CNTA_ML_Project\docs\plans\2026-03-31-xr-panel-simplify-design.md`
- Create: `D:\CNTDATA\CNTA_ML_Project\docs\plans\2026-03-31-xr-panel-simplify.md`

**Step 1: Write the approved layout and retained metrics**

Record that the panel must contain only:
- Original ROI
- Mask
- L2 Overlay
- Global + L2 summary text

**Step 2: Confirm removed elements**

Record that `L1 / L3 / L4` overlays and text sections are intentionally removed from the panel layout while remaining available in raw JSON/CSV outputs.

### Task 2: Limit the summary builder to Global and L2

**Files:**
- Modify: `D:\CNTDATA\CNTA_ML_Project\tools\generate_xr_slice_standard_batch.py`

**Step 1: Update `build_standard_summary_sections()`**

Keep the existing global block, then append only the `L2` section using `REFERENCE_THRESHOLD_LABEL`.

**Step 2: Preserve existing metric formatting**

Do not change numeric sources or field names; only reduce the displayed scope.

### Task 3: Redesign the panel layout

**Files:**
- Modify: `D:\CNTDATA\CNTA_ML_Project\tools\generate_xr_slice_standard_batch.py`

**Step 1: Update `render_panel()` grid**

Switch from the current `3 x 3` layout to a cleaner layout that shows:
- top-left: original ROI
- bottom-left: mask
- center column spanning two rows: L2 Overlay
- right column spanning two rows: summary text

**Step 2: Render only L2 overlay**

Use `threshold_profiles[REFERENCE_THRESHOLD_LABEL]` and `THRESHOLD_COLORS[REFERENCE_THRESHOLD_LABEL]`.

**Step 3: Update panel title**

Rename the title so it reflects the simplified layout and L2-focused presentation.

### Task 4: Regenerate a single sample for visual verification

**Files:**
- Modify: `D:\CNTDATA\CNTA_ML_Project\tools\generate_xr_slice_standard_batch.py`
- Output: sample report directory under `D:\CNTDATA\CNTA_ML_Project\reports`

**Step 1: Run one-sample regeneration**

Run the XR report script for the target image with a dedicated output directory.

**Step 2: Visually inspect the generated panel**

Confirm the panel is no longer crowded and that the L2 data remains readable.

### Task 5: Verify outputs

**Files:**
- Output: regenerated `panel.png`
- Output: regenerated `features.json`

**Step 1: Verify panel structure**

Check that only three image panes are present.

**Step 2: Verify summary structure**

Check that the right column contains only `Global` and `L2`.

**Step 3: Verify no analysis regressions**

Confirm `features.json` still retains the full threshold payload even though the panel only displays L2.
