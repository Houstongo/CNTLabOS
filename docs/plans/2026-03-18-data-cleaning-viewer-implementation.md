# Data Cleaning Viewer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rebuild the data cleaning page into an image-first viewer with a large original image, step thumbnails, and click-to-zoom preview.

**Architecture:** Keep the existing clean page data loading and scoring logic, but replace the current dense card layout with a two-tier viewer layout. Add a reusable lightbox modal for both the original image and intermediate step images, and keep the review panel compact so images remain readable.

**Tech Stack:** Static `index.html`, Tailwind utility classes, existing inline JavaScript state/render functions, browser `fetch`, existing node contract tests.

---

### Task 1: Add failing UI contract for the new viewer layout

**Files:**
- Modify: `D:\CNTDATA\CNTA_ML_Project\tests\test_data_cleaning_ui_contract.mjs`
- Test: `D:\CNTDATA\CNTA_ML_Project\tests\test_data_cleaning_ui_contract.mjs`

**Step 1: Write the failing test**

Add assertions for:
- `id="clean-viewer-stage"`
- `id="clean-lightbox"`
- `id="clean-step-strip"`
- `id="clean-open-original"`

**Step 2: Run test to verify it fails**

Run: `node --input-type=module -e "import('./tests/test_data_cleaning_ui_contract.mjs')"`
Expected: FAIL with missing new viewer ids.

**Step 3: Write minimal implementation**

Do not implement yet. This task only defines the contract before HTML changes.

**Step 4: Run test to verify it still fails**

Run the same command and confirm the failure is because the new ids are missing.

### Task 2: Rebuild the clean page layout around the original image

**Files:**
- Modify: `D:\CNTDATA\CNTA_ML_Project\index.html`

**Step 1: Implement the new viewer structure**

Add:
- `clean-viewer-stage` for the upper image-first area
- large original image panel with button `clean-open-original`
- compact right review column
- lower `clean-step-strip` for thumbnail cards
- `clean-lightbox` modal shell

**Step 2: Keep existing data hooks alive**

Reuse existing ids where possible:
- `clean-original-image`
- `clean-step-grid`
- `clean-review-content`
- `clean-quick-note`

**Step 3: Run the UI contract test**

Run: `node --input-type=module -e "import('./tests/test_data_cleaning_ui_contract.mjs')"`
Expected: PASS

### Task 3: Add lightbox interaction for original and step images

**Files:**
- Modify: `D:\CNTDATA\CNTA_ML_Project\index.html`

**Step 1: Write minimal JS behavior**

Add:
- `openCleanLightbox(src, title)`
- `closeCleanLightbox()`
- click handlers for original image and each step thumbnail

**Step 2: Ensure graceful fallback**

If no image source exists, keep the placeholder visible and do not open the lightbox.

**Step 3: Re-run the contract test**

Run: `node --input-type=module -e "import('./tests/test_data_cleaning_ui_contract.mjs')"`
Expected: PASS

### Task 4: Compact the review panel and step cards for readability

**Files:**
- Modify: `D:\CNTDATA\CNTA_ML_Project\index.html`

**Step 1: Reduce review noise**

Show only:
- confidence chip
- score
- five metrics
- top three reasons

**Step 2: Make step thumbnails legible**

Use larger cards with image-first layout and short descriptions.

**Step 3: Manually verify in browser**

Check:
- original image is visibly larger than before
- step thumbnails are readable
- review panel no longer crowds the image

### Task 5: Verify behavior end to end

**Files:**
- Test: `D:\CNTDATA\CNTA_ML_Project\tests\test_data_cleaning_ui_contract.mjs`
- Test: `D:\CNTDATA\CNTA_ML_Project\tests\test_algorithm_visualizer.py`

**Step 1: Run tests**

Run:
- `node --input-type=module -e "import('./tests/test_data_cleaning_ui_contract.mjs')"`
- `python -m unittest tests.test_algorithm_visualizer`

**Step 2: Browser verify**

Confirm:
- clean page opens
- original image shows fully
- original image opens in modal
- step thumbnails open in modal
- right panel remains readable

**Step 3: Commit**

```bash
git add D:\\CNTDATA\\CNTA_ML_Project\\index.html D:\\CNTDATA\\CNTA_ML_Project\\tests\\test_data_cleaning_ui_contract.mjs D:\\CNTDATA\\CNTA_ML_Project\\tests\\test_algorithm_visualizer.py D:\\CNTDATA\\CNTA_ML_Project\\backend\\core\\algorithm_visualizer.py D:\\CNTDATA\\CNTA_ML_Project\\docs\\plans\\2026-03-18-data-cleaning-viewer-design.md D:\\CNTDATA\\CNTA_ML_Project\\docs\\plans\\2026-03-18-data-cleaning-viewer-implementation.md
git commit -m "feat: redesign data cleaning viewer"
```
