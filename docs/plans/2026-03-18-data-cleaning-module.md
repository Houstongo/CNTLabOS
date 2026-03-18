# Data Cleaning Module Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a new front-end "数据清洗" module that helps review image-derived feature quality using existing image and visualization APIs.

**Architecture:** Extend the single-file `index.html` app with a new navigation entry and page container. Reuse existing `/api/images`, `/api/images/{id}/visualize`, and feature fields already returned by the backend to build a cleaning-oriented review UI with filters, image panels, and confidence heuristics computed on the client.

**Tech Stack:** Static HTML/JS in `index.html`, existing FastAPI JSON APIs, Node test for UI contract.

---

### Task 1: Add a failing UI contract test

**Files:**
- Create: `tests/test_data_cleaning_ui_contract.mjs`
- Modify: none
- Test: `tests/test_data_cleaning_ui_contract.mjs`

**Step 1: Write the failing test**

Add a Node test that reads `index.html` and asserts:
- there is a nav label `数据清洗`
- there is a page container `id="clean-page"`
- there are key containers for filters, review summary, and image panel

**Step 2: Run test to verify it fails**

Run: `node --test tests/test_data_cleaning_ui_contract.mjs`
Expected: FAIL because the module does not exist yet

**Step 3: Write minimal implementation**

Add the new nav item, page skeleton, and placeholder containers.

**Step 4: Run test to verify it passes**

Run: `node --test tests/test_data_cleaning_ui_contract.mjs`
Expected: PASS

### Task 2: Add front-end data cleaning page skeleton and navigation

**Files:**
- Modify: `index.html`
- Test: `tests/test_data_cleaning_ui_contract.mjs`

**Step 1: Add nav and page routing**

Update `showPage()` and sidebar nav so the app can open the cleaning page.

**Step 2: Add module layout**

Create:
- top filter toolbar
- left image panel
- right review panel
- bottom sample table/list

**Step 3: Keep styling consistent**

Reuse current design language while making the page readable and cleaning-focused.

### Task 3: Implement client-side cleaning data loader

**Files:**
- Modify: `index.html`

**Step 1: Load image rows from existing `/api/images`**

Fetch enough rows for browsing and filtering.

**Step 2: Normalize fields**

Map existing feature fields:
- `density`
- `alignment`
- `diameter`
- `curvature`
- `tortuosity`

**Step 3: Compute review flags**

Add client-side heuristics:
- low/high density warnings
- invalid alignment warnings
- missing diameter warnings
- curvature/tortuosity anomaly warnings

### Task 4: Implement review interaction

**Files:**
- Modify: `index.html`

**Step 1: Filter + select sample**

Allow filtering by source, processed status, and confidence level.

**Step 2: Render image review panes**

Show original image preview and visualization steps from `/api/images/{id}/visualize`.

**Step 3: Render review details**

Show feature values, confidence tier, and reasons.

### Task 5: Verify behavior

**Files:**
- Modify: none
- Test: `tests/test_data_cleaning_ui_contract.mjs`

**Step 1: Run Node contract test**

Run: `node --test tests/test_data_cleaning_ui_contract.mjs`
Expected: PASS

**Step 2: Manual browser verification**

Open the app and confirm:
- nav opens `数据清洗`
- list loads
- selecting a sample updates the image and rule panel
- visualization images render from existing API
