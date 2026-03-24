# WCNTSegNET Primary Backend Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Promote `WCNTSegNET` to the primary algorithm across backend APIs, frontend defaults, and batch processing while preserving `threshold` as a compatibility alias.

**Architecture:** Add a small normalization layer so `wcntsegnet` becomes the canonical public backend name and `threshold` becomes a legacy alias. Update backend response defaults, batch defaults, and frontend initial state/labels to use the canonical name without changing the underlying traditional segmentation implementation.

**Tech Stack:** Python, FastAPI, SQLite-backed backend utilities, static frontend JavaScript/HTML

---

### Task 1: Backend visualization/API normalization

**Files:**
- Modify: `D:\CNTDATA\CNTA_ML_Project\backend\main.py`

**Step 1: Add backend normalization helpers**

- Introduce canonical backend constants/helpers for:
  - `wcntsegnet`
  - `threshold` alias
  - `cntsegnet`
  - `both`

**Step 2: Update API defaults**

- Change default query parameter from `threshold` to `wcntsegnet`
- Make branch logic use normalized backend values

**Step 3: Update response naming**

- Return `wcntsegnet` as the main traditional backend identifier
- Keep `threshold` compatibility where transition safety is needed

**Step 4: Verify syntax**

Run: `python -m py_compile D:\CNTDATA\CNTA_ML_Project\backend\main.py`
Expected: no errors

### Task 2: Batch processor defaults and aliasing

**Files:**
- Modify: `D:\CNTDATA\CNTA_ML_Project\backend\core\batch_processor.py`

**Step 1: Add backend normalization**

- Accept both `wcntsegnet` and `threshold`
- Route both to the same traditional extraction path

**Step 2: Change defaults**

- Change function default and CLI default to `wcntsegnet`
- Update CLI help text to present `WCNTSegNET` as the primary method

**Step 3: Verify syntax**

Run: `python -m py_compile D:\CNTDATA\CNTA_ML_Project\backend\core\batch_processor.py`
Expected: no errors

### Task 3: Frontend default state and labels

**Files:**
- Modify: `D:\CNTDATA\CNTA_ML_Project\index.html`
- Review: `D:\CNTDATA\CNTA_ML_Project\frontend\modules\details\index.js`

**Step 1: Update initial backend state**

- Replace default `threshold` selection with `wcntsegnet`
- Update cache keys and switch handlers to accept canonical naming

**Step 2: Update labels**

- Replace visible "传统阈值" wording with `WCNTSegNET`

**Step 3: Preserve comparison mode**

- Ensure `both` still loads traditional plus CNTSegNet data

**Step 4: Verify syntax**

- Do a targeted scan for remaining hard-coded default `threshold` values in active UI paths

### Task 4: Verification

**Files:**
- No code changes required

**Step 1: Compile changed Python files**

Run:

```bash
python -m py_compile D:\CNTDATA\CNTA_ML_Project\backend\main.py
python -m py_compile D:\CNTDATA\CNTA_ML_Project\backend\core\batch_processor.py
```

Expected: both commands succeed

**Step 2: Verify backend default string changes**

Run targeted searches for:

- `backend: str = "wcntsegnet"`
- `segmentation_backend: str = "wcntsegnet"`
- frontend default `let currentBackend = 'wcntsegnet'`

Expected: canonical default naming present in active files

**Step 3: Verify compatibility alias remains**

- Confirm `threshold` is still accepted through normalization logic
- Confirm branch logic still routes legacy calls successfully
