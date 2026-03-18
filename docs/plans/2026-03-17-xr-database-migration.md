# XR Database Migration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Migrate XR data from the overloaded `images` table into XR-specific run, image, and target tables built around temperature, gas flow, and catalyst concentration.

**Architecture:** Add XR-specific tables beside the current schema, populate them from existing XR rows and folder metadata, then gradually switch backend and frontend reads to the new normalized structure. Preserve current behavior during migration so the app remains usable.

**Tech Stack:** Python, SQLite, FastAPI, vanilla JavaScript, unittest

---

### Task 1: Add XR schema regression tests

**Files:**
- Modify: `D:\CNTDATA\CNTA_ML_Project\tests\test_xr_temperature_backfill.py`
- Create: `D:\CNTDATA\CNTA_ML_Project\tests\test_xr_schema_migration.py`

**Step 1: Write the failing test**

Create tests that verify:
- XR folder metadata parses catalyst concentration from names like `250524 T800 3H L250 0.5g`
- migration creates one `xr_runs` row per XR folder
- migrated `xr_images` rows reference `run_id`
- migrated `xr_targets` rows preserve morphology labels

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_xr_schema_migration`

Expected: FAIL because XR schema migration code does not exist yet.

**Step 3: Write minimal implementation**

Add only the code required to support the new parser and migration entry points.

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_xr_schema_migration`

Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_xr_schema_migration.py tests/test_xr_temperature_backfill.py
git commit -m "test: cover xr schema migration"
```

### Task 2: Extend XR folder metadata parsing

**Files:**
- Modify: `D:\CNTDATA\CNTA_ML_Project\backend\core\data_manager.py`

**Step 1: Write the failing test**

Add parser assertions for:
- `T750/T800/T850` -> temperature
- `L200/L250/L300` -> Ar flow
- `0.5g/1.0g/1.5g/2g` -> catalyst concentration placeholder

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_xr_temperature_backfill`

Expected: FAIL on new flow and concentration assertions.

**Step 3: Write minimal implementation**

Update `parse_xr_folder_metadata()` to return:
- `growth_temp`
- `ar_flow`
- parsed concentration field for XR migration use

Keep existing behavior intact for current XR ingestion.

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_xr_temperature_backfill`

Expected: PASS

**Step 5: Commit**

```bash
git add backend/core/data_manager.py tests/test_xr_temperature_backfill.py
git commit -m "feat: parse xr folder process metadata"
```

### Task 3: Create XR-specific tables

**Files:**
- Create: `D:\CNTDATA\CNTA_ML_Project\tools\maintenance\migrate_xr_schema.py`

**Step 1: Write the failing test**

Add migration test assertions that the script creates:
- `xr_runs`
- `xr_images`
- `xr_targets`

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_xr_schema_migration`

Expected: FAIL because the migration script and tables do not exist.

**Step 3: Write minimal implementation**

Implement schema creation SQL for:
- `xr_runs`
- `xr_images`
- `xr_targets`

Include foreign keys and unique constraints that fit the design doc.

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_xr_schema_migration`

Expected: PASS for table creation checks.

**Step 5: Commit**

```bash
git add tools/maintenance/migrate_xr_schema.py tests/test_xr_schema_migration.py
git commit -m "feat: add xr normalized schema"
```

### Task 4: Migrate XR run-level data

**Files:**
- Modify: `D:\CNTDATA\CNTA_ML_Project\tools\maintenance\migrate_xr_schema.py`

**Step 1: Write the failing test**

Add a test using temporary SQLite data that expects:
- one run per folder
- correct `set_temp_c`
- correct `ar_flow`
- correct catalyst concentration value

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_xr_schema_migration`

Expected: FAIL because population logic is incomplete.

**Step 3: Write minimal implementation**

Populate `xr_runs` by grouping XR rows on folder name and inserting:
- `folder_name`
- `experiment_date`
- `set_temp_c`
- `growth_time_h`
- `ar_flow`
- `catalyst_concentration`
- fixed `carbon_source`
- fixed `catalyst_precursor`

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_xr_schema_migration`

Expected: PASS

**Step 5: Commit**

```bash
git add tools/maintenance/migrate_xr_schema.py tests/test_xr_schema_migration.py
git commit -m "feat: migrate xr run metadata"
```

### Task 5: Migrate XR image-level and target-level data

**Files:**
- Modify: `D:\CNTDATA\CNTA_ML_Project\tools\maintenance\migrate_xr_schema.py`

**Step 1: Write the failing test**

Add assertions that:
- `xr_images` rows retain file, position, magnification, and actual temperature
- `xr_targets` rows retain diameter, density, alignment, curvature, and tortuosity

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_xr_schema_migration`

Expected: FAIL because row migration is incomplete.

**Step 3: Write minimal implementation**

Insert:
- one `xr_images` row for each XR image
- one `xr_targets` row for each migrated image with labels

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_xr_schema_migration`

Expected: PASS

**Step 5: Commit**

```bash
git add tools/maintenance/migrate_xr_schema.py tests/test_xr_schema_migration.py
git commit -m "feat: migrate xr image and target records"
```

### Task 6: Add backend XR read endpoint support

**Files:**
- Modify: `D:\CNTDATA\CNTA_ML_Project\backend\main.py`
- Test: `D:\CNTDATA\CNTA_ML_Project\tests\test_xr_schema_migration.py`

**Step 1: Write the failing test**

Add a backend-facing test that expects XR list responses to include normalized run-level fields:
- `set_temp_c`
- `ar_flow`
- `catalyst_concentration`
- `actual_temp_c`

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_xr_schema_migration`

Expected: FAIL because backend still reads only legacy `images`.

**Step 3: Write minimal implementation**

Update XR data reads to join `xr_runs` and `xr_images` while preserving current response shape where possible.

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_xr_schema_migration`

Expected: PASS

**Step 5: Commit**

```bash
git add backend/main.py tests/test_xr_schema_migration.py
git commit -m "feat: serve xr normalized data"
```

### Task 7: Update frontend XR parameter display

**Files:**
- Modify: `D:\CNTDATA\CNTA_ML_Project\index.html`

**Step 1: Write the failing test**

If practical, add a small UI text sanity test. If no browser test harness exists, define manual verification steps in the task notes before implementation.

**Step 2: Run test to verify it fails**

Run the available UI sanity test, or document that no automated UI harness exists and use a manual verification checklist.

Expected: missing XR concentration field in table/detail view.

**Step 3: Write minimal implementation**

Display XR run-level fields in the list and detail panel:
- temperature
- Ar flow
- catalyst concentration

Avoid exposing legacy process-family fields for XR if they are not meaningful.

**Step 4: Run test to verify it passes**

Manual verify in UI:
- XR list shows concentration
- XR detail panel shows concentration
- non-XR views remain readable

**Step 5: Commit**

```bash
git add index.html
git commit -m "feat: show xr normalized process parameters"
```

### Task 8: Run migration on a safe database copy

**Files:**
- Modify: `D:\CNTDATA\CNTA_ML_Project\tools\maintenance\migrate_xr_schema.py`
- Output: `D:\CNTDATA\CNTA_ML_Project\database\cnta_experiments_xr_migrated.sqlite`

**Step 1: Write the failing test**

Add a smoke test or script assertion that the migration can process a database copy and produce non-empty XR tables.

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_xr_schema_migration`

Expected: FAIL until migration entry point is complete.

**Step 3: Write minimal implementation**

Implement a CLI path that:
- copies the source database
- creates XR normalized tables
- migrates XR rows into them
- leaves the source untouched

**Step 4: Run test to verify it passes**

Run:
- `python -m unittest tests.test_xr_schema_migration`
- `python tools/maintenance/migrate_xr_schema.py`

Expected:
- tests pass
- migrated database file is created

**Step 5: Commit**

```bash
git add tools/maintenance/migrate_xr_schema.py tests/test_xr_schema_migration.py
git commit -m "feat: migrate xr schema into database copy"
```

### Task 9: Verify migrated data integrity

**Files:**
- No code changes required unless issues are found

**Step 1: Run verification query**

Run checks for:
- XR folder count vs `xr_runs` row count
- XR image count vs `xr_images` row count
- labeled XR row count vs `xr_targets` row count

**Step 2: Run grouped sample validation**

Check representative folders:
- `250313 T750 3h L250`
- `250301 T800 3h L250`
- `250309 T850 3h L250`

Confirm:
- correct `set_temp_c`
- correct `ar_flow`
- correct catalyst concentration
- sensible `actual_temp_c`

**Step 3: Run full test suite touched by the migration**

Run:
- `python -m unittest tests.test_xr_temperature_backfill`
- `python -m unittest tests.test_xr_schema_migration`

Expected: PASS

**Step 4: Commit**

```bash
git add .
git commit -m "chore: verify xr schema migration"
```
