# ZZY 9000 Threshold Design

**Goal:** Raise the ZZY import threshold from 5000 to 9000 and logically delete existing active ZZY records below 9000 magnification.

**Scope**

- Apply the new rule only to `source = 'ZZY'`.
- Keep the existing position rule: only `mid*` images are eligible.
- Change the magnification rule from `>= 5000` to `>= 9000`.
- For existing database rows, only update active ZZY rows with `magnification < 9000`.
- Use logical deletion (`is_deleted = 1`), not physical deletion.

**Behavior**

- New imports:
  - `ZZY` rows are inserted only when `position_label` matches `mid*` and `magnification >= 9000`.
  - Rows below 9000 are skipped during import.
- Existing records:
  - Active `ZZY` rows with `magnification < 9000` are soft-deleted in place.
  - Already deleted rows are left unchanged.
  - `XR` rows are untouched.

**Files Affected**

- `D:\CNTDATA\CNTA_ML_Project\backend\core\data_manager.py`
- `D:\CNTDATA\CNTA_ML_Project\backend\core\populate_db.py`
- `D:\CNTDATA\CNTA_ML_Project\tests\test_zzy_import_rules.py`

**Verification**

- Parser tests confirm `mid 9000-1` is included and `mid 5000-1` is excluded.
- Import script still respects the `mid*` filter.
- Database verification confirms active ZZY rows below 9000 were logically deleted.

**Expected Data Impact**

- Current active ZZY rows below 9000: `177`
- These rows should be updated to `is_deleted = 1`.
