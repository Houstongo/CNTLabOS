# XR Database Redesign

## Goal

Reorganize the XR dataset schema around the real process variables for the Ar + toluene + ferrocene workflow, so the database cleanly separates experiment-level parameters from image-level records and is suitable for modeling.

## Confirmed XR Process Assumptions

- XR process uses argon as carrier gas.
- Carbon source is toluene.
- Catalyst precursor is ferrocene.
- The only intended XR process variables are:
  - temperature
  - gas flow
  - catalyst concentration

These should be treated as first-class structured fields. Other chemistry descriptors should remain fixed metadata, not primary modeling variables.

## Current Problems

- The `images` table mixes experiment parameters and image records.
- Folder-level parameters are duplicated across many images from the same run.
- Existing fields such as `h2_flow`, `c2h4_flow`, `al2o3_power`, and `fe_thickness` reflect a different process family and are not a good fit for XR.
- `catalyst_weight` is not specific enough for XR semantics and is currently incorrect for XR rows.
- Current model preparation risks leakage because image rows are acting like independent experiments.

## Recommended Schema

### 1. `xr_runs`

One row per XR experiment batch, typically one folder.

Suggested fields:

- `run_id` INTEGER PRIMARY KEY
- `folder_name` TEXT UNIQUE NOT NULL
- `experiment_date` TEXT
- `source` TEXT NOT NULL DEFAULT `'XR'`
- `set_temp_c` REAL NOT NULL
- `growth_time_h` REAL
- `ar_flow` REAL NOT NULL
- `catalyst_concentration` REAL NOT NULL
- `carbon_source` TEXT NOT NULL DEFAULT `'toluene'`
- `catalyst_precursor` TEXT NOT NULL DEFAULT `'ferrocene'`
- `notes` TEXT
- `created_at` TEXT
- `updated_at` TEXT

### 2. `xr_images`

One row per XR image.

Suggested fields:

- `image_id` INTEGER PRIMARY KEY
- `run_id` INTEGER NOT NULL
- `file_path` TEXT UNIQUE NOT NULL
- `sample_id` TEXT
- `membrane_id` INTEGER
- `position_label` TEXT
- `horizontal_pos` TEXT
- `vertical_pos` INTEGER
- `membrane_pos_cm` REAL
- `magnification` INTEGER
- `actual_temp_c` REAL
- `processed` INTEGER DEFAULT 0

Foreign key:

- `run_id` REFERENCES `xr_runs(run_id)`

### 3. `xr_targets`

One row per labeled target set. This can stay image-level unless you later decide some labels are batch-level.

Suggested fields:

- `target_id` INTEGER PRIMARY KEY
- `image_id` INTEGER NOT NULL
- `diameter` REAL
- `density` REAL
- `alignment` REAL
- `curvature` REAL
- `tortuosity` REAL
- `label_source` TEXT
- `label_version` TEXT
- `updated_at` TEXT

Foreign key:

- `image_id` REFERENCES `xr_images(image_id)`

## Modeling Implications

This design makes the intended learning problem much cleaner:

- process variables live at the run level
- morphology labels live at the image level
- dataset splits can group by `run_id`
- image rows from the same batch no longer masquerade as independent experiments

Recommended modeling inputs:

- `xr_runs.set_temp_c`
- `xr_runs.ar_flow`
- `xr_runs.catalyst_concentration`
- optionally `xr_images.actual_temp_c`
- optionally image-derived features

## Mapping From Current Schema

Current `images` table fields should map like this for XR:

- `growth_temp` -> `xr_runs.set_temp_c`
- `growth_time` -> `xr_runs.growth_time_h`
- `ar_flow` -> `xr_runs.ar_flow`
- `file_path` parent folder -> `xr_runs.folder_name`
- `actual_temp` -> `xr_images.actual_temp_c`
- `membrane_pos_cm` -> `xr_images.membrane_pos_cm`
- `sample_id` -> `xr_images.sample_id`
- `membrane_id` -> `xr_images.membrane_id`
- `position_label` -> `xr_images.position_label`
- `horizontal_pos` -> `xr_images.horizontal_pos`
- `vertical_pos` -> `xr_images.vertical_pos`
- `magnification` -> `xr_images.magnification`
- `diameter`, `density`, `alignment`, `curvature`, `tortuosity` -> `xr_targets`

## Open Migration Rule

The one XR parameter that still needs a precise source rule is `catalyst_concentration`.

If current folder names like `0.5g`, `1.0g`, `1.5g`, `2g` are intended to represent catalyst concentration in your workflow, then migration can map them directly into `catalyst_concentration`.

If they represent catalyst mass rather than concentration, then:

- store them temporarily in a migration helper field or notes
- do not label them as concentration until the experiment meaning is confirmed

## Transition Strategy

Recommended path:

1. Keep the current `images` table working for the app.
2. Add the new XR-specific tables alongside it.
3. Migrate XR records into the new structure.
4. Update XR reads in the backend to use the new tables.
5. Update the frontend to display run-level parameters from `xr_runs`.
6. Retire XR use of overloaded legacy fields only after validation.

## Recommendation

Adopt the XR-specific three-table structure:

- `xr_runs`
- `xr_images`
- `xr_targets`

This is the smallest redesign that matches the real XR process, supports your modeling goals, and reduces future schema confusion.
