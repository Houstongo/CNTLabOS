# WCNTSegNET Primary Backend Design

## Goal

Make `WCNTSegNET` the system's primary segmentation algorithm for:

- frontend feature-extraction experience
- backend image visualization/extraction interfaces
- batch processing defaults

while preserving future extensibility and keeping legacy `threshold` requests compatible.

## Current State

- Backend single-image visualization API defaults to `threshold`
- Batch processor defaults to `threshold`
- Frontend algorithm visualization buttons and initial state default to `threshold`
- Internally, `threshold` is the same traditional segmentation route now being renamed to `WCNTSegNET`

## Design Decision

Adopt `wcntsegnet` as the new canonical public identifier and keep `threshold` as a legacy alias.

### Why this approach

- It avoids breaking existing callers immediately
- It lets the UI and future API docs present `WCNTSegNET` as the main algorithm
- It provides a clean extension point for additional backends later

## Scope

### Backend

- Introduce a small backend-normalization layer:
  - `wcntsegnet` -> canonical traditional route
  - `threshold` -> legacy alias for `wcntsegnet`
  - `cntsegnet` -> deep model route
  - `both` -> comparison mode
- Change default backend parameters from `threshold` to `wcntsegnet`
- Return `wcntsegnet` in backend response payloads where the primary traditional route is reported
- Keep comparison payloads readable by exposing `wcntsegnet` and preserving compatibility where needed

### Batch processing

- Change CLI/default processing backend from `threshold` to `wcntsegnet`
- Normalize legacy `threshold` input to the same internal route
- Keep future backend expansion localized to a single normalization path

### Frontend

- Change default selection state from `threshold` to `wcntsegnet`
- Update labels from "传统阈值" to `WCNTSegNET`
- Keep comparison mode available
- Ensure frontend can still render backend data if legacy keys appear during transition

## Compatibility Strategy

- Accept both `threshold` and `wcntsegnet` in backend and batch entrypoints
- Prefer `wcntsegnet` in all new defaults, UI labels, and newly emitted response metadata
- Avoid changing the underlying algorithm implementation itself in this step

## Risks

- Existing frontend code may assume `threshold` as the cache key or button id
- Comparison-mode payload consumers may assume `comparison.threshold`
- Hidden scripts may still invoke `--segmentation-backend threshold`

## Mitigations

- Normalize legacy values centrally before branching logic
- Keep legacy alias acceptance
- Update the main UI entrypoints and batch CLI help text in the same change
- Verify both default behavior and legacy alias behavior

## Verification Plan

- Single-image visualization without explicit backend should return `wcntsegnet`
- Explicit `backend=threshold` should still work
- Batch processor CLI default should print/use `wcntsegnet`
- Frontend default state should select `WCNTSegNET`
