# Intact Skeleton Probe Gray Consistency Design

**Goal:** Add a medium-strength grayscale consistency constraint to junction exit selection in the intact skeleton probe.

**Scope**
- Use only the preprocessed image as the grayscale reference.
- Keep current angle hard-cutoff and geometric tracing rules.
- Compare each candidate exit segment against the recent mean grayscale of the already traced main path.

**Design**
- Build a recent grayscale signature from the last section of the current path.
- Build a candidate grayscale signature from the first section of the proposed exit segment.
- Convert grayscale difference into a soft consistency score in `[0, 1]`.
- Inject that score into junction candidate ranking and local beam gain.

**Constraint Strength**
- Medium-strength only:
  - no grayscale hard block
  - visible effect in candidate ordering
  - lower priority than hard angle rejection and broad forward progress

**Outputs**
- Record grayscale consistency metrics in candidate diagnostics and summary output for inspection.
