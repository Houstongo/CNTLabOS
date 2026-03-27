# Intact Skeleton Probe Curvature Export Design

**Goal:** Add Fiji-like curvature export on top of the existing intact skeleton probe, while keeping tracing logic unchanged.

**Scope**
- Keep current tracing, dedupe, and candidate ranking unchanged.
- Add three smoothing profiles for selected paths: `conservative`, `balanced`, `visual`.
- Use binary segmentation as the path-overlay background.
- Export both per-path curvature curves and per-path aggregate curvature statistics.

**Smoothing**
- Treat current traced path as the geometry baseline.
- Preserve a `raw` profile plus three smoothed profiles.
- Use shared arc-length resampling so all profiles can be compared point-by-point.
- Keep endpoints fixed during smoothing.

**Curvature**
- Use a Fiji-like local tangent method: estimate tangent direction along the resampled centerline, then compute curvature as `|dtheta/ds|`.
- Export both `1/px` and `1/nm` curvature.
- Summarize each path with mean, median, p95, max curvature, plus length/span/tortuosity.

**Outputs**
- Binary-background path overlays for `conservative`, `balanced`, and `visual`.
- One curvature curve plot per kept path, comparing `raw/conservative/balanced/visual`.
- One CSV per kept path with sampled coordinates and curvature series.
- One summary CSV/JSON across all kept paths.

**Non-Goals**
- No change to tracing seeds, beam search, junction decisions, or dedupe behavior.
- No attempt to merge curvature back into the production `feature_extractor` flow yet.
