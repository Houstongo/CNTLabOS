# Branch Graph Tracing V1c Design

## Goal

Shift the tracing prototype from "reconstruct several plausible local paths" to "recover one main CNT-like path that tends to traverse the full image from boundary to boundary."

## Problem Statement

The current v1b prototype can cross some junctions, but it still behaves like a local linker:

- it starts from generic endpoints or long branches
- it ranks states mostly by local continuation cost
- it marks branches globally visited too early
- it returns several medium-length paths instead of one dominant main path

That behavior produces paths that are directionally plausible in small neighborhoods, but still too short for CNT images where the target structure often spans most of the frame.

## V1c Target Behavior

The new prototype should explicitly prefer a single path that:

- starts near the image boundary
- initially points inward
- continues through junctions with physically plausible turning angles
- keeps a stable global direction
- extends toward the opposite side of the image

The intended mental model is a CNT entering from one side of the ROI and continuing through the scene with moderate local bending, not a path that optimizes only local branch compatibility.

## Scope

- single-image research prototype only
- preserve the existing branch-graph representation from v1b
- keep the lightweight beam-search structure
- output one primary `main_path`
- keep auxiliary candidate paths only for debugging and inspection
- no batch integration in v1c
- no database integration in v1c

## Recommended Approach

Use **boundary-seeded directional beam search**.

### Why this approach

- It matches the physical prior better than generic endpoint tracing.
- It reuses the v1b graph, tangent, and transition infrastructure.
- It makes the search objective explicit: extend a main path across the image instead of collecting many local paths.

### Alternatives considered

#### Boundary endpoint pairing

Search directly between pairs of boundary endpoints.

This makes the "edge-to-edge" goal explicit, but it depends heavily on stable boundary endpoint detection and grows combinatorially when many seeds exist.

#### Global longest-path heuristic

Pick a longest skeleton path first and only later add angle constraints.

This is simpler, but it is easier for complex junctions and local loops to dominate the result, which does not match the desired CNT prior.

## Seed Selection

Tracing should start only from boundary-adjacent endpoints.

### Boundary seed definition

An endpoint is a valid seed when its minimum distance to the image border is below a configurable boundary margin.

Recommended initial default:

- `edge_seed_margin_px = max(2.0 * expected_tube_px, 0.03 * min(image_height, image_width))`

### Inward direction rule

Each endpoint seed must have an initial tangent that points into the image rather than sliding along or out of the border.

Use the endpoint tangent and the local inward border normal to compute an inward score:

- strongly positive inward score: preferred
- weak inward score: lower priority
- negative inward score: reject

### Seed deduplication

If multiple seeds lie very close to one another and have similar initial directions, collapse them into one seed candidate to avoid tracing the same CNT entry multiple times.

### Seed ranking

Valid seeds should be ranked by:

1. smaller border distance
2. stronger inward direction
3. longer first branch
4. lower local width variation

The search should launch from the top `K` seeds rather than a single seed, then choose one best final `main_path`.

## Directional Beam Search

Retain beam search, but change its objective.

### Beam state

Each beam state should track:

- ordered `branch_ids`
- ordered `coords`
- current node id
- used branches within this beam
- current direction estimate
- cumulative turning penalty
- cumulative path length
- current span
- current boundary progress summary

### Key behavioral change

Beam ranking should no longer be based mainly on local continuation cost.

Instead, it should favor states that already look like a main CNT:

- continuing in a stable direction
- increasing image-wide span
- moving away from the entry boundary and toward the opposite side
- avoiding repeated high-angle turns

### Junction decision rule

At each junction, evaluate all available continuation branches that:

- do not create a loop in the same beam
- satisfy the hard turning-angle limit
- are not obvious short spurs

Then rank them primarily by directional consistency:

1. angle to current direction
2. angle to running global direction
3. span gain or opposite-boundary progress
4. branch length
5. width and intensity consistency

The first two angle terms should dominate.

## Turn Constraints

Use a soft and hard angle rule.

- `angle_soft_deg`: still acceptable, but increasingly penalized as the path bends
- `angle_hard_deg`: invalid continuation

Recommended initial defaults:

- `angle_soft_deg = 40` to `45`
- `angle_hard_deg = 65` to `75`

The exact values can stay user-configurable through the demo script.

## Main Path Scoring

After all seed-driven beam searches finish, score each completed candidate and keep one `main_path`.

### Score components

The final score should explicitly reward global traversal rather than only local smoothness.

- `edge_reward`
  Reward both path ends being near image boundaries.
- `span_reward`
  Reward large endpoint-to-endpoint span.
- `length_reward`
  Reward large accumulated path length, but less than span.
- `direction_stability_reward`
  Reward low cumulative direction drift and limited zigzagging.
- `angle_penalty`
  Penalize repeated high-angle transitions.
- `zigzag_penalty`
  Penalize frequent alternating turns and small local detours.

### Recommended weighting

Initial weighting:

```text
score_main
= 0.35 * edge_reward
+ 0.30 * normalized_span
+ 0.20 * normalized_length
+ 0.15 * direction_stability
- angle_penalty
- zigzag_penalty
```

This intentionally favors boundary-to-boundary reach and span ahead of raw path length.

## Stop Conditions

Stop extending a beam when any of the following are true:

- no candidate branch satisfies the hard angle limit
- extending would create a loop or reuse a branch within the same beam
- the next branch is too short and adds too little span
- several consecutive steps exceed the soft-angle comfort zone
- cumulative turning exceeds a configured safety limit
- the path has already reached the opposite boundary with good directional stability

Recommended initial guard rails:

- `max_branch_steps`
- `max_cumulative_turn_deg`
- `min_span_gain_px` for accepting another short continuation

## Global Branch Ownership

Do **not** use global `visited_branches` while searching for the main path.

That rule is appropriate for partitioning the graph into many non-overlapping paths, but it directly harms the v1c objective by forcing early local decisions that block better boundary-to-boundary candidates.

Instead:

- prevent branch reuse only within one beam state
- let different seeds compete over overlapping subpaths
- choose the best final candidate after the search completes

## Output Changes

The result structure should prioritize a single primary path.

### Required outputs

- `main_path`
- `main_path_score`
- `main_path_reasoning` or debug score breakdown
- `candidate_paths` for inspection
- existing geometric metrics for the chosen path

### Demo output

The demo script should highlight:

- selected boundary seeds
- the winning `main_path`
- optional alternative candidate paths in lighter colors
- score breakdown terms that explain why the chosen path won

## Success Criteria

V1c is successful if, on representative single-image demos:

- the selected path usually starts from a boundary seed
- the selected path tends to extend substantially farther across the ROI than v1b
- the chosen path visually resembles a single CNT main trajectory
- junction choices are explainable from direction continuity
- runtime remains practical for interactive single-image experimentation

## Non-Goals

- perfect reconstruction of all CNTs in the image
- robust handling of every strongly coiled CNT in v1c
- batch-scale production deployment
- irreversible replacement of existing curvature or segmentation paths

## Risk Notes

- Strong edge bias can miss true CNTs that are truncated away from the border.
- Strong direction-stability bias can miss genuinely curved but valid CNTs.
- Weak seed filtering can still cause duplicate or noisy starts.

These are acceptable v1c trade-offs because the primary objective is to recover one plausible image-spanning main path first.
