# Branch Graph Tracing V1 Design

## Goal

Prototype a research-only algorithm that reconstructs more plausible single-CNT paths from a skeleton graph by linking branch segments through junctions using direction continuity and lightweight geometry cues.

## Motivation

The current V2/V3 pipeline avoids junction ambiguity by cutting junctions and analyzing fragments. That makes the metric stable, but it prevents any attempt to reconstruct longer, more single-CNT-like paths.

This prototype explores a middle ground:

- keep the skeleton graph
- remove only obviously bad short branches
- do not delete junctions entirely
- decide locally which branch should continue through a junction

## Scope

- single-image research prototype
- no batch integration
- no database integration
- no global Monte Carlo search in v1
- `100000x` real-image validation target

## Core Idea

Convert the skeleton into a `branch graph`:

- nodes:
  - endpoints
  - junctions
- edges:
  - branch segments between nodes

Then reconstruct longer paths by traversing the graph. At each junction, choose the continuation branch that best matches the current direction and local geometry.

## Why Not Full Monte Carlo First

A full Monte Carlo search is possible, but it is not the best v1:

- harder to debug
- more expensive
- more parameters
- lower interpretability

The recommended v1 is deterministic local linking with optional future extension to beam search or local Monte Carlo at ambiguous junctions.

## Data Structures

### Node

- `node_id`
- `kind`: `endpoint` or `junction`
- `coord`
- `degree`
- connected `branch_ids`

### Branch Edge

- `branch_id`
- ordered `coords`
- `node_start`
- `node_end`
- `length_px`
- tangent direction at both ends
- mean width
- local intensity summary
- optional curvature summary

## Pre-Filtering

Before graph tracing:

- remove very short branches
- remove obvious spur/noise branches

This keeps the graph smaller and reduces bad continuation choices.

## Junction Linking Rule

At a junction, the next branch is chosen by a local cost function.

### Required hard rule

- reject candidates whose turning angle exceeds `45°`

This encodes the user-provided physical prior that a CNT typically should not make a very sharp direction change at one local junction decision.

### Cost terms

Use a weighted sum:

- `angle_cost`
- `width_cost`
- `intensity_cost`
- `short_branch_penalty`

`angle_cost` should dominate in v1.

## Tracing Procedure

1. start from an endpoint branch
2. follow the branch to the next node
3. if the next node is not ambiguous, continue
4. if the node is a junction:
   - enumerate candidate continuation branches
   - reject those above `45°`
   - score remaining candidates
   - choose the minimum-cost candidate
5. stop when:
   - no valid continuation exists
   - the best candidate confidence is too low
   - an endpoint is reached

## Output

### Reconstructed path records

- `path_id`
- ordered `branch_ids`
- confidence
- path length
- span
- `L/D`
- mean curvature
- p90 curvature
- total turning angle

### Visualization

Generate a validation figure with:

- original ROI
- skeleton graph
- node types
- reconstructed paths in distinct colors
- path table

## Performance Strategy

The design stays efficient by:

- working at branch level instead of pixel level
- pre-filtering short branches
- applying a hard angle cutoff
- scoring only at junctions
- caching branch features once

This should keep v1 in the “research practical” range for single-image validation.

## Success Criteria

The prototype is successful if:

- reconstructed paths look more like human-interpretable single CNTs
- obvious junction mislinks are reduced by the angle rule
- the output is explainable and visually inspectable
- runtime remains practical for single-image experimentation

