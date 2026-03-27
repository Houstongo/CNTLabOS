# Branch Graph Tracing V1c Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rework branch-graph tracing so each image produces one boundary-seeded, directionally stable main CNT path that preferentially spans the frame.

**Architecture:** Reuse the v1b branch graph and transition helpers, add boundary-seed discovery plus directional state features, remove global branch ownership during search, score complete candidates by boundary reach and span, and update the demo to visualize the chosen `main_path` and why it won.

**Tech Stack:** Python, NumPy, OpenCV, scikit-image, matplotlib, existing `FeatureExtractor`

---

### Task 1: Add boundary seed discovery helpers

**Files:**
- Modify: `D:\CNTDATA\CNTA_ML_Project\src\analysis\feature_extractor.py`

**Step 1: Write the failing test**

- Add or extend a feature-extractor tracing test file to cover:
  - endpoints near the border are selected as seeds
  - endpoints far from the border are rejected
  - seeds with outward tangents are rejected

Suggested run:

```bash
pytest D:\CNTDATA\CNTA_ML_Project\tests\test_feature_extractor_branch_graph_tracing.py -k seed -v
```

Expected before implementation: failing assertions or missing helper errors.

**Step 2: Implement minimal seed helpers**

- Add helpers in `D:\CNTDATA\CNTA_ML_Project\src\analysis\feature_extractor.py` for:
  - endpoint border distance
  - inward border normal
  - inward tangent score
  - boundary seed filtering and ranking

**Step 3: Run the seed tests**

```bash
pytest D:\CNTDATA\CNTA_ML_Project\tests\test_feature_extractor_branch_graph_tracing.py -k seed -v
```

Expected: seed-focused tests pass.

### Task 2: Rework beam-state ranking around main-path behavior

**Files:**
- Modify: `D:\CNTDATA\CNTA_ML_Project\src\analysis\feature_extractor.py`
- Test: `D:\CNTDATA\CNTA_ML_Project\tests\test_feature_extractor_branch_graph_tracing.py`

**Step 1: Write the failing test**

- Add tests that verify:
  - a straighter continuation outranks a short side branch
  - beam ranking prefers larger span gain over locally cheap but stagnant continuations
  - the search does not depend on global `visited_branches`

Suggested run:

```bash
pytest D:\CNTDATA\CNTA_ML_Project\tests\test_feature_extractor_branch_graph_tracing.py -k beam -v
```

Expected before implementation: failing ranking/search assertions.

**Step 2: Implement minimal ranking changes**

- Extend beam state tracking with:
  - running direction estimate
  - cumulative turn
  - span
  - boundary progress
- Replace local-cost-only beam ordering with a main-path-oriented ranking score.
- Remove global branch blocking from candidate expansion.

**Step 3: Run the beam tests**

```bash
pytest D:\CNTDATA\CNTA_ML_Project\tests\test_feature_extractor_branch_graph_tracing.py -k beam -v
```

Expected: beam-focused tests pass.

### Task 3: Add final main-path scoring and selection

**Files:**
- Modify: `D:\CNTDATA\CNTA_ML_Project\src\analysis\feature_extractor.py`
- Test: `D:\CNTDATA\CNTA_ML_Project\tests\test_feature_extractor_branch_graph_tracing.py`

**Step 1: Write the failing test**

- Add tests that verify:
  - a boundary-to-boundary candidate outranks a shorter internal candidate
  - larger span wins over local zigzag length when other factors are similar
  - the returned structure exposes `main_path` and score breakdown fields

Suggested run:

```bash
pytest D:\CNTDATA\CNTA_ML_Project\tests\test_feature_extractor_branch_graph_tracing.py -k main_path -v
```

Expected before implementation: missing fields or incorrect candidate ordering.

**Step 2: Implement minimal final scoring**

- Add final candidate scoring terms for:
  - boundary reward
  - span reward
  - length reward
  - direction stability
  - angle penalty
  - zigzag penalty
- Select one `main_path` from all seed-driven completed candidates.

**Step 3: Run the main-path tests**

```bash
pytest D:\CNTDATA\CNTA_ML_Project\tests\test_feature_extractor_branch_graph_tracing.py -k main_path -v
```

Expected: main-path tests pass.

### Task 4: Update tracing outputs and stop conditions

**Files:**
- Modify: `D:\CNTDATA\CNTA_ML_Project\src\analysis\feature_extractor.py`
- Test: `D:\CNTDATA\CNTA_ML_Project\tests\test_feature_extractor_branch_graph_tracing.py`

**Step 1: Write the failing test**

- Add tests that verify:
  - tracing stops when only hard-angle-invalid branches remain
  - tracing stops on repeated low-value short continuations
  - tracing may terminate early after reaching the opposite boundary with stable direction

Suggested run:

```bash
pytest D:\CNTDATA\CNTA_ML_Project\tests\test_feature_extractor_branch_graph_tracing.py -k stop -v
```

Expected before implementation: failing stop-condition assertions.

**Step 2: Implement minimal stopping logic**

- Add:
  - repeated-soft-turn guard
  - min span gain guard
  - max branch steps guard
  - max cumulative turn guard
  - opposite-boundary completion check

**Step 3: Run the stop tests**

```bash
pytest D:\CNTDATA\CNTA_ML_Project\tests\test_feature_extractor_branch_graph_tracing.py -k stop -v
```

Expected: stop-condition tests pass.

### Task 5: Update the single-image demo for v1c inspection

**Files:**
- Modify: `D:\CNTDATA\CNTA_ML_Project\tools\generate_branch_graph_tracing_demo.py`

**Step 1: Write the failing check**

- Define the expected outputs:
  - highlighted boundary seeds
  - highlighted `main_path`
  - candidate-path overlay
  - score breakdown in `summary.json`

Suggested run:

```bash
conda run -n LAB_AGENT python D:\CNTDATA\CNTA_ML_Project\tools\generate_branch_graph_tracing_demo.py --angle-limit-deg 45 --angle-hard-deg 70 --beam-width 4 --max-paths 20
```

Expected before implementation: no seed overlay, no `main_path`, no score breakdown.

**Step 2: Implement the demo updates**

- Draw boundary seeds distinctly.
- Render the winning `main_path` more prominently than debug candidates.
- Include score breakdown fields in `summary.json`.

**Step 3: Run the demo**

```bash
conda run -n LAB_AGENT python D:\CNTDATA\CNTA_ML_Project\tools\generate_branch_graph_tracing_demo.py --angle-limit-deg 45 --angle-hard-deg 70 --beam-width 4 --max-paths 20
```

Expected: a new report directory with updated visuals and summary fields.

### Task 6: Verify against the image-spanning objective

**Files:**
- Output: `D:\CNTDATA\CNTA_ML_Project\reports\branch_graph_tracing_demo_<timestamp>\`

**Step 1: Run syntax verification**

```bash
@'
import ast
from pathlib import Path
for path in [
    Path(r"D:\CNTDATA\CNTA_ML_Project\src\analysis\feature_extractor.py"),
    Path(r"D:\CNTDATA\CNTA_ML_Project\tools\generate_branch_graph_tracing_demo.py"),
    Path(r"D:\CNTDATA\CNTA_ML_Project\tests\test_feature_extractor_branch_graph_tracing.py"),
]:
    ast.parse(path.read_text(encoding="utf-8"))
    print(path.name)
'@ | conda run -n LAB_AGENT python -
```

Expected: all listed files print successfully.

**Step 2: Run the focused tracing tests**

```bash
pytest D:\CNTDATA\CNTA_ML_Project\tests\test_feature_extractor_branch_graph_tracing.py -v
```

Expected: all tracing tests pass.

**Step 3: Run the demo on the representative image**

```bash
conda run -n LAB_AGENT python D:\CNTDATA\CNTA_ML_Project\tools\generate_branch_graph_tracing_demo.py --angle-limit-deg 45 --angle-hard-deg 70 --beam-width 4 --max-paths 20
```

Expected: a new report directory is printed.

**Step 4: Inspect the output summary**

- Confirm the chosen `main_path`:
  - starts from a boundary seed
  - spans farther than v1b on the same image
  - is explainable by direction continuity instead of local branch greed

Plan complete and saved to `docs/plans/2026-03-26-branch-graph-tracing-v1c.md`. Two execution options:

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

Which approach?
