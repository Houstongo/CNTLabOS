# Branch Graph Tracing V1 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a single-image branch-graph tracing prototype that links skeleton branches across junctions using a `45°` direction prior and lightweight local scoring.

**Architecture:** Reuse the current ordered-branch extraction to build graph edges, derive endpoint/junction nodes from the skeleton, precompute branch-end features, traverse the branch graph greedily from endpoints, and render a validation panel of reconstructed paths.

**Tech Stack:** Python, NumPy, OpenCV, scikit-image, matplotlib, existing `FeatureExtractor`

---

### Task 1: Add graph-building helpers

**Files:**
- Modify: `D:\CNTDATA\CNTA_ML_Project\src\analysis\feature_extractor.py`

**Step 1: Add node and branch feature helpers**

- Build endpoint/junction node records.
- Build branch edge records with:
  - ordered coords
  - endpoint directions
  - length
  - width summary

**Step 2: Add short-branch filtering**

- Remove obviously short noise edges before tracing.

### Task 2: Add tracing helpers

**Files:**
- Modify: `D:\CNTDATA\CNTA_ML_Project\src\analysis\feature_extractor.py`

**Step 1: Add a junction scoring helper**

- Score continuation candidates with:
  - angle
  - width consistency
  - intensity consistency
  - short-branch penalty

**Step 2: Add the greedy tracing loop**

- Start from endpoints
- Traverse until stop conditions
- Produce reconstructed path records

### Task 3: Create a single-image validation script

**Files:**
- Create: `D:\CNTDATA\CNTA_ML_Project\tools\generate_branch_graph_tracing_demo.py`

**Step 1: Load one real `100000x` image**

- Use a known representative sample.

**Step 2: Run the tracing prototype**

- Extract skeleton graph and reconstructed paths.

**Step 3: Render a validation panel**

- original ROI
- skeleton graph
- node overlay
- reconstructed paths
- path metrics table

### Task 4: Verify

**Files:**
- Output: a new report directory

**Step 1: Run the single-image demo**

- Ensure outputs are produced successfully.

**Step 2: Visually inspect plausibility**

- Confirm reconstructed paths look directionally coherent.
