# Intact Skeleton Probe Gray Consistency Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the intact skeleton probe prefer junction exits whose preprocessed grayscale stays consistent with the recent traced path.

**Architecture:** Keep tracing topology unchanged, retain the current angle and anti-hook logic, and add a post-preprocess grayscale sampling layer that feeds a soft consistency score into junction exit ranking.

**Tech Stack:** Python, NumPy, OpenCV, Matplotlib, existing `FeatureExtractor` preprocessing helpers

---

### Task 1: Preserve preprocess image and add grayscale sampling helpers

**Files:**
- Modify: `D:\CNTDATA\CNTA_ML_Project\tools\generate_intact_skeleton_probe.py`

### Task 2: Add grayscale consistency to junction expansion

**Files:**
- Modify: `D:\CNTDATA\CNTA_ML_Project\tools\generate_intact_skeleton_probe.py`

### Task 3: Smoke test on the selected No28 bottom image

**Files:**
- Output: `D:\CNTDATA\CNTA_ML_Project\reports\...`
