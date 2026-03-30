# Exp C Orientation Finetune Design

**Date:** 2026-03-29

**Goal:** 基于原始标准 `Exp C` 的训练终点权重，额外进行 `3` 轮仅含 `Dice + Orientation` 的微调实验，并与原始 `Exp C` 在 `test10` 全图预测上进行对比。

## Context

- 当前主线模型位于 `experiments/cnt_paper_repro`。
- 标准 `Exp C` 配置为 `configs/paper_100000x_cldice.yaml`。
- 原始 `Exp C` run 目录为 `runs/cnt_paper_repro_100000x_center768_cldice_seed42`。
- 该 run 已保留 `pre_resume_last_model.pth` / `pre_resume_best_model.pth`，可作为“未续训前原始 Exp C”的干净起点。

## Experiment Definition

### Baseline for comparison

- 比较对象使用原始 `Exp C`：
  - checkpoint: `pre_resume_best_model.pth`
  - training endpoint for resume source: `pre_resume_last_model.pth`

### New finetune experiment

- 新实验名称：`cnt_paper_repro_100000x_center768_cldice_orifinetune`
- 起始权重：原始 `Exp C` 的 `pre_resume_last_model.pth`
- 额外轮次：`3`
- 微调 loss：
  - `dice_weight = 0.6`
  - `orientation_weight = 3e-7`
  - `lambda_cl = 0.0`
  - `lambda_ridge = 0.0`
- 不覆盖原始 run，输出到新的 run 目录

## Why this experiment

- 原始 `Exp C` 的后段训练中包含 `clDice`，用户希望验证：
  - 若在结构约束之后，额外用更保守的 `Dice + Orientation` 做短程收尾，
  - 是否能让预测结果从“偏厚”向“更收一些”方向调整。
- `3e-7` 的 `orientation_weight` 比标准 `Exp C` 中的 `1e-7` 略强，但仍属于轻约束，不会完全盖过 `Dice`。

## Output Plan

- 训练输出：
  - 新 run 的 `summary.json`
  - 新 run 的 `history.csv`
  - `best_model.pth` / `last_model.pth`
- 可视化输出：
  - 使用桌面 `C:\Users\clearlove\Desktop\text10`
  - 自动裁掉图像底部黑色信息条
  - 生成 `Original / ExpC-Orig / ExpC-OriFinetune` 的全图对比面板
  - 同时保留新模型单独 mask 结果

## Risk Notes

- 由于这是基于既有 checkpoint 的短程微调，改善幅度可能有限。
- 如果 `Orientation` 约束过弱，则结果可能几乎等同于纯 `Dice` 收尾。
- 如果需要进一步实验，可在本轮之后再比较 `3e-7` 与 `1e-6` 的差异，但本轮先不扩展扫描。
