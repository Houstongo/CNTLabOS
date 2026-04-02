# XR Junction Features Design

**Date:** 2026-03-31

**Goal**

为 XR 审核/建模链路补充一组轻量拓扑特征，用于描述网络交叉复杂度，而不改变现有曲率与 waviness 算法。

**Approved Direction**

采用最小接入方案：

- 保持现有 `curvature / waviness / alignment / diameter / density` 不变
- 新增 `junction_count`
- 新增归一化指标 `junction_ratio`

**Why This Design**

- `junction` 与弯曲、缠结、成网复杂度有关，但不是同一个物理量
- 比起手动“修正”曲率，把 `junction` 作为并列特征输入模型更稳健
- 这次只做轻量接入，优先满足建模分析，不引入新的曲率口径

**Definition**

- `junction_mask`: skeleton 中邻居数 `>= 3` 的像素
- `junction_count`: 被 `minus junction` 步骤去掉的 junction 像素个数
- `skeleton_length_px`: skeleton 前景像素总数
- `skeleton_length_um = skeleton_length_px / px_per_um`
- `junction_ratio = junction_count / skeleton_length_px`

这样定义的优点：

- `junction_count` 与当前去 junction 操作完全一致，解释直观
- `junction_ratio` 能在不同样本覆盖度之间做更公平比较

**Scope**

接入位置：

- `tools/generate_xr_slice_standard_batch.py`
  - 分析阶段计算 junction 特征
  - `features.json` 写入
  - `summary.csv/json` 展平写入
- 新增回填脚本
  - 用于把现有 XR 批次结果补上 junction 字段

**Out of Scope**

- 不修改 `minus junction` 分支切割策略
- 不引入 `junction halo exclusion`
- 不改动现有曲率、waviness 数值定义

**Verification**

- 单张样本生成后，`features.json` 中出现新字段
- `summary.csv` 中出现新列
- 对现有 `slice_standard_batch_20260331_005741` 完成回填
