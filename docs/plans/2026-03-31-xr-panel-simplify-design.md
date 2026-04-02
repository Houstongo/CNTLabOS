# XR Panel Simplify Design

**Date:** 2026-03-31

**Goal**

将 XR 标准报告面板从“6 张图 + 全部 L1-L4 文本”改成更适合人工审核的精简版，降低信息密度，提升单张样本的可读性。

**Approved Direction**

用户确认采用方案 B：

- 左侧只保留 3 个可视化区块
  - Original ROI
  - Mask
  - L2 Overlay
- 右侧只保留 `Global` 和 `L2` 摘要
- 不再显示 `L1 / L3 / L4` overlay
- 不再显示 `L1 / L3 / L4` 文本统计

**Why This Design**

- 原始 6 图面板在桌面查看时过于拥挤，字号偏小，真正做“是否入库”判断时反而难以阅读。
- 当前审核主口径已经固定偏向 `L2`，继续同时展示 L1/L3/L4 会增加视觉负担，但不会显著提升决策效率。
- 保留 `Original + Mask + L2 Overlay` 可以同时满足：
  - 看原始纹理
  - 看分割质量
  - 看最终用于分析的主阈值骨架/分支结果

**Layout**

- 使用 `2 x 2` 主网格
- 左列：
  - 第 1 行：Original ROI
  - 第 2 行：Mask
- 中列：
  - 上下合并显示 L2 Overlay
- 右列：
  - 上下合并显示文本摘要

**Summary Content**

保留：

- Global
  - sample_id
  - magnification
  - model/device
  - status/elapsed
  - patches/threshold
  - density
  - alignment
  - mean_phi_deg
  - diameter
  - diam mean/p50/p75

- L2
  - min_length_factor
  - branch_count
  - curvature_label
  - p70 sqrt/len
  - p75 sqrt/len
  - mean sqrt/len
  - trimmed mean sqrt/len
  - waviness/tortuosity
  - diam p30/p50/p75
  - curv pts/diam pts

移除：

- 全部 `L1 / L3 / L4` 文本段落
- 全部非 L2 overlay

**Implementation Notes**

- 主改动集中在 `tools/generate_xr_slice_standard_batch.py`
- `build_standard_summary_sections()` 需要只输出 `Global + L2`
- `render_panel()` 需要改版布局并只绘制 `L2 Overlay`
- 为了快速验证，先用单张样本重生成，再视需要整批 rerender

**Verification**

- 运行单张 `--image-id` 生成，确认新 panel 只包含 3 张图
- 检查右侧只含 `Global` 和 `L2`
- 检查字号与图块尺寸是否比旧版明显更清晰
