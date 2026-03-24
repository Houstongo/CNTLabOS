# 100000x CNT 结构保持损失实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**目标：** 在现有 `100000x` paper reproduction 基线上增加一条结构保持实验线，对比 `Baseline-Paper`、`Exp-C (clDice)` 和 `Exp-D (clDice + RidgeAux)`，同时保持 backbone、数据、split、optimizer 和训练日程不变。

**架构：** 继续复用 `experiments/cnt_paper_repro` 下现有的数据、模型和 staged training 流程。在不破坏 baseline 的前提下，扩展 loss 模块，加入可选的 `clDice` 和 ridge-guided auxiliary loss，再补齐结构诊断指标和 hard-case 可视化导出。

**技术栈：** Python、PyTorch、torchvision、OpenCV、NumPy、已有 `lab_agent` conda 环境

---

### 任务 1：补齐结构实验配置文件

**文件：**
- 修改：`D:\CNTDATA\CNTA_ML_Project\experiments\cnt_paper_repro\configs\paper_100000x.yaml`
- 新建：`D:\CNTDATA\CNTA_ML_Project\experiments\cnt_paper_repro\configs\paper_100000x_cldice.yaml`
- 新建：`D:\CNTDATA\CNTA_ML_Project\experiments\cnt_paper_repro\configs\paper_100000x_cldice_ridge.yaml`
- 测试：`D:\CNTDATA\CNTA_ML_Project\tests\test_cnt_paper_repro_config.py`

**步骤 1：先写失败测试**

扩展配置测试，断言：

- baseline 配置仍能正常加载
- `clDice` 配置中显式包含 `lambda_cl`
- `clDice + RidgeAux` 配置中显式包含 `lambda_cl` 和 `lambda_ridge`
- 三个配置的数据路径、patch 大小、threshold、seed、backbone 都保持一致

**步骤 2：运行测试，确认先失败**

运行：

```powershell
conda run -n lab_agent python -m unittest D:\CNTDATA\CNTA_ML_Project\tests\test_cnt_paper_repro_config.py -v
```

预期：失败，因为新的结构实验配置还不存在。

**步骤 3：写最小实现**

新建两个配置文件，只允许 phase-2 loss 权重不同，其余条件全部和 baseline 保持一致。

**步骤 4：再次运行测试，确认通过**

运行同样的 unittest 命令，预期 `PASS`。

---

### 任务 2：补齐结构辅助模块

**文件：**
- 新建：`D:\CNTDATA\CNTA_ML_Project\experiments\cnt_paper_repro\structural.py`
- 新建：`D:\CNTDATA\CNTA_ML_Project\tests\test_cnt_paper_repro_structural.py`

**步骤 1：先写失败测试**

添加测试，验证：

- soft skeleton helper 返回有限值
- ridge-response helper 输出范围被归一化到 `[0, 1]`
- 输出空间尺寸和输入一致
- 对全零图、简单直线图、随机图都能稳定运行

**步骤 2：运行测试，确认先失败**

运行：

```powershell
conda run -n lab_agent python -m unittest D:\CNTDATA\CNTA_ML_Project\tests\test_cnt_paper_repro_structural.py -v
```

预期：失败，因为 `structural.py` 还不存在。

**步骤 3：写最小实现**

实现：

- 预测概率图的 soft skeleton 近似
- 从灰度 patch 提 ridge-response map
- 响应归一化和数值稳定辅助函数

第一版重点是可解释、稳定、可调试。

**步骤 4：再次运行测试，确认通过**

运行同样的 unittest 命令，预期 `PASS`。

---

### 任务 3：实现 clDice 和 RidgeAux

**文件：**
- 修改：`D:\CNTDATA\CNTA_ML_Project\experiments\cnt_paper_repro\losses.py`
- 使用：`D:\CNTDATA\CNTA_ML_Project\experiments\cnt_paper_repro\structural.py`
- 新建：`D:\CNTDATA\CNTA_ML_Project\tests\test_cnt_paper_repro_losses_structural.py`

**步骤 1：先写失败测试**

添加测试，验证：

- `clDice` 返回有限标量
- 对 toy case，连通版本的 `clDice loss` 低于断裂版本
- `RidgeAux` 返回有限标量
- 若预测骨架落在亮脊线上，`RidgeAux` 应优于明显偏移的预测
- phase-2 的组合损失可以正常 backward

**步骤 2：运行测试，确认先失败**

运行：

```powershell
conda run -n lab_agent python -m unittest D:\CNTDATA\CNTA_ML_Project\tests\test_cnt_paper_repro_losses_structural.py -v
```

预期：失败，因为结构损失尚未接入 `losses.py`。

**步骤 3：写最小实现**

扩展 `compute_phase_loss`，支持读取：

- `lambda_cl`
- `lambda_ridge`

并实现：

- `cldice_loss_from_probs(...)`
- `ridge_aux_loss_from_logits(...)`

要求：

- 当两个权重都为 `0` 时，baseline 路径完全不变
- baseline 的数值行为不能被新代码影响

**步骤 4：再次运行测试，确认通过**

运行同样的 unittest 命令，预期 `PASS`。

---

### 任务 4：扩展训练日志和 checkpoint 元数据

**文件：**
- 修改：`D:\CNTDATA\CNTA_ML_Project\experiments\cnt_paper_repro\train.py`
- 测试：`D:\CNTDATA\CNTA_ML_Project\tests\test_cnt_paper_repro_train_smoke.py`

**步骤 1：先写失败测试**

扩展 smoke test，断言：

- 结构实验时返回的 metrics 包含 `loss_cldice` 和 `loss_ridge`
- history 行里会写入这些字段
- checkpoint 仍保留 epoch、phase_name、best_metric 等元数据

**步骤 2：运行测试，确认先失败**

运行：

```powershell
conda run -n lab_agent python -m unittest D:\CNTDATA\CNTA_ML_Project\tests\test_cnt_paper_repro_train_smoke.py -v
```

预期：失败，因为当前 trainer 还没有记录结构损失字段。

**步骤 3：写最小实现**

修改 trainer，使其：

- baseline 仍只记录 `dice / orientation / total`
- 结构实验组会额外记录 `cldice / ridge`
- summary、history、checkpoint 都兼容旧 baseline

**步骤 4：再次运行测试，确认通过**

运行同样的 unittest 命令，预期 `PASS`。

---

### 任务 5：补充结构诊断指标

**文件：**
- 修改：`D:\CNTDATA\CNTA_ML_Project\experiments\cnt_paper_repro\metrics.py`
- 修改：`D:\CNTDATA\CNTA_ML_Project\experiments\cnt_paper_repro\evaluate.py`
- 新建：`D:\CNTDATA\CNTA_ML_Project\tests\test_cnt_paper_repro_metrics_structural.py`

**步骤 1：先写失败测试**

添加测试，验证：

- endpoint count 是有限且非负
- connected-component count 是有限且非负
- foreground ratio 有界
- 对 toy case，碎片化 mask 的 fragment/endpoint 数量高于连通 mask

**步骤 2：运行测试，确认先失败**

运行：

```powershell
conda run -n lab_agent python -m unittest D:\CNTDATA\CNTA_ML_Project\tests\test_cnt_paper_repro_metrics_structural.py -v
```

预期：失败，因为当前评估脚本只输出像素级指标。

**步骤 3：写最小实现**

增加结构诊断项：

- foreground ratio
- connected-component count
- fragment count
- endpoint count
- skeleton length

要求在导出的 CSV 中清楚区分：

- 弱标签拟合指标
- 结构诊断指标

**步骤 4：再次运行测试，确认通过**

运行同样的 unittest 命令，预期 `PASS`。

---

### 任务 6：实现 hard-case 对照板导出

**文件：**
- 修改：`D:\CNTDATA\CNTA_ML_Project\experiments\cnt_paper_repro\visualize.py`
- 新建：`D:\CNTDATA\CNTA_ML_Project\tools\build_cnt_hardcase_manifest.py`
- 新建：`D:\CNTDATA\CNTA_ML_Project\tests\test_cnt_paper_repro_visualize_structural.py`

**步骤 1：先写失败测试**

添加可视化 smoke test，断言：

- 每个请求 patch 都会导出一张 panel
- panel 至少包含 `original / weak / probability / thresholded mask`
- 当同时提供 baseline 和 variant 时，可以导出差异对照图

**步骤 2：运行测试，确认先失败**

运行：

```powershell
conda run -n lab_agent python -m unittest D:\CNTDATA\CNTA_ML_Project\tests\test_cnt_paper_repro_visualize_structural.py -v
```

预期：失败，因为当前 visualizer 还不支持 cross-run 对照。

**步骤 3：写最小实现**

扩展 visualizer，使其支持：

- 同一 patch 下导出 baseline 和 variant 的对照
- 可选 difference map
- 按固定 hard-case manifest 批量导出

同时增加一个 helper，用于从当前 test patch manifest 生成第一版 hard-case manifest，后面可人工再筛。

**步骤 4：再次运行测试，确认通过**

运行同样的 unittest 命令，预期 `PASS`。

---

### 任务 7：先确认 baseline 链路不变

**文件：**
- 使用：`D:\CNTDATA\CNTA_ML_Project\experiments\cnt_paper_repro\configs\paper_100000x.yaml`
- 使用：`D:\CNTDATA\CNTA_ML_Project\experiments\cnt_paper_repro\train.py`
- 使用：`D:\CNTDATA\CNTA_ML_Project\experiments\cnt_paper_repro\evaluate.py`
- 使用：`D:\CNTDATA\CNTA_ML_Project\experiments\cnt_paper_repro\visualize.py`

**步骤 1：运行 baseline 验证**

重新确认 baseline 仍然可以：

- 训练得到 best checkpoint
- 导出评估结果
- 导出 test patch 可视化

**步骤 2：记录对照路径**

把 baseline 的 run 目录和 review 目录固定下来，作为后续 `Exp-C` 和 `Exp-D` 的对照基准。

---

### 任务 8：运行 Exp-C

**文件：**
- 使用：`D:\CNTDATA\CNTA_ML_Project\experiments\cnt_paper_repro\configs\paper_100000x_cldice.yaml`

**步骤 1：训练**

运行 `lambda_cl > 0, lambda_ridge = 0` 的结构实验。

**步骤 2：评估**

导出：

- 弱标签拟合指标
- 结构诊断指标
- test patch review panels

**步骤 3：和 baseline 对比**

重点检查：

- fragment count 是否下降
- endpoint count 是否下降
- foreground ratio 是否失控上升
- hard-case 上是否明显更连贯

---

### 任务 9：运行 Exp-D

**文件：**
- 使用：`D:\CNTDATA\CNTA_ML_Project\experiments\cnt_paper_repro\configs\paper_100000x_cldice_ridge.yaml`

**步骤 1：训练**

运行 `lambda_cl > 0` 且 `lambda_ridge > 0` 的结构实验。

**步骤 2：评估**

导出与 `Exp-C` 同样的指标和可视化结果。

**步骤 3：和 baseline、Exp-C 对比**

重点检查 `RidgeAux` 是否：

- 比 `clDice` 单独使用时更能补上细桥
- 没有在明显空隙里大面积误连
- 在“弱标签略断但原图仍有亮脊线”的区域更合理

---

### 任务 10：输出结构实验总结

**文件：**
- 新建：`D:\CNTDATA\CNTA_ML_Project\experiments\cnt_paper_repro\reports\structural_compare_100000x\summary.md`
- 新建：`D:\CNTDATA\CNTA_ML_Project\experiments\cnt_paper_repro\reports\structural_compare_100000x\metrics_comparison.csv`
- 新建：`D:\CNTDATA\CNTA_ML_Project\experiments\cnt_paper_repro\reports\structural_compare_100000x\hardcase_review_manifest.csv`

**步骤 1：汇总结果**

把 baseline、`Exp-C`、`Exp-D` 的关键结果汇总到一个统一目录。

**步骤 2：写结论**

总结：

- `clDice` 是否让断裂更贵
- `RidgeAux` 是否让“有图像证据的补连”更明显
- 还存在哪些典型误连
- 这条实验线是否值得进入下一轮更严格验证

**步骤 3：核对交付物**

确认最终目录至少包含：

- 配置快照
- summary markdown
- 指标对比 CSV
- hard-case 对照图

---

计划已保存到 `docs/plans/2026-03-25-cnt-structural-loss.md`。后续有两种执行方式：

**1. 本会话继续执行**：我按这份计划逐项实现并每个阶段给你回报

**2. 单独新会话执行**：开一个新会话，专门按这份计划批量推进

如果你愿意，我建议就在这个会话里继续，从“任务 1：补齐结构实验配置文件”开始。
