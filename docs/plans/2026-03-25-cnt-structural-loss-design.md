# 100000x CNT 结构保持损失实验设计

**日期：** 2026-03-25

**目标**

在当前 `100000x` CNT-SEM 数据上，基于已经搭好的 `ResNet34-U-Net` paper reproduction 基线，设计一条新的结构保持实验线。该实验线继续使用 `WCNTSegNET` 弱标签，但额外比较 `clDice` 与图像驱动的 `RidgeAux`，验证它们能否在“优先补连、允许少量误连”的前提下改善细长 CNT 的连续性。

---

## 1. 问题定义

当前 `experiments/cnt_paper_repro` 这条 paper 风格基线虽然已经能较好拟合弱标签，但仍然会出现：

- 细长 CNT 中间断裂
- 原图看起来连续的细桥没有被连上
- 概率图里似乎有响应，但经过二值化后仍被切断

这和当前损失设计本身有关：

- `Dice` 主要约束区域重叠
- `Orientation MSE` 主要约束全局方向分布
- 这两项都不会直接要求“局部连通性必须保持”

用户已经明确了这轮实验的优先级：

- 优先补连
- 宁可多连一点，也尽量不要把本来连续的 CNT 断开

这个优先级将直接决定新实验的损失设计与评价标准。

---

## 2. 固定不变的实验条件

为了让结果可解释，这条新实验线必须固定以下条件不变：

- 源数据集：`D:\CNTDATA\CNTA_ML_Project\experiments\cnt_loss_compare\datasets\zzy_mid_100000_train50_test50_v2`
- 派生 patch 数据集：`D:\CNTDATA\CNTA_ML_Project\experiments\cnt_paper_repro\datasets\zzy_mid_100000_patch768_center_paper_v1`
- 数据划分：`train 40 / val 10 / test 50 / reserve 2`
- 监督来源：当前 `WCNTSegNET` 弱标签
- patch 大小：`768x768`
- backbone：`ResNet34-U-Net`
- optimizer：`Adam`
- 学习率：`5e-4`
- 默认二值化阈值：`0.7`
- 随机种子：`42`

这条实验线不能同时再改这些因素：

- 不能换弱标签生成方式
- 不能换 backbone
- 不能换倍率
- 不能换 train/test split

真正变化的变量只能是结构损失。

---

## 3. 实验组设计

这轮实验先做三组。

### 3.1 Baseline-Paper

这组就是当前已经实现的 paper 风格基线：

- Phase 1：`Dice`
- Phase 2：`0.6 * Dice + 1e-7 * OrientationMSE`

作用：

- 保留一个干净的 CNTSegNet 风格参考组
- 作为后续所有结构损失实验的对照组

### 3.2 Exp-C：Paper + clDice

训练阶段设计：

- Phase 1：`Dice`
- Phase 2：`0.6 * Dice + 1e-7 * OrientationMSE + lambda_cl * clDice`

作用：

- 单独验证 `clDice` 是否能让“断裂”变得更贵
- 先看拓扑约束本身有没有帮助

### 3.3 Exp-D：Paper + clDice + RidgeAux

训练阶段设计：

- Phase 1：`Dice`
- Phase 2：`0.6 * Dice + 1e-7 * OrientationMSE + lambda_cl * clDice + lambda_ridge * RidgeAux`

作用：

- 保留 `clDice` 的连通性压力
- 再加入来自原图的结构证据，让“补连”不是盲目补，而是沿着原图真实可见的 CNT 脊线去补

这组是本轮实验的主目标。

---

## 4. 数学直觉

### 4.1 为什么 Dice 和 Orientation 还不够

当前分割项可以写成：

```text
L_dice = 1 - (2 * sum(P * Y) + s) / (sum(P) + sum(Y) + s)
```

其中：

- `P = sigmoid(logits)`
- `Y` 是 `WCNTSegNET` 弱标签

它关注的是区域重叠，不敏感于“这根丝是不是被掐断成两段”。

当前方向项可以写成：

```text
L_ori = sum_b (h_pred(b) - h_img(b))^2
```

其中：

- `h_pred` 是预测 likelihood map 的 FFT 方向直方图
- `h_img` 是原图的 FFT 方向直方图

这项是全局统计量。只要整体方向分布差不多，局部桥接缺失也可能不被明显惩罚。

### 4.2 clDice

`clDice` 关注骨架级重叠，而不是单纯区域重叠：

```text
Tprec = |Skel(P_bin) intersect Y| / |Skel(P_bin)|
Trec  = |Skel(Y) intersect P_bin| / |Skel(Y)|
clDice = 2 * Tprec * Trec / (Tprec + Trec)
L_cl = 1 - clDice
```

直觉上：

- 连续的中心线更重要
- 被切断的细丝会被更明显地惩罚
- 单纯把前景涂粗，并不能很好掩盖拓扑错误

但它也有已知风险：

- 如果两根本来分开的 CNT 很近，`clDice` 也可能倾向于错误桥接
- 所以它不能单独使用

### 4.3 RidgeAux

辅助结构项必须直接看原图，而不是只看弱标签：

```text
L_ridge = 1 - sum( S(P) * R(X) ) / (sum(S(P)) + eps)
```

其中：

- `X` 是原始灰度 patch
- `R(X)` 是从原图提取出的 ridge response map
- `S(P)` 是预测的软骨架表示

直觉上：

- 如果预测骨架正好落在原图真实的亮脊线上，loss 变小
- 如果模型想在没有图像证据的空隙里胡乱连桥，loss 不会鼓励它
- 如果原图中 CNT 还在延续，但预测中断掉了，这项会对这种断裂更敏感

也就是说，这一项的核心价值在于：

- `clDice` 负责让“断裂变贵”
- `RidgeAux` 负责让“补连有图像证据”

---

## 5. RidgeAux 的实现取向

第一版 RidgeAux 不追求最复杂，而追求：

- 结构清晰
- 数值稳定
- 易于解释
- 与现有 `cnt_paper_repro` 框架容易集成

### 建议的第一版实现

- 从灰度原图提一个 ridge-response map
- 把响应归一化到 `[0, 1]`
- 不把 ridge map 当作额外输入通道
- 只在 loss 中使用它

### 推荐的图像证据来源

第一版建议采用 Frangi-like 或 Hessian-based 的 ridge response。

原因：

- CNT 在 SEM 图里本质上就是细长亮脊线
- ridge filter 比普通 edge loss 更贴近“补连”目标
- 它给的是局部结构证据，不只是全局方向统计

### 暂不纳入第一版的方案

以下方法先不进第一轮：

- endpoint continuation loss
- structure-tensor coherence loss
- learned auxiliary branch

这些都可以在第一版跑通后再展开。

---

## 6. 训练日程

三组都维持同一个两阶段框架：

- Phase 1：`6 epochs`
- Phase 2：`3 epochs`

三组定义如下：

- `Baseline-Paper`
  - Phase 1：`Dice`
  - Phase 2：`0.6 * Dice + 1e-7 * OrientationMSE`
- `Exp-C`
  - Phase 1：`Dice`
  - Phase 2：`0.6 * Dice + 1e-7 * OrientationMSE + lambda_cl * clDice`
- `Exp-D`
  - Phase 1：`Dice`
  - Phase 2：`0.6 * Dice + 1e-7 * OrientationMSE + lambda_cl * clDice + lambda_ridge * RidgeAux`

初始权重策略：

- `0.6` 和 `1e-7` 保持不变
- `lambda_cl` 与 `lambda_ridge` 做成显式配置项
- 第一版先用保守值，后续允许小范围扫参

第一版不要把结构损失权重硬编码在 Python 里。

---

## 7. 评价思路

由于监督是弱标签，常规 Dice/IoU 只能作为拟合诊断，不能作为最终科学结论。

### 7.1 主要证据

主要证据应该是定性可视化，重点看：

- 原本连续的 CNT 是否更容易连上
- 弱对比区里的细桥是否更容易保住
- 是否把相邻平行 CNT 过度糊成一片
- 交叉区和局部复杂结构是否更自然

### 7.2 次级诊断指标

仍然保留弱标签拟合指标，但只当作辅证：

- Dice
- IoU
- Precision
- Recall

同时增加更符合目标的结构诊断指标：

- connected-component count
- fragment count
- endpoint count
- skeleton length
- foreground ratio

这些指标虽然仍不是 GT 指标，但能帮助判断一个方法究竟是在“改善连通性”，还是单纯把前景变白。

### 7.3 固定 hard-case 对照板

应固定一批 hard-case patch，作为这条实验线的主评审集。优先挑：

- 细丝易断区域
- 交叉明显区域
- 噪声强区域
- 原图仍可见连续脊线、但弱标签偏断的区域

每张图导出：

- original patch
- weak mask
- probability map
- thresholded mask
- 可选 baseline vs variant difference map

这个 hard-case 对照板才是本轮实验最重要的输出之一。

---

## 8. 成功标准

这条实验线若要算成功，至少要同时满足：

- 与 `Baseline-Paper` 相比，hard-case 上的连续性更好
- 这种提升不能只是因为整张图更白
- 在“弱标签略断但原图仍有连续脊线”的区域，`Exp-D` 比 `Exp-C` 更合理

重点观察的失败模式：

- `clDice` 系统性误连相邻平行 CNT
- `RidgeAux` 太弱，几乎不起作用
- `RidgeAux` 太强，导致顺着噪声脊线瞎连
- probability map 改善了，但一到 `0.7` 阈值后又断掉

---

## 9. 范围边界

这一版设计明确不包含：

- 新增人工 GT 标注
- 混合倍率训练
- backbone 改动
- `WCNTSegNET` 弱标签生成改动
- endpoint loss 的第一版实现

这些都可以在第一轮结构实验有结果后再考虑。

---

## 10. 推荐下一步

建议继续沿用 `experiments/cnt_paper_repro` 这条线实现新实验，而不是新开一套重复框架。这样可以保证：

- baseline 和结构增强组直接可比
- 数据、模型、训练入口共用
- 只在 loss 与配置层面引入新变量

推荐执行顺序：

1. 先确认 baseline 导出链路不变
2. 实现 `clDice`
3. 跑 `Exp-C`
4. 实现 `RidgeAux`
5. 跑 `Exp-D`
6. 用固定 hard-case 对照板比较三组结果
