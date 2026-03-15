# CNTA特征提取系统性评估启动指南

**版本**：v1.0
**日期**：2026-03-15
**目标**：全面评估v2.1算法在现有数据上的表现，识别失败模式和改进方向

---

## 一、评估脚本说明

### 1.1 完整评估脚本：`systematic_evaluator.py`

**功能**：
- 批量运行特征提取算法
- 记录详细的失败原因和成功指标
- 分析失败样品的特征分布（倍率、来源、密度等）
- 生成三类评估报告：文本报告、JSON数据、CSV统计表

**运行方式**：
```bash
cd d:\CNTDATA\CNTA_ML_Project
python backend/core/systematic_evaluator.py [选项]
```

**可选参数**：
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--reprocess` | 重新处理已处理过的图像 | False |
| `--limit N` | 只处理前 N 张图像 | 全部 |
| `--source ZZY/XR` | 只处理指定来源 | 全部 |
| `--output DIR` | 输出目录 | reports/eval_YYYYMMDD_HHMMSS |

**示例**：
```bash
# 完整评估所有未处理的图像
python backend/core/systematic_evaluator.py

# 只评估ZZY数据集的前100张
python backend/core/systematic_evaluator.py --source ZZY --limit 100

# 重新评估所有图像
python backend/core/systematic_evaluator.py --reprocess
```

---

### 1.2 快速测试脚本：`quick_eval.py`

**功能**：
- 只评估前50张图像
- 用于快速检查算法表现和调试
- 适合在完整评估前进行小规模测试

**运行方式**：
```bash
cd d:\CNTDATA\CNTA_ML_Project
python backend/core/quick_eval.py
```

---

## 二、评估指标说明

### 2.1 状态分类

| 状态 | 说明 | 处理方式 |
|------|------|----------|
| **success** | 特征提取成功，所有指标正常 | 记录统计信息 |
| **warning** | 特征提取成功，但部分指标异常 | 记录警告信息，计入统计数据 |
| **error** | 特征提取失败 | 记录错误详情，不计入统计 |
| **skipped** | 文件不存在或图像读取失败 | 跳过处理 |

---

### 2.2 质量检查规则

评估脚本会自动检查以下质量指标：

#### Diameter（管径）
| 条件 | 警告信息 | 可能原因 |
|------|----------|----------|
| `diameter = N/A` | 低倍率或提取失败 | 倍率 < 20kx |
| `diameter < 0` | 提取失败 | 骨架化或距离变换失败 |
| `diameter < 5nm` | 过小（可能是噪声） | 误判噪声为CNT |
| `diameter > 50nm` | 过大（可能是束径或团簇） | CNT粘连、闭运算连桥 |

#### Density（密度）
| 条件 | 警告信息 | 可能原因 |
|------|----------|----------|
| `density = 0%` | 二值化失败 | 阈值选择错误 |
| `density > 95%` | 背景误判 | 样品过密或背景过亮 |

#### Alignment（对齐度）
| 条件 | 警告信息 | 可能原因 |
|------|----------|----------|
| `alignment = 0` | 各向同性或提取失败 | 骨架分支数为0 |
| `alignment < -0.5` | 超出理论范围 | 算法错误 |
| `alignment > 1.0` | 超出理论范围 | 算法错误 |
| `hof_method = structure_tensor` | 低倍率（系统偏置） | 倍率 < 20kx |

#### Curvature（曲率）
| 条件 | 警告信息 | 可能原因 |
|------|----------|----------|
| `curvature = Unknown` | 骨架提取失败 | 密度过低或阈值错误 |
| `curvature = N/A` | 低倍率 | 倍率 < 20kx |

#### 辅助指标
| 条件 | 警告信息 | 可能原因 |
|------|----------|----------|
| `n_branches = 0` | 骨架化失败 | 二值化失败 |
| `coherence < 0.1` | 结构张量信号弱 | 对比度低或噪声大 |
| `tortuosity > 5` | 骨架追踪错误 | 连通域异常 |

---

## 三、评估输出说明

### 3.1 输出目录结构

```
reports/eval_20260315_143052/
├── evaluation_report.txt    # 文本报告（总体统计、按来源/倍率统计、错误/警告详情）
├── evaluation_data.json    # JSON数据（用于后续分析或可视化）
└── statistics.csv          # CSV统计表（所有问题样品的详细信息）
```

---

### 3.2 文本报告示例

```
================================================================================
CNTA 特征提取系统性评估报告
================================================================================

生成时间: 2026-03-15 14:30:52
评估版本: FeatureExtractor v2.1

一、总体统计
--------------------------------------------------------------------------------
总图像数: 1523
成功:     1354 (88.9%)
警告:     120  (7.9%)
错误:     28   (1.8%)
跳过:     21   (1.4%)

二、按来源统计
--------------------------------------------------------------------------------
来源          总计   成功   警告   错误   成功率
--------------------------------------------------------------------------------
ZZY            987    870     88     18   88.1%
XR             536    484     32     10   90.3%

三、按倍率统计
--------------------------------------------------------------------------------
倍率          总计   成功   警告   错误   成功率
--------------------------------------------------------------------------------
500x            12      8      3      1   66.7%
1000x           45     35      7      3   77.8%
5000x           89     72     12      5   80.9%
10000x         234    198     28      8   84.6%
50000x         675    634     33      8   93.9%

四、成功样品特征统计
--------------------------------------------------------------------------------
density             均值:   45.234  中位数:   46.120  最小:    2.340  最大:   95.670
alignment           均值:    0.345  中位数:    0.378  最小:   -0.120  最大:    0.892
diameter            均值:   18.456  中位数:   17.890  最小:    5.120  最大:   52.340
tortuosity          均值:    1.234  中位数:    1.180  最小:    1.020  最大:    3.450
mean_phi_deg        均值:   35.670  中位数:   34.500  最小:    5.200  最大:   78.900

五、错误详情
--------------------------------------------------------------------------------
共 28 个错误

ID: 1245  |  No26 200w 5.0nm 5w 0.75nm 400 200 100 600 750 15min 180min mid 1000-1.png
  Error: cv2.imread 返回 None

ID: 1246  |  No26 200w 5.0nm 5w 0.75nm 400 200 100 600 750 15min 180min mid 1000-2.png
  Error: 文件不存在: d:\CNTDATA\ZZY\...\*.png

... 还有 26 个错误

六、警告详情
--------------------------------------------------------------------------------
共 120 个警告

警告类型分布:
  78: diameter 过大: X.Xnm (可能是束径或团簇)
  32: alignment 使用结构张量法（低倍率，系统偏置）
  18: density > 95% (可能是背景误判)
   8: curvature = N/A (低倍率)
   4: tortuosity > 5.0 (可能骨架追踪错误)

具体警告列表 (前20个):

ID: 123  |  No26 200w 5.0nm 5w 0.75nm 400 200 100 600 750 15min 180min mid 50000-1.png
  - diameter 过大: 45.6nm (可能是束径或团簇)

ID: 124  |  No26 200w 5.0nm 5w 0.75nm 400 200 100 600 750 15min 180min mid 50000-2.png
  - diameter 过大: 52.3nm (可能是束径或团簇)

... 还有 118 个警告
```

---

### 3.3 JSON数据说明

`evaluation_data.json` 包含所有评估数据的结构化信息，可用于：

- 用Python脚本进行后续分析
- 用Tableau/PowerBI等工具可视化
- 用Excel/Pandas进行统计

数据结构：
```json
{
  "timestamp": "2026-03-15T14:30:52",
  "summary": {
    "total": 1523,
    "success": 1354,
    "warning": 120,
    "error": 28,
    "skipped": 21
  },
  "by_source": {...},
  "by_magnification": {...},
  "errors": [...],
  "warnings": [...],
  "success_stats": {...}
}
```

---

### 3.4 CSV统计表说明

`statistics.csv` 包含所有问题样品的详细信息：

| 列名 | 说明 |
|------|------|
| id | 数据库ID |
| path | 图像文件路径 |
| source | 数据来源（ZZY/XR） |
| magnification | 倍率 |
| sample_id | 样品ID |
| status | 状态（warning/error） |
| error | 错误信息（如果有） |
| warnings | 警告列表（分号分隔） |
| density | 密度值 |
| alignment | 对齐度值 |
| diameter | 管径值 |
| curvature | 曲率标签 |

可以用Excel/Pandas打开分析：
```python
import pandas as pd
df = pd.read_csv('reports/eval_20260315_143052/statistics.csv', encoding='utf-8-sig')

# 按倍率分析警告类型
warnings_by_mag = df[df['warnings'] != ''].groupby('magnification')['warnings'].apply(list)

# 按来源统计
warnings_by_source = df[df['warnings'] != ''].groupby('source').size()
```

---

## 四、评估流程建议

### 4.1 快速测试（推荐先做）

```bash
# 第一步：快速测试前50张
python backend/core/quick_eval.py

# 检查输出：
# - 查看 reports/eval_*/evaluation_report.txt
# - 检查是否有明显的系统性错误
# - 确认脚本运行正常
```

---

### 4.2 完整评估

```bash
# 第二步：完整评估所有未处理的图像
python backend/core/systematic_evaluator.py

# 第三步：如果需要重新评估所有图像
python backend/core/systematic_evaluator.py --reprocess
```

---

### 4.3 分析结果

1. **阅读文本报告**
   - 总体成功率
   - 按来源/倍率统计
   - 错误和警告类型分布

2. **分析CSV统计表**
   - 用Excel/Pandas分析问题样品的特征
   - 找出问题模式（如高倍率高密度样品的diameter异常）

3. **提取创新点**
   - 根据失败模式设计改进方案
   - 参考 `research_review_CNTA_image_analysis.md` 中的创新点

---

## 五、常见问题

### Q1: 脚本运行很慢怎么办？
A: 使用 `--limit` 参数限制处理数量，或使用 `quick_eval.py` 进行快速测试。

### Q2: 如何只评估某个倍率的图像？
A: 修改脚本中的SQL查询，添加 `AND magnification = 50000` 等条件。

### Q3: 如何可视化评估结果？
A: 使用 `evaluation_data.json` 和 Python 的 matplotlib/seaborn 库进行可视化。

### Q4: 评估数据会保存在哪里？
A: 默认保存在 `reports/eval_YYYYMMDD_HHMMSS/` 目录，可用 `--output` 参数指定。

---

## 六、下一步行动

1. **运行快速测试**
   ```bash
   python backend/core/quick_eval.py
   ```

2. **检查评估结果**
   - 查看 `evaluation_report.txt`
   - 分析主要问题类型

3. **运行完整评估**
   ```bash
   python backend/core/systematic_evaluator.py
   ```

4. **提取创新点**
   - 根据评估结果设计改进方案
   - 参考 `research_review_CNTA_image_analysis.md`

---

**文档版本**：v1.0
**最后更新**：2026-03-15
**维护者**：用户 / Claude Code
