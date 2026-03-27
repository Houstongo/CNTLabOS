# 第三章实验：知识建模与检索增强方法

## 实验概述

本目录包含论文第三章"面向碳纳米管阵列研究的知识建模与检索增强方法"的所有实验脚本。

## 实验结构

```
chapter3/
├── __init__.py                          # 模块初始化
├── README.md                             # 本文件
├── exp_01_data_statistics.py           # 实验 1: 知识库数据统计
├── exp_02_depth_extension.py           # 实验 2: 深度扩展功能验证
├── exp_03_task_comparison.py           # 实验 3: 任务类型对比实验
├── exp_04_ablation_study.py             # 实验 4: 消融实验
├── run_all_experiments.py               # 运行所有实验
└── results/                             # 实验结果输出目录
```

## 实验说明

### 实验 1: 知识库数据统计 (exp_01_data_statistics.py)

**目标**: 统计知识库的基础数据，为后续实验提供数据概况

**主要指标**:
- 文档统计：总文档数、PDF 文档数、平均每文档 Chunk 数
- Chunk 统计：总 Chunk 数、平均长度、长度范围、知识类型分布
- Link 统计：总 Link 数、关系类型分布、置信度统计、平均每文档 Link 数
- MSFU 字段统计：各字段（源实体、目标实体、工艺因子、形貌因子等）的完整性

**输出**:
- 控制台输出：格式化的统计报告
- JSON 文件：`results/exp_01_data_statistics.json`

### 实验 2: 深度扩展功能验证 (exp_02_depth_extension.py)

**目标**: 验证 KB LINK 深度扩展功能的有效性

**子实验**:
1. **深度对比实验**: 对比 max_depth=1（原始）vs max_depth=2（扩展后）的检索效果
2. **三层关系链构建实验**: 验证能否构建 Process → Mechanism → Morphology 三层关系链
3. **深度扫描实验**: 测试 max_depth 从 1 到 3 的性能变化

**主要指标**:
- 路径数、平均深度、最大深度、最小深度
- 平均评分、最小评分、最大评分

**输出**:
- 控制台输出：详细的实验结果
- JSON 文件：`results/exp_02_1_depth_comparison.json`
- JSON 文件：`results/exp_02_2_three_layer_chain.json`
- JSON 文件：`results/exp_02_3_depth_sweep.json`

### 实验 3: 任务类型对比实验 (exp_03_task_comparison.py)

**目标**: 对比不同任务类型（工艺分析、形貌解释、预测解释）下的检索性能

**任务类型**:
- `morphology_interpretation`: 形貌解释
- `process_analysis`: 工艺分析
- `prediction_explanation`: 预测解释

**主要指标**:
- **EHR** (Evidence Hit Rate): 证据命中率，命中证据的比例
- **LCI** (Link Coverage Index): 链路覆盖指数，链路完整性
- **Recall@K**: 召回率
- **Precision@K**: 精确率
- **平均深度**: 检索路径的平均深度
- **平均评分**: 检索路径的平均评分
- **平均时间**: 平均查询时间

**输出**:
- 控制台输出：各任务类型的性能对比
- JSON 文件：`results/exp_03_task_comparison_depth2.json`

### 实验 4: 消融实验 (exp_04_ablation_study.py)

**目标**: 验证 TCCER 各模块的贡献，通过移除关键组件分析性能变化

**变体实验**:
1. **基准实验**: 完整 TCCER 算法
2. **去除关系约束扩展**: 模拟仅单跳检索（max_depth=1）
3. **去除条件一致性检查**: 不考虑条件约束的检索
4. **去除方向一致性检查**: 不考虑方向约束的检索

**主要指标**:
- 平均深度变化
- 平均评分变化
- 深度变化百分比
- 评分变化百分比

**输出**:
- 控制台输出：消融实验对比结果
- JSON 文件：`results/exp_04_ablation_study.json`

## 运行方式

### 运行单个实验

```bash
# 进入实验目录
cd CNTA_ML_Project/experiments/chapter3

# 激活 conda 环境
conda activate lab_agent

# 运行单个实验
python exp_01_data_statistics.py
python exp_02_depth_extension.py
python exp_03_task_comparison.py
python exp_04_ablation_study.py
```

### 运行所有实验

```bash
# 运行所有实验
python run_all_experiments.py
```

## 预期结果

### 实验 1 预期结果

```
文档统计:
  总文档数: ~76
  PDF 文档数: ~76
  平均每文档 Chunk 数: ~135

Link 统计:
  总 Link 数: ~415
  平均每文档 Link 数: ~5.5
  关系类型分布:
    process_to_morphology: 119 (28.7%)
    mechanism_evidence: 119 (28.7%)
    process_to_mechanism: 69 (16.6%)
    ...
```

### 实验 2 预期结果

```
实验 2.1: 深度对比实验
  max_depth=1:
    路径数: 6
    平均深度: 1.0
    最大深度: 1

  max_depth=2:
    路径数: 6
    平均深度: 1.8
    最大深度: 2

  深度提升: 1 → 2 (+1)
```

### 实验 3 预期结果

```
各任务类型性能:
  morphology_interpretation:
    Recall@K: ~78.5%
    Precision@K: ~82.3%
    EHR: ~81.2%
    LCI: ~0.78
    平均深度: ~1.6

  process_analysis:
    Recall@K: ~82.1%
    Precision@K: ~75.6%
    EHR: ~85.9%
    LCI: ~0.84
    平均深度: ~2.1
```

### 实验 4 预期结果

```
消融实验对比:
  基准: 平均深度=1.8, 平均评分=0.511

  去除关系约束扩展:
    深度变化: -44.4% (1.0)
    评分变化: +0.7%

  去除条件一致性检查:
    深度变化: -5.3%
    评分变化: -2.1%

  去除方向一致性检查:
    深度变化: -6.2%
    评分变化: -4.2%
```

## 数据要求

运行实验前需要确保以下数据可用：

1. **知识库数据库**: `CNTA_ML_Project/database/cnta_knowledge_base.sqlite`
   - 包含至少 70+ 篇学术文献
   - 包含 10,000+ 个文本 Chunks
   - 包含 400+ 个关系 Links

2. **Python 环境**: conda 环境 `lab_agent`
   - 需要安装依赖: FastAPI, sentence-transformers, torch

## 故障排查

### 问题: 知识库数据库不存在

```
错误：知识库数据库不存在: CNTA_ML_Project/database/cnta_knowledge_base.sqlite
```

**解决方案**: 先运行知识库初始化
```bash
cd CNTA_ML_Project
python manage.py kb-bootstrap
python manage.py kb-import-core
```

### 问题: 模块导入错误

```
ModuleNotFoundError: No module named 'backend'
```

**解决方案**: 确保在正确的目录下运行脚本
```bash
cd CNTA_ML_Project
python experiments/chapter3/exp_01_data_statistics.py
```

### 问题: 记忆不足

某些实验（如实验 2 和实验 3）需要加载大模型，可能占用较多内存。

**解决方案**: 减少并发实验数量，或者分批运行

## 结果分析

实验运行完成后，可以在 `results/` 目录查看 JSON 格式的实验结果。建议使用以下方式分析：

1. **查看统计摘要**: 使用 `cat` 或文本编辑器查看 JSON 文件
2. **数据分析**: 使用 Python 或 Jupyter Notebook 加载 JSON 进行深入分析
3. **论文撰写**: 将结果整理为表格，用于论文第三章

## 引用

如果本实验代码用于论文，请引用本项目：

```
本文实验基于 CNTA 项目知识库系统
项目地址: d:\CNTDATA\CNTA_ML_Project
版本: 1.0.0
```
