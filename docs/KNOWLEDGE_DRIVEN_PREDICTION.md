# 知识驱动预测模型使用指南

## 架构概述

知识驱动预测模型采用三层架构，结合RAG文献、专家知识和机器学习：

```
知识层 → 数据层 → 模型层
```

### 核心特点

- **知识基线预测**：基于相似实验的加权平均
- **ML残差预测**：基于知识增强特征的机器学习模型
- **物理约束检查**：违反约束时返回保守估计
- **RAG文献支持**：自动检索相关文献作为证据

## 快速开始

### 1. 初始化预测器

```python
from backend.core.knowledge_driven_predictor import KnowledgeDrivenPredictor
from backend.core.knowledge_rag import RAGRetriever

# 初始化
db_path = r'd:\CNTDATA\CNTA_ML_Project\database\cnta_experiments.sqlite'
kb_db_path = r'd:\CNTDATA\CNTA_ML_Project\database\cnta_knowledge_base.sqlite'

rag_retriever = RAGRetriever(db_path, knowledge_db_path=kb_db_path)
predictor = KnowledgeDrivenPredictor(db_path, rag_retriever)
```

### 2. 单参数预测（ZZY数据）

```python
# ZZY数据参数
zzy_params = {
    'source': 'ZZY',
    'growth_temp': 750,      # 生长温度 ℃
    'growth_time': 3,         # 生长时间 h
    'fe_thickness': 1.5,     # Fe厚度 nm
    'al2o3_thickness': 10,   # Al2O3厚度 nm
    'ar_flow': 500,          # Ar流量 sccm
    'h2_flow': 100,          # H2流量 sccm
    'c2h4_flow': 50,         # C2H4流量 sccm
}

# 预测直径
result = predictor.predict(zzy_params, target='diameter')

print(f"预测直径: {result.predicted_value:.2f}nm")
print(f"置信度: {result.confidence:.2f}")
print(f"知识基线: {result.knowledge_baseline:.2f}")
print(f"ML残差: {result.ml_residual:.2f}")
print(f"相似实验: {len(result.similar_experiments)} 条")
```

### 3. 梯度预测（XR数据）

```python
# XR数据参数
xr_params = {
    'source': 'XR',
    'actual_temp': 750,      # 实际温度 ℃
    'growth_time': 3,         # 生长时间 h
    'inlet_distance_cm': 18.0, # 距进气口距离 cm；兼容旧字段 membrane_pos_cm
}

# 预测密度
result = predictor.predict(xr_params, target='density')

print(f"预测密度: {result.predicted_value:.2f}%")
```

### 4. 带RAG文献支持的预测

```python
# 使用RAG检索相关文献
query = "temperature effect on CNT diameter"
result = predictor.predict(zzy_params, target='diameter', query=query)

print(f"RAG证据: {len(result.rag_evidence)} 条")
for evidence in result.rag_evidence:
    print(f"- {evidence['filename']}: {evidence['text'][:100]}...")
```

### 5. 批量预测

```python
# 批量预测不同温度下的直径
params_list = [
    {'source': 'ZZY', 'growth_temp': 700, 'growth_time': 3, 'fe_thickness': 1.5},
    {'source': 'ZZY', 'growth_temp': 750, 'growth_time': 3, 'fe_thickness': 1.5},
    {'source': 'ZZY', 'growth_temp': 800, 'growth_time': 3, 'fe_thickness': 1.5},
]

results = predictor.batch_predict(params_list, target='diameter')

for i, result in enumerate(results):
    temp = params_list[i]['growth_temp']
    print(f"{temp}℃: {result.predicted_value:.2f}nm (置信度: {result.confidence:.2f})")
```

## API接口使用

### POST /api/predict

单次预测接口。

**请求体示例：**

```json
{
  "source": "ZZY",
  "growth_temp": 750,
  "growth_time": 3,
  "fe_thickness": 1.5,
  "al2o3_thickness": 10,
  "ar_flow": 500,
  "h2_flow": 100,
  "c2h4_flow": 50,
  "target": "diameter",
  "query": "temperature effect on CNT diameter"
}
```

**响应示例：**

```json
{
  "status": "success",
  "prediction": {
    "target": "diameter",
    "predicted_value": 23.06,
    "confidence": 0.85,
    "knowledge_baseline": 21.56,
    "ml_residual": 1.50
  },
  "evidence": {
    "similar_experiments": [...],
    "rag_evidence": [...],
    "physical_constraints": []
  }
}
```

### POST /api/predict/batch

批量预测接口。

**请求体示例：**

```json
{
  "params_list": [
    {"source": "ZZY", "growth_temp": 700, "growth_time": 3, "fe_thickness": 1.5},
    {"source": "ZZY", "growth_temp": 750, "growth_time": 3, "fe_thickness": 1.5},
    {"source": "ZZY", "growth_temp": 800, "growth_time": 3, "fe_thickness": 1.5}
  ],
  "target": "diameter"
}
```

### POST /api/ml/train

训练机器学习模型接口。

**请求参数：**
- `source` (可选): 指定数据源，默认所有数据

## 预测目标说明

系统支持以下形貌特征的预测：

| 目标 | 单位 | 范围 | 说明 |
|------|------|------|------|
| `diameter` | nm | 10-200 | CNT管径 |
| `density` | % | 0-100 | 面密度/覆盖率 |
| `alignment` | 无量纲 | 0-1 | 取向度 |
| `curvature` | 无量纲 | 0.9-2.5 | 波曲度/曲率指标 |

## 物理约束检查

系统会自动检查工艺参数是否符合物理约束：

### 温度约束
- **< 600℃**：温度过低，碳源裂解不足，形核困难
- **> 900℃**：温度过高，催化剂易团聚，密度下降

### 催化剂约束
- **Fe厚度 < 0.5nm**：过薄，催化活性低，生长困难
- **Fe厚度 > 3.0nm**：过厚，颗粒团聚，密度降低

### 气流约束
- **Ar/C2H4比 < 5**：碳浓度过高，易致无定形碳
- **Ar/C2H4比 > 50**：碳浓度不足，密度降低

违反约束时，系统会返回低置信度的保守估计。

## 知识增强特征

系统基于专家知识构造复合特征：

- **temp_normalized**: 归一化温度 (temp / 750)
- **fe_thickness_sq**: 厚度平方（影响团聚）
- **catalyst_ratio**: Fe/Al2O3比（催化效率）
- **temp_stability**: 温度/时间（温度稳定性）
- **carbon_supply**: C2H4/Ar比（碳供应能力）
- **reduction_ratio**: H2/Ar比（还原氛围）

## 置信度计算

置信度由以下因素决定：

1. **相似实验数量**：越多越高（+0.1 * 数量，最高+0.3）
2. **RAG证据数量**：越多越高（+0.05 * 数量，最高+0.1）
3. **参数范围**：在训练范围内加0.05
4. **基础置信度**：0.5

置信度范围：0.3（违反约束）~ 1.0

## 预测流程

```
1. 物理约束检查
   ↓ 违反？ → 返回保守估计
2. RAG文献检索
3. 相似实验检索（知识基线）
4. 计算知识基线预测（加权平均）
5. ML残差预测（知识增强特征）
6. 混合预测 = 知识基线 + ML残差
7. 计算置信度
8. 返回结果 + 证据
```

## 模型训练（可选）

当数据量充足（> 50条）时，可以训练ML模型：

```python
# 训练所有数据
predictor.train_models()

# 只训练ZZY数据
predictor.train_models(source='ZZY')

# 只训练XR数据
predictor.train_models(source='XR')
```

模型包括：
- **Random Forest**: 随机森林回归
- **Gradient Boosting**: 梯度提升回归

## 常见问题

### Q: 为什么预测置信度很低？
A: 可能是参数违反了物理约束，或者相似实验数量太少。检查返回的`physical_constraints`字段。

### Q: 知识基线和ML残差的区别是什么？
A:
- **知识基线**：基于相似实验的加权平均，代表历史数据的平均趋势
- **ML残差**：基于知识增强特征的预测，代表偏离平均趋势的修正量

### Q: 如何提高预测准确性？
A:
1. 上传更多相关PDF文献，丰富RAG知识库
2. 确保工艺参数在合理范围内
3. 积累更多实验数据
4. 定期训练ML模型

### Q: XR和ZZY数据源的区别是什么？
A:
- **ZZY**：多参数组合（14个参数），适合参数敏感性分析
- **XR**：梯度参数（温度/时间/位置），适合趋势分析

## 性能优化建议

1. **批量预测**：使用批量接口提高效率
2. **缓存相似实验**：相同参数范围的预测可以复用相似实验
3. **限制RAG检索**：对于高频预测，可以减少RAG检索频率

## 未来扩展

- [ ] 支持自定义预测目标
- [ ] 增加更多物理约束规则
- [ ] 支持实验设计推荐
- [ ] 集成在线学习机制
- [ ] 支持不确定性量化

## 技术栈

- **RAG检索**: BM25 + SQLite
- **ML模型**: scikit-learn (RandomForest, GradientBoosting)
- **知识库**: 规则引擎 + 专家知识模式匹配
- **API**: FastAPI
- **数据处理**: NumPy
