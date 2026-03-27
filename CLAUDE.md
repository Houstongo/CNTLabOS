# CNTA 项目开发指南

## 项目概述

### 简介
碳纳米管阵列（CNTA）实验数据管理与分析平台。通过 FastAPI 后端提供 REST API，配合 HTML5 前端展示实验图像数据、工艺参数及 OpenCV 提取的形貌特征。

### 技术栈
- **后端**：FastAPI + SQLite
- **前端**：HTML5 + Tailwind CSS（单页应用）
- **机器学习**：PyTorch + Transformers
- **语义检索**：Sentence-Transformers
- **图像处理**：OpenCV + CNTA 分割

## 架构概述

### 核心模块
```
CNTA_ML_Project/
├── backend/
│   ├── main.py                 # FastAPI 入口，定义所有 API 路由
│   └── core/
│       ├── knowledge_base.py       # 知识库服务 - TCCER 算法核心
│       ├── rag_retriever.py       # RAG 检索器 - 链面层
│       ├── knowledge_driven_predictor.py  # 知识驱动预测
│       └── ...
├── src/
│   └── analysis/
│       ├── feature_extractor.py    # OpenCV 特征提取
│       └── ...
└── database/
    ├── cnta_experiments.sqlite    # 实验数据库
    └── cnta_knowledge_base.sqlite  # 知识库数据库
```

### API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/images` | 查询图像列表，支持过滤、分页、排序 |
| POST | `/api/images` | 创建图像记录 |
| PUT | `/api/images/{id}` | 更新图像记录 |
| DELETE | `/api/images/{id}` | 删除图像记录 |
| POST | `/api/images/{id}/analyze` | 触发 AI 重新分析 |
| POST | `/api/chat` | 流式对话接口（SSE） |
| POST | `/api/predict` | 知识驱动预测 |
| POST | `/api/tccer/query` | TCCER 检索 |
| POST | `/api/tccer/full` | TCCER 完整检索 |

## 关键组件

### TCCER 检索算法
**文件**：`backend/core/knowledge_base.py`
- `tccer_retrieve()` - TCCER 约束链式证据检索
- `visualize_paths()` - 路径可视化
- `generate_evidence_explanation()` - 证据解释生成

**核心功能**：
1. 查询解析器 - 解析实体、条件、方向、目标
2. 关系转换矩阵 - 控制路径扩展
3. 混合召回 - 稀疏+稠密+任务+条件评分
4. 约束路径扩展 - 基于关系链接的智能扩展
5. 路径评分 - 深度、一致性、方向综合评分
6. 冗余抑制 - 去除高度重复的路径

### 任务类型
- `morphology_interpretation` - 形貌解释
- `process_analysis` - 工艺分析
- `prediction_explanation` - 预测解释

## 数据目录

### 实验数据集
- **XR/**** - 梯度参数，TIFF 格式
- **ZZY/**** - 工艺参数，PNG 格式

### 开发指南

### 添加新功能
1. 在 `backend/core/` 添加新模块时，记得：
   - 在 `backend/main.py` 添加对应的 API 路由
   - 在 `index.html` 添加对应的前端界面

2. 数据库修改：
   - 使用参数化查询避免 SQL 注入
   - 在 `backend/core/database_helpers.py` 中添加辅助函数

3. 特征提取算法在 `src/analysis/feature_extractor.py` 中维护

4. 测试：
   - 在 `tests/` 目录添加测试文件
   - 运行 `python -m pytest tests/ -v`

5. 知识库管理：
   - 使用 `python manage.py kb-import-core` 导入核心知识
   - 使用 `python manage.py kb-search "query"` 搜索知识库

## 常见任务

### 数据导入
```bash
python manage.py init-db [--clear]
python manage.py sync-mag
python manage.py data-etl
```

### 特征提取
```bash
python manage.py analyze [--reprocess] [--limit N] [--source ZZY|XR]
```

### 知识库操作
```bash
python manage.py kb-bootstrap
python manage.py kb-search "查询语句"
```

### 部署
```bash
python manage.py run-backend          # 启动后端 (8000)
python manage.py run-frontend         # 启动前端 (8080)
```

## 注意事项

1. 路径转义：Windows 路径使用 `\\` 或原始字符串
2. 编码规范：遵循 PEP 8，使用类型注解
3. 错误处理：使用 try-except，记录详细日志
4. 性能优化：使用参数化查询，避免 N+1 问题

## 联系维护

- **文档**：`CLAUDE.md`（本文件）
- **计划**：`docs/plans/` 目录中的计划文件
- **报告**：`docs/reports/` 目录中的总结报告
- **记忆**：`memory/` 目录中的项目记忆

当前实现状态：
- ✅ TCCER 检索算法完整实现
- ✅ 查询解析、关系转换矩阵、混合召回
- ✅ 路径可视化、证据解释生成
- ✅ 知识驱动预测模型
- ✅ 图像处理和特征提取
