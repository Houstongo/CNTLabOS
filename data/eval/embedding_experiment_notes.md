# Embedding模型扩展实验记录（CNTA RAG）

## 实验设置
- 评测集：`data/eval/retrieval_eval_set.json`（当前3条已标注query）
- 指标：`Recall@5/10`, `MRR@5/10`, `nDCG@5/10`
- 检索语料：`database/cnta_knowledge_base.sqlite`

## 扩展模型清单
### 通用模型
- `sentence-transformers/all-MiniLM-L6-v2`
- `intfloat/multilingual-e5-base`
- `BAAI/bge-small-zh-v1.5`

### 化学/材料相关模型
- `sentence-transformers/allenai-specter`
- `m3rg-iitd/matscibert`
- `pritamdeka/S-PubMedBert-MS-MARCO`

## 实验结果（离线环境）
- 输出文件：`data/eval/retrieval_model_comparison_extended.csv`
- `bm25`：可运行，当前表现最好。
- `all-MiniLM-L6-v2`：可运行，但当前指标为0（语料中文+专业噪声场景下不匹配）。
- 其余扩展模型：离线环境下未缓存或缓存不完整，状态为 `error`。

## 失败原因归纳
- 无法访问 HuggingFace 下载模型：`couldn't connect to https://huggingface.co`
- 本地缓存不完整（例如 `BAAI/bge-large-zh-v1.5` 缺少权重文件）

## 结论与下一步
- 你提出的“通用 + 化学材料”模型扩展对比流程已经搭好，并完成了一轮批量实验。
- 要得到完整可比结果，需要联网预下载模型后再跑一次同样命令。
