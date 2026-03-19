"""
AI interpretation module for CNT experiment analysis.
Supports OpenAI-compatible providers and SSE streaming output.
"""

import json
from typing import Generator, Optional

PROVIDER_CONFIGS = {
    "glm": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4/",
        "default_model": "glm-4-flash",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "default_model": "deepseek-chat",
    },
}

CNT_DOMAIN_KNOWLEDGE = """你是碳纳米管（CNT）阵列制备与表征方向的研究助手。
请基于工艺参数、图像特征、相似实验和检索证据进行严谨分析。
回答要求：
1. 使用中文，术语准确，逻辑清晰。
2. 优先给出证据驱动的结论，不要夸大因果。
3. 区分“趋势判断”和“定量结论”。
4. 如果证据不足要明确指出。"""


class AIInterpreter:
    def __init__(self, provider: str, api_key: str, model: Optional[str] = None):
        if provider not in PROVIDER_CONFIGS:
            raise ValueError(f"不支持的 provider: {provider}，请使用 'glm' 或 'deepseek'")
        cfg = PROVIDER_CONFIGS[provider]
        self.provider = provider
        self.model = model or cfg["default_model"]

        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, base_url=cfg["base_url"])

    @staticmethod
    def build_interpret_prompt(
        features: dict,
        params: dict,
        similar_exps: list,
        pdf_passages: list,
        knowledge_links: Optional[list] = None,
    ) -> str:
        lines = ["## 当前实验信息\n"]

        lines.append("### 工艺参数")
        param_map = {
            "sample_id": "样品编号",
            "source": "数据源",
            "growth_temp": "设定温度(℃)",
            "actual_temp": "实际生长温度(℃)",
            "membrane_pos_cm": "炉管位置(cm)",
            "growth_time": "生长时间(h)",
            "anneal_temp": "退火温度(℃)",
            "anneal_time": "退火时间(h)",
            "ar_flow": "Ar流量(sccm)",
            "h2_flow": "H2流量(sccm)",
            "c2h4_flow": "C2H4流量(sccm)",
            "fe_thickness": "Fe厚度(nm)",
            "al2o3_thickness": "Al2O3厚度(nm)",
            "position_label": "位置标签",
            "magnification": "放大倍率",
        }
        for key, label in param_map.items():
            value = params.get(key)
            if value is not None:
                lines.append(f"- {label}: {value}")

        lines.append("\n### 图像特征")
        feature_map = {
            "density": "面密度(%)",
            "alignment": "取向度",
            "diameter": "表观直径(nm)",
            "curvature": "波曲度/曲率指标",
            "tortuosity": "曲折度",
        }
        for key, label in feature_map.items():
            value = features.get(key)
            if value is not None:
                lines.append(f"- {label}: {value}")

        if similar_exps:
            lines.append("\n### 相似实验")
            for i, exp in enumerate(similar_exps[:3], 1):
                lines.append(
                    f"{i}. 样品={exp.get('sample_id', '?')}, 温度={exp.get('growth_temp', '?')}, "
                    f"Fe={exp.get('fe_thickness', '?')}, 密度={exp.get('density', '?')}, "
                    f"取向={exp.get('alignment', '?')}, 直径={exp.get('diameter', '?')}"
                )

        if pdf_passages:
            lines.append("\n### 文献证据（RAG）")
            for i, passage in enumerate(pdf_passages[:3], 1):
                title = passage.get("title") or passage.get("filename") or "unknown"
                text = passage.get("text", "")
                lines.append(f"[{i}] {title}: {text[:220]}")

        if knowledge_links:
            lines.append("\n### 关系证据（工艺-形貌-性能）")
            for i, item in enumerate(knowledge_links[:4], 1):
                lines.append(
                    f"{i}. process={item.get('process_factor') or '-'}, "
                    f"morphology={item.get('morphology_factor') or '-'}, "
                    f"performance={item.get('performance_factor') or '-'}, "
                    f"direction={item.get('effect_direction') or '-'}"
                )
                evidence = item.get("evidence_text") or ""
                if evidence:
                    lines.append(f"   evidence: {evidence[:200]}")

        lines.append(
            """
## 输出格式
### 1) 特征解读
逐项说明当前形貌特征含义与风险。

### 2) 综合判断
给出样品质量和工艺状态判断，并说明不确定性。

### 3) 工艺改进建议
给出 3-5 条可执行建议，写明预期影响。

### 4) 证据对照
引用相似实验、文献证据和关系证据，说明为什么这样判断。
"""
        )

        return "\n".join(lines)

    def interpret_stream(
        self,
        features: dict,
        params: dict,
        similar_exps: list,
        pdf_passages: list,
        knowledge_links: Optional[list] = None,
        temperature: float = 0.5,
    ) -> Generator[str, None, None]:
        user_prompt = self.build_interpret_prompt(
            features,
            params,
            similar_exps,
            pdf_passages,
            knowledge_links=knowledge_links,
        )

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": CNT_DOMAIN_KNOWLEDGE},
                {"role": "user", "content": user_prompt},
            ],
            stream=True,
            temperature=max(0.0, min(1.0, temperature)),
            max_tokens=2048,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                data = json.dumps({"type": "content", "text": delta.content}, ensure_ascii=False)
                yield f"data: {data}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    def chat_stream(
        self,
        history: list,
        user_message: str,
        context: Optional[dict] = None,
        temperature: float = 0.5,
    ) -> Generator[str, None, None]:
        system_content = CNT_DOMAIN_KNOWLEDGE

        if context:
            ctx_lines = ["\n\n## 当前对话上下文"]
            params = context.get("params", {})
            features = context.get("features", {})
            similar = context.get("similar_experiments", [])

            if params.get("sample_id"):
                ctx_lines.append(f"- 样品: {params['sample_id']} ({params.get('source', '')})")
            if params.get("growth_temp") is not None:
                ctx_lines.append(f"- 生长温度: {params['growth_temp']}℃")
            if features.get("density") is not None:
                ctx_lines.append(f"- 密度: {features['density']}%")
            if features.get("alignment") is not None:
                ctx_lines.append(f"- 取向度: {features['alignment']}")
            if features.get("diameter") is not None:
                ctx_lines.append(f"- 直径: {features['diameter']}nm")
            if features.get("curvature") is not None:
                ctx_lines.append(f"- 曲率/波曲度: {features['curvature']}")
            if similar:
                ctx_lines.append(f"- 相似实验数量: {len(similar)}")

            system_content += "\n".join(ctx_lines)

        messages = [{"role": "system", "content": system_content}]
        for msg in history[-10:]:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": user_message})

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
            temperature=max(0.0, min(1.0, temperature)),
            max_tokens=1024,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                data = json.dumps({"type": "content", "text": delta.content}, ensure_ascii=False)
                yield f"data: {data}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"
