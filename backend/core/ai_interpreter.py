"""
AI 可解释性分析模块
支持 GLM-4（智谱AI）和 DeepSeek，均通过 OpenAI 兼容接口调用。
提供流式 SSE 生成器，供 FastAPI StreamingResponse 使用。
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

# ────────────────────────────────────────────────
#  CNT 领域专家知识（System Prompt 注入）
# ────────────────────────────────────────────────
CNT_DOMAIN_KNOWLEDGE = """你是碳纳米管（CNT）阵列制备领域的专家，熟悉化学气相沉积（CVD）工艺。
以下是你掌握的核心领域知识，请在分析时严格依据这些知识给出科学准确的解读：

## 四个形貌特征的物理含义与正常值范围
- **密度（density）**：CNT阵列的填充率（0-100%）。典型高质量样品：25-60%。
  过低（<15%）说明生长稀疏，可能催化剂活化不足或气体浓度偏低；
  过高（>70%）可能出现管间聚束、形貌紊乱。
- **取向度（alignment，S值）**：-1到1的序参数，越接近1说明CNT越垂直取向。
  S>0.7 为优质；0.4-0.7 为中等；<0.4 说明取向较差，需优化气流或温度均匀性。
- **管径（diameter）**：平均管径（nm）。MWCNT典型范围：8-30nm；SWCNT：1-3nm。
  管径主要由Fe催化剂层厚度控制：Fe层越厚→催化剂颗粒越大→管径越大。
- **弯曲度（curvature）**：分类为 Straight / Wavy / Coiled。
  Straight（笔直）最优；Wavy（波浪）说明生长速率不均；Coiled（螺旋）说明存在内应力或催化剂毒化。

## 工艺参数对形貌特征的影响规律
| 参数 | 影响 |
|------|------|
| 生长温度↑ | 管径减小，取向改善，但温度过高（>850℃）导致催化剂聚集→管径增大 |
| C2H4流量↑ | 密度↑，但过多导致碳膜沉积→管径增大，取向恶化 |
| H2流量↑ | 催化剂还原更充分，密度↑，管径↓ |
| Ar流量↑ | 惰性气体稀释，密度↓，取向改善（减少湍流） |
| Fe层厚↑ | 催化剂颗粒增大→管径↑，密度可能↑ |
| Al2O3层厚↑ | 支撑层更稳定，催化剂不易聚集→管径↓，密度↑ |
| 退火时间↑ | 催化剂颗粒重排，密度↑，取向改善 |
| 生长时间↑ | 阵列高度↑，但长时间生长可能导致催化剂中毒→curvature恶化 |

## 常见问题诊断
- **取向差（S<0.4）**：检查气流均匀性、温度梯度、H2/C2H4比例
- **管径过大（>25nm）**：降低Fe层厚度或提高生长温度
- **密度过低（<20%）**：增加C2H4浓度或退火时间，检查催化剂活化步骤
- **Coiled形态**：降低C2H4流量，排查催化剂毒化（含硫/水分污染）

## ZZY vs XR 样品特点
- ZZY样品：PNG格式，参数变化范围广（梯度实验），关注参数-形貌相关性
- XR样品：TIFF格式，使用标准工艺（T800, 3h, 250L Ar），重点关注位置均匀性

请用中文回答，语言专业但易于理解，适合材料科学研究生阅读。"""


class AIInterpreter:
    def __init__(self, provider: str, api_key: str, model: Optional[str] = None):
        """
        provider: 'glm' 或 'deepseek'
        api_key:  对应平台的 API Key
        model:    可选，覆盖默认模型名称
        """
        if provider not in PROVIDER_CONFIGS:
            raise ValueError(f"不支持的提供商: {provider}，请选择 'glm' 或 'deepseek'")
        cfg = PROVIDER_CONFIGS[provider]
        self.provider = provider
        self.model = model or cfg["default_model"]

        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=cfg["base_url"])

    # ------------------------------------------------------------------ #
    #  Prompt 构建
    # ------------------------------------------------------------------ #
    @staticmethod
    def build_interpret_prompt(
        features: dict,
        params: dict,
        similar_exps: list,
        pdf_passages: list,
    ) -> str:
        lines = ["## 当前实验信息\n"]

        # 工艺参数
        lines.append("### 工艺参数")
        param_map = {
            "sample_id": "样品编号", "source": "数据集",
            "growth_temp": "设定温度(℃)", "actual_temp": "实际生长温度(℃ - 已校准)",
            "membrane_pos_cm": "炉管物理位置(cm)",
            "growth_time": "生长时间(h)",
            "anneal_temp": "退火温度(℃)", "anneal_time": "退火时间(h)",
            "ar_flow": "Ar流量(sccm)", "h2_flow": "H2流量(sccm)",
            "c2h4_flow": "C2H4流量(sccm)",
            "fe_thickness": "Fe层厚(nm)", "fe_power": "Fe溅射功率(W)",
            "al2o3_thickness": "Al2O3层厚(nm)", "al2o3_power": "Al2O3溅射功率(W)",
            "position_label": "位置标签", "magnification": "放大倍率",
        }
        lines.append("> 注意：实际生长温度是根据热场分布校准后的物理真实温度，优于设定温度，请优先参考。")
        for key, label in param_map.items():
            val = params.get(key)
            if val is not None:
                lines.append(f"- {label}: {val}")

        # 特征值
        lines.append("\n### OpenCV提取的形貌特征")
        feature_map = {
            "density": "密度(%)",
            "alignment": "取向度S值",
            "diameter": "平均管径(nm)",
            "curvature": "弯曲度分类",
        }
        for key, label in feature_map.items():
            val = features.get(key)
            if val is not None:
                lines.append(f"- {label}: {val}")

        # 相似实验
        if similar_exps:
            lines.append("\n### 数据库中相似实验（供对比参考）")
            for i, exp in enumerate(similar_exps[:3], 1):
                lines.append(
                    f"{i}. 样品{exp.get('sample_id','?')} "
                    f"| 生长温度{exp.get('growth_temp','?')}℃ "
                    f"| Fe层{exp.get('fe_thickness','?')}nm "
                    f"| 密度{exp.get('density','?')} "
                    f"| 取向S={exp.get('alignment','?')} "
                    f"| 管径{exp.get('diameter','?')}nm "
                    f"| 弯曲:{exp.get('curvature','?')}"
                )

        # PDF文献
        if pdf_passages:
            lines.append("\n### 相关文献片段（RAG检索）")
            for i, p in enumerate(pdf_passages[:2], 1):
                lines.append(f"[文献{i}: {p.get('filename','?')}]\n{p.get('text','')}\n")

        lines.append("""
## 请按以下结构输出分析报告（使用Markdown格式）

### 🔬 特征解读
逐一解读四个特征的物理含义，结合当前数值给出定性评价（优/中/差）。

### 📊 综合质量评估
给出整体CNT阵列质量的综合评价（1-5分），并说明主要优势和不足。

### 💡 工艺改进建议
根据特征数值和工艺参数，给出3-5条具体可操作的改进建议，每条注明预期效果。

### 🔗 与相似实验的对比
（如有相似实验数据）对比分析异同，指出可借鉴之处。
""")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    #  流式生成 SSE
    # ------------------------------------------------------------------ #
    def interpret_stream(
        self,
        features: dict,
        params: dict,
        similar_exps: list,
        pdf_passages: list,
        temperature: float = 0.5,
    ) -> Generator[str, None, None]:
        """生成 SSE 格式的流式响应"""
        user_prompt = self.build_interpret_prompt(features, params, similar_exps, pdf_passages)

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
        """
        对话流式响应。
        history: [{"role": "user"|"assistant", "content": "..."}]
        context: 当前图像的参数+特征+相似实验（可选）
        """
        system_content = CNT_DOMAIN_KNOWLEDGE

        if context:
            ctx_lines = ["\n\n## 当前对话上下文（用户正在查看的实验）"]
            params = context.get("params", {})
            features = context.get("features", {})
            similar = context.get("similar_experiments", [])

            if params.get("sample_id"):
                ctx_lines.append(f"- 样品: {params['sample_id']} ({params.get('source','')})")
            if params.get("growth_temp"):
                ctx_lines.append(f"- 生长温度: {params['growth_temp']}℃")
            if features.get("density") is not None:
                ctx_lines.append(f"- 密度: {features['density']}%")
            if features.get("alignment") is not None:
                ctx_lines.append(f"- 取向度S: {features['alignment']}")
            if features.get("diameter") is not None:
                ctx_lines.append(f"- 管径: {features['diameter']}nm")
            if features.get("curvature"):
                ctx_lines.append(f"- 弯曲度: {features['curvature']}")
            if similar:
                ctx_lines.append(f"- 数据库中找到 {len(similar)} 条相似实验可供参考")

            system_content += "\n".join(ctx_lines)

        messages = [{"role": "system", "content": system_content}]
        for msg in history[-10:]:  # 保留最近10轮对话
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
