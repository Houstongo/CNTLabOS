from typing import Dict, List, Any
import json

class AnalysisReportEngine:
    """
    语义报告生成引擎：
    负责将 CV 提取的结构化数据、数据库中的工艺参数与 RAG 检索的机理知识进行深度融合，
    生成具有学术深度的诊断报告。
    """
    
    def __init__(self):
        # 定义关键指标的物理阈值（可根据文献进一步调优）
        self.thresholds = {
            "diameter": {"high": 25.0, "low": 5.0},
            "alignment": {"good": 0.7, "poor": 0.4},
            "density": {"high": 30.0, "low": 5.0}
        }

    def generate_prompt(self, cv_data: Dict[str, Any], process_data: Dict[str, Any], rag_context: str) -> str:
        """
        构造增强型 Prompt，引导 LLM 进行“机理-实验”对齐分析
        """
        
        # 1. 结构化特征描述
        feature_desc = f"""
### 1. 图像特征量化结果 (CV感知层)
- 平均管径: {cv_data.get('diameter', '未知')} nm
- 垂直取向度 (S): {cv_data.get('alignment', '未知')}
- 面填充密度: {cv_data.get('density', '未知')}%
- 骨架曲率评价: {cv_data.get('curvature', '未知')}
"""

        # 2. 工艺背景描述
        process_desc = f"""
### 2. 原始工艺参数 (溯源)
- 生长温度: {process_data.get('growth_temp', '未知')} ℃
- 催化剂(Fe): {process_data.get('fe_thickness', '未知')} nm
- 载气流量: Ar={process_data.get('ar_flow', '未知')}, H2={process_data.get('h2_flow', '未知')}, C2H4={process_data.get('c2h4_flow', '未知')} sccm
- 反应时长: {process_data.get('growth_time', '未知')} h
"""

        # 3. 构造终极分析指令
        prompt = f"""
作为一名碳纳米管(CNT)生长领域的资深科学家，请结合以下数据与参考机理，为我写一份学术分析简报。

{feature_desc}
{process_desc}

### 3. 参考研究文献机理 (RAG 知识层)
{rag_context}

### 4. 任务要求
请你完成以下逻辑推演：
1. **异常关联诊断**：观察图像特征，利用文献机理分析当前的工艺参数是否匹配。例如：如果管径异常偏大，是否由Fe催化剂厚度或温度引起？
2. **机理失效分析**：根据取向度和密度，推测炉内流场或化学势能是否达到了生长森林的最佳平衡点。
3. **闭环实验建议**：基于"知识驱动"原则，给出下一批次实验的改进策略（具体的参数增减建议）。

请使用严谨的学术口吻，并以 Markdown 格式输出。
"""
        return prompt

    def analyze_anomalies(self, cv_data: Dict[str, Any]) -> List[str]:
        """
        基于硬代码规则的基础诊断（作为 LLM 的辅助参考）
        """
        anomalies = []
        dia = cv_data.get('diameter', 0)
        if dia > self.thresholds["diameter"]["high"]:
            anomalies.append("管径显著偏大，可能存在催化剂颗粒团聚现象")
        
        ali = cv_data.get('alignment', 1.0)
        if ali < self.thresholds["alignment"]["poor"]:
            anomalies.append("取向度极差，生长模式可能偏向随机缠绕而非定向森林")
            
        return anomalies

if __name__ == "__main__":
    # 测试脚本
    engine = AnalysisReportEngine()
    test_cv = {"diameter": 32.5, "alignment": 0.35, "density": 12.4, "curvature": "High"}
    test_process = {"growth_temp": 750, "fe_thickness": 2.0, "c2h4_flow": 200}
    test_rag = "文献指出：当乙烯浓度超过 15% 时，非晶碳沉积加速，会导致催化剂中毒和管径非均匀增厚..."
    
    print(engine.generate_prompt(test_cv, test_process, test_rag))
