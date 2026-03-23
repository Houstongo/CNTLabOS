"""
LLM辅助MSFU提取器
使用LLM进行结构化提取，解决规则提取器的覆盖率问题
特点：
1. 处理复杂句式（否定、转折、复合关系）
2. 理解隐含关系
3. 支持中英文混合
4. 生成结构化MSFU
"""

import json
import re
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from backend.core.msfu_extractor import (
    MSFU, MSFUMetadata, Assertion, Evidence, Condition,
    RelationType, Direction, ExtractionMethod,
    classify_entity, normalize_entity
)


@dataclass
class LLMExtractedRelation:
    """LLM提取的关系"""
    source_entity: str
    target_entity: str
    relation_type: str
    direction: str
    condition: Optional[Dict] = None
    confidence: float = 0.8
    explanation: str = ""


class LLMMSFUExtractor:
    """LLM辅助MSFU提取器"""

    def __init__(self, llm_client, confidence_threshold: float = 0.6):
        """
        Args:
            llm_client: LLM客户端（AIInterpreter）
            confidence_threshold: 置信度阈值
        """
        self.llm_client = llm_client
        self.confidence_threshold = confidence_threshold

    def extract(
        self,
        chunk: str,
        metadata: MSFUMetadata,
        doc_title: str = ""
    ) -> List[MSFU]:
        """
        使用LLM提取MSFU

        Args:
            chunk: 文本块
            metadata: 元数据
            doc_title: 文档标题

        Returns:
            MSFU列表
        """
        # 分句（每次处理一个句子，提高准确性）
        sentences = self._split_sentences(chunk)

        all_msfus = []
        for sentence in sentences:
            if len(sentence.strip()) < 20:  # 跳过太短的句子
                continue

            # LLM提取
            extracted_relations = self._llm_extract_single(sentence)

            # 转换为MSFU
            for relation in extracted_relations:
                if relation.confidence < self.confidence_threshold:
                    continue

                msfu = self._create_msfu(
                    sentence,
                    metadata,
                    doc_title,
                    relation
                )

                if msfu:
                    all_msfus.append(msfu)

        return all_msfus

    def _split_sentences(self, text: str) -> List[str]:
        """分句（改进版，支持中英文混合）"""
        # 中文句号
        text = re.sub(r'([。！？；])', r'\1\n', text)
        # 英文句号
        text = re.sub(r'([.!?;])\s+', r'\1\n', text)
        # 分号
        text = re.sub(r';\s*', ';\n', text)

        sentences = [s.strip() for s in text.split('\n') if s.strip()]
        return sentences

    def _llm_extract_single(self, sentence: str) -> List[LLMExtractedRelation]:
        """使用LLM从单个句子提取关系"""
        prompt = self._build_extraction_prompt(sentence)

        try:
            response = self._call_llm(prompt)
            return self._parse_llm_response(response, sentence)
        except Exception as e:
            print(f"LLM提取失败: {e}")
            return []

    def _build_extraction_prompt(self, sentence: str) -> str:
        """构建提取Prompt"""
        prompt = f"""
你是一个专业的碳纳米管阵列研究领域的知识抽取助手。请从以下文本中提取工艺参数、形貌特征、性能指标、机理之间的关系。

文本：{sentence}

任务要求：
1. 识别关系类型（正向/负向/因果/影响）
2. 提取源实体和目标实体
3. 识别条件约束（温度、时间、流量等）
4. 判断影响方向（positive/negative/neutral）
5. 判断否定关系（does not affect, 没有影响）
6. 处理复合关系（A affects B and C）

实体分类：
- process（工艺参数）：temperature, time, flow, thickness, catalyst等
- morphology（形貌特征）：alignment, density, diameter, curvature等
- performance（性能指标）：conductivity, resistivity, strength等
- mechanism（机理实体）：diffusion, nucleation, deactivation, ripening等

输出JSON格式：
{{
    "relations": [
        {{
            "source_entity": "process:growth_temp",
            "target_entity": "morphology:alignment",
            "relation_type": "increases",
            "direction": "positive",
            "condition": {{
                "parameter": "temperature",
                "operator": ">",
                "value": 750,
                "unit": "°C"
            }},
            "confidence": 0.85,
            "explanation": "Growth temperature increases alignment"
        }}
    ]
}}

注意事项：
1. 如果没有明确关系，返回空数组
2. 否定关系（does not affect）的direction设为"neutral"
3. 复合关系拆分为多个独立关系
4. 条件值提取数值
5. 置信度根据明确程度评估（0.5-1.0）
6. 只返回文本中明确陈述的关系，不要臆造
"""
        return prompt

    def _call_llm(self, prompt: str) -> str:
        """调用LLM"""
        # 假设llm_client有generate方法
        # 根据实际的AIInterpreter接口调整
        response = self.llm_client.generate(
            prompt=prompt,
            temperature=0.1,  # 低温度提高一致性
            max_tokens=1000
        )
        return response

    def _parse_llm_response(self, response: str, sentence: str) -> List[LLMExtractedRelation]:
        """解析LLM响应"""
        relations = []

        try:
            # 提取JSON部分（处理可能的markdown格式）
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                response = json_match.group(1)

            # 清理响应
            response = response.strip()
            if response.startswith('```'):
                response = response[3:]
            if response.endswith('```'):
                response = response[:-3]

            # 解析JSON
            data = json.loads(response)

            if "relations" in data:
                for rel_data in data["relations"]:
                    try:
                        relation = self._parse_relation_data(rel_data, sentence)
                        relations.append(relation)
                    except Exception as e:
                        print(f"解析关系失败: {e}")
                        continue

        except json.JSONDecodeError as e:
            print(f"JSON解析失败: {e}, 响应: {response[:200]}")
        except Exception as e:
            print(f"LLM响应解析失败: {e}")

        return relations

    def _parse_relation_data(self, rel_data: dict, sentence: str) -> LLMExtractedRelation:
        """解析单个关系数据"""
        source_entity = rel_data.get("source_entity", "")
        target_entity = rel_data.get("target_entity", "")
        relation_type = rel_data.get("relation_type", "")
        direction = rel_data.get("direction", "unknown")
        condition_data = rel_data.get("condition")
        confidence = rel_data.get("confidence", 0.8)
        explanation = rel_data.get("explanation", "")

        # 规范化实体
        source_entity = self._normalize_entity_with_llm(source_entity, sentence)
        target_entity = self._normalize_entity_with_llm(target_entity, sentence)

        # 规范化关系类型
        relation_type = self._normalize_relation_type(relation_type)

        # 解析条件
        condition = None
        if condition_data:
            condition = Condition(
                parameter=condition_data.get("parameter", ""),
                operator=condition_data.get("operator", ""),
                value=condition_data.get("value", 0),
                unit=condition_data.get("unit")
            )

        return LLMExtractedRelation(
            source_entity=source_entity,
            target_entity=target_entity,
            relation_type=relation_type,
            direction=direction,
            condition=condition,
            confidence=confidence,
            explanation=explanation
        )

    def _normalize_entity_with_llm(self, entity: str, sentence: str) -> str:
        """规范化实体（结合规则和LLM输出）"""
        # 1. 如果已经是分类格式，直接返回
        if ":" in entity:
            return entity

        # 2. 尝试规则分类
        classification = classify_entity(sentence)
        if classification:
            category, entity_type = classification
            return f"{category}:{entity_type}"

        # 3. LLM输出的分类格式化
        if "_" in entity:
            parts = entity.split("_")
            if len(parts) >= 2:
                category, entity_type = parts[0], "_".join(parts[1:])
                valid_categories = {"process", "morphology", "performance", "mechanism"}
                if category in valid_categories:
                    return f"{category}:{entity_type}"

        # 4. 默认格式
        return f"unknown:{entity.lower()}"

    def _normalize_relation_type(self, relation_type: str) -> str:
        """规范化关系类型"""
        valid_types = [
            "increases", "decreases", "causes", "affects",
            "promotes", "inhibits", "depends_on", "correlates_with"
        ]

        # 直接匹配
        if relation_type in valid_types:
            return relation_type

        # 映射相似词汇
        type_mapping = {
            "increase": "increases",
            "improve": "increases",
            "enhance": "increases",
            "reduce": "decreases",
            "decrease": "decreases",
            "lower": "decreases",
            "cause": "causes",
            "lead to": "causes",
            "result in": "causes",
            "affect": "affects",
            "influence": "affects",
            "promote": "promotes",
            "facilitate": "promotes",
            "inhibit": "inhibits",
            "suppress": "inhibits",
        }

        return type_mapping.get(relation_type.lower(), "affects")

    def _create_msfu(
        self,
        sentence: str,
        metadata: MSFUMetadata,
        doc_title: str,
        relation: LLMExtractedRelation
    ) -> Optional[MSFU]:
        """创建MSFU对象"""
        # 验证实体格式
        if not self._is_valid_entity(relation.source_entity):
            return None
        if not self._is_valid_entity(relation.target_entity):
            return None

        # 跳过自引用
        if relation.source_entity == relation.target_entity:
            return None

        # 创建断言
        assertion = Assertion(
            source_entity=relation.source_entity,
            relation_type=relation.relation_type,
            target_entity=relation.target_entity,
            condition=relation.condition,
            direction=relation.direction
        )

        # 创建证据
        evidence = Evidence(
            text_snippet=sentence[:200],
            doc_title=doc_title or metadata.doc_title,
            confidence=relation.confidence,
            extraction_method=ExtractionMethod.LLM.value,
            page_num=metadata.page_num,
            chunk_id=int(metadata.chunk_id) if metadata.chunk_id else None
        )

        return MSFU(
            content=sentence[:500],
            metadata=metadata,
            assertion=assertion,
            evidence=evidence
        )

    def _is_valid_entity(self, entity: str) -> bool:
        """检查实体是否有效"""
        valid_categories = {"process", "morphology", "performance", "mechanism", "unknown"}

        if ":" in entity:
            category = entity.split(":")[0]
            return category in valid_categories

        if 3 <= len(entity) <= 100:
            return not entity.isdigit()

        return False


class HybridMSFUExtractor:
    """混合提取器：规则 + LLM"""

    def __init__(
        self,
        rule_extractor=None,
        llm_client=None,
        confidence_threshold: float = 0.5,
        use_llm: bool = True
    ):
        """
        Args:
            rule_extractor: 规则提取器
            llm_client: LLM客户端
            confidence_threshold: 置信度阈值
            use_llm: 是否使用LLM
        """
        self.rule_extractor = rule_extractor
        self.llm_extractor = LLMMSFUExtractor(llm_client, confidence_threshold) if llm_client and use_llm else None
        self.confidence_threshold = confidence_threshold

    def extract(
        self,
        chunk: str,
        metadata: MSFUMetadata,
        doc_title: str = ""
    ) -> List[MSFU]:
        """混合提取"""
        all_msfus = []

        # 1. 规则提取
        if self.rule_extractor:
            rule_msfus = self.rule_extractor.extract(chunk, metadata, doc_title)
            all_msfus.extend(rule_msfus)

        # 2. LLM提取
        if self.llm_extractor:
            llm_msfus = self.llm_extractor.extract(chunk, metadata, doc_title)
            all_msfus.extend(llm_msfus)

        # 3. 去重融合
        merged_msfus = self._merge_and_deduplicate(all_msfus)

        return merged_msfus

    def _merge_and_deduplicate(self, msfus: List[MSFU]) -> List[MSFU]:
        """合并去重"""
        unique_msfus = []
        seen = set()

        for msfu in msfus:
            # 创建唯一键
            key = (
                msfu.assertion.source_entity,
                msfu.assertion.target_entity,
                msfu.assertion.relation_type
            )

            if key not in seen:
                seen.add(key)
                unique_msfus.append(msfu)
            else:
                # 如果已存在，保留置信度更高的
                for i, existing in enumerate(unique_msfus):
                    existing_key = (
                        existing.assertion.source_entity,
                        existing.assertion.target_entity,
                        existing.assertion.relation_type
                    )
                    if existing_key == key:
                        if msfu.evidence.confidence > existing.evidence.confidence:
                            unique_msfus[i] = msfu
                        break

        return unique_msfus
