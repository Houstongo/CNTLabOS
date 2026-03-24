"""
MSFU (Minimal Semantic Fact Unit) Extractor for CNTA RAG System.

Implements MSFU data structure and extraction logic:
- MSFU = (content, metadata, assertion, evidence)
- Assertion = (source_entity, relation_type, target_entity, condition, direction)
"""

import re
import json
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Set, Dict, Any, Union, Tuple
from enum import Enum


class RelationType(Enum):
    """关系类型枚举"""
    CAUSES = "causes"
    INCREASES = "increases"
    DECREASES = "decreases"
    AFFECTS = "affects"
    PROMOTES = "promotes"
    INHIBITS = "inhibits"
    DEPENDS_ON = "depends_on"
    CORRELATES_WITH = "correlates_with"


class Direction(Enum):
    """影响方向"""
    POSITIVE = "positive"    # 正相关/增加
    NEGATIVE = "negative"    # 负相关/减少
    NEUTRAL = "neutral"      # 无明确方向
    UNKNOWN = "unknown"      # 方向未知


class ExtractionMethod(Enum):
    """提取方法"""
    RULE = "rule"
    LLM = "llm"
    HYBRID = "hybrid"


@dataclass
class MSFUMetadata:
    """MSFU元数据"""
    doc_id: str
    chunk_id: str
    doc_title: str
    doc_type: str = "pdf"  # pdf, txt, html
    page_num: Optional[int] = None
    section: Optional[str] = None


@dataclass
class Condition:
    """条件约束"""
    parameter: str              # "temperature", "pressure", "time"
    operator: str              # ">", "<", "=", ">=", "<=", "in_range"
    value: Union[float, int, Tuple[float, float]]  # 500 or (400, 600)
    unit: Optional[str] = None  # "°C", "min", "sccm"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parameter": self.parameter,
            "operator": self.operator,
            "value": self.value,
            "unit": self.unit
        }

    def matches(self, param: str, value: float, unit: Optional[str] = None) -> bool:
        """检查给定值是否满足条件"""
        if self.parameter.lower() != param.lower():
            return False
        if unit and self.unit and self.unit.lower() != unit.lower():
            # 尝试单位转换（简单实现）
            pass

        if self.operator == ">":
            return value > self.value
        elif self.operator == "<":
            return value < self.value
        elif self.operator == ">=":
            return value >= self.value
        elif self.operator == "<=":
            return value <= self.value
        elif self.operator == "=":
            return abs(value - self.value) < 0.01
        elif self.operator == "in_range":
            if isinstance(self.value, tuple):
                return self.value[0] <= value <= self.value[1]
        return False

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Condition":
        return cls(
            parameter=data.get("parameter", ""),
            operator=data.get("operator", ""),
            value=data.get("value", 0),
            unit=data.get("unit")
        )


@dataclass
class Assertion:
    """关系断言"""
    source_entity: str         # "process:temperature" or "process:fe_thickness"
    relation_type: str         # RelationType value as string
    target_entity: str         # "morphology:density" or "performance:conductivity"
    condition: Optional[Condition] = None
    direction: str = Direction.UNKNOWN.value

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_entity": self.source_entity,
            "relation_type": self.relation_type,
            "target_entity": self.target_entity,
            "condition": self.condition.to_dict() if self.condition else None,
            "direction": self.direction
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Assertion":
        condition_data = data.get("condition")
        condition = Condition.from_dict(condition_data) if condition_data else None
        return cls(
            source_entity=data.get("source_entity", ""),
            relation_type=data.get("relation_type", ""),
            target_entity=data.get("target_entity", ""),
            condition=condition,
            direction=data.get("direction", Direction.UNKNOWN.value)
        )

    def reverse(self) -> "Assertion":
        """反转关系方向"""
        reversed_direction = {
            Direction.POSITIVE.value: Direction.NEGATIVE.value,
            Direction.NEGATIVE.value: Direction.POSITIVE.value,
        }.get(self.direction, self.direction)

        return Assertion(
            source_entity=self.target_entity,
            relation_type=self.relation_type,
            target_entity=self.source_entity,
            condition=self.condition,
            direction=reversed_direction
        )


@dataclass
class Evidence:
    """证据信息"""
    text_snippet: str
    doc_title: str
    confidence: float = 0.5
    extraction_method: str = ExtractionMethod.RULE.value
    page_num: Optional[int] = None
    chunk_id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Evidence":
        return cls(**data)


@dataclass
class MSFU:
    """最小语义事实单元 (Minimal Semantic Fact Unit)"""
    content: str              # 原始文本内容
    metadata: MSFUMetadata
    assertion: Assertion
    evidence: Evidence
    msfu_id: Optional[int] = None  # 数据库ID

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "content": self.content,
            "metadata": asdict(self.metadata),
            "assertion": self.assertion.to_dict(),
            "evidence": self.evidence.to_dict(),
        }
        if self.msfu_id:
            result["msfu_id"] = self.msfu_id
        return result

    def to_db_row(self) -> Dict[str, Any]:
        """转换为数据库行格式"""
        condition = self.assertion.condition

        # 从 source_entity 和 target_entity 提取 factor 类型
        source_parts = self.assertion.source_entity.split(":") if self.assertion.source_entity else []
        target_parts = self.assertion.target_entity.split(":") if self.assertion.target_entity else []

        process_factor = None
        morphology_factor = None
        performance_factor = None

        if len(source_parts) >= 2:
            if source_parts[0] == "process":
                process_factor = source_parts[1]
            elif source_parts[0] == "morphology":
                morphology_factor = source_parts[1]
            elif source_parts[0] == "performance":
                performance_factor = source_parts[1]

        if len(target_parts) >= 2:
            if target_parts[0] == "process":
                process_factor = target_parts[1]
            elif target_parts[0] == "morphology":
                morphology_factor = target_parts[1]
            elif target_parts[0] == "performance":
                performance_factor = target_parts[1]

        return {
            "chunk_id": self.metadata.chunk_id,
            "source_entity": self.assertion.source_entity,
            "relation_type": self.assertion.relation_type,
            "target_entity": self.assertion.target_entity,
            "condition_param": condition.parameter if condition else None,
            "condition_op": condition.operator if condition else None,
            "condition_value": json.dumps(condition.value) if condition else None,
            "condition_unit": condition.unit if condition else None,
            "direction": self.assertion.direction,
            "content": self.content,
            "confidence": self.evidence.confidence,
            "extraction_method": self.evidence.extraction_method,
            "doc_title": self.metadata.doc_title,
            "page_num": self.metadata.page_num,
            # 兼容 kb_links 表结构的额外字段
            "process_factor": process_factor,
            "morphology_factor": morphology_factor,
            "performance_factor": performance_factor,
            "effect_direction": self.assertion.direction,
            "mechanism_summary": self.content[:220] if self.content else None,
            "evidence_text": self.evidence.text_snippet[:320] if self.evidence.text_snippet else None,
        }

    @classmethod
    def from_db_row(cls, row: Dict[str, Any]) -> "MSFU":
        """从数据库行创建MSFU"""
        condition = None
        if row.get("condition_param"):
            try:
                value = json.loads(row["condition_value"])
            except (json.JSONDecodeError, TypeError):
                value = row.get("condition_value")
            condition = Condition(
                parameter=row["condition_param"],
                operator=row["condition_op"] or "",
                value=value,
                unit=row["condition_unit"]
            )

        assertion = Assertion(
            source_entity=row["source_entity"] or "",
            relation_type=row["relation_type"] or "",
            target_entity=row["target_entity"] or "",
            condition=condition,
            direction=row.get("direction", Direction.UNKNOWN.value)
        )

        metadata = MSFUMetadata(
            doc_id=str(row.get("doc_id", "")),
            chunk_id=str(row.get("chunk_id", "")),
            doc_title=row.get("doc_title", ""),
            doc_type="pdf",
            page_num=row.get("page_num")
        )

        evidence = Evidence(
            text_snippet=row.get("content", "")[:200],
            doc_title=row.get("doc_title", ""),
            confidence=row.get("confidence", 0.5),
            extraction_method=row.get("extraction_method", ExtractionMethod.RULE.value),
            page_num=row.get("page_num"),
            chunk_id=row.get("chunk_id")
        )

        return cls(
            content=row.get("content", ""),
            metadata=metadata,
            assertion=assertion,
            evidence=evidence,
            msfu_id=row.get("id")
        )


# ==================== 条件提取模式 ====================

CONDITION_PATTERNS = {
    # 温度条件
    "temperature_above": [
        r"(?:温度|temperature)\s*[>高于]\s*(\d+\.?\d*)\s*°?[C℃]?",
        r"(?:温度|temperature)\s*above\s*(\d+\.?\d*)\s*°?[C℃]?",
    ],
    "temperature_below": [
        r"(?:温度|temperature)\s*[<低于]\s*(\d+\.?\d*)\s*°?[C℃]?",
        r"(?:温度|temperature)\s*below\s*(\d+\.?\d*)\s*°?[C℃]?",
    ],
    "temperature_range": [
        r"(?:温度|temperature)\s*(?:在|between)\s*(\d+\.?\d*)\s*[°~\-]\s*(\d+\.?\d*)\s*°?[C℃]?",
        r"(?:温度|temperature)\s*(?:在|between)\s*(\d+\.?\d*)\s*and\s*(\d+\.?\d*)\s*°?[C℃]?",
    ],
    "temperature_equal": [
        r"(?:温度|temperature)\s*[=等于]\s*(\d+\.?\d*)\s*°?[C℃]?",
        r"(?:at\s+)?(\d+\.?\d*)\s*°?[C℃]?\s*(?:温度|temperature)",
    ],
    # 时间条件
    "time_above": [
        r"(?:时间|time|duration)\s*[>超过]\s*(\d+\.?\d*)\s*(?:min|分钟|h|小时)",
        r"(?:时间|time|duration)\s*above\s*(\d+\.?\d*)\s*(?:min|h)",
    ],
    "time_range": [
        r"(?:时间|time|duration)\s*(?:在|between)\s*(\d+\.?\d*)\s*[~\-to]\s*(\d+\.?\d*)\s*(?:min|h|分钟|小时)",
    ],
    # 流量条件
    "flow_above": [
        r"(?:流量|flow)\s*[>超过]\s*(\d+\.?\d*)\s*sccm",
    ],
    "flow_range": [
        r"(?:流量|flow)\s*(?:在|between)\s*(\d+\.?\d*)\s*[~\-to]\s*(\d+\.?\d*)\s*sccm",
    ],
    # 厚度条件
    "thickness_above": [
        r"(?:厚度|thickness)\s*[>超过]\s*(\d+\.?\d*)\s*nm",
    ],
    "thickness_range": [
        r"(?:厚度|thickness)\s*(?:在|between)\s*(\d+\.?\d*)\s*[~\-to]\s*(\d+\.?\d*)\s*nm",
    ],
    # 压力条件
    "pressure_above": [
        r"(?:压力|pressure)\s*[>超过]\s*(\d+\.?\d*)\s*Pa",
    ],
}


def extract_conditions(text: str) -> List[Condition]:
    """从文本中提取条件"""
    conditions = []

    for cond_type, patterns in CONDITION_PATTERNS.items():
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                cond = _parse_condition_from_match(cond_type, match, text)
                if cond:
                    conditions.append(cond)

    return conditions


def _parse_condition_from_match(cond_type: str, match: re.Match, text: str) -> Optional[Condition]:
    """根据匹配类型解析条件"""
    groups = match.groups()

    if "above" in cond_type:
        param = _get_param_from_type(cond_type)
        value = float(groups[0])
        unit = _get_unit_from_type(cond_type, text, match)
        return Condition(parameter=param, operator=">", value=value, unit=unit)

    elif "below" in cond_type:
        param = _get_param_from_type(cond_type)
        value = float(groups[0])
        unit = _get_unit_from_type(cond_type, text, match)
        return Condition(parameter=param, operator="<", value=value, unit=unit)

    elif "equal" in cond_type:
        param = _get_param_from_type(cond_type)
        value = float(groups[0])
        unit = _get_unit_from_type(cond_type, text, match)
        return Condition(parameter=param, operator="=", value=value, unit=unit)

    elif "range" in cond_type and len(groups) >= 2:
        param = _get_param_from_type(cond_type)
        min_val = float(groups[0])
        max_val = float(groups[1])
        unit = _get_unit_from_type(cond_type, text, match)
        return Condition(parameter=param, operator="in_range", value=(min_val, max_val), unit=unit)

    return None


def _get_param_from_type(cond_type: str) -> str:
    """从条件类型获取参数名"""
    if "temperature" in cond_type:
        return "temperature"
    elif "time" in cond_type or "duration" in cond_type:
        return "time"
    elif "flow" in cond_type:
        return "flow"
    elif "thickness" in cond_type:
        return "thickness"
    elif "pressure" in cond_type:
        return "pressure"
    return "parameter"


def _get_unit_from_type(cond_type: str, text: str, match: re.Match) -> Optional[str]:
    """从上下文获取单位"""
    if "temperature" in cond_type:
        if "°C" in text[match.start():match.end()+10] or "℃" in text[match.start():match.end()+10]:
            return "°C"
        return "°C"
    elif "time" in cond_type or "duration" in cond_type:
        if "min" in text[match.start():match.end()+10] or "分钟" in text[match.start():match.end()+10]:
            return "min"
        if "h" in text[match.start():match.end()+10] or "小时" in text[match.start():match.end()+10]:
            return "h"
    elif "flow" in cond_type:
        return "sccm"
    elif "thickness" in cond_type:
        return "nm"
    elif "pressure" in cond_type:
        return "Pa"
    return None


# ==================== 实体分类 ====================

ENTITY_CATEGORIES = {
    "process": {
        "growth_temp": r"(?:生长|growth)\s*(?:温度|temperature)",
        "growth_time": r"(?:生长|growth)\s*(?:时间|time|duration)",
        "anneal_temp": r"(?:退火|anneal(?:ing)?)\s*(?:温度|temperature)",
        "anneal_time": r"(?:退火|anneal(?:ing)?)\s*(?:时间|time)",
        "ar_flow": r"(?:Ar|氩(?:气)?)\s*(?:流量|flow)",
        "h2_flow": r"(?:H2|氢(?:气)?)\s*(?:流量|flow)",
        "c2h4_flow": r"(?:C2H4|乙烯)\s*(?:流量|flow)",
        "fe_thickness": r"(?:Fe|铁|催化剂)\s*(?:厚度|thickness)",
        "al2o3_thickness": r"(?:Al2O3|氧化铝|支撑层)\s*(?:厚度|thickness)",
    },
    "morphology": {
        "density": r"(?:密度|density|覆盖率)",
        "alignment": r"(?:取向|alignment|对齐)",
        "diameter": r"(?:直径|diameter|管径)",
        "curvature": r"(?:曲率|curvature|弯曲|波曲)",
        "tortuosity": r"(?:曲折度|tortuosity)",
        "length": r"(?:长度|length)",
    },
    "performance": {
        "conductivity": r"(?:电导|conductiv)",
        "resistivity": r"(?:电阻率|resistiv)",
        "sheet_resistance": r"(?:方阻|sheet resistance)",
        "tensile_strength": r"(?:抗拉强度|tensile|strength)",
        "modulus": r"(?:模量|modulus)",
    },
    "mechanism": {
        "catalyst": r"(?:催化剂|catalyst)",
        "nucleation": r"(?:成核|nucleation)",
        "growth_mode": r"(?:生长模式|growth mode)",
        "diffusion": r"(?:扩散|diffusion)",
    }
}


def classify_entity(text: str) -> Optional[Tuple[str, str]]:
    """分类实体文本，返回 (category, entity_type)"""
    for category, entities in ENTITY_CATEGORIES.items():
        for entity_type, pattern in entities.items():
            if re.search(pattern, text, re.IGNORECASE):
                return (category, entity_type)
    return None


def normalize_entity(text: str) -> str:
    """规范化实体文本为标准格式"""
    result = classify_entity(text)
    if result:
        category, entity_type = result
        return f"{category}:{entity_type}"
    return text.strip().lower()


# ==================== 句子分割 ====================

SENTENCE_SPLITTERS = [
    r'[。！？]\s*',  # 中文句号
    r'[.!?]\s+',    # 英文句号
    r';\s*',        # 分号
    r'\n\s*',       # 换行
]


def split_sentences(text: str) -> List[str]:
    """将文本分割为句子"""
    sentences = [text]
    for pattern in SENTENCE_SPLITTERS:
        new_sentences = []
        for s in sentences:
            new_sentences.extend(re.split(pattern, s))
        sentences = [s.strip() for s in new_sentences if s.strip()]
    return sentences


# ==================== 规则提取器 ====================

RELATION_PATTERNS = [
    # 正向关系
    {
        "type": RelationType.INCREASES.value,
        "direction": Direction.POSITIVE.value,
        "patterns": [
            r"(?P<src>.+?)\s*(?:增加|提高|enhance|increase|improve)\s*(?P<tgt>.+?)",
            r"(?P<tgt>.+?)\s*(?:随着|随着|with)\s*(?P<src>.+?)\s*(?:增加|提高|increase)",
            r"(?:higher|higher|greater)\s*(?P<src>.+?)\s*(?:leads to|leads to|results in)\s*(?:higher|higher|greater)\s*(?P<tgt>.+?)",
        ],
    },
    {
        "type": RelationType.DECREASES.value,
        "direction": Direction.NEGATIVE.value,
        "patterns": [
            r"(?P<src>.+?)\s*(?:减少|降低|decrease|reduce)\s*(?P<tgt>.+?)",
            r"(?P<tgt>.+?)\s*(?:随着|随着|with)\s*(?P<src>.+?)\s*(?:减少|降低|decrease)",
            r"(?:lower|lower)\s*(?P<src>.+?)\s*(?:leads to|leads to|results in)\s*(?:lower|lower)\s*(?P<tgt>.+?)",
        ],
    },
    {
        "type": RelationType.CAUSES.value,
        "direction": Direction.POSITIVE.value,
        "patterns": [
            r"(?P<src>.+?)\s*(?:导致|引起|cause|lead to|result in)\s*(?P<tgt>.+?)",
            r"(?P<tgt>.+?)\s*(?:是由于|due to|caused by)\s*(?P<src>.+?)",
        ],
    },
    {
        "type": RelationType.AFFECTS.value,
        "direction": Direction.UNKNOWN.value,
        "patterns": [
            r"(?P<src>.+?)\s*(?:影响|affect|influence)\s*(?P<tgt>.+?)",
            r"(?P<tgt>.+?)\s*(?:is affected by|受|受...影响)\s*(?P<src>.+?)",
        ],
    },
    {
        "type": RelationType.PROMOTES.value,
        "direction": Direction.POSITIVE.value,
        "patterns": [
            r"(?P<src>.+?)\s*(?:促进|promote|facilitate)\s*(?P<tgt>.+?)",
            r"(?P<tgt>.+?)\s*(?:促进|promote)\s*(?P<src>.+?)",
        ],
    },
    {
        "type": RelationType.INHIBITS.value,
        "direction": Direction.NEGATIVE.value,
        "patterns": [
            r"(?P<src>.+?)\s*(?:抑制|inhibit|suppress)\s*(?P<tgt>.+?)",
            r"(?P<tgt>.+?)\s*(?:抑制|inhibit)\s*(?P<src>.+?)",
        ],
    },
]


class RuleBasedExtractor:
    """基于规则的MSFU提取器"""

    # 从 knowledge_base 导入模式（类级别）
    PROCESS_FACTOR_PATTERNS = None
    MORPHOLOGY_FACTOR_PATTERNS = None
    PERFORMANCE_FACTOR_PATTERNS = None
    MECHANISM_FACTOR_PATTERNS = None
    INCREASE_PATTERNS = None
    DECREASE_PATTERNS = None

    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        # 延迟导入模式定义
        from .knowledge_base import (
            PROCESS_FACTOR_PATTERNS as _PROCESS,
            MORPHOLOGY_FACTOR_PATTERNS as _MORPHOLOGY,
            PERFORMANCE_FACTOR_PATTERNS as _PERFORMANCE,
            MECHANISM_FACTOR_PATTERNS as _MECHANISM,
            INCREASE_PATTERNS as _INCREASE,
            DECREASE_PATTERNS as _DECREASE,
        )
        RuleBasedExtractor.PROCESS_FACTOR_PATTERNS = _PROCESS
        RuleBasedExtractor.MORPHOLOGY_FACTOR_PATTERNS = _MORPHOLOGY
        RuleBasedExtractor.PERFORMANCE_FACTOR_PATTERNS = _PERFORMANCE
        RuleBasedExtractor.MECHANISM_FACTOR_PATTERNS = _MECHANISM
        RuleBasedExtractor.INCREASE_PATTERNS = _INCREASE
        RuleBasedExtractor.DECREASE_PATTERNS = _DECREASE

        # 合并所有模式
        RuleBasedExtractor.ALL_FACTOR_PATTERNS = {
            **RuleBasedExtractor.PROCESS_FACTOR_PATTERNS,
            **RuleBasedExtractor.MORPHOLOGY_FACTOR_PATTERNS,
            **RuleBasedExtractor.PERFORMANCE_FACTOR_PATTERNS,
            **RuleBasedExtractor.MECHANISM_FACTOR_PATTERNS,
        }
        # 方向检测模式
        RuleBasedExtractor.DIRECTION_PATTERNS = RuleBasedExtractor.INCREASE_PATTERNS + RuleBasedExtractor.DECREASE_PATTERNS

    def extract(
        self,
            chunk: str,
            metadata: MSFUMetadata,
            doc_title: str = ""
        ) -> List[MSFU]:
            """
            从文本块提取MSFU，            复用 knowledge_base 的规则逻辑
        """
            msfus = []
            sentences = split_sentences(chunk)

            for sentence in sentences:
                # 使用与 kb_links 相同的提取逻辑
                process_hits = self._match_factors(sentence, self.PROCESS_FACTOR_PATTERNS)
                morph_hits = self._match_factors(sentence, self.MORPHOLOGY_FACTOR_PATTERNS)
                perf_hits = self._match_factors(sentence, self.PERFORMANCE_FACTOR_PATTERNS)
                mech_hits = self._match_factors(sentence, self.MECHANISM_FACTOR_PATTERNS)

                direction = self._detect_direction(sentence)
                if not direction:
                    continue

                # 检测是否有机理关键词
                has_mechanism = bool(mech_hits) or any(
                    re.search(p, sentence.lower())
                    for patterns in self.MECHANISM_FACTOR_PATTERNS.values()
                    for p in patterns
                )

                confidence = 0.55 + 0.1 * int(has_mechanism)

                # 生成关系
                for process_factor in process_hits:
                    for morphology_factor in morph_hits:
                        msfus.append(self._create_msfu(
                            sentence, metadata, doc_title,
                            source_entity=f"process:{process_factor}",
                            relation_type="increases" if direction == "positive" else "decreases",
                            target_entity=f"morphology:{morphology_factor}",
                            direction=direction,
                            confidence=confidence
                        ))

                    for mechanism_factor in mech_hits:
                        msfus.append(self._create_msfu(
                            sentence, metadata, doc_title,
                            source_entity=f"process:{process_factor}",
                            relation_type="causes",
                            target_entity=f"mechanism:{mechanism_factor}",
                            direction=direction,
                            confidence=min(confidence + 0.08, 0.95)
                        ))

                for mechanism_factor in mech_hits:
                    for morphology_factor in morph_hits:
                        msfus.append(self._create_msfu(
                            sentence, metadata, doc_title,
                            source_entity=f"mechanism:{mechanism_factor}",
                            relation_type="affects",
                            target_entity=f"morphology:{morphology_factor}",
                            direction=direction,
                            confidence=min(confidence + 0.08, 0.95)
                        ))

                for morphology_factor in morph_hits:
                    for performance_factor in perf_hits:
                        msfus.append(self._create_msfu(
                            sentence, metadata, doc_title,
                            source_entity=f"morphology:{morphology_factor}",
                            relation_type="increases" if direction == "positive" else "decreases",
                            target_entity=f"performance:{performance_factor}",
                            direction=direction,
                            confidence=confidence
                        ))

            return msfus

    def _match_factors(self, text: str, patterns: dict) -> set:
        """匹配因子"""
        hits = set()
        for factor_type, pattern_list in patterns.items():
            for pattern in pattern_list:
                if re.search(pattern, text.lower()):
                    hits.add(factor_type)
        return hits

    def _detect_direction(self, text: str) -> Optional[str]:
        """检测影响方向"""
        text_lower = text.lower()
        for pattern in self.INCREASE_PATTERNS:
            if re.search(pattern, text_lower):
                return "positive"
        for pattern in self.DECREASE_PATTERNS:
            if re.search(pattern, text_lower):
                return "negative"
        return None

    def _create_msfu(
        self,
        sentence: str,
        metadata: MSFUMetadata,
        doc_title: str,
        source_entity: str,
        relation_type: str,
        target_entity: str,
        direction: str,
        confidence: float
    ) -> MSFU:
        """创建MSFU对象"""
        assertion = Assertion(
            source_entity=source_entity,
            relation_type=relation_type,
            target_entity=target_entity,
            direction=direction
        )
        evidence = Evidence(
            text_snippet=sentence[:200],
            doc_title=doc_title or metadata.doc_title,
            confidence=confidence,
            extraction_method=ExtractionMethod.RULE.value,
            page_num=metadata.page_num,
            chunk_id=int(metadata.chunk_id) if metadata.chunk_id else None
        )
        return MSFU(
            content=sentence[:500],
            metadata=metadata,
            assertion=assertion,
            evidence=evidence
        )

    def _extract_from_sentence(
        self,
        sentence: str,
        metadata: MSFUMetadata,
        doc_title: str = ""
    ) -> List[MSFU]:
        """从单个句子提取MSFU"""
        msfus = []

        for rel_config in RELATION_PATTERNS:
            relation_type = rel_config["type"]
            direction = rel_config["direction"]

            for pattern in rel_config["patterns"]:
                try:
                    match = re.search(pattern, sentence, re.IGNORECASE | re.DOTALL)
                    if not match:
                        continue

                    # 提取源和目标实体
                    groups = match.groupdict()
                    src_text = groups.get("src", "").strip()
                    tgt_text = groups.get("tgt", "").strip()

                    if not src_text or not tgt_text:
                        continue

                    # 规范化实体
                    source_entity = normalize_entity(src_text)
                    target_entity = normalize_entity(tgt_text)

                    # 跳过无效实体
                    if not self._is_valid_entity(source_entity) or not self._is_valid_entity(target_entity):
                        continue

                    # 提取条件
                    conditions = extract_conditions(sentence)
                    condition = conditions[0] if conditions else None

                    # 计算置信度
                    confidence = self._calculate_confidence(sentence, match, pattern)

                    if confidence < self.confidence_threshold:
                        continue

                    # 创建断言
                    assertion = Assertion(
                        source_entity=source_entity,
                        relation_type=relation_type,
                        target_entity=target_entity,
                        condition=condition,
                        direction=direction
                    )

                    # 创建证据
                    evidence = Evidence(
                        text_snippet=sentence[:200],
                        doc_title=doc_title or metadata.doc_title,
                        confidence=confidence,
                        extraction_method=ExtractionMethod.RULE.value,
                        page_num=metadata.page_num,
                        chunk_id=int(metadata.chunk_id) if metadata.chunk_id.isdigit() else None
                    )

                    # 创建MSFU
                    msfu = MSFU(
                        content=sentence,
                        metadata=metadata,
                        assertion=assertion,
                        evidence=evidence
                    )
                    msfus.append(msfu)

                except Exception:
                    # 匹配失败，跳过该模式
                    continue

        return msfus

    def _is_valid_entity(self, entity: str) -> bool:
        """检查实体是否有效"""
        # 排除太短或太长的实体
        if len(entity) < 3 or len(entity) > 100:
            return False

        # 排除纯数字
        if entity.isdigit():
            return False

        # 排除常见停用词
        stopwords = {"it", "this", "that", "the", "a", "an", "的", "这", "那"}
        if entity.lower() in stopwords:
            return False

        return True

    def _calculate_confidence(
        self,
        sentence: str,
        match: re.Match,
        pattern: str
    ) -> float:
        """计算提取置信度"""
        base_score = 0.6

        # 匹配长度占比
        match_length = len(match.group(0))
        sentence_length = len(sentence)
        if sentence_length > 0:
            base_score += 0.2 * (match_length / sentence_length)

        # 包含数值（提高置信度）
        if re.search(r'\d+\.?\d*', sentence):
            base_score += 0.1

        # 包含单位（提高置信度）
        if re.search(r'(°C|nm|min|h|sccm|Pa)', sentence):
            base_score += 0.1

        # 限制范围
        return min(1.0, max(0.0, base_score))


class MSFUExtractor:
    """MSFU提取器主类（规则+LLM混合）"""

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        use_llm_refinement: bool = False,
        confidence_threshold: float = 0.5
    ):
        self.rule_extractor = RuleBasedExtractor(confidence_threshold)
        self.llm_client = llm_client
        self.use_llm_refinement = use_llm_refinement and (llm_client is not None)

    def extract(
        self,
        chunk: str,
        metadata: MSFUMetadata,
        doc_title: str = ""
    ) -> List[MSFU]:
        """
        从文本块提取MSFU

        先用规则提取，再可选地用LLM精炼
        """
        # 1. 规则提取
        msfus = self.rule_extractor.extract(chunk, metadata, doc_title)

        # 2. LLM精炼（可选）
        if self.use_llm_refinement and msfus:
            msfus = self._refine_with_llm(msfus, chunk)

        return msfus

    def _refine_with_llm(self, candidates: List[MSFU], context: str) -> List[MSFU]:
        """
        使用LLM精炼提取结果

        验证、修正或删除候选MSFU
        """
        if not self.llm_client or not candidates:
            return candidates

        try:
            # 将 MSFU 转换为字典格式
            candidate_dicts = [msfu.to_dict() for msfu in candidates]

            # 调用 AIInterpreter 的精炼方法
            refined = self.llm_client.refine_msfu_batch(candidate_dicts, context)

            if not refined:
                return candidates

            # 将精炼结果转换回 MSFU 对象
            refined_msfus = []
            for item in refined:
                if not item.get("valid", True):
                    continue

                # 构建精炼后的 MSFU
                # 优先从顶层获取字段，                # 注意：LLM 可能返回扁平化的字段
                source_entity = item.get("source_entity", "")
                relation_type = item.get("relation_type", "")
                target_entity = item.get("target_entity", "")
                condition = item.get("condition")
                direction = item.get("direction", "unknown")

                # 如果 assertion 子对象存在，使用断言中的值
                if "assertion" in item and isinstance(item["assertion"], dict):
                    source_entity = item["assertion"].get("source_entity", "")
                    relation_type = item["assertion"].get("relation_type", "")
                    target_entity = item["assertion"].get("target_entity", "")
                    condition = item["assertion"].get("condition")
                    direction = item["assertion"].get("direction", "unknown")
                else:
                    # 直接从顶层获取（兼容扁平格式）
                    source_entity = item.get("source_entity", "")
                    relation_type = item.get("relation_type", "")
                    target_entity = item.get("target_entity", "")
                    condition = item.get("condition")
                    direction = item.get("direction", "unknown")

                # 跳过空实体
                if not source_entity or not target_entity:
                    continue

                # 找到原始候选以获取 metadata
                original = candidates[0] if candidates else None
                metadata = original.metadata if original else None

                if not metadata:
                    continue

                assertion = Assertion(
                    source_entity=assertion_data["source_entity"],
                    relation_type=assertion_data["relation_type"],
                    target_entity=assertion_data["target_entity"],
                    condition=Condition.from_dict(assertion_data["condition"]) if assertion_data["condition"] else None,
                    direction=assertion_data["direction"],
                )

                evidence = Evidence(
                    text_snippet=context[:200],
                    doc_title=metadata.doc_title,
                    confidence=item.get("confidence", 0.7),
                    extraction_method=ExtractionMethod.HYBRID.value,
                    page_num=metadata.page_num,
                    chunk_id=int(metadata.chunk_id) if metadata.chunk_id else None,
                )

                refined_msfu = MSFU(
                    content=candidates[0].content[:500] if candidates else context[:500],
                    metadata=metadata,
                    assertion=assertion,
                    evidence=evidence,
                )
                refined_msfus.append(refined_msfu)

            return refined_msfus if refined_msfus else candidates

        except Exception as e:
            # LLM 调用失败，返回原始候选
            print(f"LLM refinement failed: {e}")
            return candidates

    def extract_batch(
        self,
        chunks: List[Tuple[str, MSFUMetadata]],
        doc_title: str = ""
    ) -> List[MSFU]:
        """批量提取"""
        all_msfus = []

        for chunk, metadata in chunks:
            msfus = self.extract(chunk, metadata, doc_title)
            all_msfus.extend(msfus)

        return all_msfus


# ==================== 数据库集成 ====================

def store_msfus_in_db(
    msfus: List[MSFU],
    db_path: str,
    doc_id: int,
    chunk_id: int
) -> List[int]:
    """
    将MSFU存储到数据库

    包含后处理过滤：
    1. 过滤自引用关系（源实体 = 目标实体）
    2. 过滤无效实体格式
    3. 只保留有效的关系类型

    Returns:
        存储的MSFU ID列表
    """
    if not msfus:
        return []

    import sqlite3
    import re

    # 有效的实体类别
    VALID_CATEGORIES = {"process", "morphology", "performance", "mechanism"}

    # 有效的实体前缀
    VALID_ENTITY_PATTERN = re.compile(r"^(process|morphology|performance|mechanism):[a-z_]+$")

    # 有效的关系类型
    VALID_RELATIONS = {"increases", "decreases", "causes", "affects", "promotes", "inhibits"}

    conn = sqlite3.connect(db_path)
    ids = []

    try:
        cursor = conn.cursor()

        for msfu in msfus:
            row_data = msfu.to_db_row()
            row_data["doc_id"] = doc_id
            row_data["chunk_id"] = chunk_id

            # 后处理过滤
            source = row_data.get("source_entity", "")
            target = row_data.get("target_entity", "")
            relation = row_data.get("relation_type", "")

            # 1. 过滤空实体
            if not source or not target:
                continue

            # 2. 过滤自引用关系
            if source == target:
                continue

            # 3. 过滤无效实体格式
            if not VALID_ENTITY_PATTERN.match(source) or not VALID_ENTITY_PATTERN.match(target):
                continue

            # 4. 过滤无效关系类型
            if relation not in VALID_RELATIONS:
                continue

            # 5. 只保留 process/morphology/mechanism → morphology/performance 的关系
            src_cat = source.split(":")[0] if ":" in source else ""
            tgt_cat = target.split(":")[0] if ":" in target else ""

            # 放宽关系链限制：只要求源和目标都是有效类别即可
            # 不再强制要求特定的关系链组合
            if src_cat not in VALID_CATEGORIES or tgt_cat not in VALID_CATEGORIES:
                continue

            cursor.execute(
                """
                INSERT INTO kb_msfu (
                    chunk_id, doc_id, source_entity, relation_type, target_entity,
                    condition_param, condition_op, condition_value, condition_unit,
                    direction, content, confidence, extraction_method,
                    process_factor, morphology_factor, performance_factor,
                    effect_direction, mechanism_summary, evidence_text,
                    doc_title, page_num
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row_data["chunk_id"],
                    row_data["doc_id"],
                    row_data["source_entity"],
                    row_data["relation_type"],
                    row_data["target_entity"],
                    row_data["condition_param"],
                    row_data["condition_op"],
                    row_data["condition_value"],
                    row_data["condition_unit"],
                    row_data["direction"],
                    row_data["content"],
                    row_data["confidence"],
                    row_data["extraction_method"],
                    row_data["process_factor"],
                    row_data["morphology_factor"],
                    row_data["performance_factor"],
                    row_data["effect_direction"],
                    row_data["mechanism_summary"],
                    row_data["evidence_text"],
                    row_data["doc_title"],
                    row_data["page_num"],
                )
            )
            ids.append(cursor.lastrowid)

        conn.commit()
    finally:
        conn.close()

    return ids


def get_msfu_stats(db_path: str) -> Dict[str, Any]:
    """获取MSFU统计信息"""
    import sqlite3

    conn = sqlite3.connect(db_path)
    stats = {}

    try:
        conn.row_factory = sqlite3.Row

        # 总数
        total = conn.execute("SELECT COUNT(*) FROM kb_msfu").fetchone()[0]
        stats["total_msfus"] = total

        # 按关系类型统计
        rel_stats = conn.execute(
            "SELECT relation_type, COUNT(*) FROM kb_msfu GROUP BY relation_type"
        ).fetchall()
        stats["by_relation_type"] = {r[0]: r[1] for r in rel_stats}

        # 按方向统计
        dir_stats = conn.execute(
            "SELECT direction, COUNT(*) FROM kb_msfu GROUP BY direction"
        ).fetchall()
        stats["by_direction"] = {d[0]: d[1] for d in dir_stats}

        # 按提取方法统计
        method_stats = conn.execute(
            "SELECT extraction_method, COUNT(*) FROM kb_msfu GROUP BY extraction_method"
        ).fetchall()
        stats["by_extraction_method"] = {m[0]: m[1] for m in method_stats}

        # 平均置信度
        avg_conf = conn.execute("SELECT AVG(confidence) FROM kb_msfu").fetchone()[0]
        stats["average_confidence"] = round(avg_conf or 0, 3)

    finally:
        conn.close()

    return stats

