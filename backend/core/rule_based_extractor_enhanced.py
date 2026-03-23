"""
增强的规则提取器 - 解决原有提取器的覆盖率问题
特点：
1. 支持名词化结构
2. 支持复合谓词
3. 支持被动语态
4. 支持否定处理
5. 支持条件绑定
6. 扩展机制实体覆盖
"""

import re
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass

from backend.core.msfu_extractor import (
    MSFU, MSFUMetadata, Assertion, Evidence, Condition,
    RelationType, Direction, ExtractionMethod,
    classify_entity, normalize_entity, split_sentences
)
from backend.core.rule_patterns_enhanced import (
    ENHANCED_RELATION_PATTERNS,
    ENHANCED_ENTITY_PATTERNS,
    ENHANCED_CONDITION_PATTERNS,
)


@dataclass
class ExtractedRelation:
    """提取的关系（中间表示）"""
    source_entity: str
    target_entity: str
    relation_type: str
    direction: str
    condition: Optional[Condition] = None
    confidence: float = 0.5
    is_negation: bool = False
    source: str = "rule"  # 来源：original/enhanced_*


class EnhancedRuleExtractor:
    """增强的规则提取器"""

    def __init__(self, confidence_threshold: float = 0.4):
        """
        Args:
            confidence_threshold: 置信度阈值，低于此值的关系将被过滤
        """
        self.confidence_threshold = confidence_threshold

        # 编译正则模式（性能优化）
        self._compiled_relation_patterns = self._compile_patterns(ENHANCED_RELATION_PATTERNS)
        self._compiled_entity_patterns = self._compile_entity_patterns(ENHANCED_ENTITY_PATTERNS)
        self._compiled_condition_patterns = self._compile_condition_patterns(ENHANCED_CONDITION_PATTERNS)

    def _compile_patterns(self, patterns_list: List[dict]) -> List[Tuple]:
        """编译关系模式"""
        compiled = []
        for pattern_config in patterns_list:
            relation_type = pattern_config["relation_type"]
            direction = pattern_config["direction"]
            source = pattern_config.get("source", "unknown")
            is_negation = pattern_config.get("is_negation", False)
            split_multiple = pattern_config.get("split_multiple", False)

            for pattern in pattern_config["patterns"]:
                try:
                    compiled_pattern = re.compile(pattern, re.IGNORECASE | re.DOTALL)
                    compiled.append((
                        compiled_pattern,
                        relation_type,
                        direction,
                        source,
                        is_negation,
                        split_multiple,
                    ))
                except re.error as e:
                    print(f"模式编译失败: {pattern[:50]}... - {e}")
                    continue

        return compiled

    def _compile_entity_patterns(self, patterns_dict: dict) -> Dict[str, List[Tuple]]:
        """编译实体模式"""
        compiled = {}
        for category, entities in patterns_dict.items():
            compiled[category] = {}
            for entity_type, pattern_list in entities.items():
                compiled_patterns = []
                for pattern in pattern_list:
                    try:
                        compiled_patterns.append((re.compile(pattern, re.IGNORECASE), pattern))
                    except re.error:
                        continue
                compiled[category][entity_type] = compiled_patterns
        return compiled

    def _compile_condition_patterns(self, patterns_dict: dict) -> Dict[str, List[Tuple]]:
        """编译条件模式"""
        compiled = {}
        for cond_type, pattern_list in patterns_dict.items():
            compiled_patterns = []
            for pattern in pattern_list:
                try:
                    compiled_patterns.append((re.compile(pattern, re.IGNORECASE), pattern))
                except re.error:
                    continue
            compiled[cond_type] = compiled_patterns
        return compiled

    def extract(
        self,
        chunk: str,
        metadata: MSFUMetadata,
        doc_title: str = ""
    ) -> List[MSFU]:
        """
        从文本块提取MSFU

        Args:
            chunk: 文本块
            metadata: 元数据
            doc_title: 文档标题

        Returns:
            MSFU列表
        """
        msfus = []

        # 分句
        sentences = split_sentences(chunk)

        for sentence in sentences:
            # 1. 提取关系
            relations = self._extract_relations(sentence)

            # 2. 提取条件
            conditions = self._extract_conditions(sentence)

            # 3. 将关系转换为MSFU
            for relation in relations:
                # 尝试匹配条件
                matched_condition = self._match_condition_to_relation(
                    conditions,
                    relation,
                    sentence
                )

                # 计算置信度
                confidence = self._calculate_confidence(
                    relation,
                    sentence,
                    matched_condition is not None
                )

                if confidence < self.confidence_threshold:
                    continue

                # 创建MSFU
                msfu = self._create_msfu(
                    sentence,
                    metadata,
                    doc_title,
                    relation,
                    matched_condition,
                    confidence
                )

                if msfu:
                    msfus.append(msfu)

        return msfus

    def _extract_relations(self, sentence: str) -> List[ExtractedRelation]:
        """从句子中提取关系"""
        relations = []

        for (pattern, rel_type, direction, source, is_negation, split_multiple) in self._compiled_relation_patterns:
            try:
                match = pattern.search(sentence)
                if not match:
                    continue

                groups = match.groupdict()

                # 提取源和目标实体
                src_text = groups.get("src", "").strip()
                tgt_text = groups.get("tgt", "").strip()

                # 处理复合谓词（需要拆分）
                if split_multiple:
                    tgt2_text = groups.get("tgt2", "").strip()
                    if tgt2_text:
                        # 拆分为两个关系
                        relations.extend([
                            self._create_relation(src_text, tgt_text, rel_type, direction, source),
                            self._create_relation(src_text, tgt2_text, rel_type, direction, source),
                        ])
                        continue

                if not src_text or not tgt_text:
                    continue

                # 创建关系
                relation = self._create_relation(src_text, tgt_text, rel_type, direction, source)
                relation.is_negation = is_negation

                relations.append(relation)

            except Exception as e:
                continue

        # 去重
        unique_relations = []
        seen = set()
        for rel in relations:
            key = (rel.source_entity, rel.target_entity, rel.relation_type)
            if key not in seen:
                seen.add(key)
                unique_relations.append(rel)

        return unique_relations

    def _create_relation(
        self,
        src_text: str,
        tgt_text: str,
        rel_type: str,
        direction: str,
        source: str
    ) -> ExtractedRelation:
        """创建关系对象（包含实体规范化）"""
        # 规范化实体
        source_entity = self._normalize_entity(src_text)
        target_entity = self._normalize_entity(tgt_text)

        return ExtractedRelation(
            source_entity=source_entity,
            target_entity=target_entity,
            relation_type=rel_type,
            direction=direction,
            source=source
        )

    def _normalize_entity(self, text: str) -> str:
        """规范化实体文本"""
        text = text.strip()

        # 1. 尝试分类
        classification = classify_entity(text)
        if classification:
            category, entity_type = classification
            return f"{category}:{entity_type}"

        # 2. 使用增强的实体模式匹配
        for category, entities in self._compiled_entity_patterns.items():
            for entity_type, compiled_patterns in entities.items():
                for (pattern, original) in compiled_patterns:
                    if pattern.fullmatch(text):
                        return f"{category}:{entity_type}"

        # 3. 返回原文（小写）
        return text.lower()

    def _extract_conditions(self, sentence: str) -> List[Condition]:
        """从句子中提取条件"""
        conditions = []

        for cond_type, compiled_patterns in self._compiled_condition_patterns.items():
            for (pattern, original) in compiled_patterns:
                matches = pattern.finditer(sentence)
                for match in matches:
                    condition = self._parse_condition_from_match(cond_type, match, sentence)
                    if condition:
                        conditions.append(condition)

        return conditions

    def _parse_condition_from_match(
        self,
        cond_type: str,
        match: re.Match,
        text: str
    ) -> Optional[Condition]:
        """根据匹配类型解析条件"""
        groups = match.groupdict()

        if "above" in cond_type:
            param = self._get_param_from_type(cond_type)
            value = float(groups.get("value", 0))
            unit = self._get_unit_from_type(cond_type, text, match)
            return Condition(parameter=param, operator=">", value=value, unit=unit)

        elif "below" in cond_type:
            param = self._get_param_from_type(cond_type)
            value = float(groups.get("value", 0))
            unit = self._get_unit_from_type(cond_type, text, match)
            return Condition(parameter=param, operator="<", value=value, unit=unit)

        elif "range" in cond_type and "min" in groups and "max" in groups:
            param = self._get_param_from_type(cond_type)
            min_val = float(groups["min"])
            max_val = float(groups["max"])
            unit = self._get_unit_from_type(cond_type, text, match)
            return Condition(parameter=param, operator="in_range", value=(min_val, max_val), unit=unit)

        return None

    def _get_param_from_type(self, cond_type: str) -> str:
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

    def _get_unit_from_type(self, cond_type: str, text: str, match: re.Match) -> Optional[str]:
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

    def _match_condition_to_relation(
        self,
        conditions: List[Condition],
        relation: ExtractedRelation,
        sentence: str
    ) -> Optional[Condition]:
        """将条件匹配到关系"""
        if not conditions:
            return None

        # 简单策略：使用第一个匹配的条件
        for cond in conditions:
            # 检查条件参数是否与源实体相关
            if self._condition_matches_entity(cond, relation.source_entity, sentence):
                return cond

        return None

    def _condition_matches_entity(self, condition: Condition, entity: str, sentence: str) -> bool:
        """检查条件是否匹配实体"""
        param_keywords = {
            "temperature": ["temperature", "temp", "温度"],
            "time": ["time", "duration", "时间"],
            "flow": ["flow", "流量"],
            "thickness": ["thickness", "厚度"],
            "pressure": ["pressure", "压力"],
        }

        keywords = param_keywords.get(condition.parameter, [])
        entity_text = entity.split(":")[1] if ":" in entity else entity

        return any(kw in entity_text or kw in sentence[:100] for kw in keywords)

    def _calculate_confidence(
        self,
        relation: ExtractedRelation,
        sentence: str,
        has_condition: bool
    ) -> float:
        """计算提取置信度"""
        base_score = 0.5

        # 来源权重（增强模式权重更高）
        source_weights = {
            "original": 0.0,
            "enhanced_nominalization": 0.15,
            "enhanced_composite": 0.10,
            "enhanced_passive": 0.10,
            "enhanced_comparison": 0.05,
            "enhanced_negation": 0.10,
            "enhanced_conditional": 0.15,
        }
        base_score += source_weights.get(relation.source, 0.0)

        # 包含数值（提高置信度）
        if re.search(r'\d+\.?\d*', sentence):
            base_score += 0.1

        # 包含单位（提高置信度）
        if re.search(r'(°C|℃|nm|min|h|sccm|Pa)', sentence):
            base_score += 0.1

        # 有条件（提高置信度）
        if has_condition:
            base_score += 0.15

        # 实体格式正确（提高置信度）
        if ":" in relation.source_entity and ":" in relation.target_entity:
            base_score += 0.1

        # 否定关系（降低置信度）
        if relation.is_negation:
            base_score -= 0.1

        # 限制范围
        return max(0.0, min(1.0, base_score))

    def _create_msfu(
        self,
        sentence: str,
        metadata: MSFUMetadata,
        doc_title: str,
        relation: ExtractedRelation,
        condition: Optional[Condition],
        confidence: float
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
            condition=condition,
            direction=relation.direction
        )

        # 创建证据
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

    def _is_valid_entity(self, entity: str) -> bool:
        """检查实体是否有效"""
        # 允许的分类格式
        valid_categories = {"process", "morphology", "performance", "mechanism"}

        # 检查分类格式
        if ":" in entity:
            category = entity.split(":")[0]
            return category in valid_categories

        # 允许原始文本（长度限制）
        if 3 <= len(entity) <= 100:
            return not entity.isdigit()

        return False


# ==================== 测试和对比工具 ====================

def compare_extractors(original_extractor, enhanced_extractor, test_sentences: List[str]):
    """
    对比原有提取器和增强提取器的效果

    Args:
        original_extractor: 原有提取器
        enhanced_extractor: 增强提取器
        test_sentences: 测试句子列表

    Returns:
        dict: 对比结果
    """
    results = {
        "test_sentences": test_sentences,
        "original_results": [],
        "enhanced_results": [],
        "statistics": {}
    }

    metadata = MSFUMetadata(
        doc_id="test",
        chunk_id="0",
        doc_title="Test Document"
    )

    for i, sentence in enumerate(test_sentences):
        # 原有提取器
        original_msfus = original_extractor.extract(sentence, metadata)
        results["original_results"].append({
            "sentence": sentence,
            "msfu_count": len(original_msfus),
            "msfus": [m.to_dict() for m in original_msfus]
        })

        # 增强提取器
        enhanced_msfus = enhanced_extractor.extract(sentence, metadata)
        results["enhanced_results"].append({
            "sentence": sentence,
            "msfu_count": len(enhanced_msfus),
            "msfus": [m.to_dict() for m in enhanced_msfus]
        })

    # 统计
    original_total = sum(r["msfu_count"] for r in results["original_results"])
    enhanced_total = sum(r["msfu_count"] for r in results["enhanced_results"])

    results["statistics"] = {
        "original_total": original_total,
        "enhanced_total": enhanced_total,
        "improvement": enhanced_total - original_total,
        "improvement_rate": (enhanced_total - original_total) / max(original_total, 1) * 100
    }

    return results


if __name__ == "__main__":
    # 测试代码
    test_sentences = [
        "The increase of growth temperature from 700°C to 800°C enhances alignment and reduces diameter.",
        "Higher temperature leads to improved density.",
        "Alignment is improved by higher temperature.",
        "Temperature does not significantly affect density.",
        "When temperature > 750°C, conductivity improves.",
    ]

    # 创建提取器
    from backend.core.msfu_extractor import RuleBasedExtractor as OriginalExtractor
    original = OriginalExtractor(confidence_threshold=0.4)
    enhanced = EnhancedRuleExtractor(confidence_threshold=0.4)

    # 对比
    results = compare_extractors(original, enhanced, test_sentences)

    # 打印结果
    print("="*60)
    print("增强提取器对比测试")
    print("="*60)

    for i in range(len(test_sentences)):
        print(f"\n句子 {i+1}: {test_sentences[i][:60]}...")
        print(f"  原有提取器: {results['original_results'][i]['msfu_count']} 个MSFU")
        print(f"  增强提取器: {results['enhanced_results'][i]['msfu_count']} 个MSFU")

    print("\n" + "="*60)
    print("统计:")
    print(f"  原有提取器总计: {results['statistics']['original_total']} 个MSFU")
    print(f"  增强提取器总计: {results['statistics']['enhanced_total']} 个MSFU")
    print(f"  改进数量: +{results['statistics']['improvement']} 个MSFU")
    print(f"  改进率: {results['statistics']['improvement_rate']:.1f}%")
    print("="*60)
