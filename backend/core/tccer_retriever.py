"""
TCCER (Task-Constrained Chain Evidence Retrieval) for CNTA RAG System.

Implements constrained chain-based evidence retrieval:
- Query parsing with entity and condition extraction
- Hybrid recall (SQL + BM25 + Semantic)
- Constrained expansion (BFS with pruning)
- Path scoring with multiple dimensions
"""

import re
import json
import sqlite3
from dataclasses import dataclass, field
from typing import Optional, List, Set, Dict, Any, Tuple
from enum import Enum

from .msfu_extractor import (
    MSFU, Condition, Assertion, MSFUMetadata,
    RelationType, Direction, classify_entity, normalize_entity,
    extract_conditions, split_sentences
)


class ChainType(Enum):
    """证据链类型"""
    PRIMARY = "primary"      # 主链（核心证据）
    SUPPORTING = "supporting"  # 辅助链（补充证据）
    CONFLICTING = "conflicting"  # 冲突链（矛盾证据）


@dataclass
class QueryConstraint:
    """查询约束"""
    entities: Set[str] = field(default_factory=set)
    conditions: List[Condition] = field(default_factory=list)
    direction: Optional[str] = None
    relation_types: Set[str] = field(default_factory=set)
    max_hops: int = 3
    min_confidence: float = 0.4

    def matches(self, msfu: MSFU) -> bool:
        """检查MSFU是否满足约束"""
        # 检查方向约束
        if self.direction and msfu.assertion.direction != self.direction:
            return False

        # 检查关系类型约束
        if self.relation_types and msfu.assertion.relation_type not in self.relation_types:
            return False

        # 检查置信度
        if msfu.evidence.confidence < self.min_confidence:
            return False

        # 检查实体约束（至少匹配一个）
        if self.entities:
            src_ok = any(e in msfu.assertion.source_entity.lower() for e in self.entities)
            tgt_ok = any(e in msfu.assertion.target_entity.lower() for e in self.entities)
            if not (src_ok or tgt_ok):
                return False

        # 检查条件约束
        if self.conditions:
            cond = msfu.assertion.condition
            if not cond:
                return False
            for q_cond in self.conditions:
                if cond.parameter == q_cond.parameter:
                    # 简单检查：参数匹配
                    continue

        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entities": list(self.entities),
            "conditions": [c.to_dict() for c in self.conditions],
            "direction": self.direction,
            "relation_types": list(self.relation_types),
            "max_hops": self.max_hops,
            "min_confidence": self.min_confidence,
        }


@dataclass
class MSFUPath:
    """MSFU路径"""
    msfus: List[MSFU] = field(default_factory=list)
    total_score: float = 0.0
    chain_type: str = ChainType.SUPPORTING.value
    reasoning: str = ""

    @property
    def length(self) -> int:
        return len(self.msfus)

    @property
    def entities(self) -> Set[str]:
        entities = set()
        for msfu in self.msfus:
            entities.add(msfu.assertion.source_entity)
            entities.add(msfu.assertion.target_entity)
        return entities

    def to_dict(self) -> Dict[str, Any]:
        return {
            "msfus": [m.to_dict() for m in self.msfus],
            "total_score": round(self.total_score, 4),
            "length": self.length,
            "chain_type": self.chain_type,
            "reasoning": self.reasoning,
            "entities": list(self.entities),
        }


@dataclass
class ScoredPath:
    """带分数的路径"""
    path: MSFUPath
    scores: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path.to_dict(),
            "scores": {k: round(v, 4) for k, v in self.scores.items()},
        }


@dataclass
class EvidenceChainResult:
    """证据链检索结果"""
    query: str
    constraint: QueryConstraint
    chains: List[ScoredPath]
    summary: str
    msfu_count: int = 0
    execution_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "constraint": self.constraint.to_dict(),
            "chains": [c.to_dict() for c in self.chains],
            "summary": self.summary,
            "msfu_count": self.msfu_count,
            "execution_time_ms": round(self.execution_time_ms, 2),
        }


class TCCERRetriever:
    """TCCER检索器"""

    def __init__(self, kb_db_path: str, task_name: Optional[str] = None):
        self.kb_db_path = kb_db_path
        self.task_name = task_name
        self._conn: Optional[sqlite3.Connection] = None

    def _connect(self) -> sqlite3.Connection:
        """获取数据库连接"""
        if self._conn is None:
            self._conn = sqlite3.connect(self.kb_db_path)
            self._conn.execute("PRAGMA journal_mode=MEMORY")
            self._conn.execute("PRAGMA temp_store=MEMORY")
        return self._conn

    def close(self):
        """关闭连接"""
        if self._conn:
            self._conn.close()
            self._conn = None

    # ==================== 查询解析 ====================

    def parse_query(self, query: str, task_name: Optional[str] = None) -> QueryConstraint:
        """
        解析自然语言查询为约束对象

        示例查询:
        - "温度高于750°C时，对CNT密度有什么影响？"
        - "Fe厚度如何影响取向度？"
        - "什么因素影响电导率？"
        """
        constraint = QueryConstraint()

        # 1. 提取实体
        entities = self._extract_entities(query)
        constraint.entities.update(entities)

        # 2. 提取条件
        conditions = extract_conditions(query)
        constraint.conditions.extend(conditions)

        # 3. 推断方向
        direction = self._infer_direction(query)
        if direction:
            constraint.direction = direction

        # 4. 选择关系类型
        relation_types = self._select_relation_types(query, task_name)
        constraint.relation_types.update(relation_types)

        return constraint

    def _extract_entities(self, text: str) -> Set[str]:
        """从文本中提取实体"""
        entities = set()

        # 匹配因子模式
        factor_patterns = {
            "growth_temp": r"(?:生长|growth)\s*(?:温度|temperature)",
            "growth_time": r"(?:生长|growth)\s*(?:时间|time)",
            "anneal_temp": r"(?:退火|anneal)\s*(?:温度|temperature)",
            "ar_flow": r"(?:Ar|氩)\s*(?:流量|flow)",
            "h2_flow": r"(?:H2|氢)\s*(?:流量|flow)",
            "fe_thickness": r"(?:Fe|铁)\s*(?:厚度|thickness)",
            "al2o3_thickness": r"(?:Al2O3|氧化铝)\s*(?:厚度|thickness)",
            "density": r"(?:密度|density)",
            "alignment": r"(?:取向|alignment)",
            "diameter": r"(?:直径|diameter)",
            "conductivity": r"(?:电导|conductiv)",
            "resistivity": r"(?:电阻率|resistiv)",
        }

        for entity_type, pattern in factor_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                # 确定实体类别
                if entity_type in ["growth_temp", "growth_time", "anneal_temp", "ar_flow", "h2_flow", "fe_thickness", "al2o3_thickness"]:
                    entities.add(f"process:{entity_type}")
                elif entity_type in ["density", "alignment", "diameter"]:
                    entities.add(f"morphology:{entity_type}")
                elif entity_type in ["conductivity", "resistivity"]:
                    entities.add(f"performance:{entity_type}")

        return entities

    def _infer_direction(self, text: str) -> Optional[str]:
        """推断影响方向"""
        positive_indicators = r"(?:增加|提高|改善|enhance|improve|increase)"
        negative_indicators = r"(?:降低|减少|恶化|decrease|reduce|worsen)"

        if re.search(positive_indicators, text, re.IGNORECASE):
            return Direction.POSITIVE.value
        elif re.search(negative_indicators, text, re.IGNORECASE):
            return Direction.NEGATIVE.value
        return None

    def _select_relation_types(self, text: str, task_name: Optional[str] = None) -> Set[str]:
        """选择关系类型"""
        relation_types = set()

        # 默认关系类型
        if "影响" in text or "affect" in text.lower():
            relation_types.add(RelationType.AFFECTS.value)
        if "导致" in text or "cause" in text.lower():
            relation_types.add(RelationType.CAUSES.value)
        if "促进" in text or "promote" in text.lower():
            relation_types.add(RelationType.PROMOTES.value)
        if "抑制" in text or "inhibit" in text.lower():
            relation_types.add(RelationType.INHIBITS.value)

        # 任务特定偏好
        if task_name == "process_analysis":
            relation_types.update([
                RelationType.CAUSES.value,
                RelationType.AFFECTS.value,
            ])
        elif task_name == "morphology_interpretation":
            relation_types.update([
                RelationType.INCREASES.value,
                RelationType.DECREASES.value,
            ])

        return relation_types

    # ==================== 混合召回 ====================

    def hybrid_recall(self, constraint: QueryConstraint) -> List[MSFU]:
        """
        混合召回：SQL精确查询 + BM25关键词召回

        返回满足约束的MSFU列表
        """
        all_msfus = []

        # 1. SQL精确查询
        sql_msfus = self._sql_recall(constraint)
        all_msfus.extend(sql_msfus)

        # 2. BM25召回（通过关键词）
        bm25_msfus = self._bm25_recall(constraint)
        for msfu in bm25_msfus:
            # 去重
            if msfu.msfu_id not in [m.msfu_id for m in all_msfus if m.msfu_id]:
                all_msfus.append(msfu)

        return all_msfus

    def _sql_recall(self, constraint: QueryConstraint) -> List[MSFU]:
        """SQL精确查询"""
        conn = self._connect()
        conn.row_factory = sqlite3.Row

        msfus = []

        try:
            # 构建查询条件
            conditions = []
            params = []

            # 实体匹配
            if constraint.entities:
                entity_list = list(constraint.entities)
                placeholders = ",".join("?" * len(entity_list))
                conditions.append(f"(source_entity IN ({placeholders}) OR target_entity IN ({placeholders}))")
                params.extend(entity_list)
                params.extend(entity_list)

            # 方向匹配
            if constraint.direction:
                conditions.append("direction = ?")
                params.append(constraint.direction)

            # 关系类型匹配
            if constraint.relation_types:
                rel_list = list(constraint.relation_types)
                placeholders = ",".join("?" * len(rel_list))
                conditions.append(f"relation_type IN ({placeholders})")
                params.extend(rel_list)

            # 置信度匹配
            if constraint.min_confidence > 0:
                conditions.append("confidence >= ?")
                params.append(constraint.min_confidence)

            # 条件参数匹配
            if constraint.conditions:
                cond_params = [c.parameter for c in constraint.conditions]
                placeholders = ",".join("?" * len(cond_params))
                conditions.append(f"condition_param IN ({placeholders})")
                params.extend(cond_params)

            # 执行查询
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            query = f"""
                SELECT m.*, d.title as doc_title_full, c.text as chunk_text
                FROM kb_msfu m
                LEFT JOIN kb_documents d ON m.doc_id = d.id
                LEFT JOIN kb_chunks c ON m.chunk_id = c.id
                WHERE {where_clause}
                ORDER BY m.confidence DESC
                LIMIT 100
            """

            rows = conn.execute(query, params).fetchall()

            for row in rows:
                row_dict = dict(row)
                # 补充完整标题
                if "doc_title_full" in row_dict and row_dict["doc_title_full"]:
                    row_dict["doc_title"] = row_dict["doc_title_full"]
                msfu = MSFU.from_db_row(row_dict)
                msfus.append(msfu)

        finally:
            pass  # 不关闭连接，复用

        return msfus

    def _bm25_recall(self, constraint: QueryConstraint) -> List[MSFU]:
        """BM25关键词召回"""
        if not constraint.entities:
            return []

        conn = self._connect()
        conn.row_factory = sqlite3.Row

        msfus = []
        keywords = []

        # 从实体中提取关键词
        for entity in constraint.entities:
            parts = entity.split(":")
            if len(parts) > 1:
                keywords.append(parts[1])

        if not keywords:
            return []

        try:
            # 构建LIKE条件
            like_conditions = []
            params = []

            for keyword in keywords:
                like_conditions.append("content LIKE ?")
                params.append(f"%{keyword}%")

            where_clause = " OR ".join(like_conditions)

            query = f"""
                SELECT m.*, d.title as doc_title_full, c.text as chunk_text
                FROM kb_msfu m
                LEFT JOIN kb_documents d ON m.doc_id = d.id
                LEFT JOIN kb_chunks c ON m.chunk_id = c.id
                WHERE {where_clause}
                AND confidence >= ?
                ORDER BY confidence DESC
                LIMIT 50
            """
            params.append(constraint.min_confidence)

            rows = conn.execute(query, params).fetchall()

            for row in rows:
                row_dict = dict(row)
                if "doc_title_full" in row_dict and row_dict["doc_title_full"]:
                    row_dict["doc_title"] = row_dict["doc_title_full"]
                msfu = MSFU.from_db_row(row_dict)

        finally:
            pass

        return msfus

    # ==================== 约束扩展 ====================

    def constrained_expansion(
        self,
        seed_msfus: List[MSFU],
        constraint: QueryConstraint
    ) -> List[MSFUPath]:
        """
        约束扩展：BFS扩展路径，基于任务约束剪枝

        Args:
            seed_msfus: 种子MSFU列表
            constraint: 查询约束

        Returns:
            扩展后的路径列表
        """
        if not seed_msfus:
            return []

        paths = []
        visited = set()

        # 为每个种子MSFU构建路径
        for seed in seed_msfus:
            path = MSFUPath(msfus=[seed])
            paths.append(path)
            visited.add(seed.msfu_id or -1)

        # BFS扩展
        for hop in range(1, constraint.max_hops + 1):
            new_paths = []

            for path in paths:
                last_msfu = path.msfus[-1]

                # 找到可以连接的MSFU
                next_msfus = self._find_connected_msfus(
                    last_msfu,
                    constraint,
                    visited
                )

                for next_msfu in next_msfus:
                    # 创建新路径
                    new_path = MSFUPath(
                        msfus=path.msfus + [next_msfu]
                    )
                    new_paths.append(new_path)
                    visited.add(next_msfu.msfu_id or -1)

            if new_paths:
                paths.extend(new_paths)

        return paths

    def _find_connected_msfus(
        self,
        msfu: MSFU,
        constraint: QueryConstraint,
        visited: Set[int]
    ) -> List[MSFU]:
        """找到与给定MSFU相连的其他MSFU"""
        conn = self._connect()
        conn.row_factory = sqlite3.Row

        results = []

        try:
            # 查找目标实体匹配源实体的MSFU
            query = """
                SELECT m.*, d.title as doc_title_full
                FROM kb_msfu m
                LEFT JOIN kb_documents d ON m.doc_id = d.id
                WHERE m.source_entity = ?
                  AND m.confidence >= ?
                  AND m.id != ?
                ORDER BY m.confidence DESC
                LIMIT 20
            """

            rows = conn.execute(query, (
                msfu.assertion.target_entity,
                constraint.min_confidence,
                msfu.msfu_id or -1
            )).fetchall()

            for row in rows:
                row_dict = dict(row)
                msfu_id = row_dict.get("id")
                if msfu_id not in visited:
                    if "doc_title_full" in row_dict and row_dict["doc_title_full"]:
                        row_dict["doc_title"] = row_dict["doc_title_full"]
                    result = MSFU.from_db_row(row_dict)
                    if constraint.matches(result):
                        results.append(result)

        finally:
            pass

        return results

    # ==================== 路径评分 ====================

    def score_paths(
        self,
        paths: List[MSFUPath],
        constraint: QueryConstraint
    ) -> List[ScoredPath]:
        """
        路径评分：多维度评分

        评分维度：
        - 约束满足度 (30%)
        - 证据强度 (30%)
        - 语义连贯性 (20%)
        - 实体覆盖率 (20%)
        """
        scored_paths = []

        for path in paths:
            scores = {}

            # 1. 约束满足度
            scores["constraint_satisfaction"] = self._score_constraint_satisfaction(path, constraint)

            # 2. 证据强度
            scores["evidence_strength"] = self._score_evidence_strength(path)

            # 3. 语义连贯性
            scores["semantic_coherence"] = self._score_semantic_coherence(path)

            # 4. 实体覆盖率
            scores["entity_coverage"] = self._score_entity_coverage(path, constraint)

            # 加权总分
            total_score = (
                scores["constraint_satisfaction"] * 0.30 +
                scores["evidence_strength"] * 0.30 +
                scores["semantic_coherence"] * 0.20 +
                scores["entity_coverage"] * 0.20
            )

            path.total_score = total_score
            scored_paths.append(ScoredPath(path=path, scores=scores))

        # 按总分排序
        scored_paths.sort(key=lambda x: x.path.total_score, reverse=True)

        return scored_paths

    def _score_constraint_satisfaction(self, path: MSFUPath, constraint: QueryConstraint) -> float:
        """约束满足度评分"""
        if not path.msfus:
            return 0.0

        satisfied = 0
        for msfu in path.msfus:
            if constraint.matches(msfu):
                satisfied += 1

        return satisfied / len(path.msfus)

    def _score_evidence_strength(self, path: MSFUPath) -> float:
        """证据强度评分（基于置信度）"""
        if not path.msfus:
            return 0.0

        confidences = [m.evidence.confidence for m in path.msfus]
        return sum(confidences) / len(confidences)

    def _score_semantic_coherence(self, path: MSFUPath) -> float:
        """语义连贯性评分"""
        if len(path.msfus) < 2:
            return 1.0

        # 检查相邻MSFU是否连接连贯
        coherent_count = 0
        for i in range(len(path.msfus) - 1):
            current = path.msfus[i]
            next_msfu = path.msfus[i + 1]

            # 检查目标实体是否等于下一个源实体
            if current.assertion.target_entity == next_msfu.assertion.source_entity:
                coherent_count += 1

        return coherent_count / (len(path.msfus) - 1)

    def _score_entity_coverage(self, path: MSFUPath, constraint: QueryConstraint) -> float:
        """实体覆盖率评分"""
        if not constraint.entities:
            return 1.0

        path_entities = set()
        for msfu in path.msfus:
            path_entities.add(msfu.assertion.source_entity)
            path_entities.add(msfu.assertion.target_entity)

        covered = 0
        for entity in constraint.entities:
            for path_entity in path_entities:
                if entity in path_entity or path_entity in entity:
                    covered += 1
                    break

        return covered / len(constraint.entities)

    # ==================== 链分类 ====================

    def classify_chains(
        self,
        scored_paths: List[ScoredPath]
    ) -> Tuple[List[ScoredPath], List[ScoredPath], List[ScoredPath]]:
        """
        分类路径为主链、辅助链、冲突链

        Returns:
            (primary, supporting, conflicting)
        """
        primary = []
        supporting = []
        conflicting = []

        if not scored_paths:
            return primary, supporting, conflicting

        # 找出最高分作为基准
        max_score = max(p.path.total_score for p in scored_paths)
        threshold = max_score * 0.7

        # 检查方向一致性
        directions = {}
        for sp in scored_paths:
            if sp.path.msfus:
                direction = sp.path.msfus[0].assertion.direction
                if direction not in directions:
                    directions[direction] = []
                directions[direction].append(sp)

        most_common_direction = max(directions.keys(), key=lambda k: len(directions[k])) if directions else None

        for sp in scored_paths:
            path = sp.path
            score = path.total_score

            # 检查方向是否一致
            if path.msfus:
                path_direction = path.msfus[0].assertion.direction
                if path_direction != most_common_direction:
                    path.chain_type = ChainType.CONFLICTING.value
                    conflicting.append(sp)
                    continue

            # 分类为主链或辅助链
            if score >= threshold and len(path.msfus) >= 2:
                path.chain_type = ChainType.PRIMARY.value
                primary.append(sp)
            else:
                path.chain_type = ChainType.SUPPORTING.value
                supporting.append(sp)

        return primary, supporting, conflicting

    # ==================== 主检索接口 ====================

    def retrieve(self, query: str, task_name: Optional[str] = None) -> EvidenceChainResult:
        """
        主检索接口

        Args:
            query: 自然语言查询
            task_name: 任务名称

        Returns:
            EvidenceChainResult
        """
        import time
        start_time = time.time()

        # 1. 解析查询
        constraint = self.parse_query(query, task_name)

        # 2. 混合召回
        msfus = self.hybrid_recall(constraint)
        msfu_count = len(msfus)

        # 3. 约束扩展
        paths = self.constrained_expansion(msfus[:20], constraint)

        # 4. 路径评分
        scored_paths = self.score_paths(paths, constraint)

        # 5. 链分类
        primary, supporting, conflicting = self.classify_chains(scored_paths)

        # 合并结果
        all_chains = primary[:3] + supporting[:5] + conflicting[:3]

        # 6. 生成摘要
        summary = self._generate_summary(query, constraint, all_chains)

        execution_time = (time.time() - start_time) * 1000

        return EvidenceChainResult(
            query=query,
            constraint=constraint,
            chains=all_chains,
            summary=summary,
            msfu_count=msfu_count,
            execution_time_ms=execution_time,
        )

    def _generate_summary(
        self,
        query: str,
        constraint: QueryConstraint,
        chains: List[ScoredPath]
    ) -> str:
        """生成结果摘要"""
        parts = []

        parts.append(f"查询: {query}")

        if constraint.entities:
            parts.append(f"识别实体: {', '.join(constraint.entities)}")

        if constraint.conditions:
            cond_desc = ", ".join(
                f"{c.parameter}{c.operator}{c.value}{c.unit or ''}"
                for c in constraint.conditions
            )
            parts.append(f"条件约束: {cond_desc}")

        primary_count = sum(1 for c in chains if c.path.chain_type == ChainType.PRIMARY.value)
        supporting_count = sum(1 for c in chains if c.path.chain_type == ChainType.SUPPORTING.value)
        conflicting_count = sum(1 for c in chains if c.path.chain_type == ChainType.CONFLICTING.value)

        parts.append(f"检索到 {len(chains)} 条证据链:")
        parts.append(f"  - 主链: {primary_count}")
        parts.append(f"  - 辅助链: {supporting_count}")
        parts.append(f"  - 冲突链: {conflicting_count}")

        if chains:
            best = chains[0]
            parts.append(f"最佳证据链评分: {round(best.path.total_score, 2)}")

        return "\n".join(parts)
