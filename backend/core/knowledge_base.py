import os
import re
import sqlite3
import csv
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

import numpy as np

PROCESS_FACTOR_PATTERNS = {
    "growth_temp": [r"\bgrowth.{0,5}temperature\b", r"\btemperature\b", r"温度", r"生长温度", r"\d+\s*°[Cc]"],
    "growth_time": [r"\bgrowth.{0,5}time\b", r"\bgrowth.{0,5}duration\b", r"\btime\b", r"生长时间"],
    "anneal_time": [r"\banneal", r"退火"],
    "ar_flow": [r"\bar\b", r"氩", r"氩气"],
    "h2_flow": [r"\bh2\b", r"氢", r"氢气"],
    "c2h4_flow": [r"\bc2h4\b", r"乙烯", r"\bethylene\b"],
    "fe_thickness": [r"\bfe\b", r"铁", r"催化剂厚度", r"\bcatalyst\b", r"\biron\b"],
    "al2o3_thickness": [r"al2o3", r"氧化铝", r"支撑层"],
}

MORPHOLOGY_FACTOR_PATTERNS = {
    "alignment": [r"\balign", r"取向", r"对齐", r"oriented"],
    "density": [r"\bdens", r"密度", r"覆盖率"],
    "diameter": [r"\bdiamet", r"管径", r"直径"],
    "curvature": [r"\bcurvat", r"弯曲", r"波曲", r"\bwav"],
    "tortuosity": [r"\btortuos", r"曲折度"],
    "height": [r"\bheight\b", r"高度", r"mm-scale"],
}

PERFORMANCE_FACTOR_PATTERNS = {
    "conductivity": [
        r"\bconductiv",
        r"\belectrical conductivity\b",
        r"\bspecific conductivity\b",
        r"\bconductance\b",
        r"电导",
        r"导电",
    ],
    "resistivity": [r"\bresistiv", r"\belectrical resist", r"电阻率"],
    "sheet_resistance": [r"\bsheet resistance\b", r"方阻"],
    "tensile_strength": [
        r"\btensile\b",
        r"\bmechanical strength\b",
        r"\bultimate strength\b",
        r"\bstrength\b",
        r"抗拉",
        r"强度",
    ],
    "modulus": [r"\bmodulus\b", r"\byoung'?s modulus\b", r"\belastic modulus\b", r"\bstiffness\b", r"模量"],
}

INVERSE_PERFORMANCE_FACTORS = {
    "resistivity": "conductivity",
    "sheet_resistance": "conductivity",
}

INCREASE_PATTERNS = [
    r"increase", r"improve", r"enhance", r"rise", r"提高", r"增大", r"增加", r"改善",
    r"promot", r"stimulat", r"boost", r"facilitate", r"favor", r"促进",
    r"higher", r"greater", r"longer", r"larger", r"better",
    r"upscal", r"elongat",
]
DECREASE_PATTERNS = [
    r"decrease", r"reduce", r"drop", r"decline", r"降低", r"减小", r"下降", r"恶化",
    r"inhibit", r"suppress", r"degrade", r"deterior", r"suppress", r"抑制",
    r"lower", r"shorter", r"smaller", r"weaker",
    r"deactiv", r"poison", r"失活", r"中毒",
]
MECHANISM_FACTOR_PATTERNS = {
    "diffusion": [r"\bdiffusion\b", r"扩散"],
    "catalyst_deactivation": [r"\bdeactivation\b", r"\bpoison", r"失活", r"中毒"],
    "catalyst_agglomeration": [r"\bripening\b", r"\bagglomer", r"\bsinter", r"烧结", r"团聚"],
    "growth_kinetics": [r"\bkinetic", r"\bactivation energy\b", r"动力学"],
    "boundary_layer_effect": [r"\bboundary layer\b", r"边界层"],
}

MECHANISM_PATTERNS = [r"mechanism", r"机理"] + [
    pattern
    for patterns in MECHANISM_FACTOR_PATTERNS.values()
    for pattern in patterns
]


DEFAULT_TASK_PROFILES = (
    (
        "morphology_interpretation",
        "morphology_interpretation",
        "characterization,process_morphology,growth_mechanism",
        "applications",
        "mechanism morphology_interpretation process_morphology",
        "alignment density diameter curvature mechanism catalyst growth",
    ),
    (
        "process_analysis",
        "process_morphology",
        "process_morphology,growth_mechanism,characterization",
        "applications",
        "process_rule process_morphology growth_mechanism",
        "temperature flow catalyst growth morphology alignment diameter density",
    ),
    (
        "prediction_explanation",
        "prediction_support",
        "ml_growth,process_morphology,growth_mechanism",
        "applications",
        "process_rule prediction_support growth_mechanism",
        "prediction process morphology mechanism evidence",
    ),
)


class KnowledgeBaseService:
    # TCCER 关系转换矩阵 - 控制允许的路径扩展
    RELATION_TRANSITION_MATRIX = {
        "process_to_morphology": ["morphology_to_performance", "mechanism_to_morphology"],
        "process_to_mechanism": ["mechanism_to_morphology", "mechanism_evidence"],
        "mechanism_to_morphology": ["morphology_to_performance", "process_to_morphology"],
        "morphology_to_performance": [],
        "process_to_performance": [],
        "mechanism_evidence": [],
    }

    # 任务类型配置
    TASK_TYPES = {
        "morphology_interpretation": {
            "preferred_relations": ["process_to_morphology", "mechanism_to_morphology"],
            "direction_bias": 0.2,
        },
        "process_analysis": {
            "preferred_relations": ["process_to_morphology", "process_to_mechanism"],
            "direction_bias": 0.1,
        },
        "prediction_explanation": {
            "preferred_relations": ["morphology_to_performance", "process_to_performance"],
            "direction_bias": 0.15,
        },
    }

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.embedding_model_name = os.getenv(
            "CNTA_KB_EMBED_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2",
        )
        self._embedding_model = None
        self._embedding_disabled_reason: Optional[str] = None
        self._semantic_cache: Optional[Dict[str, object]] = None
        self.init_schema()

    def _invalidate_semantic_cache(self):
        self._semantic_cache = None

    def _prepare_db_path(self):
        if os.path.exists(self.db_path) and os.path.getsize(self.db_path) == 0:
            os.remove(self.db_path)
        journal_path = self.db_path + "-journal"
        if os.path.exists(journal_path) and not os.path.exists(self.db_path):
            os.remove(journal_path)
        if os.path.exists(journal_path) and os.path.getsize(self.db_path) == 0:
            os.remove(journal_path)

    def _connect(self):
        self._prepare_db_path()
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=MEMORY")
        conn.execute("PRAGMA temp_store=MEMORY")
        return conn

    def init_schema(self):
        conn = self._connect()
        try:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS kb_documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    theme TEXT,
                    language TEXT DEFAULT 'unknown',
                    file_path TEXT,
                    is_core INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT (datetime('now','localtime'))
                );

                CREATE TABLE IF NOT EXISTS kb_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id INTEGER NOT NULL REFERENCES kb_documents(id) ON DELETE CASCADE,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    keywords TEXT,
                    knowledge_type TEXT DEFAULT 'general'
                );

                CREATE TABLE IF NOT EXISTS kb_links (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id INTEGER REFERENCES kb_documents(id) ON DELETE CASCADE,
                    relation_type TEXT,
                    source_node TEXT,
                    target_node TEXT,
                    process_factor TEXT,
                    morphology_factor TEXT,
                    performance_factor TEXT,
                    effect_direction TEXT,
                    confidence REAL DEFAULT 0.5,
                    mechanism_summary TEXT,
                    evidence_text TEXT
                );

                CREATE TABLE IF NOT EXISTS kb_task_profiles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_name TEXT NOT NULL UNIQUE,
                    preferred_theme TEXT,
                    preferred_themes TEXT,
                    discouraged_themes TEXT,
                    preferred_knowledge_types TEXT,
                    preferred_keywords TEXT
                );

                CREATE TABLE IF NOT EXISTS kb_msfu (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chunk_id INTEGER NOT NULL REFERENCES kb_chunks(id) ON DELETE CASCADE,
                    doc_id INTEGER REFERENCES kb_documents(id) ON DELETE CASCADE,

                    -- Assertion fields
                    source_entity TEXT NOT NULL,
                    relation_type TEXT NOT NULL,
                    target_entity TEXT NOT NULL,

                    -- Condition fields
                    condition_param TEXT,
                    condition_op TEXT,
                    condition_value TEXT,
                    condition_unit TEXT,

                    -- Other fields
                    direction TEXT NOT NULL,
                    content TEXT NOT NULL,
                    confidence REAL DEFAULT 0.5,
                    extraction_method TEXT DEFAULT 'rule',

                    -- Inherited from kb_links for compatibility
                    process_factor TEXT,
                    morphology_factor TEXT,
                    performance_factor TEXT,
                    effect_direction TEXT,
                    mechanism_summary TEXT,
                    evidence_text TEXT,

                    -- Metadata
                    doc_title TEXT,
                    page_num INTEGER,

                    created_at TEXT DEFAULT (datetime('now','localtime'))
                );

                CREATE INDEX IF NOT EXISTS idx_msfu_chunk_id ON kb_msfu(chunk_id);
                CREATE INDEX IF NOT EXISTS idx_msfu_doc_id ON kb_msfu(doc_id);
                CREATE INDEX IF NOT EXISTS idx_msfu_source ON kb_msfu(source_entity);
                CREATE INDEX IF NOT EXISTS idx_msfu_target ON kb_msfu(target_entity);
                CREATE INDEX IF NOT EXISTS idx_msfu_relation ON kb_msfu(relation_type);
                CREATE INDEX IF NOT EXISTS idx_msfu_direction ON kb_msfu(direction);
                CREATE INDEX IF NOT EXISTS idx_msfu_condition ON kb_msfu(condition_param, condition_op);

                CREATE TABLE IF NOT EXISTS kb_conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    rag_context TEXT,
                    sources TEXT,
                    created_at TEXT DEFAULT (datetime('now','localtime'))
                );

                CREATE TABLE IF NOT EXISTS kb_qa_templates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    category TEXT,
                    question TEXT NOT NULL,
                    icon TEXT,
                    created_at TEXT DEFAULT (datetime('now','localtime'))
                );
                """
            )
            self._ensure_task_profile_columns(conn)
            self._ensure_link_columns(conn)
            self._ensure_msfu_columns(conn)
            self._ensure_conversation_columns(conn)
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _ensure_task_profile_columns(conn):
        existing = {
            row[1]
            for row in conn.execute("PRAGMA table_info(kb_task_profiles)").fetchall()
        }
        if "preferred_themes" not in existing:
            conn.execute("ALTER TABLE kb_task_profiles ADD COLUMN preferred_themes TEXT")
        if "discouraged_themes" not in existing:
            conn.execute("ALTER TABLE kb_task_profiles ADD COLUMN discouraged_themes TEXT")

    @staticmethod
    def _ensure_link_columns(conn):
        existing = {
            row[1]
            for row in conn.execute("PRAGMA table_info(kb_links)").fetchall()
        }
        if "relation_type" not in existing:
            conn.execute("ALTER TABLE kb_links ADD COLUMN relation_type TEXT")
        if "source_node" not in existing:
            conn.execute("ALTER TABLE kb_links ADD COLUMN source_node TEXT")
        if "target_node" not in existing:
            conn.execute("ALTER TABLE kb_links ADD COLUMN target_node TEXT")
        if "performance_factor" not in existing:
            conn.execute("ALTER TABLE kb_links ADD COLUMN performance_factor TEXT")
        if "confidence" not in existing:
            conn.execute("ALTER TABLE kb_links ADD COLUMN confidence REAL DEFAULT 0.5")

    @staticmethod
    def _ensure_msfu_columns(conn):
        existing = {
            row[1]
            for row in conn.execute("PRAGMA table_info(kb_msfu)").fetchall()
        }
        if "doc_title" not in existing:
            conn.execute("ALTER TABLE kb_msfu ADD COLUMN doc_title TEXT")
        if "page_num" not in existing:
            conn.execute("ALTER TABLE kb_msfu ADD COLUMN page_num INTEGER")
        if "created_at" not in existing:
            conn.execute("ALTER TABLE kb_msfu ADD COLUMN created_at TEXT DEFAULT (datetime('now','localtime'))")

    @staticmethod
    def _ensure_conversation_columns(conn):
        # 确保对话表和索引存在
        # 创建索引（如果不存在）
        try:
            conn.execute("CREATE INDEX IF NOT EXISTS idx_conv_session ON kb_conversations(session_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_conv_created ON kb_conversations(created_at)")
        except sqlite3.OperationalError:
            # 如果表还不存在，会在 init_schema 中创建
            pass

    def bootstrap_task_profiles(self):
        conn = self._connect()
        try:
            conn.executemany(
                """
                INSERT INTO kb_task_profiles (
                    task_name, preferred_theme, preferred_themes, discouraged_themes,
                    preferred_knowledge_types, preferred_keywords
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(task_name) DO UPDATE SET
                    preferred_theme = excluded.preferred_theme,
                    preferred_themes = excluded.preferred_themes,
                    discouraged_themes = excluded.discouraged_themes,
                    preferred_knowledge_types = excluded.preferred_knowledge_types,
                    preferred_keywords = excluded.preferred_keywords
                """,
                DEFAULT_TASK_PROFILES,
            )
            conn.commit()
        finally:
            conn.close()

    def ingest_text(
        self,
        title: str,
        text: str,
        source_type: str,
        theme: Optional[str] = None,
        is_core: bool = False,
        file_path: Optional[str] = None,
        language: str = "unknown",
    ) -> Dict[str, int]:
        cleaned = re.sub(r"\s+", " ", text or "").strip()
        if not cleaned:
            raise ValueError("text is empty")

        chunks = self._split_text(cleaned)
        conn = self._connect()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO kb_documents (title, source_type, theme, language, file_path, is_core)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (title, source_type, theme, language, file_path, int(bool(is_core))),
            )
            doc_id = cursor.lastrowid

            for index, chunk in enumerate(chunks):
                cursor.execute(
                    """
                    INSERT INTO kb_chunks (doc_id, chunk_index, text, keywords, knowledge_type)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        doc_id,
                        index,
                        chunk,
                        " ".join(self._extract_keywords(chunk)),
                        self._infer_knowledge_type(theme),
                    ),
                )

                for relation in self._extract_relations_from_chunk(chunk):
                    cursor.execute(
                        """
                        INSERT INTO kb_links (
                            doc_id, relation_type, source_node, target_node,
                            process_factor, morphology_factor, performance_factor,
                            effect_direction, confidence, mechanism_summary, evidence_text
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            doc_id,
                            relation.get("relation_type"),
                            relation.get("source_node"),
                            relation.get("target_node"),
                            relation.get("process_factor"),
                            relation.get("morphology_factor"),
                            relation.get("performance_factor"),
                            relation.get("effect_direction"),
                            relation.get("confidence", 0.5),
                            relation.get("mechanism_summary"),
                            relation.get("evidence_text"),
                            chunk_id,
                        ),
                    )

                # MSFU提取（如果kb_msfu表存在）
                try:
                    from .msfu_extractor import MSFUExtractor, MSFUMetadata

                    msfu_metadata = MSFUMetadata(
                        doc_id=str(doc_id),
                        chunk_id=str(index),
                        doc_title=title,
                        doc_type=source_type
                    )
                    msfu_extractor = MSFUExtractor(use_llm_refinement=False)
                    msfus = msfu_extractor.extract(chunk, msfu_metadata, title)

                    for msfu in msfus:
                        row_data = msfu.to_db_row()
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
                                index,
                                doc_id,
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
                                title,
                                None
                            )
                        )
                except ImportError:
                    # msfu_extractor模块不可用，跳过
                    pass
                except Exception:
                    # MSFU提取失败，不影响正常流程
                    pass
            conn.commit()
        finally:
            conn.close()

        self._invalidate_semantic_cache()
        return {"doc_id": doc_id, "chunk_count": len(chunks)}

    def list_documents(self) -> List[Dict[str, object]]:
        conn = self._connect()
        try:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT d.id, d.title, d.source_type, d.theme, d.language, d.file_path,
                       d.is_core, d.created_at, COUNT(c.id) AS chunk_count
                FROM kb_documents d
                LEFT JOIN kb_chunks c ON c.doc_id = d.id
                GROUP BY d.id
                ORDER BY d.id DESC
                """
            ).fetchall()
        finally:
            conn.close()
        return [dict(row) for row in rows]

    def delete_document(self, doc_id: int):
        conn = self._connect()
        try:
            conn.execute("DELETE FROM kb_documents WHERE id = ?", (doc_id,))
            conn.commit()
        finally:
            conn.close()
        self._invalidate_semantic_cache()

    def update_document_theme(self, doc_id: int, theme: str) -> None:
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE kb_documents SET theme = ? WHERE id = ?",
                (theme, doc_id),
            )
            conn.execute(
                "UPDATE kb_chunks SET knowledge_type = ? WHERE doc_id = ?",
                (self._infer_knowledge_type(theme), doc_id),
            )
            conn.commit()
        finally:
            conn.close()
        self._invalidate_semantic_cache()

    def search(
        self,
        query: str,
        task_name: Optional[str] = None,
        top_k: int = 5,
    ) -> List[Dict[str, object]]:
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        task_profile = self._get_task_profile(task_name)
        conn = self._connect()
        try:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT c.id, c.text, c.keywords, c.knowledge_type,
                       d.id AS doc_id, d.title, d.theme, d.source_type, d.is_core
                FROM kb_chunks c
                JOIN kb_documents d ON d.id = c.doc_id
                """
            ).fetchall()
        finally:
            conn.close()

        row_dicts = [dict(row) for row in rows]
        semantic_scores = self._semantic_scores(query, row_dicts)
        relation_scores = self._relation_constraint_scores(query, row_dicts)

        scored = []
        for row in row_dicts:
            lexical_score = self._score_row(row, query_tokens, task_profile)
            semantic_score = semantic_scores.get(int(row["id"]), 0.0)
            relation_score = relation_scores.get(int(row["doc_id"]), 0.0)
            score = lexical_score + (semantic_score * 6.0) + relation_score
            if score > 0:
                scored.append((score, row, semantic_score, relation_score))

        scored.sort(key=lambda item: item[0], reverse=True)
        results = []
        for score, row, semantic_score, relation_score in scored[:top_k]:
            results.append(
                {
                    "chunk_id": row["id"],
                    "doc_id": row["doc_id"],
                    "title": row["title"],
                    "theme": row["theme"],
                    "source_type": row["source_type"],
                    "knowledge_type": row["knowledge_type"],
                    "text": row["text"][:400],
                    "score": round(score, 4),
                    "semantic_score": round(semantic_score, 4),
                    "relation_score": round(relation_score, 4),
                }
            )
        return results

    def parse_query(self, query: str) -> Dict[str, object]:
        """
        解析查询语句，提取实体、条件、方向、目标
        z(q) = (Eq, Cq, Dq, Yq)
        """
        result = {
            "entities": [],
            "conditions": [],
            "direction": None,
            "targets": [],
            "raw_query": query,
        }

        # 提取实体 - 从查询中提取工艺/形貌/性能因子
        all_patterns = {
            **PROCESS_FACTOR_PATTERNS,
            **MORPHOLOGY_FACTOR_PATTERNS,
            **PERFORMANCE_FACTOR_PATTERNS,
        }

        for factor_name, patterns in all_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    if factor_name not in result["entities"]:
                        result["entities"].append(factor_name)

        # 提取方向 - 正向/负向相关
        if re.search(r"(促进|增加|提高|增强|increase|promote|enhance|improve)", query, re.IGNORECASE):
            result["direction"] = "positive"
        elif re.search(r"(抑制|减少|降低|减弱|decrease|reduce|suppress|inhibit)", query, re.IGNORECASE):
            result["direction"] = "negative"

        # 提取条件 - 数值范围、比较关系
        numeric_matches = re.finditer(r"\d+", query)
        for match in numeric_matches:
            result["conditions"].append({"type": "numeric", "value": match.group()})

        # 提取目标 - 用户想要得到的信息类型
        if any(kw in query.lower() for kw in ["为什么", "why", "how", "机制", "mechanism"]):
            result["targets"].append("explanation")
        if any(kw in query.lower() for kw in ["影响", "effect", "impact", "关系", "relation"]):
            result["targets"].append("relation")

        return result

    def tccer_retrieve(
        self,
        query: str,
        task_name: Optional[str] = None,
        top_k: int = 5,
        max_depth: Optional[int] = None,
    ) -> Dict[str, object]:
        """
        TCCER: 面向任务的约束链式证据检索

        Args:
            max_depth: 路径扩展最大深度，未指定时根据任务类型自动设置
                      - process_analysis: H=2
                      - morphology_interpretation: H=3
                      - prediction_explanation: H=3
        """
        parsed_query = self.parse_query(query)
        task_profile = self.TASK_TYPES.get(task_name or "morphology_interpretation", {})

        # 根据任务类型设置默认 max_depth
        depth_config = {
            "process_analysis": 2,
            "morphology_interpretation": 3,
            "prediction_explanation": 3,
        }
        actual_max_depth = max_depth or depth_config.get(task_name, 2)

        # 步骤1: 初始召回 - 稀疏检索 + 稠密检索
        initial_chunks = self._mixed_recall_initial(
            query, parsed_query, task_profile, top_k * 2
        )

        # 步骤2: 约束路径扩展
        expanded_paths = self._constrained_path_expansion(
            initial_chunks, parsed_query, task_profile, actual_max_depth
        )

        # 步骤3: 路径评分与排序
        scored_paths = self._score_paths(expanded_paths, parsed_query, task_profile)

        # 步骤4: 冗余抑制
        final_results = self._suppress_redundancy(scored_paths, top_k)

        return {
            "query": query,
            "parsed_query": parsed_query,
            "task_name": task_name,
            "max_depth": actual_max_depth,
            "results": final_results,
            "path_count": len(scored_paths),
        }

    def _mixed_recall_initial(
        self,
        query: str,
        parsed_query: Dict[str, object],
        task_profile: Dict[str, object],
        top_k: int,
    ) -> List[Dict[str, object]]:
        """混合召回：稀疏检索 + 稠密检索"""
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        conn = self._connect()
        try:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT c.id, c.text, c.keywords, c.knowledge_type,
                       d.id AS doc_id, d.title, d.theme, d.source_type, d.is_core
                FROM kb_chunks c
                JOIN kb_documents d ON d.id = c.doc_id
                """
            ).fetchall()
        finally:
            conn.close()

        row_dicts = [dict(row) for row in rows]
        semantic_scores = self._semantic_scores(query, row_dicts)

        # 混合评分: S0 = α·sparse + (1-α)·dense + β·task + γ·condition
        alpha, beta, gamma = 0.4, 0.3, 0.2

        scored = []
        for row in row_dicts:
            sparse_score = self._score_row(row, query_tokens, task_profile)
            dense_score = semantic_scores.get(int(row["id"]), 0.0)

            # 任务相关性评分
            task_score = 0.0
            if task_profile.get("preferred_relations"):
                task_score = self._task_relevance_score(row, task_profile)

            # 条件匹配评分
            condition_score = self._condition_match_score(row, parsed_query)

            total_score = (
                alpha * sparse_score +
                (1 - alpha) * dense_score +
                beta * task_score +
                gamma * condition_score
            )

            if total_score > 0:
                scored.append({
                    "chunk_id": row["id"],
                    "doc_id": row["doc_id"],
                    "text": row["text"],
                    "score": total_score,
                    "sparse_score": sparse_score,
                    "dense_score": dense_score,
                    "task_score": task_score,
                    "condition_score": condition_score,
                })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def _constrained_path_expansion(
        self,
        initial_chunks: List[Dict[str, object]],
        parsed_query: Dict[str, object],
        task_profile: Dict[str, object],
        max_depth: int,
    ) -> List[Dict[str, object]]:
        """约束路径扩展：基于关系转换矩阵扩展证据链"""
        paths = []

        # 预先加载所有初始chunks的关系信息
        chunk_relations_map = self._get_chunk_relations_map(
            [chunk["chunk_id"] for chunk in initial_chunks]
        )

        for chunk in initial_chunks:
            chunk_id = chunk["chunk_id"]

            # 为初始chunk添加关系信息
            chunk_with_relation = dict(chunk)
            chunk_with_relation["relation"] = chunk_relations_map.get(chunk_id, {}).get("relation", {})

            path = {
                "chunks": [chunk_with_relation],
                "relations": [chunk_with_relation.get("relation")],
                "depth": 1,
            }
            paths.append(path)

            # 深度优先扩展，使用路径级别的 visited 集合
            path_visited = {chunk_id}
            for depth_step in range(1, max_depth):
                last_chunk = path["chunks"][-1]
                next_chunks = self._get_related_chunks(
                    last_chunk["chunk_id"],
                    parsed_query,
                    task_profile,
                    max_hops=1,  # 每次扩展只查找1跳邻居
                )

                if not next_chunks:
                    break

                # 尝试找到第一个未访问的chunk
                found_next = False
                for next_chunk in next_chunks:
                    next_chunk_id = next_chunk["chunk_id"]
                    if next_chunk_id not in path_visited:
                        # 为新chunk加载关系信息
                        relations_map = self._get_chunk_relations_map([next_chunk_id])
                        if next_chunk_id in relations_map:
                            next_chunk["relation"] = relations_map[next_chunk_id]["relation"]

                        path["chunks"].append(next_chunk)
                        path["relations"].append(next_chunk.get("relation", {}))
                        path["depth"] += 1
                        path_visited.add(next_chunk_id)
                        found_next = True
                        break

                if not found_next:
                    break

        return paths

    def _get_chunk_relations_map(self, chunk_ids: List[int]) -> Dict[int, Dict[str, object]]:
        """获取多个chunks的关系信息

        Args:
            chunk_ids: chunk ID列表

        Returns:
            chunk_id -> 关系信息的映射
        """
        if not chunk_ids:
            return {}

        conn = self._connect()
        try:
            conn.row_factory = sqlite3.Row
            placeholders = ",".join("?" for _ in chunk_ids)
            rows = conn.execute(
                f"""
                SELECT l.chunk_id,
                       l.relation_type, l.process_factor, l.morphology_factor,
                       l.performance_factor, l.effect_direction, l.confidence
                FROM kb_links l
                WHERE l.chunk_id IN ({placeholders})
                ORDER BY l.confidence DESC
                """,
                tuple(chunk_ids),
            ).fetchall()
        finally:
            conn.close()

        relations_map = {}
        for row in rows:
            chunk_id = row["chunk_id"]
            relations_map[chunk_id] = {
                "relation": {
                    "type": row["relation_type"],
                    "process_factor": row["process_factor"],
                    "morphology_factor": row["morphology_factor"],
                    "performance_factor": row["performance_factor"],
                    "effect_direction": row["effect_direction"],
                    "confidence": row["confidence"],
                }
            }

        return relations_map

    def _get_related_chunks(
        self,
        chunk_id: int,
        parsed_query: Dict[str, object],
        task_profile: Dict[str, object],
        max_hops: int = 1,
    ) -> List[Dict[str, object]]:
        """获取相关的 chunks（基于关系链接）

        Args:
            chunk_id: 当前chunk的ID
            parsed_query: 解析后的查询
            task_profile: 任务配置
            max_hops: 最大跳数（目前只支持1跳）

        Returns:
            相关chunks列表，按相关性排序
        """
        # 只支持1跳查询，多跳通过迭代实现
        if max_hops != 1:
            max_hops = 1

        conn = self._connect()
        try:
            conn.row_factory = sqlite3.Row

            # 首先获取当前chunk的所有links
            current_links = conn.execute(
                "SELECT source_node, target_node FROM kb_links WHERE chunk_id = ?",
                (chunk_id,)
            ).fetchall()

            if not current_links:
                return []

            # 收集所有相关的因子名称
            related_factors = set()
            for link in current_links:
                if link["source_node"]:
                    related_factors.add(link["source_node"])
                if link["target_node"]:
                    related_factors.add(link["target_node"])

            if not related_factors:
                return []

            # 查找与这些因子相关的其他chunks
            placeholders = ",".join("?" for _ in related_factors)
            rows = conn.execute(
                f"""
                SELECT DISTINCT c.id, c.text, c.knowledge_type,
                       l.relation_type, l.process_factor, l.morphology_factor,
                       l.performance_factor, l.effect_direction, l.confidence
                FROM kb_links l
                JOIN kb_chunks c ON c.id = l.chunk_id
                WHERE (l.source_node IN ({placeholders}) OR l.target_node IN ({placeholders}))
                AND c.id != ?
                ORDER BY l.confidence DESC
                LIMIT 50
                """,
                tuple(list(related_factors) + list(related_factors) + [chunk_id]),
            ).fetchall()
        finally:
            conn.close()

        related = []
        for row in rows:
            score = self._relation_relevance_score(row, parsed_query, task_profile)
            # 如果没有任务配置和查询实体，使用置信度作为基础分数
            if score == 0 and row["confidence"]:
                score = row["confidence"] * 0.5

            # 降低评分阈值到0.001，允许更多结果通过
            if score > 0.001:  # 进一步降低阈值，允许更多结果通过
                related.append({
                    "chunk_id": row["id"],
                    "text": row["text"],
                    "relation": {
                        "type": row["relation_type"],
                        "process_factor": row["process_factor"],
                        "morphology_factor": row["morphology_factor"],
                        "performance_factor": row["performance_factor"],
                        "effect_direction": row["effect_direction"],
                        "confidence": row["confidence"],
                    },
                    "score": score,
                })

        return related

    def _score_paths(
        self,
        paths: List[Dict[str, object]],
        parsed_query: Dict[str, object],
        task_profile: Dict[str, object],
    ) -> List[Dict[str, object]]:
        """路径评分：综合路径深度、相关性、一致性"""
        scored = []
        for path in paths:
            # 基础评分：所有 chunk 评分的平均
            avg_chunk_score = sum(c["score"] for c in path["chunks"]) / len(path["chunks"])

            # 深度奖励：更深的路径获得更高分数
            depth_bonus = 0.1 * path["depth"]

            # 一致性评分：检查路径中关系的连贯性
            consistency_score = self._path_consistency_score(path)

            # 方向一致性：与查询方向是否一致
            direction_score = self._direction_consistency_score(path, parsed_query)

            total_score = (
                avg_chunk_score * 0.6 +
                depth_bonus * 0.1 +
                consistency_score * 0.2 +
                direction_score * 0.1
            )

            scored.append({
                "path": path,
                "score": total_score,
                "avg_chunk_score": avg_chunk_score,
                "depth_bonus": depth_bonus,
                "consistency_score": consistency_score,
                "direction_score": direction_score,
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored

    def _suppress_redundancy(
        self,
        scored_paths: List[Dict[str, object]],
        top_k: int,
    ) -> List[Dict[str, object]]:
        """冗余抑制：去除高度重复的路径"""
        selected = []
        selected_path_chunk_sets = []

        for path_data in scored_paths:
            path = path_data["path"]
            chunk_ids = {c["chunk_id"] for c in path["chunks"]}

            # 计算与已选择路径的重叠度
            max_overlap = 0.0
            for selected_ids in selected_path_chunk_sets:
                if selected_ids:
                    overlap = len(chunk_ids & selected_ids) / len(chunk_ids)
                    max_overlap = max(max_overlap, overlap)

            # 重叠度超过阈值则跳过
            if max_overlap > 0.6:
                continue

            selected.append({
                "chunks": path["chunks"],
                "relations": path["relations"],
                "depth": path["depth"],
                "score": path_data["score"],
                "consistency": path_data["consistency_score"],
            })

            selected_path_chunk_sets.append(chunk_ids)

            if len(selected) >= top_k:
                break

        return selected

    def visualize_paths(self, tccer_result: Dict[str, object]) -> Dict[str, object]:
        """
        路径可视化：生成关系图谱和路径展示

        Returns:
            {
                "graph": {...},           # 关系图谱数据
                "paths": [...],            # 可视化路径列表
                "summary": {...}           # 可视化摘要
            }
        """
        if not tccer_result.get("results"):
            return {"graph": {}, "paths": [], "summary": {"error": "No paths to visualize"}}

        # 1. 构建关系图谱节点和边
        nodes = []
        edges = []
        node_map = {}

        path_index = 0
        for path_result in tccer_result["results"]:
            chunks = path_result.get("chunks", [])
            relations = path_result.get("relations", [])

            # 为每个 chunk 创建节点
            for chunk_idx, chunk in enumerate(chunks):
                node_id = f"p{path_index}_c{chunk_idx}"
                if node_id not in node_map:
                    node_map[node_id] = {
                        "id": node_id,
                        "path_index": path_index,
                        "chunk_index": chunk_idx,
                        "text": chunk.get("text", "")[:200],
                        "score": chunk.get("score", 0.0),
                        "type": "chunk",
                    }
                    nodes.append(node_map[node_id])

            # 创建关系边
            for rel_idx, relation in enumerate(relations):
                if rel_idx < len(chunks) - 1:
                    source_id = f"p{path_index}_c{rel_idx}"
                    target_id = f"p{path_index}_c{rel_idx + 1}"

                    edge_id = f"p{path_index}_e{rel_idx}"
                    edges.append({
                        "id": edge_id,
                        "source": source_id,
                        "target": target_id,
                        "path_index": path_index,
                        "relation_type": relation.get("type", "unknown"),
                        "process_factor": relation.get("process_factor"),
                        "morphology_factor": relation.get("morphology_factor"),
                        "performance_factor": relation.get("performance_factor"),
                        "effect_direction": relation.get("effect_direction"),
                        "confidence": relation.get("confidence", 0.5),
                    })

            path_index += 1

        # 2. 构建可视化路径
        visual_paths = []
        for path_idx, path_result in enumerate(tccer_result["results"]):
            chunks = path_result.get("chunks", [])
            relations = path_result.get("relations", [])

            visual_path = {
                "path_index": path_idx,
                "depth": path_result.get("depth", 1),
                "score": path_result.get("score", 0.0),
                "consistency": path_result.get("consistency", 0.0),
                "nodes": [],
                "edges": [],
            }

            # 添加路径节点
            for chunk_idx, chunk in enumerate(chunks):
                node_id = f"p{path_idx}_c{chunk_idx}"
                visual_path["nodes"].append({
                    "id": node_id,
                    "position": chunk_idx,
                    "text": chunk.get("text", "")[:150],
                    "score": chunk.get("score", 0.0),
                })

            # 添加路径边
            for rel_idx, relation in enumerate(relations):
                if rel_idx < len(chunks) - 1:
                    visual_path["edges"].append({
                        "from": f"p{path_idx}_c{rel_idx}",
                        "to": f"p{path_idx}_c{rel_idx + 1}",
                        "relation_type": relation.get("type", "unknown"),
                        "direction": relation.get("effect_direction"),
                        "label": self._create_edge_label(relation),
                    })

            visual_paths.append(visual_path)

        # 3. 生成关系图谱数据
        graph_data = self._create_graph_data(nodes, edges)

        # 4. 可视化摘要
        summary = {
            "total_paths": len(visual_paths),
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "avg_depth": sum(p["depth"] for p in visual_paths) / max(len(visual_paths), 1),
            "relation_types": list(set(e["relation_type"] for e in edges)),
            "query": tccer_result.get("query"),
            "task_name": tccer_result.get("task_name"),
        }

        return {
            "graph": graph_data,
            "paths": visual_paths,
            "summary": summary,
        }

    def _create_graph_data(self, nodes: List[Dict[str, object]], edges: List[Dict[str, object]]) -> Dict[str, object]:
        """创建关系图谱数据，适配 D3.js 等可视化库"""
        node_list = []
        edge_list = []

        # 节点样式配置
        node_styles = {
            "chunk": {"color": "#4CAF50", "size": 20},
            "query": {"color": "#2196F3", "size": 25},
        }

        for node in nodes:
            node_list.append({
                "id": node["id"],
                "label": f"Chunk {node['chunk_index']}",
                "text": node["text"],
                "score": node["score"],
                "color": node_styles[node["type"]]["color"],
                "size": node_styles[node["type"]]["size"],
                "type": node["type"],
            })

        # 边样式配置
        edge_styles = {
            "process_to_morphology": {"color": "#FF9800", "width": 2},
            "process_to_mechanism": {"color": "#9C27B0", "width": 2},
            "mechanism_to_morphology": {"color": "#E91E63", "width": 2},
            "morphology_to_performance": {"color": "#00BCD4", "width": 2},
            "process_to_performance": {"color": "#795548", "width": 2},
            "mechanism_evidence": {"color": "#607D8B", "width": 1, "style": "dashed"},
        }

        for edge in edges:
            style = edge_styles.get(edge["relation_type"], {"color": "#999", "width": 1})
            edge_list.append({
                "id": edge["id"],
                "source": edge["source"],
                "target": edge["target"],
                "label": self._create_edge_label(edge),
                "color": style["color"],
                "width": style["width"],
                "style": style.get("style", "solid"),
                "confidence": edge.get("confidence", 0.5),
            })

        return {
            "nodes": node_list,
            "edges": edge_list,
            "layout": "force_directed",  # 推荐布局算法
        }

    def _create_edge_label(self, relation: Dict[str, object]) -> str:
        """创建边的标签"""
        parts = []
        if relation.get("process_factor"):
            parts.append(relation["process_factor"])
        if relation.get("morphology_factor"):
            parts.append("→")
            parts.append(relation["morphology_factor"])
        if relation.get("performance_factor"):
            parts.append("→")
            parts.append(relation["performance_factor"])

        direction_symbol = {
            "positive": "↑",
            "negative": "↓",
            "neutral": "→"
        }.get(relation.get("effect_direction"), "→")

        if parts:
            return f"{' '.join(parts)} {direction_symbol}"
        return relation.get("relation_type", "unknown")

    def generate_evidence_explanation(self, tccer_result: Dict[str, object]) -> Dict[str, object]:
        """
        生成证据解释：自动生成检索路径的文字解释

        Returns:
            {
                "summary": "...",           # 总体摘要
                "chain_explanation": "...",   # 链式推理解释
                "confidence_explanation": "...", # 置信度说明
                "evidence_integration": "...", # 证据整合
                "detailed_paths": [...]      # 详细路径解释
            }
        """
        if not tccer_result.get("results"):
            return {"error": "No evidence to explain"}

        query = tccer_result.get("query", "")
        task_name = tccer_result.get("task_name", "general")
        results = tccer_result.get("results", [])

        # 1. 生成总体摘要
        summary = self._generate_summary(query, task_name, results)

        # 2. 生成链式推理解释
        chain_explanation = self._generate_chain_explanation(query, results)

        # 3. 生成置信度说明
        confidence_explanation = self._generate_confidence_explanation(results)

        # 4. 生成证据整合
        evidence_integration = self._generate_evidence_integration(query, results)

        # 5. 生成详细路径解释
        detailed_paths = []
        for path_idx, path_result in enumerate(results):
            detailed_paths.append(self._generate_detailed_path_explanation(path_idx, path_result))

        return {
            "summary": summary,
            "chain_explanation": chain_explanation,
            "confidence_explanation": confidence_explanation,
            "evidence_integration": evidence_integration,
            "detailed_paths": detailed_paths,
        }

    def _generate_summary(self, query: str, task_name: str, results: List[Dict[str, object]]) -> str:
        """生成总体摘要"""
        path_count = len(results)
        avg_score = sum(r.get("score", 0) for r in results) / max(path_count, 1)
        max_depth = max(r.get("depth", 1) for r in results) if results else 1

        task_chinese = {
            "morphology_interpretation": "形貌解释",
            "process_analysis": "工艺分析",
            "prediction_explanation": "预测解释"
        }.get(task_name, "通用分析")

        summary = f"""
基于"{query}"的查询，系统在{task_chinese}任务中检索到了{path_count}条相关证据路径。

这些路径的平均相关性评分为{avg_score:.3f}，最大推理深度为{max_depth}层。
检索结果涵盖了工艺参数、形貌特征和性能指标之间的因果关系链。
        """.strip()

        return summary

    def _generate_chain_explanation(self, query: str, results: List[Dict[str, object]]) -> str:
        """生成链式推理解释"""
        if not results:
            return "未找到相关证据链。"

        explanations = []
        for path_idx, path_result in enumerate(results):
            relations = path_result.get("relations", [])
            if not relations:
                continue

            path_explanation = f"证据链 {path_idx + 1}："

            # 构建推理链
            reasoning_steps = []
            for rel_idx, relation in enumerate(relations):
                process = relation.get("process_factor", "未知工艺")
                morph = relation.get("morphology_factor", "未知形貌")
                perf = relation.get("performance_factor")
                direction = relation.get("effect_direction", "neutral")
                confidence = relation.get("confidence", 0.5)

                if process and morph:
                    direction_text = {"positive": "促进", "negative": "抑制", "neutral": "影响"}.get(direction, "影响")
                    step = f"{process}{direction_text}{morph}（置信度:{confidence:.2f}）"
                    reasoning_steps.append(step)

                if morph and perf:
                    direction_text = {"positive": "提高", "negative": "降低", "neutral": "影响"}.get(direction, "影响")
                    step = f"{morph}{direction_text}{perf}（置信度:{confidence:.2f}）"
                    reasoning_steps.append(step)

            if reasoning_steps:
                path_explanation += " → ".join(reasoning_steps)
                explanations.append(path_explanation)

        if explanations:
            return "\n".join(explanations)
        return "无法生成链式推理解释。"

    def _generate_confidence_explanation(self, results: List[Dict[str, object]]) -> str:
        """生成置信度说明"""
        if not results:
            return "无可评估的置信度。"

        total_score = sum(r.get("score", 0) for r in results)
        avg_confidence = total_score / len(results)

        consistency_scores = [r.get("consistency", 0) for r in results if r.get("consistency")]
        avg_consistency = sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0.0

        confidence_level = "高" if avg_confidence > 0.7 else "中" if avg_confidence > 0.4 else "低"
        consistency_level = "高" if avg_consistency > 0.8 else "中" if avg_consistency > 0.5 else "低"

        explanation = f"""
检索结果的总体置信度为{confidence_level}（平均评分: {avg_confidence:.3f}），
证据链的一致性为{consistency_level}（平均一致性: {avg_consistency:.3f}）。

置信度评估综合考虑了：
- 稀疏检索（关键词匹配）
- 稠密检索（语义相似度）
- 任务相关性（与查询任务的匹配度）
- 条件匹配度（数值条件满足情况）

一致性评估检查了关系链的连贯性和方向一致性。
        """.strip()

        return explanation

    def _generate_evidence_integration(self, query: str, results: List[Dict[str, object]]) -> str:
        """生成证据整合说明"""
        if not results:
            return "无可整合的证据。"

        # 收集所有证据节点
        all_chunks = []
        for path_result in results:
            chunks = path_result.get("chunks", [])
            all_chunks.extend(chunks)

        # 统计关键实体
        process_factors = set()
        morphology_factors = set()
        performance_factors = set()

        for path_result in results:
            for relation in path_result.get("relations", []):
                if relation.get("process_factor"):
                    process_factors.add(relation["process_factor"])
                if relation.get("morphology_factor"):
                    morphology_factors.add(relation["morphology_factor"])
                if relation.get("performance_factor"):
                    performance_factors.add(relation["performance_factor"])

        integration_text = f"""
证据整合分析：

覆盖的关键实体：
- 工艺因子: {len(process_factors)}个
- 形貌因子: {len(morphology_factors)}个
- 性能因子: {len(performance_factors)}个

证据来源统计：
- 独立证据块: {len(all_chunks)}个
- 证据链路: {len(results)}条
- 平均链路深度: {sum(r.get('depth', 1) for r in results) / max(len(results), 1):.1f}层

基于这些证据，我们可以形成对"{query}"的综合理解，
通过多角度证据验证提高结论的可靠性。
        """.strip()

        return integration_text

    def _generate_detailed_path_explanation(self, path_idx: int, path_result: Dict[str, object]) -> Dict[str, object]:
        """生成单个路径的详细解释"""
        chunks = path_result.get("chunks", [])
        relations = path_result.get("relations", [])
        depth = path_result.get("depth", 1)
        score = path_result.get("score", 0.0)
        consistency = path_result.get("consistency", 0.0)

        # 生成路径描述
        path_description = f"路径 {path_idx + 1}（深度:{depth}层，评分:{score:.3f}）"

        # 生成步骤解释
        steps = []
        for i in range(len(chunks)):
            chunk = chunks[i]
            step = {
                "step_number": i + 1,
                "chunk_text": chunk.get("text", "")[:200],
                "score": chunk.get("score", 0.0),
                "relation": relations[i] if i < len(relations) else None,
            }

            if i < len(relations):
                relation = relations[i]
                step["explanation"] = self._generate_relation_explanation(relation)

            steps.append(step)

        # 生成路径总结
        path_summary = {
            "path_index": path_idx + 1,
            "description": path_description,
            "depth": depth,
            "score": score,
            "consistency": consistency,
            "steps": steps,
            "quality_assessment": self._assess_path_quality(score, consistency, depth),
        }

        return path_summary

    def _generate_relation_explanation(self, relation: Dict[str, object]) -> str:
        """生成单个关系的解释"""
        process = relation.get("process_factor")
        morph = relation.get("morphology_factor")
        perf = relation.get("performance_factor")
        direction = relation.get("effect_direction", "neutral")
        confidence = relation.get("confidence", 0.5)

        if not process and not morph and not perf:
            return f"未知关系（置信度:{confidence:.2f}）"

        parts = []
        if process:
            parts.append(f"工艺参数: {process}")
        if morph:
            parts.append(f"形貌特征: {morph}")
        if perf:
            parts.append(f"性能指标: {perf}")

        direction_text = {"positive": "正向促进", "negative": "负向抑制", "neutral": "中性影响"}.get(direction, "影响")

        base = "，".join(parts)
        return f"{base}，{direction_text}（置信度:{confidence:.2f}）"

    def _assess_path_quality(self, score: float, consistency: float, depth: int) -> str:
        """评估路径质量"""
        quality_score = score * 0.6 + consistency * 0.3 + min(depth / 3, 1.0) * 0.1

        if quality_score > 0.8:
            return "高质量：证据链完整，一致性强"
        elif quality_score > 0.6:
            return "良好：证据链较为完整，一致性较好"
        elif quality_score > 0.4:
            return "中等：证据链基本完整，一致性一般"
        else:
            return "较低：证据链不完整，一致性较差"

    def _task_relevance_score(self, row: Dict[str, object], task_profile: Dict[str, object]) -> float:
        """任务相关性评分"""
        if not task_profile.get("preferred_relations"):
            return 0.0

        # 检查 chunk 是否包含任务偏好的关系类型
        conn = self._connect()
        try:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT relation_type, confidence
                FROM kb_links
                WHERE chunk_id = ?
                """,
                (row["id"],),
            ).fetchall()
        finally:
            conn.close()

        if not rows:
            return 0.0

        preferred = task_profile["preferred_relations"]
        score = 0.0
        for r in rows:
            if r["relation_type"] in preferred:
                score += r["confidence"]

        return min(score, 1.0)

    def _condition_match_score(self, row: Dict[str, object], parsed_query: Dict[str, object]) -> float:
        """条件匹配评分"""
        if not parsed_query.get("conditions"):
            return 0.0

        text = row["text"].lower()
        matches = 0
        for condition in parsed_query["conditions"]:
            if condition.get("type") == "numeric":
                if condition["value"] in text:
                    matches += 1

        return matches / max(len(parsed_query["conditions"]), 1)

    def _relation_relevance_score(
        self,
        row: Dict[str, object],
        parsed_query: Dict[str, object],
        task_profile: Dict[str, object],
    ) -> float:
        """关系相关性评分"""
        score = 0.0

        # 检查关系类型是否匹配任务偏好
        if task_profile.get("preferred_relations"):
            if row["relation_type"] in task_profile["preferred_relations"]:
                score += 0.3

        # 检查因子是否匹配查询实体
        entities = parsed_query.get("entities", [])
        for entity in entities:
            if entity in [row["process_factor"], row["morphology_factor"], row["performance_factor"]]:
                score += 0.2

        # 检查方向是否一致
        query_direction = parsed_query.get("direction")
        if query_direction and row["effect_direction"]:
            if query_direction == row["effect_direction"]:
                score += 0.1

        # 置信度加权
        confidence = row["confidence"] if "confidence" in row.keys() else 0.5
        score *= confidence

        return min(score, 1.0)

    def _path_consistency_score(self, path: Dict[str, object]) -> float:
        """路径一致性评分"""
        if len(path["relations"]) < 2:
            return 1.0

        consistency = 0.0
        for i in range(len(path["relations"]) - 1):
            rel1 = path["relations"][i]
            rel2 = path["relations"][i + 1]

            # 检查是否符合关系转换矩阵
            allowed_transitions = self.RELATION_TRANSITION_MATRIX.get(
                rel1.get("type"), []
            )
            if rel2.get("type") in allowed_transitions:
                consistency += 1.0

        return consistency / max(len(path["relations"]) - 1, 1)

    def _direction_consistency_score(self, path: Dict[str, object], parsed_query: Dict[str, object]) -> float:
        """方向一致性评分"""
        if not parsed_query.get("direction"):
            return 1.0

        query_direction = parsed_query["direction"]
        consistent_count = 0
        total_count = 0

        for rel in path["relations"]:
            if rel.get("effect_direction"):
                total_count += 1
                if rel["effect_direction"] == query_direction:
                    consistent_count += 1

        if total_count == 0:
            return 1.0

        return consistent_count / total_count

    def get_stats(self) -> Dict[str, object]:
        conn = self._connect()
        try:
            doc_count = conn.execute("SELECT COUNT(*) FROM kb_documents").fetchone()[0]
            chunk_count = conn.execute("SELECT COUNT(*) FROM kb_chunks").fetchone()[0]
            core_count = conn.execute(
                "SELECT COUNT(*) FROM kb_documents WHERE is_core = 1"
            ).fetchone()[0]
            link_count = conn.execute("SELECT COUNT(*) FROM kb_links").fetchone()[0]
            relation_rows = conn.execute(
                "SELECT relation_type, COUNT(*) FROM kb_links GROUP BY relation_type"
            ).fetchall()
            source_type_rows = conn.execute(
                "SELECT source_type, COUNT(*) FROM kb_documents GROUP BY source_type"
            ).fetchall()
            source_doc_rows = conn.execute(
                "SELECT source_type, title, file_path FROM kb_documents"
            ).fetchall()
        finally:
            conn.close()

        relation_counts = {
            "process_to_morphology": 0,
            "morphology_to_performance": 0,
            "process_to_performance": 0,
            "process_to_mechanism": 0,
            "mechanism_to_morphology": 0,
            "mechanism_evidence": 0,
        }
        for rel_type, rel_count in relation_rows:
            key = str(rel_type or "")
            if key in relation_counts:
                relation_counts[key] = int(rel_count)

        source_type_counts: Dict[str, int] = {}
        for source_type, count in source_type_rows:
            key = str(source_type or "").strip() or "unknown"
            source_type_counts[key] = int(count)
        source_type_counts = dict(
            sorted(source_type_counts.items(), key=lambda item: (-item[1], item[0]))
        )

        domain_counter: Counter[str] = Counter()
        for source_type, title, file_path in source_doc_rows:
            domain_key = self._infer_source_domain(
                source_type=str(source_type or "").strip(),
                title=str(title or ""),
                file_path=str(file_path or ""),
            )
            domain_counter[domain_key] += 1
        source_domain_counts = dict(
            sorted(domain_counter.items(), key=lambda item: (-item[1], item[0]))[:12]
        )
        candidate_stats = self._load_candidate_source_stats()

        return {
            "document_count": doc_count,
            "chunk_count": chunk_count,
            "core_document_count": core_count,
            "link_count": link_count,
            "relation_counts": relation_counts,
            "source_type_counts": source_type_counts,
            "source_domain_counts": source_domain_counts,
            "candidate_document_count": candidate_stats["candidate_document_count"],
            "candidate_download_status_counts": candidate_stats["candidate_download_status_counts"],
            "candidate_source_domain_counts": candidate_stats["candidate_source_domain_counts"],
        }

    @staticmethod
    def _candidate_csv_path() -> str:
        default_csv = os.path.normpath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "..",
                "RagDocument",
                "CORE",
                "13. 工艺-形貌-性能补充",
                "文献候选清单.csv",
            )
        )
        return os.getenv("CNTA_LITERATURE_CANDIDATE_CSV", default_csv)

    @classmethod
    def _load_candidate_source_stats(cls) -> Dict[str, object]:
        csv_path = cls._candidate_csv_path()
        if not os.path.exists(csv_path):
            return {
                "candidate_document_count": 0,
                "candidate_download_status_counts": {},
                "candidate_source_domain_counts": {},
            }

        rows = []
        for encoding in ("utf-8-sig", "utf-8", "gb18030"):
            try:
                with open(csv_path, "r", encoding=encoding, newline="") as handle:
                    rows = list(csv.DictReader(handle))
                break
            except UnicodeDecodeError:
                continue
        if not rows:
            return {
                "candidate_document_count": 0,
                "candidate_download_status_counts": {},
                "candidate_source_domain_counts": {},
            }

        status_counter: Counter[str] = Counter()
        domain_counter: Counter[str] = Counter()
        for row in rows:
            status = str((row or {}).get("download_status", "")).strip() or "unknown"
            status_counter[status] += 1
            source_url = str((row or {}).get("source_url", "")).strip()
            domain = cls._domain_from_url(source_url) or "unknown"
            domain_counter[domain] += 1

        return {
            "candidate_document_count": len(rows),
            "candidate_download_status_counts": dict(
                sorted(status_counter.items(), key=lambda item: (-item[1], item[0]))
            ),
            "candidate_source_domain_counts": dict(
                sorted(domain_counter.items(), key=lambda item: (-item[1], item[0]))[:12]
            ),
        }

    @staticmethod
    def _domain_from_url(raw_value: str) -> Optional[str]:
        value = (raw_value or "").strip()
        if not value:
            return None
        parsed = urlparse(value)
        host = (parsed.netloc or "").lower().strip()
        if not host and value.startswith("www."):
            host = value.lower().split("/")[0]
        if not host:
            return None
        if host.startswith("www."):
            host = host[4:]
        return host or None

    @classmethod
    def _infer_source_domain(cls, source_type: str, title: str, file_path: str) -> str:
        for candidate in (file_path, title):
            domain = cls._domain_from_url(candidate)
            if domain:
                return domain

        normalized_type = (source_type or "").strip().lower() or "unknown"
        return f"local_{normalized_type}"

    def rebuild_links(self, doc_ids: Optional[List[int]] = None, clear_existing: bool = True) -> Dict[str, int]:
        conn = self._connect()
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            if clear_existing:
                if doc_ids:
                    placeholders = ",".join("?" for _ in doc_ids)
                    cursor.execute(
                        f"DELETE FROM kb_links WHERE doc_id IN ({placeholders})",
                        tuple(doc_ids),
                    )
                else:
                    cursor.execute("DELETE FROM kb_links")

            if doc_ids:
                placeholders = ",".join("?" for _ in doc_ids)
                chunk_rows = cursor.execute(
                    f"""
                    SELECT id as chunk_id, doc_id, text
                    FROM kb_chunks
                    WHERE doc_id IN ({placeholders})
                    ORDER BY doc_id, chunk_index
                    """,
                    tuple(doc_ids),
                ).fetchall()
            else:
                chunk_rows = cursor.execute(
                    """
                    SELECT id as chunk_id, doc_id, text
                    FROM kb_chunks
                    ORDER BY doc_id, chunk_index
                    """
                ).fetchall()

            link_count = 0
            doc_set = set()
            for row in chunk_rows:
                doc_id = int(row["doc_id"])
                chunk_id = int(row["chunk_id"])
                doc_set.add(doc_id)
                for relation in self._extract_relations_from_chunk(row["text"] or ""):
                    cursor.execute(
                        """
                        INSERT INTO kb_links (
                            doc_id, chunk_id, relation_type, source_node, target_node,
                            process_factor, morphology_factor, performance_factor,
                            effect_direction, confidence, mechanism_summary, evidence_text
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            doc_id,
                            chunk_id,
                            relation.get("relation_type"),
                            row["source_node"],
                            row["target_node"],
                            row["process_factor"],
                            row["morphology_factor"],
                            row["performance_factor"],
                            row["effect_direction"],
                            row["confidence"] if row["confidence"] is not None else 0.5,
                            row["mechanism_summary"],
                            row["evidence_text"] if row["evidence_text"] is not None else "",
                        ),
                    )
                    link_count += 1

            conn.commit()
            return {
                "doc_count": len(doc_set),
                "link_count": link_count,
            }
        finally:
            conn.close()
            self._invalidate_semantic_cache()

    def _load_link_rows(self) -> List[Dict[str, object]]:
        conn = self._connect()
        try:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT l.id, l.process_factor, l.morphology_factor, l.performance_factor,
                       l.relation_type, l.source_node, l.target_node, l.effect_direction,
                       l.confidence, l.mechanism_summary, l.evidence_text,
                       d.id AS doc_id, d.title, d.theme
                FROM kb_links l
                JOIN kb_documents d ON d.id = l.doc_id
                """
            ).fetchall()
        finally:
            conn.close()
        return [dict(row) for row in rows]

    def search_links(self, query: str, top_k: int = 5) -> List[Dict[str, object]]:
        query_tokens = self._expand_link_query_tokens(query)
        if not query_tokens:
            return []

        scored = []
        profile = self._build_query_relation_profile(query)
        for row_dict in self._load_link_rows():
            haystack = " ".join(
                str(row_dict.get(key) or "")
                for key in (
                    "process_factor",
                    "morphology_factor",
                    "performance_factor",
                    "relation_type",
                    "source_node",
                    "target_node",
                    "effect_direction",
                    "mechanism_summary",
                    "evidence_text",
                    "theme",
                    "title",
                )
            ).lower()
            token_hits = sum(1 for token in query_tokens if token in haystack)
            if token_hits <= 0:
                continue

            relation_boost = self._score_link_against_query_profile(row_dict, profile)
            confidence = float(row_dict.get("confidence") or 0.0)
            final_score = float(token_hits) + relation_boost + confidence * 0.35
            enriched = {**row_dict, "_match_score": round(final_score, 4)}
            scored.append((final_score, enriched))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [row for _, row in scored[:top_k]]

    def _expand_link_query_tokens(self, query: str) -> set:
        tokens = set(self._tokenize(query))
        if not query:
            return tokens

        for factor_patterns in (
            PROCESS_FACTOR_PATTERNS,
            MORPHOLOGY_FACTOR_PATTERNS,
            PERFORMANCE_FACTOR_PATTERNS,
            MECHANISM_FACTOR_PATTERNS,
        ):
            tokens.update(self._match_factors(query, factor_patterns))

        profile = self._build_query_relation_profile(query)
        tokens.update(profile.get("relation_types") or set())
        if profile.get("process_hits"):
            tokens.update({"process", "process_factor"})
        if profile.get("morph_hits"):
            tokens.update({"morphology", "morphology_factor"})
        if profile.get("perf_hits"):
            tokens.update({"performance", "performance_factor"})
        if profile.get("mechanism_hits") or any(
            re.search(pattern, query.lower()) for pattern in MECHANISM_PATTERNS
        ):
            tokens.update({"mechanism", "mechanism_summary", "evidence", "literature"})
        return tokens

    def get_relation_chain_summary(self, query: str, top_k: int = 20) -> Dict[str, List[Dict[str, object]]]:
        links = self.search_links(query, top_k=top_k)
        grouped = {
            "process_to_morphology": [],
            "morphology_to_performance": [],
            "process_to_performance": [],
            "process_to_mechanism": [],
            "mechanism_to_morphology": [],
            "mechanism_evidence": [],
        }
        for row in links:
            rel_type = str(row.get("relation_type") or "")
            if rel_type in grouped:
                grouped[rel_type].append(row)
        return grouped

    @staticmethod
    def _node_suffix(node_id: str) -> str:
        raw = str(node_id or "")
        if ":" not in raw:
            return raw
        return raw.split(":", 1)[1]

    @staticmethod
    def _node_category(node_id: str) -> str:
        raw = str(node_id or "")
        if ":" not in raw:
            return "other"
        return raw.split(":", 1)[0]

    def retrieve_local_relation_subgraph(
        self,
        query: str,
        top_k: int = 20,
        max_hops: int = 1,
        max_expanded_edges: int = 60,
    ) -> Dict[str, object]:
        seeds = self.search_links(query, top_k=top_k)
        if not seeds:
            return {
                "query": query,
                "seed_count": 0,
                "edge_count": 0,
                "node_count": 0,
                "edges": [],
                "nodes": [],
                "relation_type_counts": {},
            }

        profile = self._build_query_relation_profile(query)
        allowed_relation_types: Set[str] = set(profile.get("relation_types") or [])
        if not allowed_relation_types:
            allowed_relation_types = {
                "process_to_morphology",
                "morphology_to_performance",
                "process_to_performance",
                "process_to_mechanism",
                "mechanism_to_morphology",
                "mechanism_evidence",
            }

        all_rows = self._load_link_rows()
        adjacency: Dict[str, List[Dict[str, object]]] = {}
        for row in all_rows:
            source_node = str(row.get("source_node") or "")
            target_node = str(row.get("target_node") or "")
            if source_node:
                adjacency.setdefault(source_node, []).append(row)
            if target_node:
                adjacency.setdefault(target_node, []).append(row)

        selected_edges: Dict[int, Dict[str, object]] = {}
        visited_nodes: Set[str] = set()
        frontier: Set[str] = set()
        for row in seeds:
            selected_edges[int(row["id"])] = row
            source_node = str(row.get("source_node") or "")
            target_node = str(row.get("target_node") or "")
            if source_node:
                frontier.add(source_node)
                visited_nodes.add(source_node)
            if target_node:
                frontier.add(target_node)
                visited_nodes.add(target_node)

        for _ in range(max(0, int(max_hops))):
            next_frontier: Set[str] = set()
            for node in list(frontier):
                for candidate in adjacency.get(node, []):
                    relation_type = str(candidate.get("relation_type") or "")
                    if relation_type not in allowed_relation_types:
                        continue
                    link_score = self._score_link_against_query_profile(candidate, profile)
                    if link_score <= 0 and relation_type not in profile.get("relation_types", set()):
                        continue

                    candidate_id = int(candidate["id"])
                    if candidate_id not in selected_edges:
                        selected_edges[candidate_id] = {
                            **candidate,
                            "_match_score": round(link_score, 4),
                        }

                    source_node = str(candidate.get("source_node") or "")
                    target_node = str(candidate.get("target_node") or "")
                    if source_node and source_node not in visited_nodes:
                        next_frontier.add(source_node)
                    if target_node and target_node not in visited_nodes:
                        next_frontier.add(target_node)

                    if len(selected_edges) >= max_expanded_edges:
                        break
                if len(selected_edges) >= max_expanded_edges:
                    break
            if not next_frontier:
                break
            visited_nodes.update(next_frontier)
            frontier = next_frontier

        edges = list(selected_edges.values())
        edges.sort(
            key=lambda row: (
                float(row.get("_match_score") or 0.0),
                float(row.get("confidence") or 0.0),
            ),
            reverse=True,
        )
        edges = edges[: max(1, int(max_expanded_edges))]

        relation_counter: Counter[str] = Counter()
        degree_counter: Counter[str] = Counter()
        for row in edges:
            relation_counter[str(row.get("relation_type") or "unknown")] += 1
            source_node = str(row.get("source_node") or "")
            target_node = str(row.get("target_node") or "")
            if source_node:
                degree_counter[source_node] += 1
            if target_node:
                degree_counter[target_node] += 1

        nodes = [
            {
                "id": node_id,
                "category": self._node_category(node_id),
                "label": self._node_suffix(node_id),
                "degree": degree_counter[node_id],
            }
            for node_id in sorted(degree_counter.keys(), key=lambda value: (-degree_counter[value], value))
        ]

        return {
            "query": query,
            "seed_count": len(seeds),
            "edge_count": len(edges),
            "node_count": len(nodes),
            "edges": edges,
            "nodes": nodes,
            "relation_type_counts": dict(
                sorted(relation_counter.items(), key=lambda item: (-item[1], item[0]))
            ),
        }

    def generate_constrained_evidence_chain(
        self,
        query: str,
        top_k: int = 5,
        min_confidence: float = 0.45,
    ) -> Dict[str, object]:
        links = self.search_links(query, top_k=max(40, top_k * 20))
        if not links:
            return {"query": query, "path_type": "process_mechanism_morphology", "items": []}

        p2m = [
            row
            for row in links
            if str(row.get("relation_type") or "") == "process_to_mechanism"
            and float(row.get("confidence") or 0.0) >= min_confidence
        ]
        m2m = [
            row
            for row in links
            if str(row.get("relation_type") or "") == "mechanism_to_morphology"
            and float(row.get("confidence") or 0.0) >= min_confidence
        ]
        p2morph = [
            row
            for row in links
            if str(row.get("relation_type") or "") == "process_to_morphology"
            and float(row.get("confidence") or 0.0) >= min_confidence
        ]

        m2m_by_mechanism: Dict[str, List[Dict[str, object]]] = {}
        for row in m2m:
            mech = self._node_suffix(str(row.get("source_node") or ""))
            if mech:
                m2m_by_mechanism.setdefault(mech, []).append(row)

        p2morph_index: Dict[Tuple[str, str], List[Dict[str, object]]] = {}
        for row in p2morph:
            process_key = str(row.get("process_factor") or self._node_suffix(str(row.get("source_node") or "")))
            morph_key = str(row.get("morphology_factor") or self._node_suffix(str(row.get("target_node") or "")))
            if not process_key or not morph_key:
                continue
            p2morph_index.setdefault((process_key, morph_key), []).append(row)

        chain_items = []
        seen_paths: Set[Tuple[str, str, str]] = set()
        for row_a in p2m:
            process_factor = str(row_a.get("process_factor") or self._node_suffix(str(row_a.get("source_node") or "")))
            mechanism_factor = self._node_suffix(str(row_a.get("target_node") or ""))
            if not process_factor or not mechanism_factor:
                continue

            for row_b in m2m_by_mechanism.get(mechanism_factor, []):
                morphology_factor = str(
                    row_b.get("morphology_factor") or self._node_suffix(str(row_b.get("target_node") or ""))
                )
                if not morphology_factor:
                    continue
                signature = (process_factor, mechanism_factor, morphology_factor)
                if signature in seen_paths:
                    continue
                seen_paths.add(signature)

                support_links = p2morph_index.get((process_factor, morphology_factor), [])
                support_link = support_links[0] if support_links else None
                scores = [
                    float(row_a.get("confidence") or 0.0),
                    float(row_b.get("confidence") or 0.0),
                ]
                if support_link is not None:
                    scores.append(float(support_link.get("confidence") or 0.0))
                score = sum(scores) / max(1, len(scores))

                chain_items.append(
                    {
                        "process_factor": process_factor,
                        "mechanism_factor": mechanism_factor,
                        "morphology_factor": morphology_factor,
                        "score": round(score, 4),
                        "steps": [
                            {
                                "relation_type": "process_to_mechanism",
                                "source_node": row_a.get("source_node"),
                                "target_node": row_a.get("target_node"),
                                "effect_direction": row_a.get("effect_direction"),
                                "confidence": row_a.get("confidence"),
                                "title": row_a.get("title"),
                                "evidence_text": row_a.get("evidence_text"),
                            },
                            {
                                "relation_type": "mechanism_to_morphology",
                                "source_node": row_b.get("source_node"),
                                "target_node": row_b.get("target_node"),
                                "effect_direction": row_b.get("effect_direction"),
                                "confidence": row_b.get("confidence"),
                                "title": row_b.get("title"),
                                "evidence_text": row_b.get("evidence_text"),
                            },
                        ],
                        "support": (
                            {
                                "relation_type": "process_to_morphology",
                                "source_node": support_link.get("source_node"),
                                "target_node": support_link.get("target_node"),
                                "effect_direction": support_link.get("effect_direction"),
                                "confidence": support_link.get("confidence"),
                                "title": support_link.get("title"),
                                "evidence_text": support_link.get("evidence_text"),
                            }
                            if support_link is not None
                            else None
                        ),
                    }
                )

        chain_items.sort(key=lambda item: item["score"], reverse=True)
        return {
            "query": query,
            "path_type": "process_mechanism_morphology",
            "items": chain_items[: max(1, top_k)],
        }

    def get_theme_level_aggregation(self, query: str, top_k: int = 30) -> Dict[str, object]:
        links = self.search_links(query, top_k=max(top_k, 20))
        bucket_map = {
            "process_to_morphology": "process_morphology",
            "process_to_mechanism": "process_mechanism",
            "morphology_to_performance": "morphology_performance",
        }
        grouped: Dict[str, Dict[Tuple[str, str, str], Dict[str, object]]] = {
            "process_morphology": {},
            "process_mechanism": {},
            "morphology_performance": {},
        }

        for row in links:
            relation_type = str(row.get("relation_type") or "")
            bucket = bucket_map.get(relation_type)
            if not bucket:
                continue
            source_node = str(row.get("source_node") or "")
            target_node = str(row.get("target_node") or "")
            effect_direction = str(row.get("effect_direction") or "unknown")
            key = (source_node, target_node, effect_direction)
            unit = grouped[bucket].get(key)
            if unit is None:
                unit = {
                    "source_node": source_node,
                    "target_node": target_node,
                    "effect_direction": effect_direction,
                    "count": 0,
                    "confidence_sum": 0.0,
                    "titles": set(),
                }
                grouped[bucket][key] = unit
            unit["count"] += 1
            unit["confidence_sum"] += float(row.get("confidence") or 0.0)
            title = str(row.get("title") or "").strip()
            if title:
                unit["titles"].add(title)

        def _finalize_bucket(bucket_name: str) -> Dict[str, object]:
            records = []
            for unit in grouped[bucket_name].values():
                count = int(unit["count"])
                avg_conf = float(unit["confidence_sum"]) / max(1, count)
                records.append(
                    {
                        "source_node": unit["source_node"],
                        "target_node": unit["target_node"],
                        "effect_direction": unit["effect_direction"],
                        "count": count,
                        "avg_confidence": round(avg_conf, 4),
                        "evidence_titles": sorted(unit["titles"])[:3],
                    }
                )
            records.sort(key=lambda item: (item["count"], item["avg_confidence"]), reverse=True)
            top_records = records[:5]
            summary = (
                f"{bucket_name} has {len(records)} grouped relation themes; top relation count="
                f"{top_records[0]['count'] if top_records else 0}."
            )
            return {"items": top_records, "summary": summary}

        return {
            "query": query,
            "process_morphology": _finalize_bucket("process_morphology"),
            "process_mechanism": _finalize_bucket("process_mechanism"),
            "morphology_performance": _finalize_bucket("morphology_performance"),
        }

    def ingest_directory(
        self,
        source_dir: str,
        source_type: str,
        theme: Optional[str] = None,
        is_core: bool = False,
        allowed_extensions: Optional[List[str]] = None,
    ) -> Dict[str, int]:
        extensions = [ext.lower() for ext in (allowed_extensions or [".txt", ".pdf"])]
        imported = 0
        for root, _, files in os.walk(source_dir):
            for entry in sorted(files):
                path = os.path.join(root, entry)
                if os.path.splitext(entry)[1].lower() not in extensions:
                    continue
                self.ingest_file(
                    file_path=path,
                    source_type=source_type,
                    theme=theme,
                    is_core=is_core,
                )
                imported += 1
        return {"document_count": imported}

    def ingest_file(
        self,
        file_path: str,
        source_type: str,
        theme: Optional[str] = None,
        is_core: bool = False,
    ) -> Dict[str, int]:
        extension = os.path.splitext(file_path)[1].lower()
        if extension == ".pdf":
            text = self._extract_text_from_pdf(file_path)
        else:
            with open(file_path, "r", encoding="utf-8") as handle:
                text = handle.read()

        return self.ingest_text(
            title=os.path.splitext(os.path.basename(file_path))[0],
            text=text,
            source_type=source_type,
            theme=theme,
            is_core=is_core,
            file_path=file_path,
            language="zh" if re.search(r"[\u4e00-\u9fff]", text) else "en",
        )

    def _get_task_profile(self, task_name: Optional[str]) -> Optional[Dict[str, str]]:
        if not task_name:
            return None
        conn = self._connect()
        try:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                """
                SELECT task_name, preferred_theme, preferred_themes, discouraged_themes,
                       preferred_knowledge_types, preferred_keywords
                FROM kb_task_profiles
                WHERE task_name = ?
                """,
                (task_name,),
            ).fetchone()
        finally:
            conn.close()
        return dict(row) if row else None

    def _score_row(
        self,
        row: Dict[str, object],
        query_tokens: List[str],
        task_profile: Optional[Dict[str, str]],
    ) -> float:
        text_tokens = set(self._tokenize(row.get("text", "")))
        keyword_tokens = set((row.get("keywords") or "").split())
        score = 0.0

        for token in query_tokens:
            if token in text_tokens:
                score += 2.0
            if token in keyword_tokens:
                score += 3.0

        if row.get("is_core"):
            score += 1.5

        if task_profile:
            preferred_theme = task_profile.get("preferred_theme")
            if preferred_theme and row.get("theme") == preferred_theme:
                score += 4.0

            preferred_themes = {
                item.strip()
                for item in (task_profile.get("preferred_themes") or "").split(",")
                if item.strip()
            }
            if row.get("theme") in preferred_themes:
                score += 2.5

            discouraged_themes = {
                item.strip()
                for item in (task_profile.get("discouraged_themes") or "").split(",")
                if item.strip()
            }
            if row.get("theme") in discouraged_themes:
                score -= 2.0

            preferred_types = set(
                self._tokenize(task_profile.get("preferred_knowledge_types", ""))
            )
            if row.get("knowledge_type") in preferred_types:
                score += 2.0

            preferred_keywords = set(
                self._tokenize(task_profile.get("preferred_keywords", ""))
            )
            score += len(preferred_keywords.intersection(text_tokens.union(keyword_tokens))) * 0.5

        return score

    @staticmethod
    def _split_text(text: str, chunk_size: int = 500, overlap: int = 80) -> List[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(text[start:end])
            if end == len(text):
                break
            start += chunk_size - overlap
        return chunks

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        tokens = re.findall(r"[a-zA-Z]+|[\u4e00-\u9fff]+", (text or "").lower())
        stopwords = {
            "the", "and", "for", "with", "from", "into", "that", "this",
            "of", "to", "in", "on", "is", "are", "by", "or", "a", "an",
            "的", "了", "和", "与", "及", "在", "对", "中", "为", "是",
        }
        return [token for token in tokens if token not in stopwords and len(token) > 1]

    def _extract_keywords(self, text: str, top_n: int = 20) -> List[str]:
        frequency = {}
        for token in self._tokenize(text):
            frequency[token] = frequency.get(token, 0) + 1
        return sorted(frequency, key=lambda token: (-frequency[token], token))[:top_n]

    def _semantic_scores(self, query: str, rows: List[Dict[str, object]]) -> Dict[int, float]:
        if not query or not rows:
            return {}

        model = self._get_embedding_model()
        if model is None:
            return {}

        cache = self._ensure_semantic_cache(rows, model)
        if cache is None:
            return {}

        query_embedding = self._encode_texts(model, [query])
        if query_embedding.size == 0:
            return {}

        query_vector = query_embedding[0]
        similarities = cache["embeddings"] @ query_vector
        return {
            int(row["id"]): float(score)
            for row, score in zip(cache["rows"], similarities)
        }

    def _relation_constraint_scores(
        self,
        query: str,
        rows: List[Dict[str, object]],
    ) -> Dict[int, float]:
        if not query or not rows:
            return {}

        profile = self._build_query_relation_profile(query)
        if not profile["relation_types"] and not any(
            profile[key] for key in ("process_hits", "morph_hits", "perf_hits", "mechanism_hits")
        ):
            return {}

        doc_ids = sorted({int(row["doc_id"]) for row in rows})
        if not doc_ids:
            return {}

        placeholders = ",".join("?" for _ in doc_ids)
        conn = self._connect()
        try:
            conn.row_factory = sqlite3.Row
            link_rows = conn.execute(
                f"""
                SELECT doc_id, relation_type, process_factor, morphology_factor,
                       performance_factor, source_node, target_node, confidence
                FROM kb_links
                WHERE doc_id IN ({placeholders})
                """,
                tuple(doc_ids),
            ).fetchall()
        finally:
            conn.close()

        scores = {doc_id: 0.0 for doc_id in doc_ids}
        matched_relation_types: Dict[int, set] = {doc_id: set() for doc_id in doc_ids}
        for row in link_rows:
            doc_id = int(row["doc_id"])
            link_score = self._score_link_against_query_profile(dict(row), profile)
            if link_score <= 0:
                continue
            scores[doc_id] += link_score
            relation_type = str(row["relation_type"] or "")
            if relation_type:
                matched_relation_types[doc_id].add(relation_type)

        for doc_id, rel_types in matched_relation_types.items():
            if rel_types:
                scores[doc_id] += min(len(rel_types) * 0.35, 1.4)
        return scores

    def _build_query_relation_profile(self, query: str) -> Dict[str, object]:
        process_hits = self._match_factors(query, PROCESS_FACTOR_PATTERNS)
        morph_hits = self._match_factors(query, MORPHOLOGY_FACTOR_PATTERNS)
        perf_hits = self._match_factors(query, PERFORMANCE_FACTOR_PATTERNS)
        mechanism_hits = self._match_factors(query, MECHANISM_FACTOR_PATTERNS)

        relation_types = set()
        if process_hits and morph_hits:
            relation_types.add("process_to_morphology")
        if process_hits and perf_hits:
            relation_types.add("process_to_performance")
        if morph_hits and perf_hits:
            relation_types.add("morphology_to_performance")
        if process_hits and mechanism_hits:
            relation_types.add("process_to_mechanism")
        if mechanism_hits and morph_hits:
            relation_types.add("mechanism_to_morphology")
        if mechanism_hits:
            relation_types.add("mechanism_evidence")

        return {
            "process_hits": set(process_hits),
            "morph_hits": set(morph_hits),
            "perf_hits": set(perf_hits),
            "mechanism_hits": set(mechanism_hits),
            "relation_types": relation_types,
        }

    @staticmethod
    def _score_link_against_query_profile(
        link: Dict[str, object],
        profile: Dict[str, object],
    ) -> float:
        score = 0.0
        relation_type = str(link.get("relation_type") or "")
        if relation_type in profile["relation_types"]:
            score += 1.0

        process_factor = str(link.get("process_factor") or "")
        if process_factor and process_factor in profile["process_hits"]:
            score += 0.7

        morphology_factor = str(link.get("morphology_factor") or "")
        if morphology_factor and morphology_factor in profile["morph_hits"]:
            score += 0.7

        performance_factor = str(link.get("performance_factor") or "")
        if performance_factor and performance_factor in profile["perf_hits"]:
            score += 0.7

        mechanism_text = " ".join(
            str(link.get(key) or "") for key in ("source_node", "target_node")
        )
        if any(hit in mechanism_text for hit in profile["mechanism_hits"]):
            score += 0.8

        confidence = float(link.get("confidence") or 0.0)
        if score > 0:
            score *= 0.7 + min(confidence, 1.0) * 0.3
        return score

    def _get_embedding_model(self):
        if self._embedding_disabled_reason:
            return None
        if self._embedding_model is not None:
            return self._embedding_model
        try:
            from sentence_transformers import SentenceTransformer

            self._embedding_model = SentenceTransformer(
                self.embedding_model_name,
                local_files_only=True,
            )
            return self._embedding_model
        except Exception as exc:
            self._embedding_disabled_reason = str(exc)
            return None

    def _ensure_semantic_cache(self, rows: List[Dict[str, object]], model) -> Optional[Dict[str, object]]:
        fingerprint = self._semantic_fingerprint(rows)
        if self._semantic_cache and self._semantic_cache.get("fingerprint") == fingerprint:
            return self._semantic_cache

        texts = [self._compose_semantic_text(row) for row in rows]
        embeddings = self._encode_texts(model, texts)
        if embeddings.size == 0:
            return None

        self._semantic_cache = {
            "fingerprint": fingerprint,
            "rows": rows,
            "embeddings": embeddings,
        }
        return self._semantic_cache

    @staticmethod
    def _encode_texts(model, texts: List[str]) -> np.ndarray:
        vectors = np.asarray(
            model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ),
            dtype=float,
        )
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        return vectors

    @staticmethod
    def _compose_semantic_text(row: Dict[str, object]) -> str:
        parts = [
            str(row.get("title") or ""),
            str(row.get("theme") or ""),
            str(row.get("knowledge_type") or ""),
            str(row.get("keywords") or ""),
            str(row.get("text") or "")[:800],
        ]
        return " ".join(part for part in parts if part).strip()

    @staticmethod
    def _semantic_fingerprint(rows: List[Dict[str, object]]) -> tuple:
        ids = [int(row["id"]) for row in rows]
        return (
            len(rows),
            ids[0] if ids else -1,
            ids[-1] if ids else -1,
            sum(ids),
            sum(len(str(row.get("text") or "")) for row in rows),
        )

    def _extract_relations_from_chunk(self, chunk: str) -> List[Dict[str, str]]:
        if not chunk:
            return []
        sentences = [
            item.strip()
            for item in re.split(r"[。！？.!?;\n]+", chunk)
            if item and item.strip()
        ]
        relations: List[Dict[str, str]] = []
        for sentence in sentences:
            process_hits = self._match_factors(sentence, PROCESS_FACTOR_PATTERNS)
            morph_hits = self._match_factors(sentence, MORPHOLOGY_FACTOR_PATTERNS)
            perf_hits = self._match_factors(sentence, PERFORMANCE_FACTOR_PATTERNS)
            mechanism_hits = self._match_factors(sentence, MECHANISM_FACTOR_PATTERNS)

            direction = self._detect_effect_direction(sentence)
            if not direction:
                continue

            has_mechanism = bool(mechanism_hits) or any(
                re.search(pattern, sentence.lower()) for pattern in MECHANISM_PATTERNS
            )
            base_confidence = 0.55 + 0.1 * int(has_mechanism)

            for process_factor in process_hits:
                for morphology_factor in morph_hits:
                    relations.append(
                        {
                            "relation_type": "process_to_morphology",
                            "source_node": f"process:{process_factor}",
                            "target_node": f"morphology:{morphology_factor}",
                            "process_factor": process_factor,
                            "morphology_factor": morphology_factor,
                            "performance_factor": None,
                            "effect_direction": direction,
                            "confidence": min(base_confidence, 0.95),
                            "mechanism_summary": sentence[:220],
                            "evidence_text": sentence[:320],
                        }
                    )

            for process_factor in process_hits:
                for mechanism_factor in mechanism_hits:
                    relations.append(
                        {
                            "relation_type": "process_to_mechanism",
                            "source_node": f"process:{process_factor}",
                            "target_node": f"mechanism:{mechanism_factor}",
                            "process_factor": process_factor,
                            "morphology_factor": None,
                            "performance_factor": None,
                            "effect_direction": direction,
                            "confidence": min(base_confidence + 0.08, 0.95),
                            "mechanism_summary": sentence[:220],
                            "evidence_text": sentence[:320],
                        }
                    )

            for mechanism_factor in mechanism_hits:
                for morphology_factor in morph_hits:
                    relations.append(
                        {
                            "relation_type": "mechanism_to_morphology",
                            "source_node": f"mechanism:{mechanism_factor}",
                            "target_node": f"morphology:{morphology_factor}",
                            "process_factor": None,
                            "morphology_factor": morphology_factor,
                            "performance_factor": None,
                            "effect_direction": direction,
                            "confidence": min(base_confidence + 0.08, 0.95),
                            "mechanism_summary": sentence[:220],
                            "evidence_text": sentence[:320],
                        }
                    )

            for morphology_factor in morph_hits:
                for performance_factor in perf_hits:
                    relations.append(
                        {
                            "relation_type": "morphology_to_performance",
                            "source_node": f"morphology:{morphology_factor}",
                            "target_node": f"performance:{performance_factor}",
                            "process_factor": None,
                            "morphology_factor": morphology_factor,
                            "performance_factor": performance_factor,
                            "effect_direction": direction,
                            "confidence": min(base_confidence + 0.05, 0.95),
                            "mechanism_summary": sentence[:220],
                            "evidence_text": sentence[:320],
                        }
                    )

            for process_factor in process_hits:
                for performance_factor in perf_hits:
                    relations.append(
                        {
                            "relation_type": "process_to_performance",
                            "source_node": f"process:{process_factor}",
                            "target_node": f"performance:{performance_factor}",
                            "process_factor": process_factor,
                            "morphology_factor": None,
                            "performance_factor": performance_factor,
                            "effect_direction": direction,
                            "confidence": min(base_confidence + 0.03, 0.95),
                            "mechanism_summary": sentence[:220],
                            "evidence_text": sentence[:320],
                        }
                    )

            if has_mechanism and (process_hits or morph_hits or perf_hits):
                # mechanism_evidence 的 source 应该总是 mechanism 类型
                source = f"mechanism:{mechanism_hits[0]}" if mechanism_hits else (
                    f"process:{process_hits[0]}"
                    if process_hits
                    else f"morphology:{morph_hits[0]}" if morph_hits else f"performance:{perf_hits[0]}"
                )
                # 注意：mechanism_evidence 的因子字段应保持为 None，因为 source 是 mechanism 类型
                relations.append(
                    {
                        "relation_type": "mechanism_evidence",
                        "source_node": source,
                        "target_node": "evidence:literature",
                        "process_factor": process_hits[0] if process_hits else None,
                        "morphology_factor": morph_hits[0] if morph_hits else None,
                        "performance_factor": perf_hits[0] if perf_hits else None,
                        "effect_direction": direction,
                        "confidence": min(base_confidence + 0.1, 0.98),
                        "mechanism_summary": sentence[:220],
                        "evidence_text": sentence[:320],
                    }
                )
        return self._expand_inverse_performance_relations(relations)

    @staticmethod
    def _match_factors(text: str, factor_patterns: Dict[str, List[str]]) -> List[str]:
        lowered = text.lower()
        hits: List[str] = []
        for factor, patterns in factor_patterns.items():
            if any(re.search(pattern, lowered) for pattern in patterns):
                hits.append(factor)
        return hits

    @staticmethod
    def _detect_effect_direction(text: str) -> Optional[str]:
        lowered = text.lower()
        has_inc = any(re.search(pattern, lowered) for pattern in INCREASE_PATTERNS)
        has_dec = any(re.search(pattern, lowered) for pattern in DECREASE_PATTERNS)
        if has_inc and has_dec:
            return "nonlinear_or_tradeoff"
        if has_inc:
            return "increase"
        if has_dec:
            return "decrease"
        return None

    @classmethod
    def _expand_inverse_performance_relations(cls, relations: List[Dict[str, str]]) -> List[Dict[str, str]]:
        expanded = list(relations)
        seen = {
            (
                relation.get("relation_type"),
                relation.get("source_node"),
                relation.get("target_node"),
                relation.get("effect_direction"),
            )
            for relation in relations
        }
        for relation in relations:
            performance_factor = str(relation.get("performance_factor") or "")
            inverse_factor = INVERSE_PERFORMANCE_FACTORS.get(performance_factor)
            if not inverse_factor:
                continue
            if relation.get("relation_type") not in {"morphology_to_performance", "process_to_performance"}:
                continue

            derived_relation = {
                **relation,
                "target_node": f"performance:{inverse_factor}",
                "performance_factor": inverse_factor,
                "effect_direction": cls._invert_effect_direction(relation.get("effect_direction")),
                "confidence": min(float(relation.get("confidence") or 0.5) + 0.04, 0.9),
            }
            signature = (
                derived_relation.get("relation_type"),
                derived_relation.get("source_node"),
                derived_relation.get("target_node"),
                derived_relation.get("effect_direction"),
            )
            if signature in seen:
                continue
            seen.add(signature)
            expanded.append(derived_relation)
        return expanded

    @staticmethod
    def _invert_effect_direction(direction: Optional[str]) -> Optional[str]:
        if direction == "increase":
            return "decrease"
        if direction == "decrease":
            return "increase"
        return direction

    @staticmethod
    def _extract_text_from_pdf(file_path: str) -> str:
        try:
            import pdfplumber
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "PDF ingestion requires the 'pdfplumber' package. Install project dependencies first."
            ) from exc

        pages = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
        return "\n".join(pages)

    @staticmethod
    def _infer_knowledge_type(theme: Optional[str]) -> str:
        if not theme:
            return "general"
        if "morphology" in theme:
            return "morphology_interpretation"
        if "process" in theme:
            return "process_rule"
        if "growth" in theme or "mechanism" in theme:
            return "mechanism"
        if "prediction" in theme:
            return "prediction_support"
        return "general"
