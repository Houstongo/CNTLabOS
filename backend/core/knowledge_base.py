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
                    SELECT doc_id, text
                    FROM kb_chunks
                    WHERE doc_id IN ({placeholders})
                    ORDER BY doc_id, chunk_index
                    """,
                    tuple(doc_ids),
                ).fetchall()
            else:
                chunk_rows = cursor.execute(
                    """
                    SELECT doc_id, text
                    FROM kb_chunks
                    ORDER BY doc_id, chunk_index
                    """
                ).fetchall()

            link_count = 0
            doc_set = set()
            for row in chunk_rows:
                doc_id = int(row["doc_id"])
                doc_set.add(doc_id)
                for relation in self._extract_relations_from_chunk(row["text"] or ""):
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
                source = (
                    f"mechanism:{mechanism_hits[0]}"
                    if mechanism_hits
                    else (
                        f"process:{process_hits[0]}"
                        if process_hits
                        else (
                            f"morphology:{morph_hits[0]}"
                            if morph_hits
                            else f"performance:{perf_hits[0]}"
                        )
                    )
                )
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
