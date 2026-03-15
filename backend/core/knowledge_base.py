import os
import re
import sqlite3
from typing import Dict, List, Optional


DEFAULT_TASK_PROFILES = (
    (
        "morphology_interpretation",
        "morphology_interpretation",
        "mechanism morphology_interpretation process_morphology",
        "alignment density diameter curvature mechanism catalyst growth",
    ),
    (
        "process_analysis",
        "process_morphology",
        "process_rule process_morphology growth_mechanism",
        "temperature flow catalyst growth morphology alignment diameter density",
    ),
    (
        "prediction_explanation",
        "prediction_support",
        "process_rule prediction_support growth_mechanism",
        "prediction process morphology mechanism evidence",
    ),
)


class KnowledgeBaseService:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_schema()

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
                    process_factor TEXT,
                    morphology_factor TEXT,
                    effect_direction TEXT,
                    mechanism_summary TEXT,
                    evidence_text TEXT
                );

                CREATE TABLE IF NOT EXISTS kb_task_profiles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_name TEXT NOT NULL UNIQUE,
                    preferred_theme TEXT,
                    preferred_knowledge_types TEXT,
                    preferred_keywords TEXT
                );
                """
            )
            conn.commit()
        finally:
            conn.close()

    def bootstrap_task_profiles(self):
        conn = self._connect()
        try:
            conn.executemany(
                """
                INSERT INTO kb_task_profiles (
                    task_name, preferred_theme, preferred_knowledge_types, preferred_keywords
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(task_name) DO UPDATE SET
                    preferred_theme = excluded.preferred_theme,
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
            conn.commit()
        finally:
            conn.close()

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

        scored = []
        for row in rows:
            score = self._score_row(dict(row), query_tokens, task_profile)
            if score > 0:
                scored.append((score, dict(row)))

        scored.sort(key=lambda item: item[0], reverse=True)
        results = []
        for score, row in scored[:top_k]:
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
                }
            )
        return results

    def get_stats(self) -> Dict[str, int]:
        conn = self._connect()
        try:
            doc_count = conn.execute("SELECT COUNT(*) FROM kb_documents").fetchone()[0]
            chunk_count = conn.execute("SELECT COUNT(*) FROM kb_chunks").fetchone()[0]
            core_count = conn.execute(
                "SELECT COUNT(*) FROM kb_documents WHERE is_core = 1"
            ).fetchone()[0]
        finally:
            conn.close()
        return {
            "document_count": doc_count,
            "chunk_count": chunk_count,
            "core_document_count": core_count,
        }

    def ingest_directory(
        self,
        source_dir: str,
        source_type: str,
        theme: Optional[str] = None,
        is_core: bool = False,
        allowed_extensions: Optional[List[str]] = None,
    ) -> Dict[str, int]:
        extensions = allowed_extensions or [".txt"]
        imported = 0
        for entry in sorted(os.listdir(source_dir)):
            path = os.path.join(source_dir, entry)
            if not os.path.isfile(path):
                continue
            if os.path.splitext(entry)[1].lower() not in extensions:
                continue
            with open(path, "r", encoding="utf-8") as handle:
                text = handle.read()
            self.ingest_text(
                title=os.path.splitext(entry)[0],
                text=text,
                source_type=source_type,
                theme=theme,
                is_core=is_core,
                file_path=path,
                language="zh" if re.search(r"[\u4e00-\u9fff]", text) else "en",
            )
            imported += 1
        return {"document_count": imported}

    def _get_task_profile(self, task_name: Optional[str]) -> Optional[Dict[str, str]]:
        if not task_name:
            return None
        conn = sqlite3.connect(self.db_path)
        try:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                """
                SELECT task_name, preferred_theme, preferred_knowledge_types, preferred_keywords
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
