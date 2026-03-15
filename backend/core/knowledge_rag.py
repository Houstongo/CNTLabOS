"""
RAG retrieval facade for CNTA.

This module preserves the current RAG-facing API shape while routing
document/chunk retrieval through the paper-oriented knowledge-base service.
"""

import io
import sqlite3
from typing import Dict, List, Optional

from backend.core.knowledge_base import KnowledgeBaseService


class RAGRetriever:
    def __init__(self, db_path: str, knowledge_db_path: Optional[str] = None):
        self.db_path = db_path
        self.knowledge_db_path = knowledge_db_path or db_path
        self.knowledge_base = KnowledgeBaseService(self.knowledge_db_path)
        self.knowledge_base.bootstrap_task_profiles()

    def add_pdf(self, file_bytes: bytes, filename: str) -> Dict[str, int]:
        import pdfplumber

        text_pages: List[str] = []
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    text_pages.append(text)

        return self.knowledge_base.ingest_text(
            title=filename,
            text="\n".join(text_pages),
            source_type="pdf",
            theme="growth_mechanism",
            is_core=False,
            file_path=filename,
        )

    def ingest_text_document(
        self,
        title: str,
        text: str,
        source_type: str = "text",
        theme: Optional[str] = None,
        is_core: bool = False,
        file_path: Optional[str] = None,
    ) -> Dict[str, int]:
        return self.knowledge_base.ingest_text(
            title=title,
            text=text,
            source_type=source_type,
            theme=theme,
            is_core=is_core,
            file_path=file_path,
        )

    def list_documents(self) -> List[Dict[str, object]]:
        return self.knowledge_base.list_documents()

    def delete_document(self, doc_id: int):
        self.knowledge_base.delete_document(doc_id)

    def get_stats(self) -> Dict[str, int]:
        return self.knowledge_base.get_stats()

    def retrieve_from_db(self, features: dict, params: dict, top_k: int = 5) -> List[Dict[str, object]]:
        try:
            conn = sqlite3.connect(self.db_path)
            try:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    """
                    SELECT id, sample_id, source, growth_temp, growth_time,
                           ar_flow, h2_flow, c2h4_flow,
                           fe_thickness, al2o3_thickness,
                           diameter, density, alignment, curvature,
                           position_label
                    FROM images
                    WHERE processed = 1
                      AND id != ?
                    ORDER BY id DESC
                    LIMIT 200
                    """,
                    (params.get("current_id", -1),),
                ).fetchall()
            finally:
                conn.close()
        except sqlite3.OperationalError:
            return []

        if not rows:
            return []

        def score(row):
            total = 0.0
            growth_temp = params.get("growth_temp")
            fe_thickness = params.get("fe_thickness")
            ar_flow = params.get("ar_flow")
            if growth_temp and row["growth_temp"]:
                total += abs(growth_temp - row["growth_temp"]) / max(growth_temp, 1) * 2.0
            if fe_thickness and row["fe_thickness"]:
                total += abs(fe_thickness - row["fe_thickness"]) / max(fe_thickness, 1) * 1.5
            if ar_flow and row["ar_flow"]:
                total += abs(ar_flow - row["ar_flow"]) / max(ar_flow, 1)
            return total

        return [dict(row) for row in sorted(rows, key=score)[:top_k]]

    def retrieve_from_pdf(
        self,
        query: str,
        top_k: int = 3,
        task_name: Optional[str] = None,
    ) -> List[Dict[str, object]]:
        return self.knowledge_base.search(query=query, task_name=task_name, top_k=top_k)

    def retrieve_all(
        self,
        features: dict,
        params: dict,
        query: str,
        task_name: Optional[str] = None,
    ) -> Dict[str, List[Dict[str, object]]]:
        return {
            "similar_experiments": self.retrieve_from_db(features, params),
            "pdf_passages": self.retrieve_from_pdf(query, task_name=task_name) if query else [],
        }
