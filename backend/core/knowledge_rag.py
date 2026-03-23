"""
RAG retrieval facade for CNTA.

This module preserves the current RAG-facing API shape while routing
document/chunk retrieval through the paper-oriented knowledge-base service.
"""

import io
import sqlite3
from typing import Dict, List, Optional, Any

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
            "knowledge_links": self.knowledge_base.search_links(query, top_k=5) if query else [],
            "relation_chain": self.knowledge_base.get_relation_chain_summary(query, top_k=20) if query else {},
        }

    def retrieve_for_qa(
        self,
        query: str,
        task_name: Optional[str] = None,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        专为问答场景的检索，返回结构化上下文和来源

        Args:
            query: 用户问题
            task_name: 任务名称（可选）
            top_k: 返回结果数量

        Returns:
            {
                "pdf_passages": [...],
                "knowledge_links": [...],
                "similar_experiments": [],
                "context_summary": "...",
                "retrieval_stats": {
                    "total_sources": 10,
                    "pdf_count": 5,
                    "link_count": 5
                }
            }
        """
        # 检索PDF文献
        pdf_passages = self.retrieve_from_pdf(query, task_name=task_name, top_k=top_k)

        # 检索专家知识链接
        knowledge_links = self.knowledge_base.search_links(query, top_k=top_k)

        # 构建上下文摘要
        context_parts = []

        if pdf_passages:
            context_parts.append(f"### 文献证据 ({len(pdf_passages)} 条)")
            for i, passage in enumerate(pdf_passages[:3], 1):
                title = passage.get("title") or passage.get("filename") or "未知文献"
                text = passage.get("text", "")
                context_parts.append(f"[{i}] {title}")
                context_parts.append(f"    {text[:200]}...")
                context_parts.append("")

        if knowledge_links:
            context_parts.append(f"### 专家知识 ({len(knowledge_links)} 条)")
            for i, link in enumerate(knowledge_links[:3], 1):
                process = link.get("process_factor") or "-"
                morph = link.get("morphology_factor") or "-"
                perf = link.get("performance_factor") or "-"
                direction = link.get("effect_direction") or "-"
                context_parts.append(f"{i}. 工艺={process} → 形貌={morph} → 性能={perf} ({direction})")
                evidence = link.get("evidence_text") or ""
                if evidence:
                    context_parts.append(f"   证据: {evidence[:150]}...")
                context_parts.append("")

        context_summary = "\n".join(context_parts)

        # 统计信息
        retrieval_stats = {
            "total_sources": len(pdf_passages) + len(knowledge_links),
            "pdf_count": len(pdf_passages),
            "link_count": len(knowledge_links),
        }

        return {
            "pdf_passages": pdf_passages,
            "knowledge_links": knowledge_links,
            "similar_experiments": [],  # QA场景通常不需要相似实验
            "context_summary": context_summary,
            "retrieval_stats": retrieval_stats,
        }
