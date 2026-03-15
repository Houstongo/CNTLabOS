import os
import sqlite3
import tempfile
import unittest

from backend.core.knowledge_base import KnowledgeBaseService
from backend.core.knowledge_rag import RAGRetriever


class KnowledgeBaseServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "kb.sqlite")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_initialization_creates_knowledge_base_tables(self):
        service = KnowledgeBaseService(self.db_path)

        conn = sqlite3.connect(self.db_path)
        try:
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
        finally:
            conn.close()

        self.assertTrue({"kb_documents", "kb_chunks", "kb_links", "kb_task_profiles"}.issubset(tables))
        self.assertIsNotNone(service)

    def test_ingest_text_creates_document_and_chunks(self):
        service = KnowledgeBaseService(self.db_path)

        result = service.ingest_text(
            title="CNT growth mechanism",
            text=(
                "Water-assisted growth improves catalyst activity and alignment. "
                "High Fe thickness often increases diameter and reduces uniformity. "
                "Boundary layer effects influence nanotube forest growth kinetics."
            ),
            source_type="paper",
            theme="growth_mechanism",
            is_core=True,
        )

        self.assertGreater(result["chunk_count"], 0)

        documents = service.list_documents()
        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0]["title"], "CNT growth mechanism")
        self.assertEqual(documents[0]["theme"], "growth_mechanism")
        self.assertEqual(documents[0]["is_core"], 1)

    def test_task_aware_retrieval_prefers_matching_theme_and_keywords(self):
        service = KnowledgeBaseService(self.db_path)
        service.bootstrap_task_profiles()
        service.ingest_text(
            title="Morphology interpretation note",
            text=(
                "Low alignment in vertically aligned CNT arrays is often related to "
                "catalyst deactivation, boundary layer instability, and nonuniform carbon supply."
            ),
            source_type="paper",
            theme="morphology_interpretation",
            is_core=True,
        )
        service.ingest_text(
            title="Sensor application note",
            text="CNT tactile sensors are useful in flexible electronics and biomimetic devices.",
            source_type="paper",
            theme="applications",
            is_core=False,
        )

        results = service.search(
            query="alignment mechanism for CNT arrays",
            task_name="morphology_interpretation",
            top_k=3,
        )

        self.assertGreaterEqual(len(results), 1)
        self.assertEqual(results[0]["title"], "Morphology interpretation note")
        self.assertEqual(results[0]["theme"], "morphology_interpretation")

    def test_ingest_directory_imports_supported_text_files(self):
        service = KnowledgeBaseService(self.db_path)
        source_dir = os.path.join(self.temp_dir.name, "source")
        os.makedirs(source_dir, exist_ok=True)
        with open(os.path.join(source_dir, "growth_notes.txt"), "w", encoding="utf-8") as handle:
            handle.write("CNT growth depends on catalyst activity and gas flow balance.")
        with open(os.path.join(source_dir, "ignore.md"), "w", encoding="utf-8") as handle:
            handle.write("This file should not be imported by the default text loader.")

        result = service.ingest_directory(
            source_dir=source_dir,
            source_type="notes",
            theme="growth_mechanism",
            is_core=True,
        )

        self.assertEqual(result["document_count"], 1)
        self.assertEqual(service.get_stats()["document_count"], 1)

    def test_initialization_recovers_from_empty_database_stub(self):
        open(self.db_path, "wb").close()
        with open(self.db_path + "-journal", "wb") as handle:
            handle.write(b"stub")

        service = KnowledgeBaseService(self.db_path)

        self.assertEqual(service.get_stats()["document_count"], 0)


class RAGRetrieverCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "rag.sqlite")
        self.kb_path = os.path.join(self.temp_dir.name, "kb.sqlite")
        self.retriever = RAGRetriever(self.db_path, knowledge_db_path=self.kb_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_retriever_lists_imported_documents(self):
        self.retriever.ingest_text_document(
            title="Process morphology relation",
            text="Higher temperature can improve alignment but excessive catalyst agglomeration increases diameter.",
            source_type="paper",
            theme="process_morphology",
            is_core=True,
        )

        documents = self.retriever.list_documents()

        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0]["title"], "Process morphology relation")

    def test_retrieve_all_returns_pdf_passages_from_knowledge_base(self):
        self.retriever.ingest_text_document(
            title="Growth kinetics review",
            text="Boundary layer thickness affects CNT forest growth kinetics and morphology evolution.",
            source_type="paper",
            theme="growth_mechanism",
            is_core=True,
        )

        results = self.retriever.retrieve_all(
            features={},
            params={},
            query="CNT growth kinetics boundary layer",
            task_name="process_analysis",
        )

        self.assertIn("pdf_passages", results)
        self.assertGreaterEqual(len(results["pdf_passages"]), 1)
        self.assertEqual(results["pdf_passages"][0]["title"], "Growth kinetics review")

    def test_retriever_can_split_experiment_db_and_knowledge_db(self):
        self.retriever.ingest_text_document(
            title="Independent KB document",
            text="Alignment interpretation should be stored in the dedicated knowledge database.",
            source_type="paper",
            theme="morphology_interpretation",
            is_core=True,
        )

        self.assertTrue(os.path.exists(self.kb_path))
        self.assertEqual(self.retriever.get_stats()["document_count"], 1)


if __name__ == "__main__":
    unittest.main()
