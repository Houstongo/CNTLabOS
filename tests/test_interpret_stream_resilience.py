import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend import main as api_main


class _BrokenInterpreter:
    def interpret_stream(self, **kwargs):
        raise RuntimeError("upstream stream failed")


class InterpretStreamResilienceTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.root = Path(self.tmpdir.name)

        self.db_path = self.root / "test.sqlite"
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """
            CREATE TABLE images (
                id INTEGER PRIMARY KEY,
                file_path TEXT,
                sample_id TEXT,
                density REAL,
                alignment REAL,
                diameter REAL,
                curvature REAL,
                tortuosity REAL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO images (id, file_path, sample_id, density, alignment, diameter, curvature, tortuosity)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (1, "dummy.png", "XR-C1", 10.0, 0.5, 20.0, 0.2, None),
        )
        conn.commit()
        conn.close()

        self.prev_db_path = api_main.DB_PATH
        api_main.DB_PATH = str(self.db_path)
        self.addCleanup(self.restore_db_path)

        self.client = TestClient(api_main.app)

    def restore_db_path(self):
        api_main.DB_PATH = self.prev_db_path

    def test_interpret_stream_returns_sse_error_instead_of_abrupt_disconnect(self):
        with patch.object(api_main, "_get_interpreter", return_value=_BrokenInterpreter()), patch.object(
            api_main.rag_retriever,
            "retrieve_all",
            return_value={"similar_experiments": [], "pdf_passages": [], "knowledge_links": []},
        ):
            response = self.client.post(
                "/api/images/1/interpret",
                headers={
                    "X-Provider": "glm",
                    "X-Api-Key": "test-api-key",
                    "X-Model": "glm-4-flash",
                    "X-Temperature": "0.5",
                },
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertIn('"type": "error"', response.text)
        self.assertIn("upstream stream failed", response.text)


if __name__ == "__main__":
    unittest.main()
