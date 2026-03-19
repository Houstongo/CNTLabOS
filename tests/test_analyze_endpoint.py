import sqlite3
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

from backend import main as api_main


class AnalyzeEndpointTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.root = Path(self.tmpdir.name)

        self.image_path = self.root / "sample.png"
        img = np.zeros((64, 64), dtype=np.uint8)
        img[:, 24:40] = 255
        cv2.imwrite(str(self.image_path), img)

        self.db_path = self.root / "test.sqlite"
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """
            CREATE TABLE images (
                id INTEGER PRIMARY KEY,
                file_path TEXT,
                magnification INTEGER,
                diameter REAL,
                density REAL,
                alignment REAL,
                curvature REAL,
                processed INTEGER
            )
            """
        )
        conn.execute(
            """
            INSERT INTO images (id, file_path, magnification, processed)
            VALUES (?, ?, ?, 0)
            """,
            (1, str(self.image_path), 50000),
        )
        conn.commit()
        conn.close()

        self.prev_db_path = api_main.DB_PATH
        api_main.DB_PATH = str(self.db_path)
        self.addCleanup(self.restore_db_path)

        self.client = TestClient(api_main.app)

    def restore_db_path(self):
        api_main.DB_PATH = self.prev_db_path

    def test_analyze_endpoint_returns_success_for_valid_image(self):
        response = self.client.post("/api/images/1/analyze")

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["status"], "success")
        self.assertIn("results", payload)


if __name__ == "__main__":
    unittest.main()
