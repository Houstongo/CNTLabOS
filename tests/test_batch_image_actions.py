import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from fastapi.testclient import TestClient

from backend import main as api_main


class DummyExtractor:
    def __init__(self, magnification=None, diameter_method=None):
        self.magnification = magnification
        self.diameter_method = diameter_method

    def extract_all(self, img):
        return {
            "diameter": 12.3,
            "density": 45.6,
            "alignment": 0.78,
            "curvature": 0.12,
            "tortuosity": 8.0,
        }


class BatchImageActionsTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.root = Path(self.tmpdir.name)

        self.image_path = self.root / "sample.png"
        img = np.zeros((48, 48), dtype=np.uint8)
        img[:, 18:30] = 255
        cv2.imwrite(str(self.image_path), img)

        self.db_path = self.root / "batch.sqlite"
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
                tortuosity REAL,
                processed INTEGER DEFAULT 0,
                is_deleted INTEGER DEFAULT 0
            )
            """
        )
        conn.executemany(
            """
            INSERT INTO images (id, file_path, magnification, processed, is_deleted)
            VALUES (?, ?, ?, 0, ?)
            """,
            [
                (1, str(self.image_path), 50000, 0),
                (2, str(self.image_path), 50000, 0),
                (3, str(self.image_path), 50000, 1),
            ],
        )
        conn.commit()
        conn.close()

        self.prev_db_path = api_main.DB_PATH
        api_main.DB_PATH = str(self.db_path)
        self.addCleanup(self.restore_db_path)

        self.client = TestClient(api_main.app)

    def restore_db_path(self):
        api_main.DB_PATH = self.prev_db_path

    def test_batch_analyze_updates_multiple_active_records(self):
        with patch.object(api_main, "FeatureExtractor", DummyExtractor):
            response = self.client.post("/api/images/batch/analyze", json={"image_ids": [1, 2]})

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["summary"]["success_count"], 2)
        self.assertEqual(payload["summary"]["failed_count"], 0)

        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            "SELECT processed, diameter, density, alignment, curvature FROM images WHERE id IN (1, 2) ORDER BY id"
        ).fetchall()
        conn.close()
        self.assertEqual(rows, [(1, 12.3, 45.6, 0.78, 0.12), (1, 12.3, 45.6, 0.78, 0.12)])

    def test_batch_analyze_skips_deleted_records(self):
        with patch.object(api_main, "FeatureExtractor", DummyExtractor):
            response = self.client.post("/api/images/batch/analyze", json={"image_ids": [1, 3]})

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["summary"]["success_count"], 1)
        self.assertEqual(payload["summary"]["failed_count"], 1)
        self.assertEqual(payload["items"][1]["status"], "skipped")

        conn = sqlite3.connect(self.db_path)
        processed = conn.execute("SELECT processed FROM images WHERE id = 3").fetchone()[0]
        conn.close()
        self.assertEqual(processed, 0)

    def test_batch_delete_marks_requested_records_deleted(self):
        response = self.client.put("/api/images/batch/delete", json={"image_ids": [1, 2]})

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["summary"]["success_count"], 2)
        self.assertEqual(payload["summary"]["failed_count"], 0)

        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("SELECT id, is_deleted FROM images WHERE id IN (1, 2) ORDER BY id").fetchall()
        conn.close()
        self.assertEqual(rows, [(1, 1), (2, 1)])

    def test_batch_actions_reject_empty_payload(self):
        response = self.client.post("/api/images/batch/analyze", json={"image_ids": []})
        self.assertEqual(response.status_code, 400, response.text)


if __name__ == "__main__":
    unittest.main()
