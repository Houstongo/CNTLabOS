import sqlite3
import shutil
import unittest
from pathlib import Path
from uuid import uuid4
from unittest.mock import patch

import cv2
import numpy as np
from fastapi.testclient import TestClient

from backend import main as api_main


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_TEST_TEMP_ROOT = _PROJECT_ROOT / "_tmp_unittest"


def _make_workspace_tempdir(prefix: str) -> Path:
    _TEST_TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    path = _TEST_TEMP_ROOT / f"{prefix}{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    return path


class AnalyzeEndpointTests(unittest.TestCase):
    def setUp(self):
        self.root = _make_workspace_tempdir("analyze_endpoint_")
        self.addCleanup(shutil.rmtree, self.root, True)

        self.image_path = self.root / "sample.png"
        img = np.zeros((64, 64), dtype=np.uint8)
        img[:, 24:40] = 255
        cv2.imwrite(str(self.image_path), img)

        self.db_path = f"file:analyze_endpoint_{uuid4().hex}?mode=memory&cache=shared"
        self._sqlite_connect = sqlite3.connect
        self._keeper_conn = self._sqlite_connect(self.db_path, uri=True, check_same_thread=False)
        self.addCleanup(self._keeper_conn.close)
        conn = self._keeper_conn
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

        self.prev_db_path = api_main.DB_PATH
        api_main.DB_PATH = self.db_path
        self.addCleanup(self.restore_db_path)
        self.sqlite_connect_patcher = patch.object(api_main.sqlite3, "connect", side_effect=self._connect_test_db)
        self.sqlite_connect_patcher.start()
        self.addCleanup(self.sqlite_connect_patcher.stop)

        self.client = TestClient(api_main.app)

    def restore_db_path(self):
        api_main.DB_PATH = self.prev_db_path

    def _connect_test_db(self, path, *args, **kwargs):
        if path == self.db_path:
            kwargs.setdefault("uri", True)
            kwargs.setdefault("check_same_thread", False)
        return self._sqlite_connect(path, *args, **kwargs)

    def test_analyze_endpoint_returns_success_for_valid_image(self):
        response = self.client.post("/api/images/1/analyze")

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["status"], "success")
        self.assertIn("results", payload)

    def test_analyze_endpoint_uses_adaptive_alignment_correction(self):
        rotated_path = self.root / "rotated_sample.png"
        img = np.zeros((220, 220), dtype=np.uint8)
        for x in range(35, 186, 22):
            cv2.line(img, (x, 20), (x, 190), 255, 5)
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite(str(rotated_path), img)

        conn = self._sqlite_connect(self.db_path, uri=True, check_same_thread=False)
        conn.execute(
            """
            INSERT INTO images (id, file_path, magnification, processed)
            VALUES (?, ?, ?, 0)
            """,
            (2, str(rotated_path), 50000),
        )
        conn.commit()
        conn.close()

        response = self.client.post("/api/images/2/analyze")

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        results = payload["results"]
        self.assertEqual(results["rotation_correction_deg"], 90)
        self.assertLess(results["alignment_raw"], 0.0)
        self.assertGreater(results["alignment"], results["alignment_raw"])

    def test_analyze_endpoint_can_read_unicode_path_images(self):
        unicode_dir = self.root / "中文目录"
        unicode_dir.mkdir()
        unicode_path = unicode_dir / "样品图.png"

        img = np.zeros((64, 64), dtype=np.uint8)
        img[:, 20:44] = 255
        ok, encoded = cv2.imencode(".png", img)
        self.assertTrue(ok)
        encoded.tofile(str(unicode_path))

        conn = self._sqlite_connect(self.db_path, uri=True, check_same_thread=False)
        conn.execute(
            """
            INSERT INTO images (id, file_path, magnification, processed)
            VALUES (?, ?, ?, 0)
            """,
            (3, str(unicode_path), 50000),
        )
        conn.commit()
        conn.close()

        original_imread = api_main.cv2.imread

        def fake_imread(path, flags):
            if "中文目录" in str(path):
                return None
            return original_imread(path, flags)

        with patch.object(api_main.cv2, "imread", side_effect=fake_imread):
            response = self.client.post("/api/images/3/analyze")

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["status"], "success")


if __name__ == "__main__":
    unittest.main()
