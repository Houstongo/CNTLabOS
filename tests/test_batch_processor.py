import os
import sqlite3
import shutil
import unittest
from pathlib import Path
from uuid import uuid4
from unittest.mock import patch

from backend.core import batch_processor


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_TEST_TEMP_ROOT = _PROJECT_ROOT / "_tmp_unittest"


def _make_workspace_tempdir(prefix: str) -> str:
    _TEST_TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    path = _TEST_TEMP_ROOT / f"{prefix}{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    return str(path)


class BatchProcessorTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = _make_workspace_tempdir("batch_processor_")
        self.addCleanup(shutil.rmtree, self.temp_dir, True)
        self.db_path = f"file:batch_processor_{uuid4().hex}?mode=memory&cache=shared"
        self._sqlite_connect = sqlite3.connect
        self._keeper_conn = self._sqlite_connect(self.db_path, uri=True, check_same_thread=False)
        self.addCleanup(self._keeper_conn.close)
        self.image_dir = os.path.join(self.temp_dir, "images")
        os.makedirs(self.image_dir, exist_ok=True)

        self.keep_path = os.path.join(self.image_dir, "C1A2.tiff")
        self.deleted_path = os.path.join(self.image_dir, "C1A3.tiff")
        for path in (self.keep_path, self.deleted_path):
            with open(path, "wb") as handle:
                handle.write(b"fake")

        conn = self._keeper_conn
        conn.execute(
            """
            CREATE TABLE images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT,
                source TEXT,
                magnification INTEGER,
                diameter REAL,
                density REAL,
                alignment REAL,
                curvature REAL,
                tortuosity REAL,
                waviness_ratio REAL,
                waviness_height_nm REAL,
                waviness_wavelength_nm REAL,
                waviness_branches INTEGER,
                processed INTEGER DEFAULT 0,
                is_deleted INTEGER DEFAULT 0
            )
            """
        )
        conn.execute(
            """
            INSERT INTO images (file_path, source, magnification, processed, is_deleted)
            VALUES (?, 'XR', 50000, 0, 0)
            """,
            (self.keep_path,),
        )
        conn.execute(
            """
            INSERT INTO images (file_path, source, magnification, processed, is_deleted)
            VALUES (?, 'XR', 50000, 0, 1)
            """,
            (self.deleted_path,),
        )
        conn.commit()

    def _connect_test_db(self, path, *args, **kwargs):
        if path == self.db_path:
            kwargs.setdefault("uri", True)
            kwargs.setdefault("check_same_thread", False)
        return self._sqlite_connect(path, *args, **kwargs)

    def test_batch_process_skips_logically_deleted_images(self):
        processed_paths = []
        diameter_methods = []

        def fake_extract_image_features(file_path, magnification, diameter_method="standard", **kwargs):
            processed_paths.append(file_path)
            diameter_methods.append(diameter_method)
            return {
                "diameter": 10.0,
                "density": 20.0,
                "alignment": 0.3,
                "curvature": 0.4,
                "curvature_nm": 0.4,
                "tortuosity": 1.2,
                "waviness_ratio": 0.25,
                "waviness_height_nm": 12.5,
                "waviness_wavelength_nm": 50.0,
                "waviness_branches": 3,
            }

        with patch.object(batch_processor, "DB_PATH", self.db_path), \
             patch.object(batch_processor.sqlite3, "connect", side_effect=self._connect_test_db), \
             patch.object(batch_processor, "_extract_image_features", side_effect=fake_extract_image_features):
            batch_processor.batch_process(source="XR")

        self.assertEqual(processed_paths, [self.keep_path])

        conn = self._sqlite_connect(self.db_path, uri=True, check_same_thread=False)
        rows = conn.execute(
            """
            SELECT
                file_path,
                processed,
                tortuosity,
                waviness_ratio,
                waviness_height_nm,
                waviness_wavelength_nm,
                waviness_branches
            FROM images
            ORDER BY id
            """
        ).fetchall()
        conn.close()

        self.assertEqual(rows, [
            (self.keep_path, 1, 1.2, 0.25, 12.5, 50.0, 3),
            (self.deleted_path, 0, None, None, None, None, None),
        ])
        self.assertEqual(diameter_methods, ["standard"])

    def test_batch_process_falls_back_to_standard_for_xr_20k_when_enhanced_requested(self):
        diameter_methods = []

        def fake_extract_image_features(file_path, magnification, diameter_method="standard", **kwargs):
            diameter_methods.append(diameter_method)
            return {
                "diameter": 10.0,
                "density": 20.0,
                "alignment": 0.3,
                "curvature": 0.4,
                "curvature_nm": 0.4,
                "tortuosity": 1.2,
                "waviness_ratio": 0.25,
                "waviness_height_nm": 12.5,
                "waviness_wavelength_nm": 50.0,
                "waviness_branches": 3,
            }

        conn = self._sqlite_connect(self.db_path, uri=True, check_same_thread=False)
        conn.execute("UPDATE images SET magnification = 20000 WHERE is_deleted = 0")
        conn.commit()
        conn.close()

        with patch.object(batch_processor, "DB_PATH", self.db_path), \
             patch.object(batch_processor.sqlite3, "connect", side_effect=self._connect_test_db), \
             patch.object(batch_processor, "_extract_image_features", side_effect=fake_extract_image_features):
            batch_processor.batch_process(source="XR", diameter_method="enhanced")

        self.assertEqual(diameter_methods, ["standard"])


if __name__ == "__main__":
    unittest.main()
