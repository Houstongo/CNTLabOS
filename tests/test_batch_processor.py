import os
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from backend.core import batch_processor


class BatchProcessorTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.db_path = os.path.join(self.temp_dir.name, "batch.sqlite")
        self.image_dir = os.path.join(self.temp_dir.name, "images")
        os.makedirs(self.image_dir, exist_ok=True)

        self.keep_path = os.path.join(self.image_dir, "C1A2.tiff")
        self.deleted_path = os.path.join(self.image_dir, "C1A3.tiff")
        for path in (self.keep_path, self.deleted_path):
            with open(path, "wb") as handle:
                handle.write(b"fake")

        conn = sqlite3.connect(self.db_path)
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
        conn.close()

    def test_batch_process_skips_logically_deleted_images(self):
        processed_paths = []
        diameter_methods = []

        class DummyExtractor:
            def __init__(self, magnification=None, diameter_method=None):
                self.magnification = magnification
                self.diameter_method = diameter_method
                diameter_methods.append(diameter_method)

            def extract_all(self, img):
                processed_paths.append(img)
                return {
                    "diameter": 10.0,
                    "density": 20.0,
                    "alignment": 0.3,
                    "curvature": 0.4,
                    "tortuosity": 1.2,
                    "waviness_ratio": 0.25,
                    "waviness_height_nm": 12.5,
                    "waviness_wavelength_nm": 50.0,
                    "waviness_branches": 3,
                }

        def fake_imread(path, flags):
            return path

        with patch.object(batch_processor, "DB_PATH", self.db_path), \
             patch.object(batch_processor, "FeatureExtractor", DummyExtractor), \
             patch.object(batch_processor.cv2, "imread", side_effect=fake_imread):
            batch_processor.batch_process(source="XR")

        self.assertEqual(processed_paths, [self.keep_path])

        conn = sqlite3.connect(self.db_path)
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

        class DummyExtractor:
            def __init__(self, magnification=None, diameter_method=None):
                diameter_methods.append(diameter_method)

            def extract_all(self, img):
                return {
                    "diameter": 10.0,
                    "density": 20.0,
                    "alignment": 0.3,
                    "curvature": 0.4,
                    "tortuosity": 1.2,
                    "waviness_ratio": 0.25,
                    "waviness_height_nm": 12.5,
                    "waviness_wavelength_nm": 50.0,
                    "waviness_branches": 3,
                }

        def fake_imread(path, flags):
            return path

        conn = sqlite3.connect(self.db_path)
        conn.execute("UPDATE images SET magnification = 20000 WHERE is_deleted = 0")
        conn.commit()
        conn.close()

        with patch.object(batch_processor, "DB_PATH", self.db_path), \
             patch.object(batch_processor, "FeatureExtractor", DummyExtractor), \
             patch.object(batch_processor.cv2, "imread", side_effect=fake_imread):
            batch_processor.batch_process(source="XR", diameter_method="enhanced")

        self.assertEqual(diameter_methods, ["standard"])


if __name__ == "__main__":
    unittest.main()
