import os
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from backend import main as api_main


class XRSimpleModelApiTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "xr_ml.sqlite")
        self._seed_db()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _seed_db(self):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT UNIQUE,
                source TEXT,
                sample_id TEXT,
                growth_temp REAL,
                actual_temp REAL,
                ar_flow REAL,
                catalyst_weight REAL,
                diameter REAL,
                density REAL,
                alignment REAL,
                curvature REAL
            )
            """
        )

        rows = [
            (
                r"d:\CNTDATA\XR\run-01\C1A1.tiff",
                "XR",
                "C1-A1",
                800.0,
                790.0,
                200.0,
                1.0,
                94.0,
                15.8,
                1.09,
                0.094,
            ),
            (
                r"d:\CNTDATA\XR\run-01\C1A2.tiff",
                "XR",
                "C1-A2",
                800.0,
                792.0,
                220.0,
                1.1,
                95.7,
                15.72,
                1.132,
                0.091,
            ),
            (
                r"d:\CNTDATA\XR\run-01\C2B1.tiff",
                "XR",
                "C2-B1",
                800.0,
                798.0,
                250.0,
                1.3,
                98.55,
                15.91,
                1.218,
                0.087,
            ),
            (
                r"d:\CNTDATA\XR\run-02\C3C1.tiff",
                "XR",
                "C3-C1",
                850.0,
                805.0,
                300.0,
                1.5,
                102.0,
                16.6,
                1.305,
                0.084,
            ),
            (
                r"d:\CNTDATA\XR\run-02\C4A1.tiff",
                "XR",
                "C4-A1",
                850.0,
                810.0,
                280.0,
                1.4,
                None,
                None,
                None,
                None,
            ),
            (
                r"d:\CNTDATA\XR\run-02\A1B1.tiff",
                "XR",
                "A1-B1",
                850.0,
                810.0,
                280.0,
                1.4,
                88.0,
                12.0,
                0.8,
                0.12,
            ),
        ]

        cur.executemany(
            """
            INSERT INTO images (
                file_path, source, sample_id, growth_temp, actual_temp,
                ar_flow, catalyst_weight, diameter, density, alignment, curvature
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
        conn.close()

    async def test_xr_simple_model_endpoint_returns_visualization_rows_and_predictions(self):
        with patch.object(api_main, "DB_PATH", self.db_path):
            result = await api_main.get_xr_simple_model_data()

        self.assertIn("summary", result)
        self.assertIn("rows", result)
        self.assertEqual(result["summary"]["total_rows"], 5)

        any_row = result["rows"][0]
        self.assertIn("sample_no", any_row)
        self.assertIn("position", any_row)
        self.assertIn("shot_no", any_row)
        self.assertIn("actual_temp", any_row)
        self.assertIn("flow_rate", any_row)
        self.assertIn("catalyst_concentration", any_row)

        unlabeled = [r for r in result["rows"] if r["sample_id"] == "C4-A1"][0]
        self.assertIsNotNone(unlabeled["diameter_pred"])
        self.assertIsNotNone(unlabeled["density_pred"])
        self.assertIsNotNone(unlabeled["alignment_pred"])
        self.assertIsNotNone(unlabeled["curvature_pred"])

        coeffs = result["coefficients"]["diameter"]
        self.assertEqual(coeffs["n_train"], 4)
        self.assertIn(coeffs["method"], {"ols", "mean"})


if __name__ == "__main__":
    unittest.main()
