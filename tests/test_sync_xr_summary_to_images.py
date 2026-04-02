import csv
import sqlite3
import tempfile
import unittest
from pathlib import Path

from tools.maintenance.sync_xr_summary_to_images import run_sync


class SyncXRSummaryToImagesTests(unittest.TestCase):
    def test_sync_overwrites_legacy_fields_and_adds_new_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            db_path = tmp_path / "test.sqlite"
            csv_path = tmp_path / "summary.csv"

            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE images (
                    id INTEGER PRIMARY KEY,
                    source TEXT,
                    density REAL,
                    alignment REAL,
                    diameter REAL,
                    curvature REAL,
                    tortuosity REAL,
                    waviness_ratio REAL,
                    diameter_mean REAL,
                    diameter_std REAL,
                    diameter_min REAL,
                    diameter_max REAL,
                    diameter_p50 REAL,
                    diameter_p75 REAL
                )
                """
            )
            cur.execute(
                """
                INSERT INTO images (
                    id, source, density, alignment, diameter, curvature, tortuosity, waviness_ratio
                ) VALUES (1, 'XR', 1.0, 0.1, 10.0, 9.0, 8.0, 7.0)
                """
            )
            cur.execute(
                """
                INSERT INTO images (
                    id, source, density, alignment, diameter, curvature, tortuosity, waviness_ratio
                ) VALUES (2, 'ZZY', 2.0, 0.2, 20.0, 19.0, 18.0, 17.0)
                """
            )
            conn.commit()
            conn.close()

            fieldnames = [
                "image_id",
                "status",
                "density",
                "alignment",
                "diameter_mean_nm",
                "diameter_std_nm",
                "diameter_min_nm",
                "diameter_max_nm",
                "diameter_p50_nm",
                "diameter_p75_nm",
                "diameter_p30_nm",
                "junction_count",
                "junction_ratio",
                "skeleton_length_px",
                "skeleton_length_um",
                "l2_branch_count",
                "l2_curvature_label",
                "l2_curvature_p70_sqrt_length_nm",
                "l2_curvature_mean_sqrt_length_nm",
                "l2_curvature_trimmed_mean_sqrt_length_nm",
                "l2_waviness_ratio_v2",
                "l2_tortuosity_v2",
            ]
            with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(
                    {
                        "image_id": 1,
                        "status": "success",
                        "density": 51.1,
                        "alignment": 0.61,
                        "diameter_mean_nm": 55.2,
                        "diameter_std_nm": 5.5,
                        "diameter_min_nm": 12.0,
                        "diameter_max_nm": 87.0,
                        "diameter_p50_nm": 54.0,
                        "diameter_p75_nm": 63.0,
                        "diameter_p30_nm": 45.0,
                        "junction_count": 321,
                        "junction_ratio": 0.123,
                        "skeleton_length_px": 5000,
                        "skeleton_length_um": 12.5,
                        "l2_branch_count": 88,
                        "l2_curvature_label": "Coiled",
                        "l2_curvature_p70_sqrt_length_nm": 0.008,
                        "l2_curvature_mean_sqrt_length_nm": 0.007,
                        "l2_curvature_trimmed_mean_sqrt_length_nm": 0.006,
                        "l2_waviness_ratio_v2": 0.15,
                        "l2_tortuosity_v2": 1.08,
                    }
                )
                writer.writerow(
                    {
                        "image_id": 2,
                        "status": "success",
                        "density": 99.0,
                        "alignment": 0.99,
                        "diameter_mean_nm": 99.0,
                        "l2_curvature_trimmed_mean_sqrt_length_nm": 0.2,
                        "l2_waviness_ratio_v2": 0.3,
                        "l2_tortuosity_v2": 2.0,
                    }
                )

            _, stats = run_sync(db_path, csv_path)

            self.assertEqual(stats["updated"], 1)
            self.assertEqual(stats["skipped_non_xr"], 1)
            self.assertEqual(stats["missing_images"], 0)

            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()

            xr = cur.execute("SELECT * FROM images WHERE id = 1").fetchone()
            self.assertAlmostEqual(xr["density"], 51.1)
            self.assertAlmostEqual(xr["alignment"], 0.61)
            self.assertAlmostEqual(xr["diameter"], 55.2)
            self.assertAlmostEqual(xr["curvature"], 6.0)
            self.assertAlmostEqual(xr["tortuosity"], 1.08)
            self.assertAlmostEqual(xr["waviness_ratio"], 0.15)
            self.assertAlmostEqual(xr["diameter_mean"], 55.2)
            self.assertAlmostEqual(xr["diameter_p50"], 54.0)
            self.assertAlmostEqual(xr["junction_ratio"], 0.123)
            self.assertEqual(xr["l2_curvature_label"], "Coiled")
            self.assertEqual(xr["curvature_label"], "Coiled")
            self.assertAlmostEqual(xr["curvature_p70"], 8.0)
            self.assertAlmostEqual(xr["curvature_mean"], 7.0)
            self.assertAlmostEqual(xr["curvature_trimmed_mean"], 6.0)

            zzy = cur.execute("SELECT * FROM images WHERE id = 2").fetchone()
            self.assertAlmostEqual(zzy["density"], 2.0)
            self.assertAlmostEqual(zzy["diameter"], 20.0)

            conn.close()


if __name__ == "__main__":
    unittest.main()
