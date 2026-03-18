import os
import sqlite3
import tempfile
import unittest

from tools.maintenance.migrate_xr_schema import create_xr_tables, migrate_xr_data


class XRSchemaMigrationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "xr_migration.sqlite")
        self._seed_legacy_schema()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _seed_legacy_schema(self):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT UNIQUE,
                source TEXT,
                sample_id TEXT,
                membrane_id INTEGER,
                growth_temp REAL,
                growth_time REAL,
                ar_flow REAL,
                catalyst_weight REAL,
                position_label TEXT,
                horizontal_pos TEXT,
                vertical_pos INTEGER,
                magnification INTEGER,
                actual_temp REAL,
                membrane_pos_cm REAL,
                diameter REAL,
                density REAL,
                alignment REAL,
                curvature REAL,
                tortuosity REAL,
                processed INTEGER DEFAULT 0
            )
            """
        )
        rows = [
            (
                r"d:\CNTDATA\XR\250524 T800 3H L250 0.5g\C4A1.tiff",
                "XR",
                "C4-A1",
                4,
                800.0,
                3.0,
                250.0,
                0.5,
                "C4-A1",
                "A",
                1,
                20000,
                801.7,
                13.7,
                55.2,
                52.1,
                0.11,
                0.17,
                None,
                1,
            ),
            (
                r"d:\CNTDATA\XR\250524 T800 3H L250 0.5g\C4A2.tiff",
                "XR",
                "C4-A2",
                4,
                800.0,
                3.0,
                250.0,
                0.5,
                "C4-A2",
                "A",
                2,
                20000,
                801.7,
                13.7,
                None,
                None,
                None,
                None,
                None,
                0,
            ),
            (
                r"d:\CNTDATA\XR\250309 T850 3h L300 1.5g\C2B1.tiff",
                "XR",
                "C2-B1",
                2,
                850.0,
                3.0,
                300.0,
                1.5,
                "C2-B1",
                "B",
                1,
                20000,
                815.0,
                7.0,
                61.0,
                50.0,
                0.20,
                0.10,
                None,
                1,
            ),
            (
                r"d:\CNTDATA\ZZY\No26 ... \50000-1.png",
                "ZZY",
                "No26-50000-1",
                None,
                750.0,
                3.0,
                400.0,
                None,
                "mid",
                None,
                None,
                50000,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                0,
            ),
        ]
        cur.executemany(
            """
            INSERT INTO images (
                file_path, source, sample_id, membrane_id, growth_temp, growth_time,
                ar_flow, catalyst_weight, position_label, horizontal_pos, vertical_pos,
                magnification, actual_temp, membrane_pos_cm, diameter, density, alignment,
                curvature, tortuosity, processed
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
        conn.close()

    def test_create_xr_tables_and_migrate_xr_rows(self):
        conn = sqlite3.connect(self.db_path)
        try:
            create_xr_tables(conn)
            migrate_xr_data(conn)

            cur = conn.cursor()
            tables = {
                row[0]
                for row in cur.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            self.assertTrue({"xr_runs", "xr_images", "xr_targets"}.issubset(tables))

            run_count = cur.execute("SELECT COUNT(*) FROM xr_runs").fetchone()[0]
            image_count = cur.execute("SELECT COUNT(*) FROM xr_images").fetchone()[0]
            target_count = cur.execute("SELECT COUNT(*) FROM xr_targets").fetchone()[0]
            self.assertEqual(run_count, 2)
            self.assertEqual(image_count, 3)
            self.assertEqual(target_count, 2)

            first_run = tuple(
                cur.execute(
                    "SELECT set_temp_c, ar_flow, catalyst_concentration FROM xr_runs WHERE folder_name = ?",
                    ("250524 T800 3H L250 0.5g",),
                ).fetchone()
            )
            self.assertEqual(first_run, (800.0, 250.0, 0.5))

            second_run = tuple(
                cur.execute(
                    "SELECT set_temp_c, ar_flow, catalyst_concentration FROM xr_runs WHERE folder_name = ?",
                    ("250309 T850 3h L300 1.5g",),
                ).fetchone()
            )
            self.assertEqual(second_run, (850.0, 300.0, 1.5))
        finally:
            conn.close()


if __name__ == "__main__":
    unittest.main()
