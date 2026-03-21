import os
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
from fastapi import HTTPException

from backend import main as api_main


class XRApiListingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "api.sqlite")
        self.image_root = os.path.join(self.temp_dir.name, "images_root")
        os.makedirs(self.image_root, exist_ok=True)

        self.xr_folder = os.path.join(self.image_root, "XR", "250524 T800 3H L250 0.5g")
        os.makedirs(self.xr_folder, exist_ok=True)
        self.xr_file = os.path.join(self.xr_folder, "C4A1.tiff")
        with open(self.xr_file, "wb") as handle:
            handle.write(b"")

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
                processed INTEGER DEFAULT 0,
                is_deleted INTEGER DEFAULT 0
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE xr_runs (
                run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                folder_name TEXT UNIQUE NOT NULL,
                source TEXT NOT NULL DEFAULT 'XR',
                set_temp_c REAL NOT NULL,
                growth_time_h REAL,
                ar_flow REAL NOT NULL,
                catalyst_concentration REAL NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE xr_images (
                image_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                file_path TEXT UNIQUE NOT NULL,
                sample_id TEXT,
                position_label TEXT,
                horizontal_pos TEXT,
                vertical_pos INTEGER,
                membrane_pos_cm REAL,
                magnification INTEGER,
                actual_temp_c REAL,
                processed INTEGER DEFAULT 0
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE xr_targets (
                target_id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id INTEGER NOT NULL,
                diameter REAL,
                density REAL,
                alignment REAL,
                curvature REAL,
                tortuosity REAL
            )
            """
        )

        cur.execute(
            """
            INSERT INTO images (
                file_path, source, sample_id, growth_temp, growth_time, ar_flow,
                catalyst_weight, position_label, horizontal_pos, vertical_pos,
                magnification, actual_temp, membrane_pos_cm, processed
            ) VALUES (?, 'XR', 'C4-A1', 800.0, 3.0, 250.0, NULL, 'C4-A1', 'A', 1, 20000, NULL, 13.7, 1)
            """,
            (self.xr_file,),
        )
        legacy_image_id = cur.lastrowid

        cur.execute(
            """
            INSERT INTO xr_runs (
                folder_name, set_temp_c, growth_time_h, ar_flow, catalyst_concentration
            ) VALUES ('250524 T800 3H L250 0.5g', 800.0, 3.0, 250.0, 0.5)
            """
        )
        run_id = cur.lastrowid

        cur.execute(
            """
            INSERT INTO xr_images (
                run_id, file_path, sample_id, position_label, horizontal_pos, vertical_pos,
                membrane_pos_cm, magnification, actual_temp_c, processed
            ) VALUES (?, ?, 'C4-A1', 'C4-A1', 'A', 1, 13.7, 20000, 801.7, 1)
            """,
            (run_id, self.xr_file),
        )
        xr_image_id = cur.lastrowid

        cur.execute(
            """
            INSERT INTO xr_targets (
                image_id, diameter, density, alignment, curvature, tortuosity
            ) VALUES (?, 55.2, 52.1, 0.11, 0.17, NULL)
            """,
            (xr_image_id,),
        )

        conn.commit()
        conn.close()
        self.legacy_image_id = legacy_image_id

    def tearDown(self):
        self.temp_dir.cleanup()

    async def test_xr_list_prefers_normalized_tables_and_keeps_legacy_id(self):
        with patch.object(api_main, "DB_PATH", self.db_path), patch.object(api_main, "IMAGE_ROOT", self.image_root):
            result = await api_main.get_image_list(source="XR", limit=10, offset=0, sort_by="id", order="desc")

        self.assertEqual(result["total"], 1)
        item = result["items"][0]
        self.assertEqual(item["id"], self.legacy_image_id)
        self.assertEqual(item["source"], "XR")
        self.assertEqual(item["growth_temp"], 800.0)
        self.assertEqual(item["ar_flow"], 250.0)
        self.assertEqual(item["catalyst_weight"], 0.5)
        self.assertEqual(item["actual_temp"], 801.7)
        self.assertEqual(item["diameter"], 55.2)
        self.assertEqual(item["density"], 52.1)

    async def test_xr_list_excludes_logically_deleted_images(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("UPDATE images SET is_deleted = 1 WHERE id = ?", (self.legacy_image_id,))
        conn.commit()
        conn.close()

        with patch.object(api_main, "DB_PATH", self.db_path), patch.object(api_main, "IMAGE_ROOT", self.image_root):
            result = await api_main.get_image_list(source="XR", limit=10, offset=0, sort_by="id", order="desc")

        self.assertEqual(result["total"], 0)
        self.assertEqual(result["items"], [])

    async def test_xr_list_can_show_deleted_view(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("UPDATE images SET is_deleted = 1 WHERE id = ?", (self.legacy_image_id,))
        conn.commit()
        conn.close()

        with patch.object(api_main, "DB_PATH", self.db_path), patch.object(api_main, "IMAGE_ROOT", self.image_root):
            result = await api_main.get_image_list(
                source="XR",
                deletion_view="deleted",
                limit=10,
                offset=0,
                sort_by="id",
                order="desc",
            )

        self.assertEqual(result["total"], 1)
        self.assertEqual(result["items"][0]["id"], self.legacy_image_id)
        self.assertEqual(result["items"][0]["is_deleted"], 1)

    async def test_xr_list_can_show_all_view(self):
        with patch.object(api_main, "DB_PATH", self.db_path), patch.object(api_main, "IMAGE_ROOT", self.image_root):
            baseline = await api_main.get_image_list(source="XR", limit=10, offset=0, sort_by="id", order="desc")

        conn = sqlite3.connect(self.db_path)
        conn.execute("INSERT INTO images (file_path, source, sample_id, magnification, is_deleted) VALUES (?, 'XR', 'C4-A2', 20000, 1)", (self.xr_file + '.deleted',))
        conn.commit()
        conn.close()

        with patch.object(api_main, "DB_PATH", self.db_path), patch.object(api_main, "IMAGE_ROOT", self.image_root):
            result = await api_main.get_image_list(
                source="XR",
                deletion_view="all",
                limit=10,
                offset=0,
                sort_by="id",
                order="desc",
            )

        self.assertEqual(baseline["total"], 1)
        self.assertEqual(result["total"], 2)
        self.assertEqual(sorted(item["is_deleted"] for item in result["items"]), [0, 1])

    async def test_hard_delete_rejects_active_record(self):
        with patch.object(api_main, "DB_PATH", self.db_path):
            with self.assertRaises(HTTPException) as ctx:
                await api_main.delete_image(self.legacy_image_id)

        self.assertEqual(ctx.exception.status_code, 409)

    async def test_hard_delete_removes_deleted_record(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("UPDATE images SET is_deleted = 1 WHERE id = ?", (self.legacy_image_id,))
        conn.commit()
        conn.close()

        with patch.object(api_main, "DB_PATH", self.db_path):
            result = await api_main.delete_image(self.legacy_image_id)

        self.assertEqual(result["status"], "success")

        conn = sqlite3.connect(self.db_path)
        remaining = conn.execute("SELECT COUNT(*) FROM images WHERE id = ?", (self.legacy_image_id,)).fetchone()[0]
        conn.close()
        self.assertEqual(remaining, 0)


if __name__ == "__main__":
    unittest.main()
