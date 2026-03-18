import os
import re
import pandas as pd
import sqlite3
from typing import Any, Dict, Optional


class CNTADataParser:
    def __init__(self):
        # ZZY filename parser.
        self.zzy_pattern = re.compile(
            r"No(\d+)\s+"                     # Sample ID
            r"([\d.]+)w?\s+"                  # Al2O3 Power
            r"([\d.]+)nm?\s+"                 # Al2O3 Thickness
            r"([\d.]+)w?\s+"                  # Fe Power
            r"([\d.]+)(?:nm)?(?:-\d+)?\s+"    # Fe Thickness (optional -N)
            r"([\d.]+)\s+"                    # Ar flow
            r"([\d.]+)\s+"                    # H2 flow
            r"([\d.]+)\s+"                    # C2H4 flow
            r"([\d.]+)\s+"                    # Anneal Temp
            r"([\d.]+)\s+"                    # Growth Temp
            r"([\d.min]+)\s+"                 # Anneal Time
            r"([\d.minh]+)\s+"                # Growth Time
            r"(top|mid|bottom|Middle|Top|Bottom|Position-\w+|all|bpttom)?\s*"
            r"(\d+)(?:\s+(\d+)-|-)(\d+)(?:-.*)?",
            re.IGNORECASE,
        )

        # XR filename parser: C<number><A/B/C><repeat>.
        self.xr_pattern = re.compile(r"C(\d+)([ABC])(\d+)", re.IGNORECASE)
        self.xr_temp_pattern = re.compile(r"[tT]\s*(\d+)")
        self.xr_flow_pattern = re.compile(r"[lL]\s*(\d+)")
        self.xr_catalyst_pattern = re.compile(r"(\d+(?:\.\d+)?)\s*g\b", re.IGNORECASE)

    def parse_zzy_filename(self, filename: str) -> Optional[Dict[str, Any]]:
        name = os.path.splitext(filename)[0]
        match = self.zzy_pattern.search(name)
        if not match:
            return None

        g = match.groups()
        mag = int(g[13])
        prefix = g[14]  # 可能为 None
        repeat = int(g[15])
        
        # sample_id 包含样品号-倍率-重复号(含前缀)
        full_repeat = f"{prefix}-{repeat}" if prefix is not None else str(repeat)
        sample_id = f"No{g[0]}-{mag}-{full_repeat}"

        return {
            "source": "ZZY",
            "sample_id": sample_id,
            "al2o3_power": float(g[1]),
            "al2o3_thickness": float(g[2]),
            "fe_power": float(g[3]),
            "fe_thickness": float(g[4]),
            "ar_flow": float(g[5]),
            "h2_flow": float(g[6]),
            "c2h4_flow": float(g[7]),
            "anneal_temp": float(g[8]),
            "growth_temp": float(g[9]),
            "anneal_time": g[10],
            "growth_time": g[11],
            "position_label": g[12] or "Unknown",
            "magnification": mag,
            "repeat_id": repeat,
        }

    def parse_xr_folder_metadata(self, folder_name: Optional[str]) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {}
        if not folder_name:
            return metadata

        temp_match = self.xr_temp_pattern.search(folder_name)
        if temp_match:
            metadata["growth_temp"] = float(temp_match.group(1))

        flow_match = self.xr_flow_pattern.search(folder_name)
        if flow_match:
            metadata["ar_flow"] = float(flow_match.group(1))

        catalyst_match = self.xr_catalyst_pattern.search(folder_name)
        if catalyst_match:
            metadata["catalyst_weight"] = float(catalyst_match.group(1))

        return metadata

    def parse_xr_filename(
        self,
        filename: str,
        folder_name: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        name = os.path.splitext(filename)[0]
        match = self.xr_pattern.search(name)
        if not match:
            return None

        membrane_id = int(match.group(1))
        horizontal_pos = match.group(2).upper()
        vertical_pos = int(match.group(3))

        data = {
            "source": "XR",
            "sample_id": f"C{membrane_id}-{horizontal_pos}{vertical_pos}",
            "membrane_id": membrane_id,
            "horizontal_pos": horizontal_pos,
            "vertical_pos": vertical_pos,
            "growth_temp": 800.0,
            "growth_time": 3.0,
            "ar_flow": 250.0,
            "catalyst_weight": 1.0,
            "position_label": f"C{membrane_id}-{horizontal_pos}{vertical_pos}",
        }
        data.update(self.parse_xr_folder_metadata(folder_name))
        return data


class DatabaseManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_db()

    def init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT UNIQUE,
                source TEXT,
                sample_id TEXT,
                membrane_id INTEGER,
                growth_temp REAL,
                actual_temp REAL,
                membrane_pos_cm REAL,
                growth_time REAL,
                ar_flow REAL,
                h2_flow REAL,
                c2h4_flow REAL,
                al2o3_power REAL,
                al2o3_thickness REAL,
                fe_power REAL,
                fe_thickness REAL,
                anneal_temp REAL,
                anneal_time REAL,
                position_label TEXT,
                magnification INTEGER,
                horizontal_pos TEXT,
                vertical_pos INTEGER,
                repeat_id INTEGER,
                catalyst_weight REAL,
                diameter REAL,
                density REAL,
                alignment REAL,
                curvature TEXT,
                tortuosity REAL,
                processed INTEGER DEFAULT 0
            )
            """
        )
        conn.commit()
        conn.close()

    def clear_tables(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM images")
        conn.commit()
        conn.close()
        print("Database tables cleared.")

    def insert_image(self, data: Dict[str, Any]):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        keys = data.keys()
        columns = ", ".join(keys)
        placeholders = ", ".join(["?"] * len(keys))
        sql = f"INSERT OR IGNORE INTO images ({columns}) VALUES ({placeholders})"

        cursor.execute(sql, list(data.values()))
        conn.commit()
        conn.close()


if __name__ == "__main__":
    parser = CNTADataParser()
    test_zzy = "No26 200w 5.0nm 5w 0.75nm 400 200 100 600 750 15min 180min mid 50000-1.png"
    print(f"Testing ZZY: {parser.parse_zzy_filename(test_zzy)}")

    test_xr = "C1A1.tiff"
    print(f"Testing XR: {parser.parse_xr_filename(test_xr)}")
