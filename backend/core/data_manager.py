import os
import re
import pandas as pd
import sqlite3
from typing import Dict, Any, Optional

class CNTADataParser:
    def __init__(self):
        # ZZY 正则表达式 (14个参数)
        # 支持: No26 200w 5.0nm 5w 0.75nm 400 200 100 600 750 15min 180min mid 1000-1.png
        # ZZY 正则表达式 (增强版)
        self.zzy_pattern = re.compile(
            r"No(\d+)\s+"                     # 序号 (Group 1)
            r"([\d.]+)w?\s+"                  # Al2O3功率 (Group 2)
            r"([\d.]+)nm?\s+"                 # Al2O3厚度 (Group 3)
            r"([\d.]+)w?\s+"                  # Fe功率 (Group 4)
            r"([\d.]+)nm?\s+"                 # Fe厚度 (Group 5)
            r"([\d.]+)\s+"                    # Ar (Group 6)
            r"([\d.]+)\s+"                    # H2 (Group 7)
            r"([\d.]+)\s+"                    # C2H4 (Group 8)
            r"([\d.]+)\s+"                    # 退火温 (Group 9)
            r"([\d.]+)\s+"                    # 生长温 (Group 10)
            r"([\d.min]+)\s+"                 # 退火时 (Group 11)
            r"([\d.minh]+)\s+"                # 生长时 (Group 12)
            r"(top|mid|bottom|Middle|Top|Bottom|Position-\w+)?\s*" # 部位 (Group 13)
            r"(\d+)-(\d+)(?:-.*)?",           # 倍率-重复 (Group 14, 15)
            re.IGNORECASE
        )
        
        # XR 正则表达式: C(序号)(A/B/C)(1/2/3)
        self.xr_pattern = re.compile(r"C(\d+)([ABC])(\d+)", re.IGNORECASE)

    def parse_zzy_filename(self, filename: str) -> Optional[Dict[str, Any]]:
        """
        解析 ZZY 的长文件名
        """
        name = os.path.splitext(filename)[0]
        match = self.zzy_pattern.search(name)
        if match:
            g = match.groups()
            return {
                'source': 'ZZY',
                'sample_id': f"No{g[0]}-{g[13]}-{g[14]}", # 包含倍率和重复ID
                'al2o3_power': float(g[1]),
                'al2o3_thickness': float(g[2]),
                'fe_power': float(g[3]),
                'fe_thickness': float(g[4]),
                'ar_flow': float(g[5]),
                'h2_flow': float(g[6]),
                'c2h4_flow': float(g[7]),
                'anneal_temp': float(g[8]),
                'growth_temp': float(g[9]),
                'anneal_time': g[10],
                'growth_time': g[11],
                'position_label': g[12] or "Unknown",
                'magnification': int(g[13]),
                'repeat_id': int(g[14])
            }
        return None

    def parse_xr_filename(self, filename: str) -> Optional[Dict[str, Any]]:
        """
        解析 XR 的文件名: C序号ABC123
        """
        name = os.path.splitext(filename)[0]
        match = self.xr_pattern.search(name)
        if match:
            g = match.groups()
            membrane_id = int(g[0])
            h_pos = g[1].upper() # A, B, C
            v_pos = int(g[2])    # 1, 2, 3
            
            # XR 默认参数
            return {
                'source': 'XR',
                'sample_id': f"C{membrane_id}-{h_pos}{v_pos}",
                'membrane_id': membrane_id,
                'horizontal_pos': h_pos,
                'vertical_pos': v_pos,
                'growth_temp': 800.0,      # 默认
                'growth_time': 3.0,        # 默认
                'ar_flow': 250.0,          # 默认
                'catalyst_weight': 1.0,    # 1g 二茂铁
                'position_label': f"C{membrane_id}-{h_pos}{v_pos}"
            }
        return None

class DatabaseManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_db()

    def init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        # 建立大一统索引表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT UNIQUE,
                source TEXT,
                sample_id TEXT,
                membrane_id INTEGER,
                growth_temp REAL,
                actual_temp REAL,           -- 实际生长温度 (校准后)
                membrane_pos_cm REAL,       -- 膜在炉管中的物理位置 (cm)
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
                
                -- 特征字段 (由 FeatureExtractor 填入)
                diameter REAL,
                density REAL,
                alignment REAL,
                curvature TEXT,
                tortuosity REAL,
                processed INTEGER DEFAULT 0
            )
        """)
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
        columns = ', '.join(keys)
        placeholders = ', '.join(['?'] * len(keys))
        sql = f"INSERT OR IGNORE INTO images ({columns}) VALUES ({placeholders})"
        
        cursor.execute(sql, list(data.values()))
        conn.commit()
        conn.close()

if __name__ == "__main__":
    # 局部测试
    parser = CNTADataParser()
    test_zzy = "No26 200w 5.0nm 5w 0.75nm 400 200 100 600 750 15min 180min mid 50000-1.png"
    # 注意: 原命名有 15min 180min, 代码正则需调整兼容
    print(f"Testing ZZY: {parser.parse_zzy_filename(test_zzy)}")
    
    test_xr = "C1A1.tiff"
    print(f"Testing XR: {parser.parse_xr_filename(test_xr)}")
