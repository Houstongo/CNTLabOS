import sqlite3
import os
import sys

# 添加项目路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.analysis.furnace_model import FurnaceModel
from backend.core.calibrator import calibrator

DB_PATH = r'd:\CNTDATA\CNTA_ML_Project\database\cnta_experiments.sqlite'

def migrate_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 尝试增加列
    columns = [
        ("actual_temp", "REAL"),
        ("membrane_pos_cm", "REAL")
    ]
    
    for col_name, col_type in columns:
        try:
            cursor.execute(f"ALTER TABLE images ADD COLUMN {col_name} {col_type}")
            print(f"Added column: {col_name}")
        except sqlite3.OperationalError:
            pass
            
    conn.commit()
    conn.close()

def recalibrate():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM images")
    rows = cursor.fetchall()
    
    print(f"Recalibrating {len(rows)} records using DataCalibrator...")
    
    for row in rows:
        data = dict(row)
        # 清除旧的校准值以强制重新计算
        data['actual_temp'] = None
        data['membrane_pos_cm'] = None
        
        calibrated = calibrator.calibrate(data)
        
        if calibrated.get('actual_temp') is not None:
            cursor.execute("""
                UPDATE images 
                SET actual_temp = ?, 
                    membrane_pos_cm = ?
                WHERE id = ?
            """, (calibrated['actual_temp'], calibrated['membrane_pos_cm'], data['id']))
            
    conn.commit()
    conn.close()
    print("Recalibration complete.")

if __name__ == "__main__":
    migrate_db()
    recalibrate()
