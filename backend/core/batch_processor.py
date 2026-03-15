"""
批量特征提取脚本  v2.0
========================
调用 src/analysis/feature_extractor.py（v2.0）对数据库中未处理的图像
逐一提取四特征，并将结果写回数据库。

运行方式：
    cd d:\\CNTDATA\\CNTA_ML_Project
    python backend/core/batch_processor.py

可选参数：
    --reprocess   重新处理已处理过的图像（processed=1 的也会被重跑）
    --limit N     只处理前 N 张（用于测试）
    --source ZZY  只处理指定来源
"""

import os
import sys
import sqlite3
import argparse
import cv2

# 确保可以 import src.analysis.feature_extractor
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.analysis.feature_extractor import FeatureExtractor

DB_PATH = r'd:\CNTDATA\CNTA_ML_Project\database\cnta_experiments.sqlite'


def batch_process(reprocess: bool = False, limit: int = None, source: str = None):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # 构建查询条件
    where_clauses = []
    params = []

    if not reprocess:
        where_clauses.append("processed = 0")
    if source:
        where_clauses.append("source = ?")
        params.append(source)

    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
    limit_sql = f"LIMIT {limit}" if limit else ""

    cursor.execute(
        f"SELECT id, file_path, source, magnification FROM images {where_sql} {limit_sql}",
        params
    )
    rows = cursor.fetchall()
    total = len(rows)
    print(f"待处理图像: {total} 张")

    success = 0
    skipped = 0
    errors = 0

    for i, row in enumerate(rows, 1):
        img_id   = row['id']
        path     = row['file_path']
        mag      = row['magnification']

        prefix = f"[{i:4d}/{total}]"

        if not os.path.exists(path):
            print(f"{prefix} SKIP (文件不存在): {os.path.basename(path)}")
            skipped += 1
            continue

        try:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"{prefix} SKIP (imread 失败): {os.path.basename(path)}")
                skipped += 1
                continue

            extractor = FeatureExtractor(magnification=int(mag) if mag else None)
            res = extractor.extract_all(img)

            cursor.execute("""
                UPDATE images
                SET diameter=?, density=?, alignment=?, curvature=?, processed=1
                WHERE id=?
            """, (
                res['diameter'],
                res['density'],
                res['alignment'],
                res['curvature'],
                img_id
            ))
            conn.commit()

            diam_str = f"{res['diameter']:.1f}nm" if res['diameter'] is not None else "N/A"
            print(
                f"{prefix} OK  density={res['density']:.1f}%  "
                f"S={res['alignment']:.3f}  diameter={diam_str}  "
                f"curv={res['curvature']}  | {os.path.basename(path)}"
            )
            success += 1

        except Exception as e:
            print(f"{prefix} ERROR: {os.path.basename(path)} → {e}")
            errors += 1

    conn.close()
    print(f"\n完成：成功 {success}，跳过 {skipped}，错误 {errors}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量提取 CNT 图像特征")
    parser.add_argument("--reprocess", action="store_true", help="重新处理已处理过的图像")
    parser.add_argument("--limit",     type=int, default=None, help="只处理前 N 张")
    parser.add_argument("--source",    type=str, default=None, help="只处理指定来源 (ZZY/XR)")
    args = parser.parse_args()

    batch_process(reprocess=args.reprocess, limit=args.limit, source=args.source)
