"""
更新管径为"管径分布"
=======================

修改内容：
1. 数据库：添加统计字段（mean, std, min, max, p25, p50, p75）
2. 算法：从二值图计算管径分布统计
3. 批量处理：更新所有未处理的ZZY数据
4. 前端：修改显示文本

数据来源：src.analysis.feature_extractor.py
"""

import sqlite3
import numpy as np
import cv2
import sys
import os
from datetime import datetime

# 添加项目根目录到 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analysis.feature_extractor import FeatureExtractor

# 数据库路径
DB_PATH = r"d:\CNTDATA\CNTA_ML_Project\database\cnta_experiments.sqlite"


def add_diameter_stats_columns():
    """添加管径分布统计字段"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    print("="*70)
    print("添加管径分布统计字段")
    print("="*70)

    # 新增列
    new_columns = [
        ('diameter_mean', 'REAL', '管径平均值（nm)'),
        ('diameter_std', 'REAL', '管径标准差（nm）'),
        ('diameter_min', 'REAL', '管径最小值（nm）'),
        ('diameter_max', 'REAL', '管径最大值（nm）'),
        ('diameter_p25', 'REAL', '管径25%分位数（nm）'),
        ('diameter_p50', 'REAL', '管径50%分位数（nm）'),
        ('diameter_p75', 'REAL', '管径75%分位数（nm）'),
        ('diameter_distribution', 'TEXT', '管径分布：JSON'),
    ]

    for col_name, col_type, col_comment in new_columns:
        try:
            cursor.execute(f"ALTER TABLE images ADD COLUMN {col_name} {col_type}")
            print(f"  添加列: {col_name} ({col_type})")
        except sqlite3.OperationalError as e:
            print(f"  列 {col_name} 已存在，跳过")

    # 修改diameter列注释
    cursor.execute("""
        PRAGMA table_info(images)
    """)
    table_info = cursor.fetchall()

    for row in table_info:
        if row[1] == 'diameter':
            cid = row[0]
            try:
                cursor.execute(f"""
                    CREATE TABLE images_{cid} (
                        id INTEGER PRIMARY KEY,
                        diameter_mean REAL,
                        diameter_std REAL,
                        diameter_min REAL,
                        diameter_max REAL,
                        diameter_p25 REAL,
                        diameter_p50 REAL,
                        diameter_p75 REAL,
                        diameter_distribution TEXT
                    ) AS SELECT * FROM images WHERE id = {cid}
                """)
                cursor.execute(f"DROP TABLE images_{cid}")
                print(f" 创建触发器: images_{cid}")
                print(f" 迁移数据到新表: {cursor.execute('SELECT COUNT(*) FROM images WHERE id = ' + str(cid) + ' AND diameter IS NOT NULL')[0]} 行")
            except Exception as e:
                print(f" 处理 {cid} 失败: {e}")

    conn.commit()
    conn.close()
    print("数据库修改完成")


def calculate_diameter_distribution(thresh, extractor):
    """
    计算管径分布统计

    Returns
    -------
    dict : {
        'mean': 平均值（nm）,
        'std': 标准差（nm）,
        'min': 最小值（nm）,
        'max': 最大值（nm）,
        'p25': 25%分位数（nm）,
        'p50': 50%分位数（nm）,
        'p75': 75%分位数（nm）,
        'distribution_json': 分布的JSON表示
    }
    """
    # 距离变换
    from scipy import ndimage as ndi
    dist = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)

    # 获取所有距离值（仅前景）
    all_distances = dist[thresh > 0].ravel()

    if len(all_distances) == 0:
        return {
            'mean': 0,
            'std': 0,
            'min': 0,
            'max': 0,
            'p25': 0,
            'p50': 0,
            'p75': 0,
            'distribution_json': '[]'
        }

    # 统计量
    mean_px = np.mean(all_distances)
    std_px = np.std(all_distances)
    min_px = np.min(all_distances)
    max_px = np.max(all_distances)

    # 百分位数
    p25_px = np.percentile(all_distances, 25)
    p50_px = np.percentile(all_distances, 50)
    p75_px = np.percentile(all_distances, 75)

    # 转换为nm
    px_per_nm = extractor.px_per_um / 1000.0

    # 确保值不为0
    px_per_nm = max(px_per_nm, 0.001)

    mean_nm = mean_px / px_per_nm
    std_nm = std_px / px_per_nm
    min_nm = min_px / px_per_nm
    max_nm = max_px / px_per_nm
    p25_nm = p25_px / px_per_nm
    p50_nm = p50_px / px_per_nm
    p75_nm = p75_px / px_per_nm

    # 分布JSON（简单直方图）
    hist, bin_edges = np.histogram(all_distances, bins=20)
    distribution = []
    for i in range(len(hist) - 1):
        distribution.append({
            'count': int(hist[i]),
            'range': f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f} nm"
        })

    return {
        'mean': mean_nm,
        'std': std_nm,
        'min': min_nm,
        'max': max_nm,
        'p25': p25_nm,
        'p50': p50_nm,
        'p75': p75_nm,
        'distribution_json': str(distribution)
    }


def update_images_batch(source='ZZY', limit=50):
    """批量更新图像的管径分布（支持ZZY和XR）"""

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 查询需要处理的图像（缺少管径分布数据）
    query = """
        SELECT id, file_path, magnification
        FROM images
        WHERE file_path LIKE ?
        AND magnification >= 20000
        AND diameter_mean IS NULL
    """
    if limit:
        query += f" LIMIT {limit}"

    cursor.execute(query, (f'%{source}%',))
    images = cursor.fetchall()

    print("="*70)
    print(f"开始批量更新{source}管径分布（共 {len(images)} 张图像）")
    print("="*70)

    updated = 0
    errors = []

    for img_id, file_path, mag in images:
        if not mag or mag < 20000:
            continue

        print(f"[{updated+1}/{len(images)}] 处理: {os.path.basename(file_path)} | mag={mag}")

        try:
            # 提取特征
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                errors.append((img_id, f"无法读取图像"))
                continue

            extractor = FeatureExtractor(magnification=mag, diameter_method='enhanced')

            # 获取二值图（用于计算分布）
            extractor._calibrate(img.shape[1])
            roi = extractor.extract_roi(img)
            processed = extractor.preprocess(roi)

            block = max(11, int(extractor.expected_tube_px * 4) | 1)
            block = min(block, 51)
            thresh = cv2.adaptiveThreshold(
                processed, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                block, 2
            )

            # 计算分布
            stats = calculate_diameter_distribution(thresh, extractor)

            # 构建distribution_json（转换numpy类型为Python原生类型）
            distribution_data = {
                'mean': float(stats['mean']),
                'std': float(stats['std']),
                'min': float(stats['min']),
                'max': float(stats['max']),
                'p25': float(stats['p25']),
                'p50': float(stats['p50']),
                'p75': float(stats['p75']),
            }
            import json
            distribution_json = json.dumps(distribution_data)

            # 更新数据库
            cursor.execute("""
                UPDATE images
                SET diameter_mean = ?,
                    diameter_std = ?,
                    diameter_min = ?,
                    diameter_max = ?,
                    diameter_p25 = ?,
                    diameter_p50 = ?,
                    diameter_p75 = ?,
                    diameter_distribution = ?
                WHERE id = ?
            """, (float(stats['mean']), float(stats['std']), float(stats['min']), float(stats['max']),
                 float(stats['p25']), float(stats['p50']), float(stats['p75']), distribution_json,
                 img_id))

            updated += 1
            print(f"  完成: mean={stats['mean']:.2f} nm, std={stats['std']:.2f} nm")

        except Exception as e:
            errors.append((img_id, str(e)))
            print(f"  错误: {e}")

    conn.commit()
    conn.close()

    print("="*70)
    print(f"批量更新完成: 成功 {updated} 张, 失败 {len(errors)} 张")
    if errors:
        print("\n错误列表:")
        for img_id, err in errors[:5]:
            print(f"  ID {img_id}: {err}")


def main():
    import sys

    print("="*70)
    print("管径分布更新任务")
    print("="*70)

    # 步骤1：添加统计字段
    add_diameter_stats_columns()

    # 步骤2：批量更新ZZY数据
    update_images_batch(source='ZZY', limit=1000)

    # 步骤3：批量更新XR数据
    update_images_batch(source='XR', limit=1000)

    print("\n步骤4: 前端修改")
    print("  需要修改 index.html 中的显示逻辑：")
    print("  1. 将文本 '平均管径' 改为 '管径分布'")
    print("  2. 在卡片中添加统计展示（mean ± std, 范围）")
    print("  3. 更新tooltip和图表（如直方图）")
    print("\n文件位置:")
    print("  - 数据库: database/cnta_experiments.sqlite")
    print("  - 后端API: backend/core/ (现有_diameter端点)")
    print("  - 前端: index.html")

    sys.exit(0)


if __name__ == "__main__":
    main()
