"""
根据曲率筛选极端图像
====================
找出曲率最大和最小的各20张图像
"""

import sqlite3
import cv2
import numpy as np
from feature_extractor import FeatureExtractor

DB_PATH = r"d:\CNTDATA\CNTA_ML_Project\database\cnta_experiments.sqlite"


def get_all_images():
    """从数据库获取所有图像"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    query = """
    SELECT id, file_path, magnification
    FROM images
    WHERE file_path IS NOT NULL
    """
    cursor.execute(query)
    results = cursor.fetchall()

    conn.close()
    return results


def extract_curvature_features(img_path, mag):
    """提取曲率相关特征"""
    if not mag or mag < 5000:
        return None

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    extractor = FeatureExtractor(magnification=mag, diameter_method='enhanced')
    res = extractor.extract_all(img)

    return {
        'curvature': res['curvature'],
        'tortuosity': res['tortuosity'],
        'alignment': res['alignment'],
        'mean_phi_deg': res['mean_phi_deg'],
        'diameter': res['diameter'],
        'density': res['density'],
    }


def main():
    print("="*70)
    print("根据曲率筛选极端图像")
    print("="*70)

    # 获取所有图像
    all_images = get_all_images()
    print(f"数据库中共有 {len(all_images)} 张图像")

    # 提取特征
    features = []
    for img_id, file_path, mag in all_images:
        feats = extract_curvature_features(file_path, mag)
        if feats is None:
            continue

        features.append({
            'id': img_id,
            'path': file_path,
            'mag': mag,
            **feats
        })

    print(f"成功提取 {len(features)} 张图像的特征")

    # 按曲率分类
    # 曲率分类：Straight(直) < Wavy(波) < Coiled(卷)
    curvature_order = {'Straight': 0, 'Wavy': 1, 'Coiled': 2}

    for feat in features:
        feat['curvature_score'] = curvature_order.get(feat['curvature'], -1)

    # 按曲率得分排序（Coiled最高）
    sorted_by_curvature = sorted(features, key=lambda x: x['curvature_score'], reverse=True)

    # 提取前20（最卷曲）和后20（最直）
    top_curved = sorted_by_curvature[:20]
    bottom_straight = sorted_by_curvature[-20:]

    print(f"\n最卷曲的20张（Coiled/Wavy）：")
    for i, f in enumerate(top_curved, 1):
        mag_str = f"{f['mag']}x" if f['mag'] else "N/A"
        print(f"  {i:2d}. {f['curvature']:10s} | {mag_str:>8s} | tort={f['tortuosity']:.3f}")

    print(f"\n最笔直的20张（Straight）：")
    for i, f in enumerate(reversed(bottom_straight), 1):
        mag_str = f"{f['mag']}x" if f['mag'] else "N/A"
        print(f"  {i:2d}. {f['curvature']:10s} | {mag_str:>8s} | tort={f['tortuosity']:.3f}")

    # 统计倍数覆盖
    mag_counts = {}
    for f in top_curved + bottom_straight:
        mag = f['mag'] if f['mag'] else 0
        mag_counts[mag] = mag_counts.get(mag, 0) + 1

    print(f"\n倍数覆盖统计：")
    for mag in sorted(mag_counts.keys()):
        print(f"  {mag if mag else 'N/A'}x: {mag_counts[mag]} 张")

    # 保存列表
    import csv

    with open(r'd:\CNTDATA\CNTA_ML_Project\src\analysis\curvature_extremes.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['ID', 'Path', 'Mag', 'Curvature', 'Tortuosity', 'Alignment', 'Diameter', 'Density'])

        for f in top_curved:
            writer.writerow([f['id'], f['path'], f['mag'], f['curvature'], f['tortuosity'], f['alignment'], f['diameter'], f['density']])

        writer.writerow([])  # 空行分隔

        for f in reversed(bottom_straight):
            writer.writerow([f['id'], f['path'], f['mag'], f['curvature'], f['tortuosity'], f['alignment'], f['diameter'], f['density']])

    print(f"\n列表已保存至: curvature_extremes.csv")

    return top_curved, bottom_straight


if __name__ == "__main__":
    top_curved, bottom_straight = main()
