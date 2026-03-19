"""
从数据库查询曲率极端图像（仅ZZY数据）
========================================
"""

import sqlite3
import os

DB_PATH = r"d:\CNTDATA\CNTA_ML_Project\database\cnta_experiments.sqlite"


def query_curvature_extremes():
    """查询曲率最大和最小的图像（仅ZZY）"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 查询ZZY图像（路径包含ZZY）且有曲率标签（高倍率）
    query = """
    SELECT id, file_path, magnification, curvature
    FROM images
    WHERE file_path IS NOT NULL
    AND file_path LIKE '%ZZY%'
    AND curvature IS NOT NULL
    AND curvature != 'Unknown'
    AND curvature != 'N/A'
    AND magnification >= 20000
    ORDER BY id
    """

    cursor.execute(query)
    results = cursor.fetchall()

    conn.close()

    return results


def main():
    print("="*70)
    print("查询曲率极端图像（ZZY数据）")
    print("="*70)

    results = query_curvature_extremes()
    print("找到 " + str(len(results)) + " 张有曲率标签的ZZY图像（倍率>=20kx）")

    # 曲率排序：Coiled(卷) > Wavy(波) > Straight(直)
    curvature_order = {'Coiled': 2, 'Wavy': 1, 'Straight': 0}

    # 添加得分
    data = []
    for img_id, file_path, mag, curvature in results:
        data.append({
            'id': img_id,
            'path': file_path,
            'mag': mag,
            'curvature': curvature,
            'score': curvature_order.get(curvature, -1)
        })

    # 按得分排序（Coiled最高）
    sorted_data = sorted(data, key=lambda x: x['score'], reverse=True)

    # 提取前20（最卷）和后20（最直）
    top_curved = sorted_data[:20]
    bottom_straight = sorted_data[-20:]

    print("\n最卷曲的20张（Coiled/Wavy）：")
    for i, d in enumerate(top_curved, 1):
        mag_str = str(d['mag']) + "x" if d['mag'] else "N/A"
        curv_str = str(d['curvature'])
        filename = os.path.basename(d['path'])
        print("  " + str(i).rjust(2) + ". " + curv_str.rjust(10) + " | " + mag_str.rjust(8) + " | " + filename)

    print("\n最笔直的20张（Straight/Wavy）：")
    for i, d in enumerate(reversed(bottom_straight), 1):
        mag_str = str(d['mag']) + "x" if d['mag'] else "N/A"
        curv_str = str(d['curvature'])
        filename = os.path.basename(d['path'])
        print("  " + str(i).rjust(2) + ". " + curv_str.rjust(10) + " | " + mag_str.rjust(8) + " | " + filename)

    # 统计倍率覆盖
    mag_counts = {}
    for d in top_curved + bottom_straight:
        mag = d['mag'] if d['mag'] else 0
        mag_counts[mag] = mag_counts.get(mag, 0) + 1

    print("\n倍率覆盖统计：")
    for mag in sorted(mag_counts.keys()):
        mag_str = str(mag) + "x" if mag else "N/A"
        print("  " + mag_str.rjust(8) + ": " + str(mag_counts[mag]) + " 张")

    # 保存路径列表
    with open(r'd:\CNTDATA\CNTA_ML_Project\src\analysis\curved_paths.txt', 'w', encoding='utf-8') as f:
        f.write("=== 最卷曲的20张 ===\n")
        for d in top_curved:
            f.write(d['path'] + "\n")

        f.write("\n=== 最笔直的20张 ===\n")
        for d in reversed(bottom_straight):
            f.write(d['path'] + "\n")

    print("\n路径列表已保存至: curved_paths.txt")

    # 返回路径列表
    curved_paths = [d['path'] for d in top_curved]
    straight_paths = [d['path'] for d in reversed(bottom_straight)]

    return curved_paths, straight_paths


if __name__ == "__main__":
    curved_paths, straight_paths = main()
