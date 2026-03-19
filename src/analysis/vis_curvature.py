"""
生成曲率极端图像的可视化
======================
供大模型直接评判，不使用算法
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


def visualize_for_judgment(img_paths, output_path, title):
    """生成评判用的可视化图"""
    n = len(img_paths)

    # 3x4布局
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    for i, path in enumerate(img_paths):
        if i >= 12:  # 最多12张
            break

        row = i // 4
        col = i % 4

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        axes[row, col].imshow(img, cmap='gray')
        axes[row, col].set_title(os.path.basename(path), fontsize=8)
        axes[row, col].axis('off')

    # 隐藏多余的子图
    for i in range(n, 12):
        row = i // 4
        col = i % 4
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    # 读取路径列表
    with open(r'd:\CNTDATA\CNTA_ML_Project\src\analysis\curved_paths.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 解析
    curved_paths = []
    straight_paths = []
    current_section = None

    for line in lines:
        line = line.strip()
        if line.startswith('==='):
            if '卷曲' in line:
                current_section = 'curved'
            elif '笔直' in line:
                current_section = 'straight'
            continue

        if not line:
            continue

        if current_section == 'curved':
            curved_paths.append(line)
        elif current_section == 'straight':
            straight_paths.append(line)

    print("卷曲图像:", len(curved_paths))
    print("笔直图像:", len(straight_paths))

    # 生成可视化
    output_dir = r"d:\CNTDATA\CNTA_ML_Project\src\analysis\vis_curvature_judgment"
    os.makedirs(output_dir, exist_ok=True)

    # 前12张卷曲
    visualize_for_judgment(
        curved_paths[:12],
        os.path.join(output_dir, "01_curved_top12.png"),
        "曲率最高（卷曲）- 前12张"
    )

    # 前12张笔直
    visualize_for_judgment(
        straight_paths[:12],
        os.path.join(output_dir, "02_straight_top12.png"),
        "曲率最低（笔直）- 前12张"
    )

    print("\n可视化已生成:")
    print(f"  1. {output_dir}/01_curved_top12.png")
    print(f"  2. {output_dir}/02_straight_top12.png")


if __name__ == "__main__":
    main()
