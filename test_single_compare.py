#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
单图对比测试 - 输出：原图 + 两种分割MASK + 五种参数对比
"""

import os
import sys
import base64
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT.parent / "VLMSAM"))

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from backend.core.algorithm_visualizer import AlgorithmVisualizer
from backend.core.cntsegnet_visualizer import CNTSegNetVisualizer


def extract_features_from_steps(steps, suffix='', debug=False):
    """从步骤描述中提取特征值"""
    import re
    features = {}

    for step in steps:
        desc = step.get('description', '')
        name = step.get('name', '')

        if debug:
            print(f"    Step: {repr(name)}")
            print(f"    Desc: {repr(desc[:100])}")

        # 密度
        if 'density' in name.lower() or '密度' in name:
            m = re.search(r'density\s*=\s*([\d.]+)', desc, re.I)
            if m:
                features[f'density{suffix}'] = float(m.group(1))

        # 取向度
        if 'alignment' in name.lower() or '对齐' in name or '方向' in name:
            m = re.search(r'alignment\s*=\s*(\d+\.\d+)', desc, re.I)
            if m:
                features[f'alignment{suffix}'] = float(m.group(1))

        # 直径
        if '直径' in name:
            m = re.search(r'平均直径\s*=\s*([\d.]+)\s*nm', desc)
            if m:
                features[f'diameter_avg{suffix}'] = float(m.group(1))
            m = re.search(r'标准差\s*=\s*([\d.]+)\s*nm', desc)
            if m:
                features[f'diameter_std{suffix}'] = float(m.group(1))

        # 曲率
        if 'curvature' in name.lower() or '曲率' in name:
            m = re.search(r'κ\s*=\s*([\d.]+)', desc)
            if m:
                features[f'curvature{suffix}'] = float(m.group(1))

        # 波曲度
        if 'tortuosity' in name.lower() or '波曲度' in name:
            m = re.search(r'τ\s*=\s*([\d.]+)', desc)
            if not m:
                m = re.search(r'tortuosity\s*=\s*([\d.]+)', desc, re.I)
            if m:
                features[f'tortuosity{suffix}'] = float(m.group(1))

    return features


def get_step_image(steps, step_name):
    """获取指定步骤的图片"""
    for step in steps:
        if step_name in step.get('name', '').lower():
            img_data = base64.b64decode(step['image'])
            img_array = np.frombuffer(img_data, dtype=np.uint8)
            return cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return None


def process_single(img_path, output_dir="./test_output"):
    """处理单张图片，输出对比图"""

    img_path = Path(img_path)
    if not img_path.exists():
        print(f"Error: {img_path} not found")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 读取图片
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Cannot read image")
        return

    h, w = img.shape
    img_name = img_path.stem
    print(f"Processing: {img_name} ({h}x{w})")

    # 解析倍率（先匹配大的数字避免误匹配）
    mag = 50000  # 默认
    if '100000' in img_name:
        mag = 100000
    elif '50000' in img_name:
        mag = 50000
    elif '10000' in img_name:
        mag = 10000
    elif '5000' in img_name:
        mag = 5000
    elif '1000' in img_name:
        mag = 1000

    print(f"  Magnification: {mag}x")

    # === Threshold 方法 ===
    print("  Running Threshold method...")
    vis_t = AlgorithmVisualizer(magnification=mag)
    steps_t = vis_t.visualize_extraction(img)
    features_t = extract_features_from_steps(steps_t, '_t')
    print(f"    Features: {features_t}")

    # === CNTSegNet 方法 ===
    print("  Running CNTSegNet method...")
    vis_c = CNTSegNetVisualizer(magnification=mag, device='cpu')
    steps_c = vis_c.visualize_extraction(img)
    features_c = extract_features_from_steps(steps_c, '_c', debug=False)
    print(f"    Features: {features_c}")

    # 获取分割MASK图
    mask_t = get_step_image(steps_t, '二值化')
    if mask_t is None:
        mask_t = get_step_image(steps_t, 'binary')
    mask_c = get_step_image(steps_c, '概率图融合')
    if mask_c is None:
        mask_c = get_step_image(steps_c, 'prob')

    # 创建对比图
    fig = plt.figure(figsize=(16, 10))

    # 上排：原图 + 两种分割
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(img, cmap='gray')
    ax1.set_title('Original', fontsize=12)
    ax1.axis('off')

    ax2 = fig.add_subplot(2, 3, 2)
    if mask_t is not None:
        ax2.imshow(cv2.cvtColor(mask_t, cv2.COLOR_BGR2RGB))
    ax2.set_title('Threshold Mask', fontsize=12)
    ax2.axis('off')

    ax3 = fig.add_subplot(2, 3, 3)
    if mask_c is not None:
        ax3.imshow(cv2.cvtColor(mask_c, cv2.COLOR_BGR2RGB))
    ax3.set_title('CNTSegNet Mask', fontsize=12)
    ax3.axis('off')

    # 下排：参数对比表
    ax_table = fig.add_subplot(2, 1, 2)
    ax_table.axis('off')

    # 构建参数表格
    all_features = {**features_t, **features_c}
    param_names = [
        ('density', 'Density (%)'),
        ('alignment', 'Alignment'),
        ('diameter_avg', 'Diameter Avg (nm)'),
        ('diameter_std', 'Diameter Std (nm)'),
        ('curvature', 'Curvature (nm⁻¹)'),
        ('tortuosity', 'Tortuosity'),
    ]

    table_data = []
    for key, label in param_names:
        t_val = features_t.get(f'{key}_t', '-')
        c_val = features_c.get(f'{key}_c', '-')
        if t_val != '-':
            t_val = f'{t_val:.4f}' if key in ['density', 'curvature'] else f'{t_val:.2f}'
        if c_val != '-':
            c_val = f'{c_val:.4f}' if key in ['density', 'curvature'] else f'{c_val:.2f}'
        table_data.append([label, t_val, c_val])

    table = ax_table.table(
        cellText=table_data,
        colLabels=['Parameter', 'Threshold', 'CNTSegNet'],
        loc='center',
        cellLoc='center',
        colWidths=[0.3, 0.2, 0.2]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # 设置表头样式
    for i in range(3):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    plt.suptitle(f'{img_name}\nMagnification: {mag}x', fontsize=14, fontweight='bold')
    plt.tight_layout()

    # 保存
    output_file = output_path / f'{img_name}_compare.png'
    plt.savefig(str(output_file), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nSaved: {output_file}")
    return output_file


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="单图对比测试")
    parser.add_argument("--input", "-i", required=True, help="输入图片路径")
    parser.add_argument("--output", "-o", default="./test_output", help="输出目录")
    args = parser.parse_args()

    process_single(args.input, args.output)
