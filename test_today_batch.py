#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
今天入库的三组实验 SEM 图批量处理脚本

输出：
1. 寏张图：原图 + Threshold分割 + CNTSegNet分割
2. 每张图：5种参数组合图
3. 对比汇总
"""

import os
import sys
import json
import time
import base64
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "VLMSAM"))

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from backend.core.algorithm_visualizer import AlgorithmVisualizer
from backend.core.cntsegnet_visualizer import CNTSegNetVisualizer


def process_batch(image_dir, output_dir, limit=None):
    """批量处理图片"""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 获取图片列表
    image_dir = Path(image_dir)
    if not image_dir.exists():
        print(f"Error: {image_dir} does not exist")
        return

    # 找到今天的实验数据
    today = datetime.now().strftime("%Y%m%d")
    images = []

    for root, dirs, files in os.walk(image_dir):
        for f in files:
                if f.lower().endswith(('.png', '.tif', '.tiff')):
                    # 检查是否包含今天的日期或                    if today in f or any(c.isdigit() for c in today.split('/')):
                        full_path = root / f
                        images.append(full_path)

    print(f"Found {len(images)} images with today's date")

    if limit:
        images = images[:limit]

    # 结果存储
    results = []

    for i, img_path in enumerate(images, 1):
        img_name = os.path.basename(img_path)
        print(f"[{i}/{len(images)}] {img_name}")

        try:
            # 读取图片
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"  Error: Cannot read image")
                continue

            h, w = img.shape
            mag = 50000  # 默认倍率，            if '10000' in img_name:
                mag = 10000
            elif '100000' in img_name:
                mag = 100000

            # === Threshold 方法 ===
            vis_t = AlgorithmVisualizer(magnification=mag)
            steps_t = vis_t.visualize_extraction(img)

            # === CNTSegNet 方法 ===
            vis_c = CNTSegNetVisualizer(magnification=mag, device='cpu')
            steps_c = vis_c.visualize_extraction(img)

            # 提取特征
            features = {
                'image': img_name,
                'size': f"{h}x{w}",
                'magnification': mag,
            }

            # 从 threshold 提取
            for step in steps_t:
                desc = step.get('description', '')
                name = step.get('name', '')

                if 'density' in name.lower() or '密度' in name:
                    import re
                    m = re.search(r'density\s*=\s*([\d.]+)', desc, re.I)
                    if m:
                        features['density_t'] = float(m.group(1))

                if 'alignment' in name.lower() or '对齐' in name or '方向' in name:
                    m = re.search(r'alignment\s*=\s*(\d+\.\d+)', desc, re.I)
                    if m:
                        features['alignment_t'] = float(m.group(1))

                if '直径' in name:
                    m = re.search(r'平均直径\s*=\s*([\d.]+)\s*nm', desc)
                    if m:
                        features['diameter_avg_t'] = float(m.group(1))
                    m = re.search(r'标准差\s*=\s*([\d.]+)\s*nm', desc)
                    if m:
                        features['diameter_std_t'] = float(m.group(1))
                    m = re.search(r'中位数\s*=\s*([\d.]+)\s*nm', desc)
                    if m:
                        features['diameter_median_t'] = float(m.group(1))

                if 'curvature' in name.lower() or '曲率' in name:
                    m = re.search(r'κ\s*=\s*([\d.]+)', desc)
                    if m:
                        features['curvature_t'] = float(m.group(1))

                if 'tortuosity' in name.lower() or '波曲度' in name:
                    m = re.search(r'τ\s*=\s*([\d.]+)', desc)
                    if not m:
                        m = re.search(r'tortuosity\s*=\s*([\d.]+)', desc, re.I)
                    if m:
                        features['tortuosity_t'] = float(m.group(1))

            # 从 CNTSegNet 提取
            for step in steps_c:
                desc = step.get('description', '')
                name = step.get('name', '')

                if 'density' in name.lower() or '密度' in name:
                    import re
                    m = re.search(r'density\s*=\s*([\d.]+)', desc, re.I)
                    if m:
                        features['density_c'] = float(m.group(1))

            results.append(features)

            # 保存单图结果
            img_output_dir = output_path / Path(img_name).stem
            img_output_dir.mkdir(parents=True, exist_ok=True)

            # 保存原图
            cv2.imwrite(str(img_output_dir / 'original.png'), img)

            # 保存 threshold 步骤
            for j, step in enumerate(steps_t):
                img_data = cv2.imdecode(np.frombuffer(base64.b64decode(step['image'])), cv2.IMREAD_COLOR)
                cv2.imwrite(str(img_output_dir / f'{j+1:02d}_{step["name"]}_t.png'), img_data)

            # 保存 CNTSegNet 步骤
            for j, step in enumerate(steps_c):
                img_data = cv2.imdecode(np.frombuffer(base64.b64decode(step['image'])), cv2.IMREAD_COLOR)
                cv2.imwrite(str(img_output_dir / f'{j+1:02d}_{step["name"]}_c.png'), img_data)

            # 生成对比图
            fig, axes = plt.subplots(2, 5, figsize=(20, 8))

            # 原图
            axes[0, 0].imshow(img, cmap='gray')
            axes[0, 0].set_title('Original')

            # Threshold 结果
            for j, step in enumerate(steps_t[:5]):
                img_data = cv2.imdecode(np.frombuffer(base64.b64decode(step['image'])), cv2.IMREAD_COLOR)
                axes[0, j+1].imshow(cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB))
                axes[0, j+1].set_title(f'T: {step["name"]}')

            # CNTSegNet 结果
            for j, step in enumerate(steps_c[:5]):
                img_data = cv2.imdecode(np.frombuffer(base64.b64decode(step['image'])), cv2.IMREAD_COLOR)
                axes[1, j].imshow(cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB))
                axes[1, j].set_title(f'C: {step["name"]}')

            # 隐藏最后一列
            for j in range(5, 6):
                axes[1, j].axis('off')

            plt.tight_layout()
            plt.savefig(str(img_output_dir / 'comparison.png'), dpi=100)
            plt.close()

        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 保存汇总
    summary_path = output_path / 'batch_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 打印统计
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Processed: {len(results)} images")

    if results:
        import pandas as pd
        df = pd.DataFrame(results)
        print(f"\nDensity: t={df['density_t'].mean():.3f}, c={df['density_c'].mean():.3f}")
        if 'diameter_avg_t' in df:
            print(f"Diameter: avg={df['diameter_avg_t'].mean():.1f}nm, std={df['diameter_std_t'].mean():.1f}nm")
        if 'alignment_t' in df:
            print(f"Alignment: {df['alignment_t'].mean():.3f}")
        if 'curvature_t' in df:
            print(f"Curvature: {df['curvature_t'].mean():.4f}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="今天入库的三组实验批量处理")
    parser.add_argument("--input", "-i", required=True, help="输入目录")
    parser.add_argument("--output", "-o", default="./output_today", help="输出目录")
    parser.add_argument("--limit", "-n", type=int, default=None, help="限制数量")
    args = parser.parse_args()

    process_batch(args.input, args.output, args.limit)
