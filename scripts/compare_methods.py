"""
对比测试脚本：传统方法 vs 改进方法
"""
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src', 'analysis'))

import cv2
import numpy as np
import pandas as pd

from feature_extractor import FeatureExtractor as FeatureExtractorV2
from feature_extractor_v3 import FeatureExtractorV3


def compare_diameter_methods():
    """对比diameter计算方法"""
    test_images = [
        r"d:\CNTDATA\ZZY\20230306 No26 0.75 1.0\No26 200w 5.0nm 5w 0.75nm 400 200 100 600 750 15min 180min mid 50000-1.png",
        r"d:\CNTDATA\ZZY\20230306 No26 0.75 1.0\No26 200w 5.0nm 5w 0.75nm 400 200 100 600 750 15min 180min mid 100000-1.png",
        r"d:\CNTDATA\ZZY\20230306 No26 0.75 1.0\No26 200w 5.0nm 5w 0.75nm 400 200 100 600 750 15min 180min mid 50000-2.png",
        r"d:\CNTDATA\ZZY\20230306 No26 0.75 1.0\No26 200w 5.0nm 5w 1.0nm 400 200 100 600 750 15min 180min mid 100000-1.png",
        r"d:\CNTDATA\ZZY\20230306 No26 0.75 1.0\No26 200w 5.0nm 5w 1.0nm 400 200 100 600 750 15min 180min mid 100000-2.png",
    ]

    results = []

    for img_path in test_images:
        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # v2.0 传统方法
        extractor_v2 = FeatureExtractorV2(magnification=50000)
        roi_v2 = extractor_v2.extract_roi(img)
        processed_v2 = extractor_v2.preprocess(roi_v2)
        _, thresh_v2 = cv2.threshold(processed_v2, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU, 0)
        diameter_v2, skel_v2 = extractor_v2.calculate_diameter(thresh_v2)

        # v3.0 改进方法（分水岭）
        extractor_v3 = FeatureExtractorV3(magnification=50000)
        results_v3 = extractor_v3.extract_all(img, use_improved=True)
        diameter_v3 = results_v3.get('diameter')

        results.append({
            'image': os.path.basename(img_path),
            'diameter_v2_traditional': diameter_v2,
            'diameter_v3_watershed': diameter_v3,
            'improvement': diameter_v2 - diameter_v3 if diameter_v2 > 0 and diameter_v3 > 0 else None,
            'improvement_pct': None
        })

    # 计算改善百分比
    for r in results:
        if r['diameter_v2_traditional'] and r['diameter_v3_watershed']:
            improvement = r['diameter_v2_traditional'] - r['diameter_v3_watershed']
            if r['diameter_v2_traditional'] > 0:
                r['improvement_pct'] = (improvement / r['diameter_v2_traditional']) * 100

    df = pd.DataFrame(results)

    print("=" * 80)
    print("diameter方法对比")
    print("=" * 80)
    print(df.to_string(index=False))

    # 保存结果
    output_path = os.path.join(PROJECT_ROOT, 'docs', 'diameter_comparison.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"\n结果已保存: {output_path}")

    return df


if __name__ == "__main__":
    df = compare_diameter_methods()
