"""
测试增强分水岭分割的直径计算
"""
import cv2
import os
import sys

# 确保可以 import src.analysis.feature_extractor
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.analysis.feature_extractor import FeatureExtractor

# 测试图像
test_images = [
    r"d:\CNTDATA\ZZY\20230306 No26 0.75 1.0\No26 200w 5.0nm 5w 0.75nm 400 200 100 600 750 15min 180min mid 50000-1.png",
    r"d:\CNTDATA\ZZY\20230306 No26 0.75 1.0\No26 200w 5.0nm 5w 0.75nm 400 200 100 600 750 15min 180min mid 100000-1.png",
]

for img_path in test_images:
    if not os.path.exists(img_path):
        print(f"SKIP: {os.path.basename(img_path)}")
        continue

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"无法读取: {img_path}")
        continue

    print(f"\n{'='*60}")
    print(f"图像: {os.path.basename(img_path)}")

    # 测试标准方法
    extractor_std = FeatureExtractor(magnification=50000, diameter_method='standard')
    res_std = extractor_std.extract_all(img)
    print(f"[标准方法]")
    print(f"  diameter = {res_std['diameter']} nm")
    print(f"  method = {res_std.get('diameter_method', 'N/A')}")

    # 测试增强方法
    extractor_enh = FeatureExtractor(magnification=50000, diameter_method='enhanced')
    res_enh = extractor_enh.extract_all(img)
    print(f"[增强方法]")
    print(f"  diameter = {res_enh['diameter']} nm")
    print(f"  method = {res_enh.get('diameter_method', 'N/A')}")

    # 对比差异
    if res_std['diameter'] and res_enh['diameter']:
        diff = abs(res_std['diameter'] - res_enh['diameter'])
        pct_diff = diff / res_std['diameter'] * 100
        print(f"[对比]")
        print(f"  差异 = {diff:.2f} nm ({pct_diff:.1f}%)")
