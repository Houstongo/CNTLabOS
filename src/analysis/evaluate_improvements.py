"""
特征提取改进系统性评估
======================

对比改进前后的效果：
1. 直径：standard vs enhanced_watershed
2. 取向度：structure_tensor vs skeleton（统一）
3. ROI：原方法 vs 多特征融合

测试图像集：
- 高倍率（50kx）: ZZY/No26 mid 50000
- 低倍率（10kx）: ZZY/No26 mid 10000
- XR格式: XR/250301 T800
"""

import cv2
import numpy as np
from feature_extractor import FeatureExtractor
import sys

# 重定向输出到文件
sys.stdout = open(r'd:\CNTDATA\CNTA_ML_Project\src\analysis\eval_output.txt', 'w', encoding='utf-8')

# 测试图像路径
TEST_IMAGES = [
    (r"d:\CNTDATA\ZZY\20230306 No26 0.75 1.0\No26 200w 5.0nm 5w 0.75nm 400 200 100 600 750 15min 180min mid 50000-1.png", 50000),
    (r"d:\CNTDATA\ZZY\20230306 No26 0.75 1.0\No26 200w 5.0nm 5w 0.75nm 400 200 100 600 750 15min 180min mid 10000-1.png", 10000),
    (r"d:\CNTDATA\XR\250301 T800\C1AN1.tiff", None),
]


def main():
    print("="*70)
    print("特征提取改进评估")
    print("="*70)

    for img_path, mag in TEST_IMAGES:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[跳过] 无法读取: {img_path}")
            continue

        name = img_path.split("\\")[-1]
        print(f"\n{'='*70}")
        print(f"图像: {name} | 倍率: {mag if mag else 'N/A'}")
        print(f"尺寸: {img.shape}")
        print(f"{'='*70}")

        # 测试1: 直径对比
        print("\n[1] 直径算法对比")
        print("-"*70)

        # 标准方法
        extractor_std = FeatureExtractor(magnification=mag, diameter_method='standard')
        res_std = extractor_std.extract_all(img)
        print(f"  标准方法 (闭运算):")
        print(f"    diameter = {res_std['diameter']} nm")
        print(f"    hof       = {res_std['alignment']:.4f} (method={res_std['hof_method']})")
        print(f"    n_branches = {res_std.get('n_branches', 'N/A')}")

        # 增强方法
        extractor_enh = FeatureExtractor(magnification=mag, diameter_method='enhanced')
        res_enh = extractor_enh.extract_all(img)
        print(f"\n  增强方法 (分水岭):")
        print(f"    diameter = {res_enh['diameter']} nm")
        print(f"    hof       = {res_enh['alignment']:.4f} (method={res_enh['hof_method']})")
        print(f"    n_branches = {res_enh.get('n_branches', 'N/A')}")

        # 对比差异
        if res_std['diameter'] and res_enh['diameter']:
            diff = res_enh['diameter'] - res_std['diameter']
            diff_pct = (diff / res_std['diameter']) * 100
            print(f"\n  差异: {diff:+.2f} nm ({diff_pct:+.1f}%)")

        # 测试2: ROI检测可视化
        print("\n[2] ROI裁切检测")
        print("-"*70)

        h_orig = img.shape[0]
        roi_std = extractor_std.extract_roi(img)
        h_roi = roi_std.shape[0]
        cut_px = h_orig - h_roi

        print(f"  原始高度: {h_orig} px")
        print(f"  ROI高度:   {h_roi} px")
        print(f"  裁切高度: {cut_px} px")
        print(f"  裁切比例: {cut_px/h_orig*100:.1f}%")

        # 综合对比表
        print("\n[3] 综合对比表")
        print("-"*70)
        print(f"{'指标':<15} {'标准':<15} {'增强':<15} {'差异':<15}")
        print("-"*70)

        for key in ['density', 'alignment', 'mean_phi_deg']:
            val_std = res_std.get(key, 'N/A')
            val_enh = res_enh.get(key, 'N/A')

            if key == 'density':
                val_std = f"{val_std}%" if val_std != 'N/A' else val_std
                val_enh = f"{val_enh}%" if val_enh != 'N/A' else val_enh
            elif key == 'alignment':
                val_std = f"{val_std:.4f}" if val_std != 'N/A' else val_std
                val_enh = f"{val_enh:.4f}" if val_enh != 'N/A' else val_enh
            elif key == 'mean_phi_deg':
                val_std = f"{val_std:.1f}°" if val_std != 'N/A' else val_std
                val_enh = f"{val_enh:.1f}°" if val_enh != 'N/A' else val_enh

            diff = "N/A"
            if isinstance(val_std, str) or isinstance(val_enh, str):
                diff = "N/A"
            else:
                diff = f"{(val_enh - val_std):+.3f}"

            print(f"{key:<15} {val_std:<15} {val_enh:<15} {diff:<15}")

        print(f"\n{'diameter':<15} {res_std['diameter'] or 'N/A':<15} {res_enh['diameter'] or 'N/A':<15} N/A")
        print(f"{'curvature':<15} {res_std['curvature']:<15} {res_enh['curvature']:<15} N/A")

    print("\n" + "="*70)
    print("评估完成")
    print("="*70)


if __name__ == "__main__":
    main()
