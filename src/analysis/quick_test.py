"""
快速测试改进效果
"""
import cv2
import sys
import os

# 设置输出文件
output_path = r'd:\CNTDATA\CNTA_ML_Project\src\analysis\quick_eval.txt'

# 切换目录
os.chdir(r"d:\CNTDATA\CNTA_ML_Project\src\analysis")

# 导入模块
from feature_extractor import FeatureExtractor

# 测试单张图像
test_img = r"d:\CNTDATA\ZZY\20230306 No26 0.75 1.0\No26 200w 5.0nm 5w 0.75nm 400 200 100 600 750 15min 180min mid 50000-1.png"
mag = 50000

img = cv2.imread(test_img, cv2.IMREAD_GRAYSCALE)

# 测试1: 标准方法
extractor_std = FeatureExtractor(magnification=mag, diameter_method='standard')
res_std = extractor_std.extract_all(img)

# 测试2: 增强方法
extractor_enh = FeatureExtractor(magnification=mag, diameter_method='enhanced')
res_enh = extractor_enh.extract_all(img)

# 写入结果
with open(output_path, 'w', encoding='utf-8') as f:
    f.write("="*70 + "\n")
    f.write("特征提取改进快速测试\n")
    f.write("="*70 + "\n\n")
    f.write(f"图像: {os.path.basename(test_img)} | 倍率: {mag}\n")
    f.write(f"尺寸: {img.shape}\n\n")

    f.write("[标准方法]\n")
    f.write(f"  diameter = {res_std['diameter']} nm\n")
    f.write(f"  alignment = {res_std['alignment']:.4f} (method={res_std['hof_method']})\n")
    f.write(f"  density = {res_std['density']}%\n")
    f.write(f"  curvature = {res_std['curvature']}\n\n")

    f.write("[增强方法]\n")
    f.write(f"  diameter = {res_enh['diameter']} nm\n")
    f.write(f"  alignment = {res_enh['alignment']:.4f} (method={res_enh['hof_method']})\n")
    f.write(f"  density = {res_enh['density']}%\n")
    f.write(f"  curvature = {res_enh['curvature']}\n\n")

    # 计算差异
    if res_std['diameter'] and res_enh['diameter']:
        diff = res_enh['diameter'] - res_std['diameter']
        diff_pct = (diff / res_std['diameter']) * 100
        f.write(f"[直径差异]\n")
        f.write(f"  差异: {diff:+.2f} nm ({diff_pct:+.1f}%)\n\n")

    f.write(f"[取向度对比]\n")
    f.write(f"  HOF变化: {(res_enh['alignment'] - res_std['alignment']):+.4f}\n")
    f.write(f"  phi变化: {(res_enh['mean_phi_deg'] - res_std['mean_phi_deg']):+.2f}°\n")

print("测试完成，结果已写入:", output_path)
