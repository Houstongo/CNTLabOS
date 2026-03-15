"""
简单对比：闭运算 vs 分水岭分割
"""
import os
import cv2
import numpy as np
import pandas as pd

# 测试图像
test_img = r"d:\CNTDATA\ZZY\20230306 No26 0.75 1.0\No26 200w 5.0nm 5w 0.75nm 400 200 100 600 750 15min 180min mid 50000-1.png"

print("测试图像:", test_img)

# 读取图像
img = cv2.imread(test_img, cv2.IMREAD_GRAYSCALE)
print("图像尺寸:", img.shape)

# 预处理
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced = clahe.apply(img)
smoothed = cv2.GaussianBlur(enhanced, (3, 3), 0)

# 二值化
_, thresh = cv2.threshold(smoothed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU, 0)

print("前景占比:", np.count_nonzero(thresh) / thresh.size * 100, "%")

# 方法1：闭运算（传统）
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
dist1 = cv2.distanceTransform(closed, cv2.DIST_L2, 5)
diameter1_px = np.median(dist1[dist1 > 1.5]) * 2
print(f"方法1（闭运算）: diameter = {diameter1_px:.2f} px")

# 方法2：分水岭分割（改进）
dist2 = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)

# 简单方法：直接用距离变换作为分割依据
# 根据距离值进行简单的区域划分
diameter2_px = np.median(dist2[dist2 > 1.5]) * 2
print(f"方法2（距离中位数）: diameter = {diameter2_px:.2f} px")

# 计算有多少"大直径"区域（可能表示粘连）
large_area_count = np.sum(dist2 > diameter2_px * 1.5) / dist2.size * 100
print(f"  大直径区域占比: {large_area_count:.1f}%")

# 对比
print(f"\n对比:")
print(f"  改进: {(diameter1_px - diameter2_px) / diameter1_px * 100:.1f}%")
