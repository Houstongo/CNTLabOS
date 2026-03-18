import cv2
import numpy as np
from scipy.ndimage import maximum_filter, label

# 读取图像
img_path = r'd:\CNTDATA\ZZY\20230306 No26 0.75 1.0\No26 200w 5.0nm 5w 0.75nm 400 200 100 600 750 15min 180min mid 100000-1.png'
img = cv2.imread(img_path, 0)

# 预处理
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced = clahe.apply(img)
smoothed = cv2.GaussianBlur(enhanced, (3, 3), 0)

# 二值化
_, thresh = cv2.threshold(smoothed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 距离变换
dist = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)

# 找种子点（局部最大值）
local_max = maximum_filter(dist, footprint=np.ones((3, 3), dtype=int))
threshold_value = dist.max() * 0.3
local_max = (local_max == dist) & (dist > threshold_value)

# 统计种子点数量
seed_count = np.sum(local_max)

print("Seed count:", seed_count)
print("Distance max:", dist.max())
print("Threshold:", threshold_value)
