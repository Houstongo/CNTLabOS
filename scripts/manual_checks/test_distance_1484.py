import cv2
import numpy as np
from scipy.ndimage import label

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

# 骨架化
from skimage.morphology import skeletonize
skel = skeletonize(thresh > 0).astype(np.uint8)

# 标记骨架的连通区域
from skimage.measure import label
labeled = label(skel, connectivity=2)
n_regions = labeled.max()

print("=== Image 1484 Analysis ===")
print(f"Skeleton connected regions: {n_regions}")

for rid in range(1, min(n_regions + 1, 11)):
    region_mask = (labeled == rid).astype(np.uint8)
    coords = np.argwhere(region_mask)
    pixel_count = len(coords)

    region_dist_vals = dist[region_mask > 0]
    avg_dist = np.mean(region_dist_vals) if len(region_dist_vals) > 0 else 0
    max_dist = np.max(region_dist_vals) if len(region_dist_vals) > 0 else 0

    print(f"  Region {rid}: {pixel_count} pixels, avg dist: {avg_dist:.2f}, max dist: {max_dist:.2f}")

px_per_nm = 0.01
avg_diameter_nm = np.mean(dist[dist > 0]) * 2 / px_per_nm
print(f"Estimated avg diameter: {avg_diameter_nm:.2f} nm")
