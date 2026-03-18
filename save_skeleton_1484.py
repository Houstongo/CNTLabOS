import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage.measure import label
from scipy.ndimage import convolve


def trace_skeleton(mask, start, end):
    """追踪骨架路径"""
    path = []
    current = start.copy()
    visited = set()
    visited.add(tuple(current))

    max_steps = 500
    step = 0

    while step < max_steps:
        y, x = int(current[0]), int(current[1])
        path.append(current)

        if np.linalg.norm(current - end) < 3:
            break

        next_point = None
        min_dist = float('inf')

        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if (ny, nx) not in visited and 0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1]:
                    if mask[ny, nx] > 0:
                        dist = np.linalg.norm([ny, nx] - end)
                        if dist < min_dist:
                            min_dist = dist
                            next_point = np.array([ny, nx])

        if next_point is None:
            break

        visited.add(tuple(next_point))
        current = next_point
        step += 1

    return np.array(path) if len(path) > 0 else np.array([])


# 读取图像
img_path = r'd:\CNTDATA\ZZY\20230306 No26 0.75 1.0\No26 200w 5.0nm 5w 0.75nm 400 200 100 600 750 15min 180min mid 100000-1.png'
img = cv2.imread(img_path, 0)

# 预处理
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced = clahe.apply(img)
smoothed = cv2.GaussianBlur(enhanced, (3, 3), 0)

# 二值化
_, thresh = cv2.threshold(smoothed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 骨架化
skel = skeletonize(thresh > 0).astype(np.uint8)

# 找最大区域
labeled = label(skel, connectivity=2)
n_regions = labeled.max()

largest_region_id = -1
max_pixel_count = 0

if n_regions > 0:
    for rid in range(1, n_regions + 1):
        region_mask = (labeled == rid)
        pixel_count = np.sum(region_mask)
        if pixel_count > max_pixel_count:
            max_pixel_count = pixel_count
            largest_region_id = rid

# 在最大区域内找最长的连续路径（无分支）
longest_path = []
longest_length = 0

if largest_region_id > 0:
    region_mask = (labeled == largest_region_id).astype(np.uint8)

    # 找端点
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0
    neighbor_count = convolve(region_mask, kernel, mode='constant', cval=0)
    endpoints = np.argwhere((region_mask > 0) & (neighbor_count == 1))

    # 找最长的路径
    if len(endpoints) >= 2:
        max_pairs = min(len(endpoints), 10)
        for i in range(max_pairs):
            for j in range(i + 1, max_pairs):
                path = trace_skeleton(region_mask, endpoints[i], endpoints[j])
                if len(path) > longest_length:
                    longest_path = path
                    longest_length = len(path)

# 只保留最长路径
max_skel = np.zeros_like(skel)
if len(longest_path) > 5:
    for point in longest_path:
        max_skel[int(point[0]), int(point[1])] = 255

# 创建可视化
skeleton_display = cv2.cvtColor(smoothed, cv2.COLOR_GRAY2BGR)
skeleton_display[max_skel > 0] = [0, 255, 0]  # 绿色骨架

# 保存
output_path = r'd:\CNTDATA\skeleton_1484_single_path.png'
cv2.imwrite(output_path, skeleton_display)

print(f"Largest region: {largest_region_id}, pixels: {max_pixel_count}")
print(f"Longest path length: {longest_length} pixels")
print(f"Saved to: {output_path}")
