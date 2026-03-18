# 临时修复文件：恢复平均曲率计算
import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage.measure import label

def analyze_image_with_curvature(img, mag):
    # 预处理
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img)
    smoothed = cv2.GaussianBlur(enhanced, (2, 2), 0)
    _, thresh = cv2.threshold(smoothed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 闭运算连接断裂CNT
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 2))
    thresh_closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 对齐计算（结构张量）
    Ix = cv2.Scharr(enhanced, cv2.CV_64F, 1, 0)
    Iy = cv2.Scharr(enhanced, cv2.CV_64F, 0, 1)
    Jxx = cv2.GaussianBlur(Ix * Ix, (3, 3), 0)
    Jyy = cv2.GaussianBlur(Iy * Iy, (3, 3), 0)
    trace = Jxx + Jyy + 1e-10
    coherence = np.abs(Jxx - Jyy) / trace
    hof = float(np.clip(np.mean(coherence), 0, 1))
    mean_phi_deg = float(np.degrees(np.arccos(np.sqrt(np.clip(np.mean(coherence), 0, 1)))))

    # 骨架追踪曲率计算
    from scipy.ndimage import convolve

    skel_raw = skeletonize(thresh_closed > 0)

    # 骨架修剪：同一连通区域只保留1条主干
    skel = prune_skeleton(skel_raw).astype(np.uint8) * 255
    labeled = label(skel, connectivity=2)

    px_per_nm = 0.05 if mag and mag == 50000 else 0.1
    all_curvatures = []

    n_regions = labeled.max()
    if n_regions > 0:
        for rid in range(1, min(n_regions + 1, 15)):
            region_mask = (labeled == rid).astype(np.uint8)
            coords = np.argwhere(region_mask).astype(float)

            if len(coords) < 10:
                continue

            # 找端点
            kernel = np.ones((3, 3), dtype=np.uint8)
            kernel[1, 1] = 0
            neighbor_count = convolve(region_mask, kernel, mode='constant', cval=0)
            endpoints = np.argwhere((region_mask > 0) & (neighbor_count == 1))

            if len(endpoints) >= 2:
                # 追踪路径并计算曲率
                path = trace_skeleton(region_mask, endpoints[0], endpoints[1])
                if len(path) > 5:
                    for i in range(2, min(len(path) - 2, 50)):
                        p0, p1, p2, p3, p4 = path[i-2], path[i-1], path[i], path[i+1], path[i+2]
                        v1 = p2 - p0
                        v2 = p4 - p2
                        angle1 = np.arctan2(v1[0], v1[1])
                        angle2 = np.arctan2(v2[0], v2[1])
                        d_theta = abs(angle2 - angle1)
                        ds = np.linalg.norm(p4 - p0)
                        if ds > 0:
                            curvature_px = d_theta / ds
                            curvature_nm = curvature_px / px_per_nm
                            if curvature_nm < 2.0:
                                all_curvatures.append(curvature_nm)

    avg_curvature_nm = float(np.median(all_curvatures)) if all_curvatures else 0.0

    # 距离变换（用于直径）
    dist = cv2.distanceTransform(thresh_closed, cv2.DIST_L2, 5)
    diameter_px = np.median(dist[dist > 1.5]) * 2
    density = np.count_nonzero(thresh_closed) / thresh.size * 100

    px_per_um = 50.0 if mag and mag == 50000 else 100.0
    diameter_nm = diameter_px / px_per_um * 1000.0

    results = {
        'diameter': float(diameter_nm),
        'density': float(density),
        'alignment': float(hof),
        'curvature': float(avg_curvature_nm),
        'mean_phi_deg': float(mean_phi_deg),
        'px_per_um': float(px_per_um)
    }

    return results


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


def prune_skeleton(skeleton, min_length=50):
    """修剪骨架：同一连通区域只保留1条主干"""
    from skimage.measure import label
    from scipy.ndimage import convolve

    skel = skeleton.copy().astype(np.uint8)

    labeled = label(skel, connectivity=2)
    n_regions = labeled.max()

    pruned = np.zeros_like(skel)

    if n_regions > 0:
        for rid in range(1, min(n_regions + 1, 20)):
            region_mask = (labeled == rid).astype(np.uint8)
            coords = np.argwhere(region_mask)

            if len(coords) < 10:
                continue

            kernel = np.ones((3, 3), dtype=np.uint8)
            kernel[1, 1] = 0
            neighbor_count = convolve(region_mask, kernel, mode='constant', cval=0)
            endpoints = np.argwhere((region_mask > 0) & (neighbor_count == 1))

            if len(endpoints) < 2:
                pruned[region_mask > 0] = 255
                continue

            longest_path = []
            longest_length = 0

            max_pairs = min(len(endpoints), 10)
            for i in range(max_pairs):
                for j in range(i + 1, max_pairs):
                    path = trace_skeleton(region_mask, endpoints[i], endpoints[j])
                    if len(path) > longest_length:
                        longest_path = path
                        longest_length = len(path)

            if len(longest_path) > 5:
                for point in longest_path:
                    pruned[int(point[0]), int(point[1])] = 255

    return pruned
