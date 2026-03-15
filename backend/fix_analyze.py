# 临时修复文件：恢复平均曲率计算
import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage.measure import label

def analyze_image_with_curvature(img, mag):
    # 预处理
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img)
    smoothed = cv2.GaussianBlur(enhanced, (3, 3), 0)
    _, thresh = cv2.threshold(smoothed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 对齐计算（结构张量）
    Ix = cv2.Scharr(enhanced, cv2.CV_64F, 1, 0)
    Iy = cv2.Scharr(enhanced, cv2.CV_64F, 0, 1)
    Jxx = cv2.GaussianBlur(Ix * Ix, (3, 3), 0)
    Jyy = cv2.GaussianBlur(Iy * Iy, (3, 3), 0)
    trace = Jxx + Jyy + 1e-10
    coherence = np.abs(Jxx - Jyy) / trace
    hof = float(np.clip(np.mean(coherence), 0, 1))
    mean_phi_deg = float(np.degrees(np.arccos(np.sqrt(np.clip(np.mean(coherence), 0, 1)))))

    # 平均曲率计算（基于骨架的三点法）
    skel = skeletonize(thresh > 0).astype(np.uint8) * 255
    labeled = label(skel, connectivity=2)
    n_regions = labeled.max()

    all_curvatures = []
    if n_regions > 0:
        for rid in range(1, min(n_regions + 1, 20)):
            coords = np.argwhere(labeled == rid).astype(float)
            if len(coords) < 15:
                continue

            sorted_indices = np.lexsort((coords[:, 1], coords[:, 0]))
            coords = coords[sorted_indices]

            step = max(2, len(coords) // 10)
            sampled = coords[::step]

            if len(sampled) > 2:
                for i in range(1, len(sampled) - 1):
                    p_prev, p_curr, p_next = sampled[i-1], sampled[i], sampled[i+1]
                    AB = p_prev - p_curr
                    BC = p_curr - p_next
                    CA = p_next - p_prev

                    a, b, c = np.linalg.norm(BC), np.linalg.norm(CA), np.linalg.norm(AB)

                    if a > 2 and b > 2 and c > 2:
                        s = (a + b + c) / 2
                        area = np.sqrt(max(0, s * (s - a) * (s - b) * (s - c)))
                        curvature = 4 * area / (a * b * c)
                        if curvature < 0.2:
                            all_curvatures.append(curvature)

    if all_curvatures:
        avg_curvature_px = np.median(all_curvatures)
    else:
        avg_curvature_px = 0.05

    px_per_nm = 50.0 / 1000.0 if mag and mag == 50000 else 100.0 / 1000.0
    avg_curvature_nm = avg_curvature_px / px_per_nm

    # 距离变换（用于直径）
    dist = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
    diameter_px = np.median(dist[dist > 1.5]) * 2
    density = np.count_nonzero(thresh) / thresh.size * 100

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
