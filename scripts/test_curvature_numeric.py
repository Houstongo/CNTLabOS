"""
简单测试：曲率 κ = 1/R (nm⁻¹)
"""
import cv2
import numpy as np

test_img = r"d:\CNTDATA\ZZY\20230306 No26 0.75 1.0\No26 200w 5.0nm 5w 0.75nm 400 200 100 600 750 15min 180min mid 50000-1.png"
magnification = 50000

print("测试图像:", test_img)

img = cv2.imread(test_img, cv2.IMREAD_GRAYSCALE)
print("图像尺寸:", img.shape)

# 像素与纳米的转换（与FeatureExtractorV3保持一致）
HFW_AT_1X_UM = 518_000
hfw_um = HFW_AT_1X_UM / magnification
px_per_um = img.shape[1] / hfw_um
px_per_nm = px_per_um / 1000.0
expected_tube_px = max(1.5, 15.0 * px_per_um * 0.001)

print(f"px_per_um: {px_per_um:.2f}")
print(f"expected_tube_px: {expected_tube_px:.2f}")

# 简单曲率计算（三点法）
def calculate_curvature_simple(skel, px_per_nm, expected_tube_px):
    """计算曲率 κ = 1/R (nm⁻¹)"""
    from skimage.measure import label

    labeled = label(skel, connectivity=2)
    n_regions = labeled.max()

    if n_regions == 0:
        return {"curvature": -1.0, "label": "Unknown"}

    all_curvatures = []
    for rid in range(1, n_regions + 1):
        coords = np.argwhere(labeled == rid).astype(float)
        n = len(coords)
        if n < 15:
            continue

        # 按y,x排序
        sorted_indices = np.lexsort((coords[:, 1], coords[:, 0]))
        coords = coords[sorted_indices]

        # 降采样
        step = max(2, int(expected_tube_px / 2))
        sampled_coords = coords[::step]

        curvature_values = []
        m = len(sampled_coords)
        for i in range(1, m - 1):
            p_prev = sampled_coords[i - 1]
            p_curr = sampled_coords[i]
            p_next = sampled_coords[i + 1]

            AB = p_prev - p_curr
            BC = p_curr - p_next
            CA = p_next - p_prev

            a = np.linalg.norm(BC)
            b = np.linalg.norm(CA)
            c = np.linalg.norm(AB)

            # 确保三点不共线，距离足够大
            if a > 2 and b > 2 and c > 2:
                s = (a + b + c) / 2
                area = np.sqrt(max(0, s * (s - a) * (s - b) * (s - c)))
                curvature = 4 * area / (a * b * c)
                curvature_values.append(curvature)

        if curvature_values:
            median_curv_px = np.median(curvature_values)
            curvature_nm = median_curv_px / px_per_nm
            all_curvatures.append(curvature_nm)

    if not all_curvatures:
        return {"curvature": -1.0, "label": "Unknown"}

    median_curvature = float(np.median(all_curvatures))

    # 分类标签
    if median_curvature < 0.05:
        label = "Straight"
    elif median_curvature < 0.15:
        label = "Wavy"
    else:
        label = "Coiled"

    return {"curvature": median_curvature, "label": label}

# 预处理
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced = clahe.apply(img)
smoothed = cv2.GaussianBlur(enhanced, (3, 3), 0)

# 二值化
_, thresh = cv2.threshold(smoothed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU, 0)

# 简单骨架化
from skimage.morphology import skeletonize
skel = skeletonize(thresh > 0)

# 计算曲率
curvature_result = calculate_curvature_simple(skel, px_per_nm, expected_tube_px)

print("\n曲率结果:")
print(f"  curvature (nm^-1): {curvature_result['curvature']:.6f}")
print(f"  label: {curvature_result['label']}")