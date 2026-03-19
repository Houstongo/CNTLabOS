"""
调试分水岭分割问题
"""
import cv2
import numpy as np
from feature_extractor import FeatureExtractor
from skimage.morphology import skeletonize
from skimage.segmentation import watershed
from skimage.measure import label
from scipy import ndimage as ndi

# 测试图像
test_cases = [
    (r"d:\CNTDATA\ZZY\20230306 No26 0.75 1.0\No26 200w 5.0nm 5w 0.75nm 400 200 100 600 750 15min 180min mid 50000-1.png", 50000, "No26 mid 50000-1"),
    (r"d:\CNTDATA\ZZY\20230306 No26 0.75 1.0\No26 200w 5.0nm 5w 1.0nm 400 200 100 600 750 15min 180min mid 50000-1.png", 50000, "No26 mid 50000-1 (1.0nm)"),
    (r"d:\CNTDATA\ZZY\20260309 No30 0.75 1.0 1.25\No30 200w 5.0nm 5w 0.75nm 400 200 100 600 750 15min 180min mid 50000 1.png", 50000, "No30 mid 50000-1"),
    (r"d:\CNTDATA\ZZY\20260309 No30 0.75 1.0 1.25\No30 200w 5.0nm 5w 1.25nm 400 200 100 600 750 15min 180min mid 50000 1.png", 50000, "No30 mid 50000-1 (1.25nm)"),
]

for img_path, mag, name in test_cases:
    print(f"\n{'='*70}")
    print(f"测试: {name}")
    print(f"{'='*70}")

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("无法读取图像")
        continue

    extractor = FeatureExtractor(magnification=mag, diameter_method='enhanced')

    # 处理
    roi = extractor.extract_roi(img)
    processed = extractor.preprocess(roi)

    # 分割
    block = max(11, int(extractor.expected_tube_px * 4) | 1)
    block = min(block, 51)
    thresh = cv2.adaptiveThreshold(
        processed, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block, 2
    )

    # 距离变换
    dist = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)

    print(f"ROI尺寸: {roi.shape}")
    print(f"前景像素: {np.count_nonzero(thresh)} / {thresh.size} = {np.count_nonzero(thresh)/thresh.size*100:.1f}%")
    print(f"距离变换 max: {dist.max():.2f}, mean: {dist[thresh>0].mean():.2f}")

    # 种子点检测
    max_dist = dist.max()
    density_factor = np.count_nonzero(thresh) / thresh.size
    threshold_base = max_dist * (0.3 + 0.05 * density_factor)
    min_distance = max(1.0, int(extractor.expected_tube_px * 0.4))

    print(f"种子点参数:")
    print(f"  threshold_base: {threshold_base:.2f} (max_dist * {0.3 + 0.05 * density_factor:.3f})")
    print(f"  min_distance: {min_distance:.1f} px")

    from scipy.ndimage import maximum_filter
    local_max = (maximum_filter(dist, size=min_distance + 1) == dist) & (dist > threshold_base)
    markers, _ = ndi.label(local_max)

    print(f"种子点检测结果:")
    print(f"  local_max: {np.count_nonzero(local_max)} 个点")
    print(f"  n_markers: {markers.max()} 个标记")

    # 分水岭
    if markers.max() > 0:
        try:
            labels = watershed(-dist, markers, mask=thresh)
            print(f"分水岭结果:")
            print(f"  labels.max: {labels.max()}")
            print(f"  labels唯一值: {np.unique(labels)[:10]}..." if len(np.unique(labels)) > 10 else f"  labels唯一值: {np.unique(labels)}")

            # 检查是否有有效区域
            if labels.max() > 0:
                print(f"  状态: ✓ 成功")
            else:
                print(f"  状态: ✗ 返回全0")
        except Exception as e:
            print(f"  状态: ✗ 异常 - {e}")
    else:
        print(f"  状态: ✗ 无种子点")

print(f"\n{'='*70}")
print("调试完成")
