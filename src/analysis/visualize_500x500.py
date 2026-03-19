"""
500x500局部对比可视化
====================
对比标准方法 vs 增强方法在局部区域的差异
"""

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from feature_extractor import FeatureExtractor
from skimage.morphology import skeletonize
from skimage.segmentation import watershed
from skimage.measure import label
from scipy import ndimage as ndi

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


def visualize_local_comparison(img_path, mag, output_path):
    """500x500局部对比"""

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return

    img_name = os.path.basename(img_path)
    extractor = FeatureExtractor(magnification=mag, diameter_method='enhanced')

    # 裁剪500x500局部区域（从中心）
    h, w = img.shape
    crop_size = 500
    y_start = max(0, (h - crop_size) // 2 - 50)
    x_start = max(0, (w - crop_size) // 2)
    crop_end_y = min(h, y_start + crop_size)
    crop_end_x = min(w, x_start + crop_size)

    img_crop = img[y_start:crop_end_y, x_start:crop_end_x]

    # 预处理
    roi = extractor.extract_roi(img_crop)
    processed = extractor.preprocess(roi)

    # 自适应阈值
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

    # 标准方法：闭运算
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed_std = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    skel_std = skeletonize(closed_std > 0)

    # 增强方法：分水岭
    max_dist = dist.max()
    density_factor = np.count_nonzero(thresh) / thresh.size
    threshold_base = max_dist * (0.3 + 0.05 * density_factor)
    min_distance = max(1.0, int(extractor.expected_tube_px * 0.4))
    from scipy.ndimage import maximum_filter
    local_max = (maximum_filter(dist, size=min_distance + 1) == dist) & (dist > threshold_base)
    markers, _ = ndi.label(local_max)

    if markers.max() > 0:
        labels = watershed(-dist, markers, mask=thresh)
        expected_area = (extractor.expected_tube_px * 2) ** 2
        valid_labels = [l for l in np.unique(labels) if l > 0 and 10 <= (labels == l).sum() <= expected_area * 3]
        n_valid = len(valid_labels)

        # 可视化：在原图上用绿色标注分水岭边界
        watershed_vis = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
        from scipy.ndimage import sobel
        labels_grad = sobel(labels) > 0
        watershed_vis[labels_grad] = [0, 255, 0]  # 绿色边界
    else:
        watershed_vis = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
        watershed_vis[thresh > 0] = [0, 255, 255]  # 黄色：种子点失败
        n_valid = 0

    # 增强方法骨架（基于分水岭labels）
    if markers.max() > 0:
        labels = watershed(-dist, markers, mask=thresh)
        skel_enh = skeletonize(labels > 0)
        n_valid = len([l for l in np.unique(labels) if l > 0])
    else:
        labels = np.zeros_like(thresh, dtype=int)
        skel_enh = np.zeros_like(thresh, dtype=bool)
        n_valid = 0

    # 创建对比图：2x2
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    fig.suptitle(f'{img_name} | mag={mag} | 500x500局部对比', fontsize=14, fontweight='bold')

    # 第1行：分割区域对比
    # 标准方法：闭运算区域（红色）
    std_overlay = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
    std_overlay[closed_std > 0] = [0, 0, 255]  # 红色
    axes[0, 0].imshow(std_overlay)
    axes[0, 0].set_title('1. 标准方法-分割区域\n红色=闭运算', fontsize=12)
    axes[0, 0].axis('off')

    # 增强方法：分水岭区域（绿色边界）
    from scipy.ndimage import sobel
    labels_grad = sobel(labels) > 0
    enh_overlay = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
    enh_overlay[labels_grad] = [0, 255, 0]  # 绿色边界
    axes[0, 1].imshow(enh_overlay)
    axes[0, 1].set_title(f'2. 增强方法-分割区域\n绿色=分水岭边界, 区域={n_valid}', fontsize=12)
    axes[0, 1].axis('off')

    # 第2行：骨架对比
    # 标准方法骨架
    axes[1, 0].imshow(skel_std, cmap='gray')
    axes[1, 0].set_title('3. 标准方法-骨架\n基于闭运算', fontsize=12)
    axes[1, 0].axis('off')

    # 增强方法骨架
    axes[1, 1].imshow(skel_enh, cmap='gray')
    axes[1, 1].set_title('4. 增强方法-骨架\n基于分水岭', fontsize=12)
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()


def main():
    import os

    test_images = [
        (r"d:\CNTDATA\ZZY\20230306 No26 0.75 1.0\No26 200w 5.0nm 5w 0.75nm 400 200 100 600 750 15min 180min mid 50000-1.png", 50000),
        (r"d:\CNTDATA\ZZY\20230306 No26 0.75 1.0\No26 200w 5.0nm 5w 1.0nm 400 200 100 600 750 15min 180min mid 50000-1.png", 50000),
        (r"d:\CNTDATA\ZZY\20260309 No30 0.75 1.0 1.25\No30 200w 5.0nm 5w 0.75nm 400 200 100 600 750 15min 180min mid 50000 1.png", 50000),
        (r"d:\CNTDATA\ZZY\20260309 No30 0.75 1.0 1.25\No30 200w 5.0nm 5w 1.25nm 400 200 100 600 750 15min 180min mid 50000 1.png", 50000),
        (r"d:\CNTDATA\ZZY\20230306 No26 0.75 1.0\No26 200w 5.0nm 5w 0.75nm 400 200 100 600 750 15min 180min mid 100000-1.png", 100000),
    ]

    output_dir = r"d:\CNTDATA\CNTA_ML_Project\src\analysis\vis_500x500"
    os.makedirs(output_dir, exist_ok=True)

    print("="*60)
    print("500x500局部对比可视化")
    print("="*60)

    for idx, (img_path, mag) in enumerate(test_images, 1):
        if not os.path.exists(img_path):
            continue

        img_name = os.path.basename(img_path)
        output_name = f"{idx:02d}_{img_name.replace('.png', '')}_500x500.png"
        output_path = os.path.join(output_dir, output_name)

        print(f"[{idx}/{len(test_images)}] {img_name}")
        visualize_local_comparison(img_path, mag, output_path)

    print("="*60)
    print("完成")
    print("="*60)


if __name__ == "__main__":
    main()
