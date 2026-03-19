"""
算法可视化脚本 - 展示每个处理步骤
====================================

对20张图片进行算法流程可视化，便于直观评判

可视化内容（9宫格）：
1. 原始图像
2. ROI裁切
3. CLAHE增强
4. 自适应阈值分割
5. 骨架化
6. 距离变换（热力图）
7. 标准方法（闭运算效果）
8. 增强方法-种子点
9. 增强方法-分水岭分割
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from feature_extractor import FeatureExtractor
from skimage.morphology import skeletonize
from skimage.segmentation import watershed
from skimage.measure import label
from scipy import ndimage as ndi
import os

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


def visualize_full_pipeline(img_path, mag, output_path):
    """完整算法流程可视化"""

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"无法读取: {img_path}")
        return

    img_name = os.path.basename(img_path)
    extractor = FeatureExtractor(magnification=mag, diameter_method='enhanced')

    # 裁剪500x500局部区域（从中心）
    h, w = img.shape
    crop_size = 500
    y_start = max(0, (h - crop_size) // 2 - 50)  # 稍微偏上，避开底部信息栏
    x_start = max(0, (w - crop_size) // 2)
    crop_end_y = min(h, y_start + crop_size)
    crop_end_x = min(w, x_start + crop_size)

    img_crop = img[y_start:crop_end_y, x_start:crop_end_x]

    # 1. 原始图像（局部）
    orig = img_crop.copy()

    # 2. ROI裁切
    roi = extractor.extract_roi(img_crop)

    # 3. 预处理（CLAHE）
    processed = extractor.preprocess(roi)

    # 4. 自适应阈值分割
    block = max(11, int(extractor.expected_tube_px * 4) | 1)
    block = min(block, 51)
    thresh = cv2.adaptiveThreshold(
        processed, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block, 2
    )

    # 5. 骨架化
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed_std = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    skel = skeletonize(closed_std > 0)

    # 6. 距离变换（热力图）
    dist = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)

    # 7. 标准方法：闭运算效果
    closed_vis = cv2.cvtColor((closed_std * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    # 在原图上叠加闭运算结果（红色）
    overlay_std = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
    overlay_std[closed_std > 0] = [0, 0, 255]  # 红色表示被识别为CNT的区域

    # 8. 增强方法：种子点
    max_dist = dist.max()
    local_max = np.zeros_like(dist, dtype=bool)

    if max_dist > 0:
        density_factor = np.count_nonzero(thresh) / thresh.size
        threshold_base = max_dist * (0.3 + 0.05 * density_factor)
        min_distance = max(1.0, int(extractor.expected_tube_px * 0.4))
        from scipy.ndimage import maximum_filter
        local_max = (maximum_filter(dist, size=min_distance + 1) == dist) & (dist > threshold_base)
        markers, _ = ndi.label(local_max)
    else:
        markers = np.zeros_like(dist, dtype=int)

    # 调试信息
    debug_info = f"max_dist={max_dist:.2f}, n_markers={markers.max()}"

    # 可视化种子点（在原图上叠加黄色点）
    seeds_vis = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
    seeds_vis[local_max] = [0, 255, 255]  # 黄色种子点

    # 9. 增强方法：分水岭分割
    if markers.max() > 0:
        try:
            labels = watershed(-dist, markers, mask=thresh)
            # 检查分割结果
            if labels.max() > 0:
                # 改进可视化：在原图上用不同颜色标注分离后的区域
                watershed_rgb = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
                # 只标注分离的边界（用绿色）
                from scipy.ndimage import sobel
                labels_grad = sobel(labels) > 0
                watershed_rgb[labels_grad] = [0, 255, 0]  # 绿色边界

                # 计算有效区域数（过滤小区域后的）
                expected_area = (extractor.expected_tube_px * 2) ** 2
                valid_labels = []
                for label_id in np.unique(labels):
                    if label_id == 0:
                        continue
                    region_size = (labels == label_id).sum()
                    if region_size >= 10 and region_size <= expected_area * 3:
                        valid_labels.append(label_id)

                debug_info += f", regions={len(valid_labels)}"
            else:
                # 分水岭返回全0，显示二值图
                watershed_rgb = cv2.cvtColor((thresh > 0).astype(np.uint8) * 100, cv2.COLOR_GRAY2BGR)
                watershed_rgb = watershed_rgb / 255.0
                debug_info += ", watershed_failed"
        except Exception as e:
            # 分水岭出错，显示二值图
            watershed_rgb = cv2.cvtColor((thresh > 0).astype(np.uint8) * 100, cv2.COLOR_GRAY2BGR)
            watershed_rgb = watershed_rgb / 255.0
            debug_info += f", watershed_error"
    else:
        # 没有种子点，显示二值图
        watershed_rgb = cv2.cvtColor((thresh > 0).astype(np.uint8) * 100, cv2.COLOR_GRAY2BGR)
        watershed_rgb = watershed_rgb / 255.0
        debug_info += ", no_markers"

    # 创建9宫格
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    fig.suptitle(f'{img_name} | mag={mag}', fontsize=16, fontweight='bold')

    # 第1行：预处理
    axes[0, 0].imshow(orig, cmap='gray')
    axes[0, 0].set_title('1. 原始图像', fontsize=12)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(roi, cmap='gray')
    axes[0, 1].set_title('2. ROI裁切', fontsize=12)
    axes[0, 1].axis('off')

    axes[0, 2].imshow(processed, cmap='gray')
    axes[0, 2].set_title('3. CLAHE增强', fontsize=12)
    axes[0, 2].axis('off')

    # 第2行：分割
    axes[1, 0].imshow(thresh, cmap='gray')
    axes[1, 0].set_title('4. 自适应阈值', fontsize=12)
    axes[1, 0].axis('off')

    axes[1, 1].imshow(skel, cmap='gray')
    axes[1, 1].set_title('5. 骨架化', fontsize=12)
    axes[1, 1].axis('off')

    axes[1, 2].imshow(dist, cmap='hot')
    axes[1, 2].set_title('6. 距离变换（热力图）', fontsize=12)
    axes[1, 2].axis('off')

    # 第3行：直径算法对比
    axes[2, 0].imshow(overlay_std)
    axes[2, 0].set_title('7. 标准方法（闭运算）\n红色=CNT区域', fontsize=12)
    axes[2, 0].axis('off')

    axes[2, 1].imshow(seeds_vis)
    axes[2, 1].set_title('8. 增强方法-种子点\n黄色=局部最大值', fontsize=12)
    axes[2, 1].axis('off')

    axes[2, 2].imshow(watershed_rgb)
    axes[2, 2].set_title(f'9. 增强方法-分水岭分割\n绿色=分离边界\n{debug_info}', fontsize=10)
    axes[2, 2].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"已保存: {output_path}")


def main():
    # 选择20张测试图片（不同倍率、不同样品）
    test_images = [
        # No26 0.75nm - 高倍率
        (r"d:\CNTDATA\ZZY\20230306 No26 0.75 1.0\No26 200w 5.0nm 5w 0.75nm 400 200 100 600 750 15min 180min mid 50000-1.png", 50000),
        (r"d:\CNTDATA\ZZY\20230306 No26 0.75 1.0\No26 200w 5.0nm 5w 0.75nm 400 200 100 600 750 15min 180min mid 100000-1.png", 100000),
        (r"d:\CNTDATA\ZZY\20230306 No26 0.75 1.0\No26 200w 5.0nm 5w 0.75nm 400 200 100 600 750 15min 180min mid 50000-2.png", 50000),
        (r"d:\CNTDATA\ZZY\20230306 No26 0.75 1.0\No26 200w 5.0nm 5w 0.75nm 400 200 100 600 750 15min 180min top 50000-1.png", 50000),
        (r"d:\CNTDATA\ZZY\20230306 No26 0.75 1.0\No26 200w 5.0nm 5w 0.75nm 400 200 100 600 750 15min 180min bottom 50000-1.png", 50000),

        # No26 1.0nm - 高倍率
        (r"d:\CNTDATA\ZZY\20230306 No26 0.75 1.0\No26 200w 5.0nm 5w 1.0nm 400 200 100 600 750 15min 180min mid 50000-1.png", 50000),
        (r"d:\CNTDATA\ZZY\20230306 No26 0.75 1.0\No26 200w 5.0nm 5w 1.0nm 400 200 100 600 750 15min 180min mid 100000-1.png", 100000),

        # No30 - 不同参数
        (r"d:\CNTDATA\ZZY\20260309 No30 0.75 1.0 1.25\No30 200w 5.0nm 5w 0.75nm 400 200 100 600 750 15min 180min mid 50000 1.png", 50000),
        (r"d:\CNTDATA\ZZY\20260309 No30 0.75 1.0 1.25\No30 200w 5.0nm 5w 1.0nm 400 200 100 600 750 15min 180min mid 50000 1.png", 50000),
        (r"d:\CNTDATA\ZZY\20260309 No30 0.75 1.0 1.25\No30 200w 5.0nm 5w 1.25nm 400 200 100 600 750 15min 180min mid 50000 1.png", 50000),

        # No32 - 不同参数
        (r"d:\CNTDATA\ZZY\20260314 No32 0.75 1.0 1.25\No32 200w 5.0nm 5w 0.75nm 400 200 100 600 750 15min 180min mid 50000 0-1.png", 50000),
        (r"d:\CNTDATA\ZZY\20260314 No32 0.75 1.0 1.25\No32 200w 5.0nm 5w 1.0nm 400 200 100 600 750 15min 180min mid 50000 0-1.png", 50000),
        (r"d:\CNTDATA\ZZY\20260314 No32 0.75 1.0 1.25\No32 200w 5.0nm 5w 1.25nm 400 200 100 600 750 15min 180min mid 50000 0-1.png", 50000),

        # 中低倍率
        (r"d:\CNTDATA\ZZY\20230306 No26 0.75 1.0\No26 200w 5.0nm 5w 0.75nm 400 200 100 600 750 15min 180min mid 10000-1.png", 10000),
        (r"d:\CNTDATA\ZZY\20230306 No26 0.75 1.0\No26 200w 5.0nm 5w 0.75nm 400 200 100 600 750 15min 180min mid 5000-1.png", 5000),
        (r"d:\CNTDATA\ZZY\20260306 No26 0.75 1.0\No26 200w 5.0nm 5w 0.75nm 400 200 100 600 750 15min 180min mid 1000-1.png", 1000),

        # 100kx 超高倍率
        (r"d:\CNTDATA\ZZY\20230306 No26 0.75 1.0\No26 200w 5.0nm 5w 1.0nm 400 200 100 600 750 15min 180min bottom 100000-3.png", 100000),

        # 顶部/底部位置差异
        (r"d:\CNTDATA\ZZY\20260314 No32 0.75 1.0 1.25\No32 200w 5.0nm 5w 1.0nm 400 200 100 600 750 15min 180min top 50000 0-1.png", 50000),
    ]

    # 创建输出目录
    output_dir = r"d:\CNTDATA\CNTA_ML_Project\src\analysis\algorithm_vis"
    os.makedirs(output_dir, exist_ok=True)

    print("="*70)
    print("算法可视化开始")
    print(f"输出目录: {output_dir}")
    print("="*70)

    # 处理每张图片
    for idx, (img_path, mag) in enumerate(test_images, 1):
        if not os.path.exists(img_path):
            print(f"[跳过] {img_path}")
            continue

        img_name = os.path.basename(img_path)
        output_name = f"{idx:02d}_{img_name.replace('.png', '')}_vis.png"
        output_path = os.path.join(output_dir, output_name)

        print(f"[{idx}/{len(test_images)}] 处理: {img_name} | mag={mag}")
        visualize_full_pipeline(img_path, mag, output_path)

    print("="*70)
    print(f"完成！生成 {len([f for f in os.listdir(output_dir) if f.endswith('.png')])} 张可视化图像")
    print("="*70)


if __name__ == "__main__":
    main()
