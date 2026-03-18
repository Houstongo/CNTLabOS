"""
算法可视化模块（改进版：分水岭分割确保CNT独立性）
生成特征提取过程的中间步骤结果
"""
import cv2
import numpy as np
import base64
from io import BytesIO


class AlgorithmVisualizer:
    """算法可视化：生成每一步的中间结果"""

    def __init__(self, magnification=50000):
        self.mag = magnification
        self.steps = []
        self.current_image = None
        self.current_step = 0

    def add_step(self, name, image, description=""):
        """添加一个步骤"""
        _, buffer = cv2.imencode('.jpg', image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        self.steps.append({
            'name': name,
            'image': img_base64,
            'description': description
        })

    def visualize_extraction(self, img_gray):
        """生成完整的特征提取可视化流程"""
        self.steps = []

        # 步骤1：原始图像
        self.add_step("原始图像", img_gray,
            "读取的灰度SEM图像。SEM（扫描电子显微镜）利用电子束扫描样品表面，通过检测反射电子成像。")

        # 步骤2：CLAHE增强
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(img_gray)
        self.add_step("CLAHE增强", enhanced,
            "自适应直方图均衡化，增强对比度。原理：将图像分成小块，对每块进行直方图均衡化，同时限制对比度增强幅度。")

        # 步骤3：高斯模糊（轻度）
        smoothed = cv2.GaussianBlur(enhanced, (2, 2), 0)
        self.add_step("高斯模糊", smoothed,
            "2×2高斯滤波，轻度去噪。原理：用高斯函数作为卷积核，减少噪声同时保留边缘细节。")

        # 步骤4：二值化
        _, thresh = cv2.threshold(smoothed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self.add_step("二值化", thresh,
            "Otsu自适应阈值分割。原理：自动寻找最佳阈值，使类内方差最小化，将图像分为前景（CNT）和背景。")

        # 步骤5：分水岭分割（分离交叉CNT）
        from scipy.ndimage import maximum_filter, label

        dist = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)

        # 找种子点
        local_max = maximum_filter(dist, footprint=np.ones((3, 3), dtype=int))
        threshold_value = dist.max() * 0.3
        local_max = (local_max == dist) & (dist > threshold_value)

        # 标记种子点
        markers = label(local_max, structure=np.ones((3, 3), dtype=int))
        markers = markers.astype(np.int32)

        # 分水岭分割
        cv2.watershed(cv2.cvtColor(smoothed, cv2.COLOR_GRAY2BGR), markers)

        # 创建分割后的二值图像（移除负标记）
        segmented = np.zeros_like(thresh)
        for i in range(1, markers.max() + 1):
            segmented[markers == i] = 255

        # 可视化分割结果（每个区域不同颜色）
        segmented_display = cv2.cvtColor(segmented, cv2.COLOR_GRAY2BGR)
        segmented_labels = label(segmented > 0, connectivity=2)
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
                  (0, 255, 255), (128, 0, 128), (0, 128, 0), (0, 0, 128), (128, 128, 0)]
        for rid in range(1, min(segmented_labels.max() + 1, 11)):
            color = colors[(rid - 1) % len(colors)]
            segmented_display[segmented_labels == rid] = color

        self.add_step("分水岭分割", segmented_display,
            "分水岭算法分离交叉CNT。原理：基于距离变换找到种子点，用分水岭算法将交叉的CNT分割为独立的连通区域，每根CNT一条骨架。")

        # 步骤6：骨架提取
        from skimage.morphology import skeletonize

        skel = skeletonize(segmented > 0).astype(np.uint8) * 255

        # 叠加显示骨架
        skeleton_display = cv2.cvtColor(smoothed, cv2.COLOR_GRAY2BGR)
        skeleton_display[skel > 0] = [0, 255, 0]  # 绿色骨架
        self.add_step("骨架提取", skeleton_display,
            "绿色曲线为CNT骨架。原理：骨架提取通过迭代腐蚀保留中心线，得到1像素宽的拓扑结构。分水岭分割确保每根CNT独立，骨架不连通。")

        # 步骤7：对齐方向场
        self._add_alignment_step(enhanced, skel)

        # 步骤8：直径测量可视化
        self._add_diameter_step(segmented, skel)

        # 步骤9：骨架追踪曲率
        self._add_curvature_step(skel)

        return self.steps

    def _add_alignment_step(self, enhanced, skel):
        """添加对齐分析步骤"""
        # 使用结构张量计算方向
        Ix = cv2.Scharr(enhanced, cv2.CV_64F, 1, 0)
        Iy = cv2.Scharr(enhanced, cv2.CV_64F, 0, 1)

        sigma = 3.0
        ksize = int(sigma * 4) | 1
        Jxx = cv2.GaussianBlur(Ix * Ix, (ksize, ksize), sigma)
        Jxy = cv2.GaussianBlur(Ix * Iy, (ksize, ksize), sigma)
        Jyy = cv2.GaussianBlur(Iy * Iy, (ksize, ksize), sigma)

        # 主方向
        theta = 0.5 * np.arctan2(2.0 * Jxy, Jxx - Jyy)

        # 创建方向场可视化（降采样）
        step = 20
        height, width = enhanced.shape
        direction_display = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

        for y in range(step//2, height, step):
            for x in range(step//2, width, step):
                angle = theta[y, x]
                dx = 15 * np.cos(angle)
                dy = 15 * np.sin(angle)
                cv2.arrowedLine(direction_display, (x, y), (int(x+dx), int(y+dy)), (255, 0, 0), 1)

        self.add_step("对齐方向场", direction_display,
            "蓝色箭头显示CNT的主要生长方向。原理：使用结构张量（Structure Tensor）计算局部梯度方向。Jxx和Jyy表示x和y方向的梯度能量，θ = 0.5·arctan(2Jxy, Jxx-Jyy)给出主方向角度。")

    def _add_diameter_step(self, thresh, skel):
        """添加直径测量步骤"""
        # 距离变换
        dist = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)

        # 在骨架上标注直径
        diameter_display = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

        # 采样一些骨架点
        skel_points = np.argwhere(skel > 0)
        if len(skel_points) > 0:
            sample_indices = np.random.choice(len(skel_points), min(50, len(skel_points)), replace=False)
            for idx in sample_indices:
                y, x = skel_points[idx]
                radius = dist[y, x]
                if radius > 0:
                    # 画圆表示直径
                    cv2.circle(diameter_display, (x, y), int(radius), (0, 255, 255), 1)

        self.add_step("直径测量", diameter_display,
            "黄色圆表示估计的CNT直径（半径）。原理：基于距离变换，CNT直径D = 2 × 距离值。距离变换给出每个前景点到背景的最短距离，对于圆柱状CNT，这个距离近似等于半径。")

    def _add_curvature_step(self, skel):
        """添加骨架追踪曲率分析步骤"""
        from skimage.measure import label
        from scipy.ndimage import convolve

        labeled = label(skel, connectivity=2)
        n_regions = labeled.max()

        # 创建曲率可视化
        curvature_display = cv2.cvtColor(skel, cv2.COLOR_GRAY2BGR)

        all_curvatures = []
        px_per_nm = 0.05 if self.mag == 50000 else 0.1

        if n_regions > 0:
            for rid in range(1, min(n_regions + 1, 10)):
                region_mask = (labeled == rid).astype(np.uint8)
                coords = np.argwhere(region_mask)

                if len(coords) < 10:
                    continue

                # 找端点
                kernel = np.ones((3, 3), dtype=np.uint8)
                kernel[1, 1] = 0
                neighbor_count = convolve(region_mask, kernel, mode='constant', cval=0)
                endpoints = np.argwhere((region_mask > 0) & (neighbor_count == 1))

                if len(endpoints) >= 2:
                    # 追踪路径并计算曲率
                    path = self._trace_skeleton(region_mask, endpoints[0], endpoints[1])
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

                                    if curvature_nm < 0.05:
                                        color = (0, 255, 0)
                                    elif curvature_nm < 0.15:
                                        color = (0, 255, 255)
                                    else:
                                        color = (0, 0, 255)

                                    cv2.circle(curvature_display, (int(p2[1]), int(p2[0])), 2, color, -1)

        # 计算平均曲率
        if all_curvatures:
            avg_curvature_nm = np.median(all_curvatures)
        else:
            avg_curvature_nm = 0.0

        if avg_curvature_nm < 0.05:
            curvature_label = "直：κ < 0.05 nm⁻¹"
        elif avg_curvature_nm < 0.15:
            curvature_label = "波：0.05 ≤ κ < 0.15 nm⁻¹"
        else:
            curvature_label = "卷曲：κ ≥ 0.15 nm⁻¹"

        cv2.putText(curvature_display, f"平均曲率 κ = {avg_curvature_nm:.3f} nm⁻¹", (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(curvature_display, curvature_label, (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        self.add_step("骨架追踪曲率", curvature_display,
            f"平均曲率 κ = {avg_curvature_nm:.3f} nm⁻¹。原理：基于骨架化提取中轴线，追踪真实路径计算曲率。"
            f"算法流程：1)识别端点（度数=1的骨架点）；2)追踪骨架路径；3)计算曲率κ = dθ/ds（角度变化/路径长度）。"
            f"优势：反映真实的CNT形变，对长波弯曲敏感。分水岭分割确保每根CNT独立。"
            f"颜色编码：绿色=直（κ < 0.05），黄色=波（0.05 ≤ κ < 0.15），红色=卷曲（κ ≥ 0.15）。")

    def _trace_skeleton(self, mask, start, end):
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

    def get_steps(self):
        """获取所有步骤"""
        return self.steps

    def get_step(self, index):
        """获取特定步骤"""
        if 0 <= index < len(self.steps):
            return self.steps[index]
        return None
