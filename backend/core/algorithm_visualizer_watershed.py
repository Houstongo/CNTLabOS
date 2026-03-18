"""
算法可视化模块
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
        # 转换为JPEG base64
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

        # 步骤5：形态学闭运算（连接断裂）
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 2))
        thresh_closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

        # 可视化闭运算效果
        closed_display = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        # 标记被连接的区域（用红色显示新增部分）
        new_parts = cv2.bitwise_and(thresh_closed, cv2.bitwise_not(thresh))
        closed_display[new_parts > 0] = [0, 0, 255]
        self.add_step("闭运算连接", closed_display,
            "红色区域为闭运算连接的部分。原理：先膨胀再腐蚀，填补CNT之间的断裂和空隙，使骨架保持连通。")

        # 步骤6：距离变换
        dist = cv2.distanceTransform(thresh_closed, cv2.DIST_L2, 5)
        # 归一化到0-255用于显示
        dist_display = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        # 应用伪彩色
        dist_color = cv2.applyColorMap(dist_display, cv2.COLORMAP_JET)
        self.add_step("距离变换", dist_color,
            "计算每个前景点到背景的距离，用于直径估计。原理：距离变换计算图像中每个像素到最近背景点的欧氏距离。")

        # 步骤6：分水岭标记（种子点）
        from scipy.ndimage import maximum_filter
        local_max = maximum_filter(dist, footprint=np.ones((3, 3), dtype=int))
        threshold_value = dist.max() * 0.3
        local_max = (local_max == dist) & (dist > threshold_value)

        # 可视化种子点
        markers_display = cv2.cvtColor(thresh_closed, cv2.COLOR_GRAY2BGR)
        markers_display[local_max] = [0, 0, 255]  # 红色标记种子点
        self.add_step("分水岭种子点", markers_display,
            "红色标记局部最大值点，作为分水岭分割的种子。原理：在距离变换图中找到局部最大值，这些点代表不同CNT的中心。")

        # 步骤7：骨架提取
        from skimage.morphology import skeletonize, remove_small_objects
        from skimage.measure import label

        skel_raw = skeletonize(thresh_closed > 0)

        # 骨架修剪：同一连通区域只保留1条主干
        skel = self._prune_skeleton(skel_raw)

        # 叠加显示骨架
        skeleton_display = cv2.cvtColor(thresh_closed, cv2.COLOR_GRAY2BGR)
        skeleton_display[skel > 0] = [0, 255, 0]  # 绿色骨架
        self.add_step("骨架提取", skeleton_display,
            f"绿色曲线为CNT骨架，用于曲率和对齐分析。原理：骨架提取通过迭代腐蚀保留中心线，得到1像素宽的拓扑结构。骨架修剪去除长度< {min_length_nm} nm 的短分支，只保留主要骨架。")

        # 步骤8：对齐方向场
        self._add_alignment_step(enhanced, skel)

        # 步骤9：直径测量可视化
        self._add_diameter_step(thresh_closed, skel)

        # 步骤10：平均曲率分析
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
        from scipy.ndimage import binary_dilation

        labeled = label(skel, connectivity=2)
        n_regions = labeled.max()

        # 创建曲率可视化
        curvature_display = cv2.cvtColor(skel, cv2.COLOR_GRAY2BGR)

        # 骨架曲率参数
        px_per_nm = 0.05  # 50,000倍率时，1像素=0.05nm
        all_curvatures = []

        if n_regions > 0:
            for rid in range(1, min(n_regions + 1, 10)):  # 最多分析10个区域
                region_mask = (labeled == rid).astype(np.uint8)
                coords = np.argwhere(region_mask).astype(float)

                if len(coords) < 10:
                    continue

                # 找到端点：度数为1的骨架点
                from scipy.ndimage import convolve
                kernel = np.ones((3, 3), dtype=np.uint8)
                kernel[1, 1] = 0
                neighbor_count = convolve(region_mask, kernel, mode='constant', cval=0)
                endpoints = np.argwhere((region_mask > 0) & (neighbor_count == 1))

                if len(endpoints) >= 2:
                    # 追踪骨架路径
                    path = self._trace_skeleton(region_mask, endpoints[0], endpoints[1])

                    if len(path) > 5:
                        # 计算路径曲率 dθ/ds
                        for i in range(2, min(len(path) - 2, 50)):  # 采样50个点
                            # 使用5点计算角度变化
                            p0, p1, p2, p3, p4 = path[i-2], path[i-1], path[i], path[i+1], path[i+2]

                            # 计算两个方向向量的角度
                            v1 = p2 - p0
                            v2 = p4 - p2

                            angle1 = np.arctan2(v1[0], v1[1])
                            angle2 = np.arctan2(v2[0], v2[1])

                            # 角度变化
                            d_theta = abs(angle2 - angle1)

                            # 路径长度
                            ds = np.linalg.norm(p4 - p0)

                            # 曲率 κ = dθ/ds
                            if ds > 0:
                                curvature_px = d_theta / ds
                                curvature_nm = curvature_px / px_per_nm

                                if curvature_nm < 2.0:  # 过滤异常值
                                    all_curvatures.append(curvature_nm)

                                    # 颜色编码
                                    if curvature_nm < 0.05:
                                        color = (0, 255, 0)  # 绿色=直
                                    elif curvature_nm < 0.15:
                                        color = (0, 255, 255)  # 黄色=波
                                    else:
                                        color = (0, 0, 255)  # 红色=卷曲

                                    y, x = int(p2[0]), int(p2[1])
                                    cv2.circle(curvature_display, (x, y), 2, color, -1)

        # 计算平均曲率
        if all_curvatures:
            avg_curvature_nm = np.median(all_curvatures)
        else:
            avg_curvature_nm = 0.0

        # 添加曲率说明文本
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
            f"优势：反映真实的CNT形变，对长波弯曲敏感。"
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

            # 检查是否到达终点
            if np.linalg.norm(current - end) < 3:
                break

            # 寻找下一个骨架点
            next_point = None
            min_dist = float('inf')

            # 3×3邻域搜索
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if (ny, nx) not in visited and 0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1]:
                        if mask[ny, nx] > 0:
                            # 选择距离终点最近的点
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

    def _prune_skeleton(self, skeleton, min_length=50):
        """修剪骨架：同一连通区域只保留1条主干"""
        from skimage.measure import label
        from scipy.ndimage import convolve

        skel = skeleton.copy().astype(np.uint8)

        # 标记连通区域
        labeled = label(skel, connectivity=2)
        n_regions = labeled.max()

        pruned = np.zeros_like(skel)

        if n_regions > 0:
            for rid in range(1, min(n_regions + 1, 20)):  # 最多处理20个区域
                region_mask = (labeled == rid).astype(np.uint8)
                coords = np.argwhere(region_mask)

                if len(coords) < 10:
                    continue

                # 找到所有端点
                kernel = np.ones((3, 3), dtype=np.uint8)
                kernel[1, 1] = 0
                neighbor_count = convolve(region_mask, kernel, mode='constant', cval=0)
                endpoints = np.argwhere((region_mask > 0) & (neighbor_count == 1))

                if len(endpoints) < 2:
                    # 没有端点或只有一个，保留整个区域
                    pruned[region_mask > 0] = 255
                    continue

                # 计算所有端点对之间的路径，选择最长的
                longest_path = []
                longest_length = 0

                # 限制端点对数量，避免计算爆炸
                max_pairs = min(len(endpoints), 10)
                for i in range(max_pairs):
                    for j in range(i + 1, max_pairs):
                        path = self._trace_skeleton(region_mask, endpoints[i], endpoints[j])
                        if len(path) > longest_length:
                            longest_path = path
                            longest_length = len(path)

                # 保留最长路径
                if len(longest_path) > 5:
                    for point in longest_path:
                        pruned[int(point[0]), int(point[1])] = 255

        return pruned
