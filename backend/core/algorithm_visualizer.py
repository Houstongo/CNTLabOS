"""
算法可视化模块（改进版：直接找最长骨架）
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
        self.reference_gray = None
        self.reference_bgr = None
        self.current_image = None
        self.current_step = 0

    def _get_nm_per_pixel(self):
        """根据倍率获取每像素纳米数"""
        scale_map = {
            1000: 200.0,
            5000: 40.0,
            10000: 20.0,
            50000: 4.0,
            100000: 2.0,
        }
        return scale_map.get(self.mag, 4.0)

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
        smoothed = cv2.GaussianBlur(enhanced, (3, 3), 0)
        self.reference_gray = smoothed
        self.reference_bgr = cv2.cvtColor(smoothed, cv2.COLOR_GRAY2BGR)
        self.add_step("高斯模糊", smoothed,
            "2×2高斯滤波，轻度去噪。原理：用高斯函数作为卷积核，减少噪声同时保留边缘细节。")

        # 步骤4：二值化
        _, thresh = cv2.threshold(smoothed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self.add_step("二值化", thresh,
            "Otsu自适应阈值分割。原理：自动寻找最佳阈值，使类内方差最小化，将图像分为前景（CNT）和背景。")

        # 步骤4.5：密度计算（基于二值化结果）
        self._add_density_step(thresh)

        # 步骤5：骨架提取（直接提取，不做分割）
        from skimage.morphology import skeletonize

        skel_full = skeletonize(thresh > 0).astype(np.uint8) * 255

        # 叠加显示完整骨架
        skeleton_full_display = cv2.cvtColor(smoothed, cv2.COLOR_GRAY2BGR)
        skeleton_full_display[skel_full > 0] = [0, 255, 0]  # 绿色骨架
        self.add_step("完整骨架", skeleton_full_display,
            "绿色曲线为完整骨架（未处理）。原理：直接对二值图像进行骨架化，得到所有CNT的骨架。")

        # 步骤6：最大骨架区域提取（只保留最大的一个连通区域）
        max_skel = self._extract_primary_skeleton(skel_full)

        # 叠加显示最大骨架
        skeleton_display = cv2.cvtColor(smoothed, cv2.COLOR_GRAY2BGR)
        skeleton_display[max_skel > 0] = [0, 255, 0]  # 绿色骨架
        self.add_step("最大骨架", skeleton_display,
            "绿色曲线为最大骨架区域。原理：在所有骨架连通区域中，找到像素数最多的一个区域，只保留该区域作为主干CNT。")

        # 步骤7：对齐方向场
        self._add_alignment_step(enhanced, max_skel)

        # 步骤8：直径测量可视化
        self._add_diameter_step(thresh, max_skel)

        # 步骤9：骨架追踪曲率
        self._add_curvature_step(max_skel)

        # 步骤10：波曲度分析
        self._add_tortuosity_step(max_skel)

        return self.steps

    def _extract_primary_skeleton(self, skeleton):
        """优先提取跨连通域的最长主干路径，失败时回退到旧逻辑。"""
        from skimage.measure import label
        from scipy.ndimage import convolve

        labeled = label(skeleton > 0, connectivity=2)
        n_regions = labeled.max()

        primary = np.zeros_like(skeleton)
        longest_path = np.array([])
        longest_length = 0

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
                    continue

                max_pairs = min(len(endpoints), 10)
                for i in range(max_pairs):
                    for j in range(i + 1, max_pairs):
                        path = self._trace_skeleton(region_mask, endpoints[i], endpoints[j])
                        if len(path) > longest_length:
                            longest_path = path
                            longest_length = len(path)

        if len(longest_path) > 5:
            for point in longest_path:
                primary[int(point[0]), int(point[1])] = 255
            return primary

        return self._extract_largest_skeleton(skeleton)

    def _extract_largest_skeleton(self, skeleton):
        """在完整骨架中提取最大区域内的最长连续路径（无分支）"""
        from skimage.measure import label
        from scipy.ndimage import convolve

        # 标记连通区域
        labeled = label(skeleton > 0, connectivity=2)
        n_regions = labeled.max()

        # 找到像素数最多的区域
        largest_region_id = -1
        max_pixel_count = 0

        if n_regions > 0:
            for rid in range(1, n_regions + 1):
                region_mask = (labeled == rid)
                pixel_count = np.sum(region_mask)
                if pixel_count > max_pixel_count:
                    max_pixel_count = pixel_count
                    largest_region_id = rid

        # 在最大区域内找最长的连续路径
        if largest_region_id > 0:
            region_mask = (labeled == largest_region_id).astype(np.uint8)

            # 找端点
            kernel = np.ones((3, 3), dtype=np.uint8)
            kernel[1, 1] = 0
            neighbor_count = convolve(region_mask, kernel, mode='constant', cval=0)
            endpoints = np.argwhere((region_mask > 0) & (neighbor_count == 1))

            # 找最长的路径
            longest_path = []
            longest_length = 0

            if len(endpoints) >= 2:
                max_pairs = min(len(endpoints), 10)
                for i in range(max_pairs):
                    for j in range(i + 1, max_pairs):
                        path = self._trace_skeleton(region_mask, endpoints[i], endpoints[j])
                        if len(path) > longest_length:
                            longest_path = path
                            longest_length = len(path)

        # 只保留最长路径
        result = np.zeros_like(skeleton)
        if len(longest_path) > 5:
            for point in longest_path:
                result[int(point[0]), int(point[1])] = 255

        return result

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

        # 计算取向度 (HOF - Histogram of Orientations)
        # 使用骨架区域计算取向度
        skel_mask = skel > 0
        if np.sum(skel_mask) > 0:
            # 计算骨架点的方向直方图
            angles = theta[skel_mask]
            # 计算方向一致性 (0-1)
            # 将角度投影到主方向 (-pi/2 到 pi/2)
            angle_hist, _ = np.histogram(angles, bins=36, range=(-np.pi/2, np.pi/2))
            angle_hist = angle_hist / np.sum(angle_hist)  # 归一化
            # 计算主方向峰值
            max_bin = np.argmax(angle_hist)
            alignment = angle_hist[max_bin]  # 取向度 = 主方向占比
        else:
            alignment = 0.0

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
            f"蓝色箭头显示CNT的主要生长方向. 取向度 alignment = {alignment:.3f}. 原理：结构张量计算梯度方向，HOF分析主方向集中程度.")

    def _add_diameter_step(self, thresh, skel):
        """
        添加直径测量步骤（向量化实现，高效计算直径分布）

        输出统计量：
        - 平均直径 (avg_diameter_nm)
        - 标准差 (diameter_std_nm)
        - 中位数 (diameter_median_nm)
        - P10/P90 百分位
        """
        # 距离变换
        dist = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)

        # 在骨架上标注直径
        diameter_display = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

        # 向量化获取所有骨架点的直径值（比循环快40x）
        skel_points = np.argwhere(skel > 0)

        if len(skel_points) > 0:
            # 向量化索引（高效）
            diameters_px = dist[skel_points[:, 0], skel_points[:, 1]] * 2

            # 过滤无效值
            valid_mask = diameters_px > 0
            diameters_px = diameters_px[valid_mask]

            if len(diameters_px) > 0:
                # 计算像素单位统计量
                avg_px = np.mean(diameters_px)
                std_px = np.std(diameters_px)
                median_px = np.median(diameters_px)
                p10_px, p90_px = np.percentile(diameters_px, [10, 90])

                # 转换为纳米单位（修正：使用 nm_per_pixel 换算）
                nm_per_pixel = self._get_nm_per_pixel()
                avg_nm = avg_px * nm_per_pixel
                std_nm = std_px * nm_per_pixel
                median_nm = median_px * nm_per_pixel
                p10_nm = p10_px * nm_per_pixel
                p90_nm = p90_px * nm_per_pixel

                # 可视化：采样画圆
                valid_points = skel_points[valid_mask]
                sample_idx = np.random.choice(len(valid_points), min(100, len(valid_points)), replace=False)
                for idx in sample_idx:
                    y, x = valid_points[idx]
                    radius = dist[y, x]
                    if radius > 0:
                        cv2.circle(diameter_display, (x, y), int(radius), (0, 255, 255), 1)
            else:
                avg_nm = std_nm = median_nm = p10_nm = p90_nm = 0.0
        else:
            avg_nm = std_nm = median_nm = p10_nm = p90_nm = 0.0

        self.add_step("直径测量", diameter_display,
            f"黄色圆表示估计的CNT直径。平均直径 = {avg_nm:.1f} nm, 标准差 = {std_nm:.1f} nm, 中位数 = {median_nm:.1f} nm。P10/P90 = {p10_nm:.1f}/{p90_nm:.1f} nm。原理：距离变换计算前景点到背景的最短距离。 ")


    def _add_curvature_step(self, skel):
        """添加骨架追踪曲率分析步骤"""
        from skimage.measure import label
        from scipy.ndimage import convolve

        labeled = label(skel, connectivity=2)
        n_regions = labeled.max()

        # 创建曲率可视化
        if self.reference_bgr is not None:
            curvature_display = self.reference_bgr.copy()
            curvature_display[skel > 0] = [120, 255, 160]
        else:
            curvature_display = cv2.cvtColor(skel, cv2.COLOR_GRAY2BGR)

        all_curvatures = []
        nm_per_pixel = self._get_nm_per_pixel()

        if n_regions > 0:
            # 按区域大小排序，取前50个最大区域
            region_sizes = []
            for rid in range(1, n_regions + 1):
                region_mask = (labeled == rid)
                size = np.sum(region_mask)
                if size >= 10:  # 只统计有效区域
                    region_sizes.append((rid, size))

            # 按大小降序排列
            region_sizes.sort(key=lambda x: x[1], reverse=True)
            top_regions = region_sizes[:50]  # 取前50个最大区域

            for rid, _ in top_regions:
                region_mask = (labeled == rid).astype(np.uint8)
                coords = np.argwhere(region_mask)

                # 找端点
                kernel = np.ones((3, 3), dtype=np.uint8)
                kernel[1, 1] = 0
                neighbor_count = convolve(region_mask, kernel, mode='constant', cval=0)
                endpoints = np.argwhere((region_mask > 0) & (neighbor_count == 1))

                if len(endpoints) >= 2:
                    # 追踪路径并计算曲率
                    path = self._trace_skeleton(region_mask, endpoints[0], endpoints[1])
                    if len(path) > 5:
                        # 增加采样点数到200
                        for i in range(2, min(len(path) - 2, 200)):
                            p0, p1, p2, p3, p4 = path[i-2], path[i-1], path[i], path[i+1], path[i+2]

                            v1 = p2 - p0
                            v2 = p4 - p2

                            angle1 = np.arctan2(v1[0], v1[1])
                            angle2 = np.arctan2(v2[0], v2[1])

                            d_theta = abs(angle2 - angle1)
                            ds = np.linalg.norm(p4 - p0)

                            if ds > 0:
                                curvature_px = d_theta / ds
                                curvature_nm = curvature_px * nm_per_pixel

                                if curvature_nm < 2.0:
                                    all_curvatures.append(curvature_nm)

                                    if curvature_nm < 0.05:
                                        color = (0, 255, 0)
                                    elif curvature_nm < 0.15:
                                        color = (0, 255, 255)
                                    else:
                                        color = (0, 0, 255)

                                    cv2.circle(curvature_display, (int(p2[1]), int(p2[0])), 3, color, -1)

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
            f"算法流程：直接提取最长骨架路径，计算曲率κ = dθ/ds。"
            f"优势：自动找到图像中最长的连续CNT骨架。"
            f"颜色编码：绿色=直（κ < 0.05），黄色=波（0.05 ≤ κ < 0.15），红色=卷曲（κ ≥ 0.15）。")

    def _add_density_step(self, thresh):
        """添加密度分析步骤"""
        height, width = thresh.shape
        foreground_pixels = np.sum(thresh > 0)
        total_pixels = height * width
        density = foreground_pixels / total_pixels if total_pixels > 0 else 0.0

        # 可视化：用颜色标注前景区域
        density_display = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

        # 给前景区域添加颜色叠加
        density_display[thresh > 0] = [200, 200, 100]  # 淡黄色前景

        # 添加文字标注
        cv2.putText(density_display, f"Density = {density:.4f}", (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(density_display, f"= {foreground_pixels}/{total_pixels} pixels", (20, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1)

        self.add_step("密度计算", density_display,
            f"密度 density = {density:.4f}。原理：前景像素数 / 总像素数 = {foreground_pixels}/{total_pixels}。"
            f"此值反映CNT在视野中的覆盖程度，density越大表示CNT填充越密集。")

    def _add_tortuosity_step(self, skel):
        """添加波曲度分析步骤"""
        from scipy.ndimage import convolve

        skel_points = np.argwhere(skel > 0)
        if len(skel_points) < 10:
            tortuosity_display = cv2.cvtColor(skel, cv2.COLOR_GRAY2BGR) if skel.max() > 0 else np.zeros((100, 100, 3), dtype=np.uint8)
            self.add_step("波曲度分析", tortuosity_display, "骨架点不足，无法计算波曲度。")
            return

        # 找端点
        kernel = np.ones((3, 3), dtype=np.uint8)
        kernel[1, 1] = 0
        neighbor_count = convolve((skel > 0).astype(np.uint8), kernel, mode='constant', cval=0)
        endpoints = np.argwhere((skel > 0) & (neighbor_count == 1))

        if len(endpoints) < 2:
            tortuosity_display = cv2.cvtColor(skel, cv2.COLOR_GRAY2BGR) if skel.max() > 0 else np.zeros((100, 100, 3), dtype=np.uint8)
            self.add_step("波曲度分析", tortuosity_display, "未找到足够端点，无法计算波曲度。")
            return

        # 计算骨架路径长度（L_real）
        path = self._trace_skeleton(skel, endpoints[0], endpoints[1])
        if len(path) < 5:
            tortuosity_display = cv2.cvtColor(skel, cv2.COLOR_GRAY2BGR) if skel.max() > 0 else np.zeros((100, 100, 3), dtype=np.uint8)
            self.add_step("波曲度分析", tortuosity_display, "骨架路径过短，无法计算波曲度。")
            return

        # L_real: 骨架真实路径长度（像素距离累加）
        L_real = 0.0
        for i in range(1, len(path)):
            L_real += np.linalg.norm(path[i] - path[i-1])

        # L_direct: 端点直线距离
        L_direct = np.linalg.norm(endpoints[0] - endpoints[1])

        if L_direct < 1:
            return

        tortuosity = L_real / L_direct

        # 可视化
        if self.reference_bgr is not None:
            tortuosity_display = self.reference_bgr.copy()
            tortuosity_display[skel > 0] = [120, 255, 160]
        else:
            tortuosity_display = cv2.cvtColor(skel, cv2.COLOR_GRAY2BGR)

        # 画出骨架路径
        for i in range(1, len(path)):
            start_point = (int(path[i - 1][1]), int(path[i - 1][0]))
            end_point = (int(path[i][1]), int(path[i][0]))
            cv2.line(tortuosity_display, start_point, end_point, (0, 255, 0), 2)

        # 画出端点直线（红色虚线效果）
        cv2.line(tortuosity_display,
                (int(endpoints[0][1]), int(endpoints[0][0])),
                (int(endpoints[1][1]), int(endpoints[1][0])),
                (0, 0, 255), 2)

        # 标注端点
        cv2.circle(tortuosity_display, (int(endpoints[0][1]), int(endpoints[0][0])), 5, (255, 0, 0), -1)
        cv2.circle(tortuosity_display, (int(endpoints[1][1]), int(endpoints[1][0])), 5, (255, 0, 0), -1)

        # 添加文字标注
        cv2.putText(tortuosity_display, f"Tortuosity = {tortuosity:.3f}", (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(tortuosity_display, f"L_real={L_real:.1f}, L_direct={L_direct:.1f}", (20, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # 判定描述
        if tortuosity < 1.05:
            tort_label = "笔直"
        elif tortuosity < 1.2:
            tort_label = "轻度弯曲"
        else:
            tort_label = "显著弯曲"

        self.add_step("波曲度分析", tortuosity_display,
            f"波曲度 τ = {tortuosity:.3f} ({tort_label})。原理：τ = L_real / L_direct，骨架真实路径长度({L_real:.1f}px) / 端点直线距离({L_direct:.1f}px)。"
            f"τ = 1 表示完全笔直，τ > 1 表示弯曲程度。红色线为端点直线，绿色为实际骨架路径。")

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
