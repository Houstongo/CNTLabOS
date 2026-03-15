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

        # 步骤3：高斯模糊
        smoothed = cv2.GaussianBlur(enhanced, (3, 3), 0)
        self.add_step("高斯模糊", smoothed,
            "3×3高斯滤波，去噪。原理：用高斯函数作为卷积核，对图像进行平滑处理，保留边缘信息的同时减少噪声。")

        # 步骤4：二值化
        _, thresh = cv2.threshold(smoothed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self.add_step("二值化", thresh,
            "Otsu自适应阈值分割。原理：自动寻找最佳阈值，使类内方差最小化，将图像分为前景（CNT）和背景。")

        # 步骤5：距离变换
        dist = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
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
        markers_display = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        markers_display[local_max] = [0, 0, 255]  # 红色标记种子点
        self.add_step("分水岭种子点", markers_display,
            "红色标记局部最大值点，作为分水岭分割的种子。原理：在距离变换图中找到局部最大值，这些点代表不同CNT的中心。")

        # 步骤7：骨架提取
        from skimage.morphology import skeletonize
        skel = skeletonize(thresh > 0).astype(np.uint8) * 255

        # 叠加显示骨架
        skeleton_display = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        skeleton_display[skel > 0] = [0, 255, 0]  # 绿色骨架
        self.add_step("骨架提取", skeleton_display,
            "绿色曲线为CNT骨架，用于曲率和对齐分析。原理：骨架提取通过迭代腐蚀保留中心线，得到1像素宽的拓扑结构。")

        # 步骤8：对齐方向场
        self._add_alignment_step(enhanced, skel)

        # 步骤9：直径测量可视化
        self._add_diameter_step(thresh, skel)

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
        """添加平均曲率分析步骤"""
        from skimage.measure import label

        labeled = label(skel, connectivity=2)
        n_regions = labeled.max()

        # 创建曲率可视化
        curvature_display = cv2.cvtColor(skel, cv2.COLOR_GRAY2BGR)

        all_curvatures = []

        if n_regions > 0:
            for rid in range(1, min(n_regions + 1, 10)):  # 最多分析10个区域
                coords = np.argwhere(labeled == rid).astype(float)
                if len(coords) < 15:
                    continue

                # 按y,x排序
                sorted_indices = np.lexsort((coords[:, 1], coords[:, 0]))
                coords = coords[sorted_indices]

                # 降采样（优化性能）
                step = max(2, int(len(coords) / 20))
                sampled = coords[::step]

                if len(sampled) > 2:
                    # 计算曲率并颜色编码（三点法）
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

                            # 异常值过滤（去除极值）
                            if curvature < 0.2:  # 避免数值不稳定
                                all_curvatures.append(curvature)

                            # 颜色编码（转换为nm⁻¹）
                            px_per_nm = 0.05  # 50,000倍率
                            curvature_nm = curvature / px_per_nm

                            # 颜色编码：绿色=直，黄色=弯，红色=卷曲
                            if curvature_nm < 0.05:
                                color = (0, 255, 0)
                            elif curvature_nm < 0.15:
                                color = (0, 255, 255)
                            else:
                                color = (0, 0, 255)

                            cv2.circle(curvature_display, (int(p_curr[1]), int(p_curr[0])), 2, color, -1)

        # 计算平均曲率
        if all_curvatures:
            avg_curvature_px = np.median(all_curvatures)
            # 转换为nm⁻¹
            px_per_nm = 0.05  # 50,000倍率
            avg_curvature_nm = avg_curvature_px / px_per_nm
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

        self.add_step("平均曲率分析", curvature_display,
            f"平均曲率 κ = {avg_curvature_nm:.3f} nm⁻¹。原理：曲率κ = 1/R（R为曲率半径）。"
            f"使用三点法计算：通过三个连续点形成的圆，κ = 4·area/(a·b·c)，其中area是三角形面积，a,b,c是三边长。"
            f"颜色编码：绿色=直（κ < 0.05），黄色=波（0.05 ≤ κ < 0.15），红色=卷曲（κ ≥ 0.15）。"
            f"物理意义：曲率与材料性能直接相关，高曲率降低电导率和力学性能。")

    def get_steps(self):
        """获取所有步骤"""
        return self.steps

    def get_step(self, index):
        """获取特定步骤"""
        if 0 <= index < len(self.steps):
            return self.steps[index]
        return None
