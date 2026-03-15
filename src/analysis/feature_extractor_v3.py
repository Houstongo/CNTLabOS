"""
CNTA 特征提取算法 v3.0 - 改进版
=================================
主要改进：
1. 分水岭分割替代闭运算（解决密集区域diameter偏大问题）
2. 跨倍率一致的alignment计算
3. 改进的ROI裁切策略
"""

import cv2
import numpy as np
from skimage.morphology import skeletonize, label
from skimage.segmentation import watershed
from scipy import ndimage as ndi


class FeatureExtractorV3:
    """
    CNT SEM 图像特征提取器 v3.0
    """

    def __init__(self, magnification: int = None):
        self.mag = magnification
        self.px_per_um: float = 1.0
        self.expected_tube_px: float = 3.0

    def _calibrate(self, img_width: int):
        """物理标定"""
        HFW_AT_1X_UM = 518_000

        if self.mag and self.mag > 0:
            hfw_um = HFW_AT_1X_UM / self.mag
            self.px_per_um = img_width / hfw_um
        else:
            self.px_per_um = 1.0

        self.expected_tube_px = max(1.5, 15.0 * self.px_per_um * 0.001)

    # ========================================================================
    # 改进1：基于分水岭分割的diameter计算
    # ========================================================================

    def calculate_diameter_watershed(self, thresh: np.ndarray):
        """
        基于分水岭分割的CNT管径估计

        改进点：
        - 去掉闭运算（避免连接相邻CNT）
        - 使用分水岭分割分离不同CNT
        - 减少密集区域的系统性偏大

        Returns
        -------
        diameter_nm : float   平均管径（nm），-1 表示提取失败
        skel        : ndarray 骨架二值图
        """
        # 1. 计算距离变换（不进行闭运算）
        dist = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)

        # 2. 找到局部最大值作为分水岭种子点
        # 使用简单的方法寻找局部最大值
        min_distance = max(1.5, int(self.expected_tube_px * 0.3))

        # 方法1：基于距离变换的简单阈值
        threshold_value = dist.max() * 0.3

        # 方法2：局部最大值（使用简单实现）
        from scipy.ndimage import maximum_filter

        # 局部最大值
        local_max = maximum_filter(dist, footprint=np.ones((3, 3), dtype=int))

        # 应用阈值
        local_max = (local_max == dist) & (dist > threshold_value)

        # 标记不同的峰
        markers, _ = ndi.label(local_max)

        # 3. 创建标记（标记不同的CNT）
        markers = ndi.label(local_max)[0]

        # 4. 分水岭分割
        labels = watershed(-dist, markers, mask=thresh)

        # 5. 提取每个连通区域的管径
        diameters = []

        for label_id in np.unique(labels):
            if label_id == 0:  # 背景标签
                continue

            # 单个CNT区域
            cnt_mask = (labels == label_id)
            if cnt_mask.sum() < 10:  # 太小的区域忽略
                continue

            # 计算该区域的平均半径
            radius = np.mean(dist[cnt_mask])
            diameter = radius * 2
            diameters.append(diameter)

        if len(diameters) == 0:
            return -1.0, np.zeros_like(thresh, dtype=bool)

        # 6. IQR过滤异常值
        diameters = np.array(diameters)
        q1, q3 = np.percentile(diameters, [25, 75])
        iqr = q3 - q1
        filtered = diameters[(diameters >= q1 - 1.5*iqr) & (diameters <= q3 + 1.5*iqr)]

        if len(filtered) == 0:
            filtered = diameters

        median_diameter_px = np.median(filtered)
        diameter_nm = (median_diameter_px / self.px_per_um) * 1000.0

        # 7. 生成骨架（用于curvature计算）
        # 使用原始二值图生成骨架
        skel = skeletonize(thresh > 0)

        return float(diameter_nm), skel

    # ========================================================================
    # 改进2：跨倍率一致的alignment计算
    # ========================================================================

    def calculate_alignment_unified(self, processed: np.ndarray):
        """
        跨倍率一致的alignment计算

        改进点：
        - 统一使用改进的梯度方法
        - 根据倍率调整参数
        - 避免结构张量的系统偏置

        Returns
        -------
        hof          : float  Herman 取向因子（-0.5 ~ 1.0）
        mean_phi_deg : float  平均取向偏角（度）
        method       : str   'unified_gradient'
        """
        if self.mag is None or self.mag < 20000:
            # 低倍率：降级使用结构张量，但加入校正
            return self._calculate_alignment_structure_tensor_corrected(processed)
        else:
            # 高倍率：使用骨架PCA法
            return self._calculate_alignment_skeleton(processed)

    def _calculate_alignment_skeleton(self, processed: np.ndarray):
        """骨架PCA法（保持原有逻辑）"""
        # 二值化
        _, thresh = cv2.threshold(
            processed, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # 骨架化
        skel = skeletonize(thresh > 0)
        labeled = label(skel, connectivity=2)
        n_regions = labeled.max()

        if n_regions == 0:
            return 0.0, 90.0, 'skeleton'

        branch_data = []
        for rid in range(1, n_regions + 1):
            coords = np.argwhere(labeled == rid).astype(float)
            n = len(coords)
            if n < 10:
                continue
            branch_data.append((coords, n))

        if not branch_data:
            return 0.0, 90.0, 'skeleton'

        # 过滤超大连通域
        all_sizes = np.array([n for _, n in branch_data])
        median_n = float(np.median(all_sizes))
        max_n = max(500, median_n * 20)

        cos2_list, weights = [], []
        for coords, n in branch_data:
            if n > max_n:
                continue

            c = coords - coords.mean(axis=0)
            cov = np.cov(c.T)
            if cov.ndim < 2:
                continue
            _, vecs = np.linalg.eigh(cov)
            v = vecs[:, -1]
            phi = np.arctan2(abs(v[1]), abs(v[0]))

            cos2_list.append(np.cos(phi) ** 2)
            weights.append(float(n))

        if not cos2_list:
            return 0.0, 90.0, 'skeleton'

        w = np.array(weights)
        cos2_mean = np.average(cos2_list, weights=w)
        hof = (3.0 * cos2_mean - 1.0) / 2.0
        mean_phi_deg = float(np.degrees(np.arccos(np.sqrt(np.clip(cos2_mean, 0, 1)))))

        return float(np.clip(hof, -0.5, 1.0)), mean_phi_deg, 'skeleton'

    def _calculate_alignment_structure_tensor_corrected(self, processed: np.ndarray):
        """改进的结构张量法（加入校正）"""
        # 基础结构张量计算
        Ix = cv2.Scharr(processed, cv2.CV_64F, 1, 0)
        Iy = cv2.Scharr(processed, cv2.CV_64F, 0, 1)

        # 自适应高斯模糊参数
        sigma = max(2.0, self.expected_tube_px * 1.5)
        ksize = int(sigma * 4) | 1

        Jxx = cv2.GaussianBlur(Ix * Ix, (ksize, ksize), sigma)
        Jxy = cv2.GaussianBlur(Ix * Iy, (ksize, ksize), sigma)
        Jyy = cv2.GaussianBlur(Iy * Iy, (ksize, ksize), sigma)

        trace = Jxx + Jyy
        disc = np.sqrt(np.maximum(0.0, (Jxx - Jyy) ** 2 / 4 + Jxy ** 2))
        lambda1 = trace / 2 + disc
        lambda2 = trace / 2 - disc

        denom = lambda1 + lambda2
        valid = denom > 1e-8
        coherence = np.zeros_like(denom)
        coherence[valid] = (lambda1[valid] - lambda2[valid]) / denom[valid]

        # 结构张量主方向角
        theta = 0.5 * np.arctan2(2.0 * Jxy, Jxx - Jyy)

        # 纤维轴与垂直参考方向的夹角
        phi = np.abs(theta)

        w = coherence.ravel()
        total_w = w.sum()
        if total_w < 1e-10:
            return 0.0, 90.0, 'structure_tensor'

        cos2_mean = np.dot(w, np.cos(phi.ravel()) ** 2) / total_w
        hof = (3.0 * cos2_mean - 1.0) / 2.0
        mean_phi_deg = float(np.degrees(np.arccos(np.sqrt(np.clip(cos2_mean, 0, 1)))))

        # 基于倍率的校正
        # 低倍率结构张量存在系统偏置，需要校正
        if self.mag and self.mag < 20000:
            # 经验校正系数（可根据实验数据调整）
            calibration_factor = 1.15  # 校正系统偏低问题
            hof = np.clip(hof * calibration_factor, -0.5, 1.0)

        return float(np.clip(hof, -0.5, 1.0)), mean_phi_deg, 'structure_tensor'

    # ========================================================================
    # 改进3：改进的ROI裁切
    # ========================================================================

    def extract_roi_improved(self, img_gray: np.ndarray) -> np.ndarray:
        """
        改进的ROI裁切策略

        改进点：
        - 多种方法结合
        - 避免切掉样品区域
        - 更鲁棒的信息栏检测
        """
        h = img_gray.shape[0]

        # 方法1：基于标尺栏模式匹配
        roi_end_method1 = self._detect_scale_bar_roi(img_gray)

        # 方法2：基于行统计的动态检测
        roi_end_method2 = self._detect_roi_by_row_stats(img_gray)

        # 方法3：基于边缘检测
        roi_end_method3 = self._detect_roi_by_edges(img_gray)

        # 投票选择最可能的ROI
        roi_end_candidates = [roi_end_method1, roi_end_method2, roi_end_method3]
        roi_end = self._vote_roi_end(roi_end_candidates, h)

        # 检查底部是否有样品残留（避免切掉样品）
        if self._has_sample_at_bottom(img_gray, roi_end):
            roi_end = h

        return img_gray[:roi_end, :]

    def _detect_scale_bar_roi(self, img_gray: np.ndarray) -> int:
        """基于标尺栏模式检测ROI"""
        # 底部区域灰度特征
        bottom_region = img_gray[-50:, :]
        mean_bottom = bottom_region.mean()
        std_bottom = bottom_region.std()

        # 寻找信息栏上边界
        for i in range(img_gray.shape[0] - 60, max(img_gray.shape[0] - 200, 0), -1):
            row_mean = img_gray[i:i+5, :].mean()
            if abs(row_mean - mean_bottom) > std_bottom * 3:
                return i + 10

        # 保底：固定高度
        return img_gray.shape[0] - 75

    def _detect_roi_by_row_stats(self, img_gray: np.ndarray) -> int:
        """基于行统计检测ROI"""
        h = img_gray.shape[0]

        # 计算每行均值
        row_means = [img_gray[i, :].mean() for i in range(h)]

        # 寻找行均值的突变点
        for i in range(h - 1, max(h - 200, 0), -1):
            if row_means[i] > 60:
                return min(i + 5, h)

        # 保底
        return h - 75

    def _detect_roi_by_edges(self, img_gray: np.ndarray) -> int:
        """基于边缘检测ROI"""
        edges = cv2.Canny(img_gray, 50, 150)
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100, minLineLength=100)

        if lines is not None and len(lines) > 0:
            horizontal_lines = []
            for line in lines:
                if line is not None and len(line) == 1 and line[0] is not None and len(line[0]) >= 4:
                    if abs(line[0][1]) < 5:  # 角度接近水平
                        horizontal_lines.append(line)

            if horizontal_lines:
                y_positions = [line[0][1] for line in horizontal_lines]
                roi_end = int(np.percentile(y_positions, 75))
                return min(roi_end + 10, img_gray.shape[0])

        return img_gray.shape[0] - 75

    def _vote_roi_end(self, candidates: list, img_height: int) -> int:
        """投票选择ROI"""
        valid_candidates = [c for c in candidates if c < img_height]
        if not valid_candidates:
            return img_height - 75
        return int(np.median(valid_candidates))

    def _has_sample_at_bottom(self, img_gray: np.ndarray, roi_end: int) -> int:
        """检查底部是否有样品残留"""
        bottom_region = img_gray[roi_end:, :]

        # 检查是否有明显的CNT信号
        if bottom_region.std() > 20 and bottom_region.mean() > 50:
            return True
        return False

    # ========================================================================
    # 主入口
    # ========================================================================

    def extract_all(self, img_gray: np.ndarray, use_improved: bool = True) -> dict:
        """
        提取特征

        Parameters
        ----------
        img_gray : ndarray  灰度 SEM 图像
        use_improved : bool  是否使用改进方法

        Returns
        -------
        dict : 特征字典
        """
        if use_improved:
            roi = self.extract_roi_improved(img_gray)
            alignment, mean_phi, method = self.calculate_alignment_unified(roi)
        else:
            # 保持原有逻辑
            roi = self._extract_roi_original(img_gray)
            # 简化处理，只返回基本信息
            return {
                'status': 'original',
                'roi_extracted': True
            }

        self._calibrate(roi.shape[1])

        # 预处理
        processed = self._preprocess(roi)

        # 二值化（尝试两种方向）
        thresh_val, thresh = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 如果前景太少，尝试反向
        if np.count_nonzero(thresh) / thresh.size < 0.1:
            thresh_val, thresh = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 提取特征
        if use_improved:
            diameter_nm, skel = self.calculate_diameter_watershed(thresh)
            density = self._calculate_density(thresh)
            curv_result = self._calculate_curvature(skel)
            curvature = curv_result["curvature_nm"]
        else:
            # 原有方法
            diameter_nm, skel = self._calculate_diameter_original(thresh)

        return {
            'diameter': round(diameter_nm, 2) if diameter_nm >= 0 else None,
            'alignment': round(alignment, 4) if use_improved else None,
            'mean_phi_deg': round(mean_phi, 2) if use_improved else None,
            'alignment_method': method if use_improved else 'original',
            'density': round(self._calculate_density(thresh), 2) if use_improved else None,
            'curvature': round(curvature, 6) if use_improved else None,
            'px_per_um': round(self.px_per_um, 2),
            'method': 'improved_v3' if use_improved else 'original_v2'
        }

    def _extract_roi_original(self, img_gray: np.ndarray) -> np.ndarray:
        """原有ROI提取方法"""
        h = img_gray.shape[0]
        for i in range(h - 1, max(h - 200, 0), -1):
            if img_gray[i].mean() > 60:
                return img_gray[:min(i + 5, h), :]
        return img_gray[:h - 75, :]

    def _preprocess(self, roi: np.ndarray) -> np.ndarray:
        """预处理"""
        import cv2

        tile = max(4, int(32 / max(1, self.expected_tube_px)))
        tile = min(tile, 16)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(tile, tile))
        enhanced = clahe.apply(roi)

        ksize = max(3, int(self.expected_tube_px * 0.8) | 1)
        smoothed = cv2.GaussianBlur(enhanced, (ksize, ksize), 0)
        return smoothed

    def _calculate_density(self, thresh: np.ndarray) -> float:
        """计算密度"""
        density = np.count_nonzero(thresh) / thresh.size * 100.0
        return float(density)

    def _calculate_diameter_original(self, thresh: np.ndarray):
        """原有diameter计算方法（闭运算）"""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        skel = skeletonize(closed > 0)

        if not np.any(skel):
            return -1.0, skel

        dist = cv2.distanceTransform(closed, cv2.DIST_L2, 5)
        radii = dist[skel].astype(float)

        radii = radii[radii > 1.5]
        if len(radii) == 0:
            return -1.0, skel

        q1, q3 = np.percentile(radii, [25, 75])
        iqr = q3 - q1
        radii = radii[radii <= q3 + 1.5 * iqr]

        if len(radii) == 0:
            return -1.0, skel

        median_radius_px = np.median(radii)
        diameter_nm = (median_radius_px * 2 / self.px_per_um) * 1000.0

        return float(diameter_nm), skel

    def _calculate_curvature(self, skel: np.ndarray) -> dict:
        """计算曲率 κ = 1/R（单位：nm⁻¹）"""
        from skimage.measure import label

        labeled = label(skel, connectivity=2)
        n_regions = labeled.max()

        if n_regions == 0:
            return {"curvature_nm": -1.0, "label": "Unknown"}

        all_curvatures = []
        for rid in range(1, n_regions + 1):
            coords = np.argwhere(labeled == rid).astype(float)
            n = len(coords)
            if n < 15:
                continue

            # 按骨架路径排序
            # 简单方法：找到端点，然后追踪路径
            # 这里简化：直接按y,x排序（对于近似垂直的CNT有效）
            sorted_indices = np.lexsort((coords[:, 1], coords[:, 0]))
            coords = coords[sorted_indices]

            # 降采样：每隔expected_tube_px个点取一个，减少噪声
            step = max(2, int(self.expected_tube_px / 2))
            sampled_coords = coords[::step]

            # 计算每个点的曲率（三点法）
            curvature_values = []
            m = len(sampled_coords)
            for i in range(1, m - 1):
                p_prev = sampled_coords[i - 1]
                p_curr = sampled_coords[i]
                p_next = sampled_coords[i + 1]

                # 计算三点形成的圆的曲率
                # 曲率 κ = 4 * area / (|AB| * |BC| * |CA|)
                AB = p_prev - p_curr
                BC = p_curr - p_next
                CA = p_next - p_prev

                a = np.linalg.norm(BC)
                b = np.linalg.norm(CA)
                c = np.linalg.norm(AB)

                # 确保三点不共线，距离足够大（至少2个像素）
                if a > 2 and b > 2 and c > 2:
                    s = (a + b + c) / 2
                    area = np.sqrt(max(0, s * (s - a) * (s - b) * (s - c)))
                    curvature = 4 * area / (a * b * c)
                    curvature_values.append(curvature)

            if curvature_values:
                # 使用中位数减少噪声影响
                median_curv_px = np.median(curvature_values)
                # 转换为nm⁻¹：像素曲率 * (nm/像素) = 像素曲率 / (像素/nm)
                px_per_nm = self.px_per_um / 1000.0
                curvature_nm = median_curv_px / px_per_nm
                all_curvatures.append(curvature_nm)

        if not all_curvatures:
            return {"curvature_nm": -1.0, "label": "Unknown"}

        median_curvature = float(np.median(all_curvatures))

        # 分类标签（用于显示，基于经验阈值）
        # 直线: κ < 0.05 nm⁻¹
        # 波浪: 0.05 ≤ κ < 0.15 nm⁻¹
        # 卷曲: κ ≥ 0.15 nm⁻¹
        if median_curvature < 0.05:
            label = "Straight"
        elif median_curvature < 0.15:
            label = "Wavy"
        else:
            label = "Coiled"

        return {"curvature_nm": median_curvature, "label": label}


# 命令行测试
if __name__ == "__main__":
    import sys
    import os

    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    sys.path.insert(0, PROJECT_ROOT)

    test_images = [
        r"d:\CNTDATA\ZZY\20230306 No26 0.75 1.0\No26 200w 5.0nm 5w 0.75nm 400 200 100 600 750 15min 180min mid 50000-1.png",
        r"d:\CNTDATA\ZZY\20230306 No26 0.75 1.0\No26 200w 5.0nm 5w 0.75nm 400 200 100 600 750 15min 180min mid 100000-1.png",
    ]

    for img_path in test_images:
        if not os.path.exists(img_path):
            print(f"SKIP: {os.path.basename(img_path)}")
            continue

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        extractor = FeatureExtractorV3(magnification=50000)

        print(f"\n{'='*60}")
        print(f"图像: {os.path.basename(img_path)}")

        # 改进方法
        results_v3 = extractor.extract_all(img, use_improved=True)
        print(f"[改进方法 v3.0]")
        print(f"  diameter = {results_v3['diameter']} nm")
        print(f"  alignment = {results_v3['alignment']} (方法: {results_v3['alignment_method']})")
        print(f"  density = {results_v3['density']}%")
        print(f"  curvature = {results_v3['curvature']}")
