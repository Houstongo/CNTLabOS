"""
CNTSEGNET深度学习分割可视化模块
生成特征提取过程的中间步骤结果
"""
import cv2
import numpy as np
import base64
import time
from io import BytesIO


class CNTSegNetVisualizer:
    """CNTSEGNET深度学习分割可视化器"""

    def __init__(self, magnification=50000, device="cpu",
                 checkpoint_path=None, tile_size=512, overlap=64, seg_threshold=0.5):
        self.mag = magnification
        self.device = device
        self.tile_size = tile_size
        self.overlap = overlap
        self.seg_threshold = seg_threshold
        self.steps = []
        self.model = None
        self.inference_time = 0.0
        self.mean = None
        self.std = None

        # 加载模型
        self._load_model(checkpoint_path)

    def _load_model(self, checkpoint_path):
        """加载CNTSegNet模型"""
        import os
        import sys

        # 导入PyTorch和CNTSegNet
        try:
            import torch
            from cntsegnet import CNTSegNet
        except Exception as e:
            raise RuntimeError(f"Failed to import CNTSegNet/Torch: {e}")

        # 设置默认checkpoint路径
        if checkpoint_path is None:
            VLMSAM_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'VLMSAM'))
            checkpoint_path = os.path.join(VLMSAM_ROOT, 'checkpoints_512_v2', 'best_model.pth')

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"CNTSegNet checkpoint not found: {checkpoint_path}")

        # 加载模型
        model = CNTSegNet(num_classes=1)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        self.model = model.to(self.device)
        self.model.eval()

        # 标准化参数
        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

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
        """生成完整的CNTSegNet分割可视化流程"""
        self.steps = []

        # 阶段1：模型初始化
        self._add_initialization_steps(img_gray)

        # 阶段2：分块推理
        mask, binary_mask, inference_time = self._add_tiling_steps(img_gray)
        self.inference_time = inference_time

        # 阶段3：特征计算
        self._add_feature_computation_steps(binary_mask)

        return self.steps

    def _add_initialization_steps(self, img_gray):
        """添加模型初始化步骤"""

        # 步骤1：原始图像
        self.add_step("原始图像", img_gray,
            "读取的灰度SEM图像，准备进行深度学习分割。")

        # 步骤2：模型配置
        config_display = self._create_config_display(img_gray)
        self.add_step("模型配置", config_display,
            f"CNTSegNet模型已加载。架构：ResNet-50编码器 + ASPP + 解码器。\n"
            f"设备：{self.device} | Tile大小：{self.tile_size}px | 重叠：{self.overlap}px | "
            f"分割阈值：{self.seg_threshold}")

        # 步骤3：ROI提取
        roi = self._extract_roi(img_gray)
        roi_display = self._visualize_roi(img_gray, roi)
        self.add_step("ROI提取", roi_display,
            f"自动提取感兴趣区域（ROI），去除底部标尺和信息栏。\n"
            f"ROI尺寸：{roi['width']}x{roi['height']}像素。")

    def _extract_roi(self, img_gray):
        """提取ROI区域"""
        # 简化的ROI提取：假设底部75像素是信息栏
        height = img_gray.shape[0]
        if height > 75:
            return {
                'y1': 0,
                'y2': height - 75,
                'x1': 0,
                'x2': img_gray.shape[1],
                'width': img_gray.shape[1],
                'height': height - 75
            }
        else:
            return {
                'y1': 0,
                'y2': height,
                'x1': 0,
                'x2': img_gray.shape[1],
                'width': img_gray.shape[1],
                'height': height
            }

    def _visualize_roi(self, img_gray, roi):
        """可视化ROI区域"""
        display = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(display, (roi['x1'], roi['y1']), (roi['x2'], roi['y2']), (0, 255, 0), 2)
        cv2.putText(display, f"ROI: {roi['width']}x{roi['height']}",
                   (roi['x1'], roi['y1'] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return display

    def _create_config_display(self, img):
        """创建配置信息显示图"""
        display = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (400, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)

        texts = [
            f"Model: CNTSegNet-ResNet50",
            f"Device: {self.device}",
            f"Tile: {self.tile_size}px | Overlap: {self.overlap}px",
            f"Threshold: {self.seg_threshold}"
        ]

        for i, text in enumerate(texts):
            cv2.putText(display, text, (20, 30 + i*25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return display

    def _add_tiling_steps(self, img_gray):
        """添加分块推理步骤"""
        roi = self._extract_roi(img_gray)
        roi_img = img_gray[roi['y1']:roi['y2'], roi['x1']:roi['x2']]
        h, w = roi_img.shape[:2]

        # 步骤4：Tile网格规划
        tile_grid_display = self._visualize_tile_grid(roi_img, h, w)
        self.add_step("Tile网格规划", tile_grid_display,
            f"分块推理策略：将{h}x{w}图像分割为{self.tile_size}x{self.tile_size}块。\n"
            f"网格大小：{self._calculate_grid_size(h, w)}，重叠区域{self.overlap}px用于边缘平滑。")

        # 步骤5：分块推理热图
        prob_map, inference_time = self._predict_with_visualization(roi_img)
        heatmap_display = self._visualize_heatmap(roi_img, prob_map)
        self.add_step("分块推理热图", heatmap_display,
            f"深度学习推理完成。推理时间：{inference_time:.2f}秒。\n"
            f"热图显示各像素属于CNT的概率（红色=高概率，蓝色=低概率）。")

        # 步骤6：概率图融合
        binary_mask = (prob_map >= self.seg_threshold).astype(np.uint8) * 255
        mask_display = self._visualize_prob_to_mask(roi_img, prob_map)
        self.add_step("概率图融合", mask_display,
            f"概率阈值化（>{self.seg_threshold}）生成二值分割结果。\n"
            f"白色区域=前景（CNT），黑色区域=背景。")

        return mask_display, binary_mask, inference_time

    def _visualize_tile_grid(self, img, h, w):
        """可视化Tile网格"""
        display = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        tile_size = self.tile_size
        overlap = self.overlap
        stride = max(1, tile_size - overlap)

        # 画出所有tile边界
        for y in range(0, max(h - tile_size + 1, 1), stride):
            for x in range(0, max(w - tile_size + 1, 1), stride):
                color = (0, 255, 255) if (x, y) != (max(w - tile_size, 0), max(h - tile_size, 0)) else (255, 0, 0)
                cv2.rectangle(display, (x, y), (x + tile_size, y + tile_size), color, 1)

        # 添加说明文字
        cv2.putText(display, f"Tile Grid: {self._calculate_grid_size(h, w)} tiles",
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        return display

    def _calculate_grid_size(self, h, w):
        """计算tile网格大小"""
        stride = max(1, self.tile_size - self.overlap)
        ny = len(range(0, max(h - self.tile_size + 1, 1), stride)) + 1
        nx = len(range(0, max(w - self.tile_size + 1, 1), stride)) + 1
        return f"{ny}x{nx}"

    def _predict_with_visualization(self, img):
        """执行分块推理并返回概率图"""
        import torch

        image_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        h, w = image_rgb.shape[:2]
        tile_size = self.tile_size
        overlap = self.overlap
        stride = max(1, tile_size - overlap)

        start_time = time.time()

        # 生成网格
        ys = list(range(0, max(h - tile_size + 1, 1), stride))
        xs = list(range(0, max(w - tile_size + 1, 1), stride))
        if ys[-1] != max(h - tile_size, 0):
            ys.append(max(h - tile_size, 0))
        if xs[-1] != max(w - tile_size, 0):
            xs.append(max(w - tile_size, 0))

        accum = np.zeros((h, w), dtype=np.float32)
        counts = np.zeros((h, w), dtype=np.float32)

        # 分块推理
        with torch.no_grad():
            for y in ys:
                for x in xs:
                    tile = image_rgb[y:y + tile_size, x:x + tile_size]
                    tile_h, tile_w = tile.shape[:2]

                    # Padding
                    if tile_h != tile_size or tile_w != tile_size:
                        padded = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                        padded[:tile_h, :tile_w] = tile
                        tile = padded

                    tensor = torch.from_numpy(tile.transpose(2, 0, 1)).float()
                    tensor = (tensor / 255.0 - self.mean) / self.std
                    tensor = tensor.unsqueeze(0).to(self.device)

                    pred = torch.sigmoid(self.model(tensor)).detach().cpu().numpy()[0, 0]
                    accum[y:y + tile_h, x:x + tile_w] += pred[:tile_h, :tile_w]
                    counts[y:y + tile_h, x:x + tile_w] += 1.0

        prob = accum / np.maximum(counts, 1.0)
        inference_time = time.time() - start_time

        return prob, inference_time

    def _visualize_heatmap(self, img, prob):
        """可视化概率热图"""
        # 概率图归一化到0-255
        prob_norm = (prob * 255).astype(np.uint8)

        # 创建热图
        heatmap = cv2.applyColorMap(prob_norm, cv2.COLORMAP_JET)

        # 叠加到原始图像
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        result = cv2.addWeighted(img_color, 0.5, heatmap, 0.5, 0)

        return result

    def _visualize_prob_to_mask(self, img, prob):
        """可视化概率图到二值mask"""
        mask = (prob >= self.seg_threshold).astype(np.uint8) * 255

        # 叠加显示
        display = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        display[mask > 0] = [200, 200, 255]  # 淡蓝色前景

        return display

    def _add_feature_computation_steps(self, mask):
        """添加特征计算步骤（复用现有逻辑）"""

        # 步骤7：密度计算
        self._add_density_step(mask)

        # 步骤8-13：骨架和高级特征
        self._add_skeleton_and_advanced_features(mask)

    def _add_curvature_step(self, skel):
        """添加骨架追踪曲率分析步骤"""
        from skimage.measure import label
        from scipy.ndimage import convolve

        labeled = label(skel, connectivity=2)
        n_regions = labeled.max()

        curvature_display = cv2.cvtColor(skel, cv2.COLOR_GRAY2BGR)
        all_curvatures = []
        nm_per_pixel = self._get_nm_per_pixel()

        # 按区域大小排序，优先处理大的连通区域
        region_sizes = []
        for rid in range(1, n_regions + 1):
            size = np.sum(labeled == rid)
            if size >= 10:
                region_sizes.append((size, rid))
        region_sizes.sort(reverse=True)

        # 最多处理前20个最大区域
        for _, rid in region_sizes[:20]:
            region_mask = (labeled == rid).astype(np.uint8)
            coords = np.argwhere(region_mask)

            if len(coords) < 10:
                continue

            kernel = np.ones((3, 3), dtype=np.uint8)
            kernel[1, 1] = 0
            neighbor_count = convolve(region_mask, kernel, mode='constant', cval=0)
            endpoints = np.argwhere((region_mask > 0) & (neighbor_count == 1))

            if len(endpoints) >= 2:
                path = self._trace_skeleton(region_mask, endpoints[0], endpoints[1])
                # 增加路径采样点数
                max_points = min(len(path) - 2, 200)
                if len(path) > 5:
                    for i in range(2, max_points):
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

        avg_curvature_nm = np.median(all_curvatures) if all_curvatures else 0.0
        self.add_step("骨架追踪曲率", curvature_display,
            f"平均曲率 κ = {avg_curvature_nm:.3f} nm⁻¹。原理：追踪骨架路径计算曲率κ = dθ/ds。采样了{len(all_curvatures)}个点。")

    def _add_tortuosity_step(self, skel):
        """添加波曲度分析步骤"""
        from scipy.ndimage import convolve

        skel_points = np.argwhere(skel > 0)
        if len(skel_points) < 10:
            self.add_step("波曲度分析", cv2.cvtColor(skel, cv2.COLOR_GRAY2BGR), "骨架点不足，无法计算波曲度。")
            return

        kernel = np.ones((3, 3), dtype=np.uint8)
        kernel[1, 1] = 0
        neighbor_count = convolve((skel > 0).astype(np.uint8), kernel, mode='constant', cval=0)
        endpoints = np.argwhere((skel > 0) & (neighbor_count == 1))

        if len(endpoints) < 2:
            self.add_step("波曲度分析", cv2.cvtColor(skel, cv2.COLOR_GRAY2BGR), "未找到足够端点，无法计算波曲度。")
            return

        path = self._trace_skeleton(skel, endpoints[0], endpoints[1])
        if len(path) < 5:
            self.add_step("波曲度分析", cv2.cvtColor(skel, cv2.COLOR_GRAY2BGR), "骨架路径过短，无法计算波曲度。")
            return

        L_real = sum(np.linalg.norm(path[i] - path[i-1]) for i in range(1, len(path)))
        L_direct = np.linalg.norm(endpoints[0] - endpoints[1])

        tortuosity = L_real / L_direct if L_direct > 1 else 1.0

        tortuosity_display = cv2.cvtColor(skel, cv2.COLOR_GRAY2BGR)
        for i in range(1, len(path)):
            cv2.line(tortuosity_display,
                    (int(path[i-1][1]), int(path[i-1][0])),
                    (int(path[i][1]), int(path[i][0])), (0, 255, 0), 2)
        cv2.line(tortuosity_display,
                (int(endpoints[0][1]), int(endpoints[0][0])),
                (int(endpoints[1][1]), int(endpoints[1][0])), (0, 0, 255), 2)

        self.add_step("波曲度分析", tortuosity_display,
            f"波曲度 τ = {tortuosity:.3f}。原理：τ = L_real / L_direct，骨架真实长度 / 端点直线距离。")

    def _add_density_step(self, mask):
        """添加密度计算步骤"""
        height, width = mask.shape
        foreground_pixels = np.sum(mask > 0)
        total_pixels = height * width
        density = foreground_pixels / total_pixels if total_pixels > 0 else 0.0

        density_display = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        density_display[mask > 0] = [200, 200, 100]

        cv2.putText(density_display, f"Density = {density:.4f}", (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        self.add_step("密度计算", density_display,
            f"密度 density = {density:.4f}。原理：前景像素数 / 总像素数 = {foreground_pixels}/{total_pixels}。")

    def _add_skeleton_and_advanced_features(self, mask):
        """添加骨架和高级特征计算步骤"""

        from skimage.morphology import skeletonize

        # 步骤8：完整骨架
        skel_full = skeletonize(mask > 0).astype(np.uint8) * 255
        skel_display = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        skel_display[skel_full > 0] = [0, 255, 0]
        self.add_step("完整骨架", skel_display, "基于CNTSegNet分割结果的骨架提取。")

        # 步骤9：最大骨架
        max_skel = self._extract_largest_skeleton(skel_full)
        max_skel_display = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        max_skel_display[max_skel > 0] = [0, 255, 0]
        self.add_step("最大骨架", max_skel_display, "提取最长连续骨架作为主干CNT。")

        # 步骤10：对齐方向场
        self._add_alignment_step(mask, max_skel)

        # 步骤11：直径测量
        self._add_diameter_step(mask, max_skel)

        # 步骤12：曲率分析
        self._add_curvature_step(skel_full)

        # 步骤13：波曲度分析
        self._add_tortuosity_step(skel_full)

    def _extract_largest_skeleton(self, skeleton):
        """提取最大骨架"""
        from skimage.measure import label
        from scipy.ndimage import convolve

        labeled = label(skeleton > 0, connectivity=2)
        n_regions = labeled.max()

        if n_regions == 0:
            return skeleton

        largest_region_id = -1
        max_pixel_count = 0

        for rid in range(1, n_regions + 1):
            region_mask = (labeled == rid)
            pixel_count = np.sum(region_mask)
            if pixel_count > max_pixel_count:
                max_pixel_count = pixel_count
                largest_region_id = rid

        if largest_region_id > 0:
            region_mask = (labeled == largest_region_id).astype(np.uint8)

            kernel = np.ones((3, 3), dtype=np.uint8)
            kernel[1, 1] = 0
            neighbor_count = convolve(region_mask, kernel, mode='constant', cval=0)
            endpoints = np.argwhere((region_mask > 0) & (neighbor_count == 1))

            if len(endpoints) >= 2:
                path = self._trace_skeleton(region_mask, endpoints[0], endpoints[1])
                result = np.zeros_like(skeleton)
                if len(path) > 5:
                    for point in path:
                        result[int(point[0]), int(point[1])] = 255
                return result

        return skeleton

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

    def _add_alignment_step(self, mask, skel):
        """添加对齐分析步骤"""
        # 使用结构张量计算方向
        smoothed = cv2.GaussianBlur(mask, (3, 3), 0)
        Ix = cv2.Scharr(smoothed, cv2.CV_64F, 1, 0)
        Iy = cv2.Scharr(smoothed, cv2.CV_64F, 0, 1)

        sigma = 3.0
        ksize = int(sigma * 4) | 1
        Jxx = cv2.GaussianBlur(Ix * Ix, (ksize, ksize), sigma)
        Jxy = cv2.GaussianBlur(Ix * Iy, (ksize, ksize), sigma)
        Jyy = cv2.GaussianBlur(Iy * Iy, (ksize, ksize), sigma)

        # 主方向
        theta = 0.5 * np.arctan2(2.0 * Jxy, Jxx - Jyy)

        # 计算取向度 (HOF方法)
        valid_mask = mask > 0
        if np.sum(valid_mask) > 0:
            angles = theta[valid_mask]
            angle_hist, _ = np.histogram(angles, bins=36, range=(-np.pi/2, np.pi/2))
            angle_hist = angle_hist / np.sum(angle_hist)
            max_bin = np.argmax(angle_hist)
            alignment = angle_hist[max_bin]
        else:
            alignment = 0.0

        # 创建方向场可视化（降采样）
        step = 20
        height, width = mask.shape
        direction_display = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        for y in range(step//2, height, step):
            for x in range(step//2, width, step):
                angle = theta[y, x]
                dx = 15 * np.cos(angle)
                dy = 15 * np.sin(angle)
                cv2.arrowedLine(direction_display, (x, y), (int(x+dx), int(y+dy)), (255, 0, 0), 1)

        self.add_step("对齐方向场", direction_display,
            f"红色箭头显示CNT的主要生长方向。取向度 alignment = {alignment:.3f}。原理：使用结构张量（Structure Tensor）计算局部梯度方向。")

    def _add_diameter_step(self, thresh, skel):
        """添加直径测量步骤"""
        # 距离变换
        dist = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)

        # 在骨架上标注直径
        diameter_display = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

        # 采样一些骨架点并计算直径统计
        skel_points = np.argwhere(skel > 0)
        avg_diameter_nm = 0.0
        std_diameter_nm = 0.0

        if len(skel_points) > 0:
            # 向量化获取所有骨架点的直径值
            diameters_px = dist[skel_points[:, 0], skel_points[:, 1]] * 2
            valid_mask = diameters_px > 0
            diameters_px = diameters_px[valid_mask]

            if len(diameters_px) > 0:
                avg_px = np.mean(diameters_px)
                std_px = np.std(diameters_px)
                median_px = np.median(diameters_px)

                # 转换为纳米 (根据倍率)
                nm_per_pixel = self._get_nm_per_pixel()
                avg_diameter_nm = avg_px * nm_per_pixel
                std_diameter_nm = std_px * nm_per_pixel
                median_diameter_nm = median_px * nm_per_pixel

                # 采样显示
                sample_indices = np.random.choice(len(skel_points), min(50, len(skel_points)), replace=False)
                for idx in sample_indices:
                    y, x = skel_points[idx]
                    radius = dist[y, x]
                    if radius > 0:
                        cv2.circle(diameter_display, (x, y), int(radius), (0, 255, 255), 1)

        self.add_step("直径测量", diameter_display,
            f"黄色圆表示估计的CNT直径（半径）。平均直径 = {avg_diameter_nm:.1f} nm，标准差 = {std_diameter_nm:.1f} nm。原理：基于距离变换，CNT直径D = 2 × 距离值。")

    def _get_nm_per_pixel(self):
        """根据倍率获取每像素纳米数"""
        # 常见倍率下的标尺换算
        scale_map = {
            1000: 200.0,
            5000: 40.0,
            10000: 20.0,
            50000: 4.0,
            100000: 2.0,
        }
        return scale_map.get(self.mag, 4.0)  # 默认50000x

    def get_steps(self):
        """获取所有步骤"""
        return self.steps

    def get_step(self, index):
        """获取特定步骤"""
        if 0 <= index < len(self.steps):
            return self.steps[index]
        return None

    def get_model_info(self):
        """获取模型信息"""
        return "CNTSegNet-ResNet50"

    def get_inference_time(self):
        """获取推理时间"""
        return self.inference_time
