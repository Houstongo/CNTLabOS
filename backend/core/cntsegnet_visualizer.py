"""
CNTSegNet-clDice 深度学习分割可视化模块。
推理步骤 (1-6) 保持不变，后处理步骤 (7-12) 委托给 FeatureVisualizationAdapter。
"""
import cv2
import numpy as np
import base64
import time
import os


class CNTSegNetVisualizer:
    """CNTSegNet-clDice 深度学习分割可视化器（ResNet34UNet, 1ch 灰度）"""

    def __init__(self, magnification=50000, device="cpu",
                 checkpoint_path=None, tile_size=768, overlap=384, seg_threshold=0.7):
        self.mag = magnification
        self.device = device
        self.tile_size = tile_size
        self.overlap = overlap
        self.seg_threshold = seg_threshold
        self.steps = []
        self.model = None
        self.inference_time = 0.0
        self._normalize_mean = 0.5
        self._normalize_std = 0.5

        self._load_model(checkpoint_path)

    def _load_model(self, checkpoint_path):
        """加载 CNTSegNet-clDice (ResNet34UNet) 模型"""
        import sys
        PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))

        try:
            import torch
            from experiments.cnt_paper_repro.config import load_config
            from experiments.cnt_paper_repro.model import ResNet34UNet
        except Exception as e:
            raise RuntimeError(f"Failed to import clDice model dependencies: {e}")

        if checkpoint_path is None:
            checkpoint_path = os.path.join(
                PROJECT_ROOT, 'experiments', 'cnt_paper_repro', 'runs',
                'cnt_paper_repro_100000x_center768_cldice_seed42', 'best_model.pth',
            )
            config_path = os.path.join(
                PROJECT_ROOT, 'experiments', 'cnt_paper_repro', 'runs',
                'cnt_paper_repro_100000x_center768_cldice_seed42', 'config_snapshot.yaml',
            )
        else:
            config_path = checkpoint_path.replace('best_model.pth', 'config_snapshot.yaml')

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"clDice checkpoint not found: {checkpoint_path}")

        cfg = load_config(config_path)
        self._normalize_mean = float(cfg["data"].get("normalize_mean", 0.5))
        self._normalize_std = float(cfg["data"].get("normalize_std", 0.5))
        self.tile_size = int(cfg["data"].get("patch_size", 768))
        self.overlap = self.tile_size // 2
        self.seg_threshold = float(cfg.get("inference", {}).get("threshold", 0.7))

        model = ResNet34UNet(
            in_channels=int(cfg["model"].get("in_channels", 1)),
            num_classes=int(cfg["model"].get("num_classes", 1)),
            encoder_weights=None,
        )

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        self.model = model.to(self.device)
        self.model.eval()

    def add_step(self, name, image, description=""):
        """添加一个步骤"""
        _, buffer = cv2.imencode('.jpg', image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        self.steps.append({
            'name': name,
            'image': img_base64,
            'description': description
        })

    # ── 主流程 ────────────────────────────────────────

    def visualize_extraction(self, img_gray):
        """生成完整的 CNTSegNet 分割可视化流程"""
        self.steps = []

        # 推理步骤 (1-6)
        self._add_initialization_steps(img_gray)
        _, binary_mask, inference_time = self._add_tiling_steps(img_gray)
        self.inference_time = inference_time

        # 后处理步骤 (7-12) — 委托给 FeatureVisualizationAdapter
        self._add_feature_computation_steps(binary_mask)

        return self.steps

    # ── 推理步骤（保持不变）───────────────────────────

    def _add_initialization_steps(self, img_gray):
        self.add_step("原始图像", img_gray,
            "读取的灰度SEM图像，准备进行深度学习分割。")

        config_display = self._create_config_display(img_gray)
        self.add_step("模型配置", config_display,
            f"CNTSegNet-clDice模型已加载。架构：ResNet34-UNet。\n"
            f"设备：{self.device} | Tile大小：{self.tile_size}px | 重叠：{self.overlap}px | "
            f"分割阈值：{self.seg_threshold}")

        roi = self._extract_roi(img_gray)
        roi_display = self._visualize_roi(img_gray, roi)
        self.add_step("ROI提取", roi_display,
            f"自动提取感兴趣区域（ROI），去除底部标尺和信息栏。\n"
            f"ROI尺寸：{roi['width']}x{roi['height']}像素。")

    def _add_tiling_steps(self, img_gray):
        roi = self._extract_roi(img_gray)
        roi_img = img_gray[roi['y1']:roi['y2'], roi['x1']:roi['x2']]
        h, w = roi_img.shape[:2]

        tile_grid_display = self._visualize_tile_grid(roi_img, h, w)
        self.add_step("Tile网格规划", tile_grid_display,
            f"分块推理策略：将{h}x{w}图像分割为{self.tile_size}x{self.tile_size}块。\n"
            f"网格大小：{self._calculate_grid_size(h, w)}，重叠区域{self.overlap}px用于边缘平滑。")

        prob_map, inference_time = self._predict_with_visualization(roi_img)
        heatmap_display = self._visualize_heatmap(roi_img, prob_map)
        self.add_step("分块推理热图", heatmap_display,
            f"深度学习推理完成。推理时间：{inference_time:.2f}秒。\n"
            f"热图显示各像素属于CNT的概率（红色=高概率，蓝色=低概率）。")

        binary_mask = (prob_map >= self.seg_threshold).astype(np.uint8) * 255
        mask_display = self._visualize_prob_to_mask(roi_img, prob_map)
        self.add_step("概率图融合", mask_display,
            f"概率阈值化（>{self.seg_threshold}）生成二值分割结果。\n"
            f"白色区域=前景（CNT），黑色区域=背景。")

        return mask_display, binary_mask, inference_time

    # ── 后处理步骤（委托给 adapter）────────────────────

    def _add_feature_computation_steps(self, binary_mask):
        """后处理步骤委托给 FeatureVisualizationAdapter。"""
        from backend.core.visualization_adapter import FeatureVisualizationAdapter
        adapter = FeatureVisualizationAdapter(magnification=self.mag)
        post_steps = adapter.visualize_post_processing(binary_mask)
        self.steps.extend(post_steps)

    # ── 推理辅助方法（保持不变）────────────────────────

    def _extract_roi(self, img_gray):
        height = img_gray.shape[0]
        if height > 75:
            return {'y1': 0, 'y2': height - 75, 'x1': 0, 'x2': img_gray.shape[1],
                    'width': img_gray.shape[1], 'height': height - 75}
        else:
            return {'y1': 0, 'y2': height, 'x1': 0, 'x2': img_gray.shape[1],
                    'width': img_gray.shape[1], 'height': height}

    def _visualize_roi(self, img_gray, roi):
        display = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(display, (roi['x1'], roi['y1']), (roi['x2'], roi['y2']), (0, 255, 0), 2)
        cv2.putText(display, f"ROI: {roi['width']}x{roi['height']}",
                    (roi['x1'], roi['y1'] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return display

    def _create_config_display(self, img):
        display = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (400, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)
        texts = [
            f"Model: CNTSegNet-clDice (ResNet34UNet)",
            f"Device: {self.device}",
            f"Tile: {self.tile_size}px | Overlap: {self.overlap}px",
            f"Threshold: {self.seg_threshold}"
        ]
        for i, text in enumerate(texts):
            cv2.putText(display, text, (20, 30 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return display

    def _visualize_tile_grid(self, img, h, w):
        display = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        stride = max(1, self.tile_size - self.overlap)
        for y in range(0, max(h - self.tile_size + 1, 1), stride):
            for x in range(0, max(w - self.tile_size + 1, 1), stride):
                color = (0, 255, 255) if (x, y) != (max(w - self.tile_size, 0), max(h - self.tile_size, 0)) else (255, 0, 0)
                cv2.rectangle(display, (x, y), (x + self.tile_size, y + self.tile_size), color, 1)
        cv2.putText(display, f"Tile Grid: {self._calculate_grid_size(h, w)} tiles",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        return display

    def _calculate_grid_size(self, h, w):
        stride = max(1, self.tile_size - self.overlap)
        ny = len(range(0, max(h - self.tile_size + 1, 1), stride)) + 1
        nx = len(range(0, max(w - self.tile_size + 1, 1), stride)) + 1
        return f"{ny}x{nx}"

    def _predict_with_visualization(self, img):
        import torch
        h, w = img.shape[:2]
        tile_size = self.tile_size
        stride = tile_size // 2
        start_time = time.time()

        ys = list(range(0, max(h - tile_size + 1, 1), stride))
        xs = list(range(0, max(w - tile_size + 1, 1), stride))
        if ys and ys[-1] != max(h - tile_size, 0):
            ys.append(max(h - tile_size, 0))
        if xs and xs[-1] != max(w - tile_size, 0):
            xs.append(max(w - tile_size, 0))

        accum = np.zeros((h, w), dtype=np.float32)
        counts = np.zeros((h, w), dtype=np.float32)

        with torch.no_grad():
            for y in ys:
                for x in xs:
                    tile_h = min(tile_size, h - y)
                    tile_w = min(tile_size, w - x)
                    tile = img[y:y + tile_h, x:x + tile_w].astype(np.float32)
                    if tile_h != tile_size or tile_w != tile_size:
                        padded = np.zeros((tile_size, tile_size), dtype=np.float32)
                        padded[:tile_h, :tile_w] = tile
                        tile = padded
                    tensor = torch.from_numpy(
                        ((tile / 255.0 - self._normalize_mean) / max(self._normalize_std, 1e-6))
                    ).unsqueeze(0).unsqueeze(0).to(self.device)
                    pred = torch.sigmoid(self.model(tensor)).detach().cpu().numpy()[0, 0]
                    accum[y:y + tile_h, x:x + tile_w] += pred[:tile_h, :tile_w]
                    counts[y:y + tile_h, x:x + tile_w] += 1.0

        prob = accum / np.maximum(counts, 1.0)
        return prob, time.time() - start_time

    def _visualize_heatmap(self, img, prob):
        prob_norm = (prob * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(prob_norm, cv2.COLORMAP_JET)
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return cv2.addWeighted(img_color, 0.5, heatmap, 0.5, 0)

    def _visualize_prob_to_mask(self, img, prob):
        mask = (prob >= self.seg_threshold).astype(np.uint8) * 255
        display = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        display[mask > 0] = [200, 200, 255]
        return display

    # ── 兼容接口 ──────────────────────────────────────

    def get_steps(self):
        return self.steps

    def get_step(self, index):
        if 0 <= index < len(self.steps):
            return self.steps[index]
        return None

    def get_model_info(self):
        return "CNTSegNet-clDice (ResNet34UNet)"

    def get_inference_time(self):
        return self.inference_time
