"""
FeatureVisualizationAdapter — 按与 FeatureExtractor.extract_all 相同的顺序
逐步调用 FeatureExtractor 各方法，捕获中间图像用于可视化。

设计原则：
- 不修改 FeatureExtractor 的返回值或内部逻辑
- 通过重放相同方法调用来保证可视化与实际提取一致
- 纯可视化用途，不影响批量处理等生产路径
"""
import sys
import os
import cv2
import numpy as np

# 确保 src/analysis 在 sys.path 中
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.feature_extractor import FeatureExtractor
from backend.core.viz_rendering import (
    encode_step, overlay_skeleton, render_roi_diff, render_density,
    render_diameter, render_branch_cleanup, render_alignment_field,
    render_curvature_v3, render_waviness,
)


class FeatureVisualizationAdapter:
    """重放 FeatureExtractor 流程，捕获中间图像。"""

    def __init__(self, magnification: int = None,
                 diameter_method: str = "enhanced"):
        self.mag = magnification
        self.extractor = FeatureExtractor(
            magnification=magnification,
            diameter_method=diameter_method,
        )
        self.steps: list = []

    # ── 完整流程（传统阈值路径）─────────────────────────

    def visualize(self, img_gray: np.ndarray,
                  external_binary_mask: np.ndarray = None) -> list:
        """
        完整特征提取流程可视化（12 步）。
        与 FeatureExtractor.extract_all 相同的调用顺序。
        """
        self.steps = []

        # Step 1: 原始图像
        self.steps.append(encode_step(
            img_gray, "原始图像",
            "读取的灰度SEM图像。SEM利用电子束扫描样品表面成像。"))

        # Step 2: ROI 裁剪
        roi = self.extractor.extract_roi(img_gray)
        roi_vis = render_roi_diff(img_gray, roi)
        self.steps.append(encode_step(
            roi_vis, "ROI 裁剪",
            f"自动提取感兴趣区域，去除底部标尺/信息栏。"
            f"裁剪后尺寸：{roi.shape[1]}x{roi.shape[0]}px。"))

        # Step 3: 物理标定
        self.extractor._calibrate(roi.shape[1])
        nm_per_pixel = 1000.0 / max(self.extractor.px_per_um, 1e-9)
        calib_img = roi.copy()
        if len(calib_img.shape) == 2:
            calib_img = cv2.cvtColor(calib_img, cv2.COLOR_GRAY2BGR)
        cv2.putText(calib_img,
                    f"mag={self.mag}x  px/um={self.extractor.px_per_um:.3f}"
                    f"  nm/px={nm_per_pixel:.3f}  tube_px={self.extractor.expected_tube_px:.1f}",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        self.steps.append(encode_step(
            calib_img, "物理标定",
            f"倍率={self.mag}x，px/um={self.extractor.px_per_um:.3f}，"
            f"nm/px={nm_per_pixel:.3f}。预估管径={self.extractor.expected_tube_px:.1f}px。"))

        # Step 4: 自适应 CLAHE 预处理
        processed = self.extractor.preprocess(roi)
        clahe_vis = processed.copy()
        if len(clahe_vis.shape) == 2:
            clahe_vis = cv2.cvtColor(clahe_vis, cv2.COLOR_GRAY2BGR)
        tile = max(4, int(32 / max(1, self.extractor.expected_tube_px)))
        tile = min(tile, 16)
        ksize = max(3, int(self.extractor.expected_tube_px * 0.8) | 1)
        cv2.putText(clahe_vis,
                    f"CLAHE tile={tile}x{tile}  Gaussian ksize={ksize}x{ksize}",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        self.steps.append(encode_step(
            clahe_vis, "自适应 CLAHE 增强",
            f"CLAHE tileGridSize=({tile},{tile})自适应管径。"
            f"高斯模糊ksize=({ksize},{ksize})。"))

        # Step 5: 自适应阈值分割
        if external_binary_mask is not None:
            mask = np.asarray(external_binary_mask)
            if mask.shape != roi.shape:
                mask = cv2.resize(mask.astype(np.uint8),
                                  (roi.shape[1], roi.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
            thresh = (mask > 0).astype(np.uint8) * 255
            density = float(np.count_nonzero(thresh) / max(thresh.size, 1) * 100.0)
        else:
            density, thresh = self.extractor.calculate_density(processed)

        block = max(11, int(self.extractor.expected_tube_px * 4) | 1)
        block = min(block, 51)
        thresh_vis = thresh.copy()
        if len(thresh_vis.shape) == 2:
            thresh_vis = cv2.cvtColor(thresh_vis, cv2.COLOR_GRAY2BGR)
        cv2.putText(thresh_vis,
                    f"Adaptive Gaussian block={block}  density={density:.2f}%",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        self.steps.append(encode_step(
            thresh_vis, "自适应阈值分割",
            f"自适应高斯阈值（blockSize={block}自适应）。密度={density:.2f}%。"))

        # Step 6: 密度计算
        self.steps.append(encode_step(
            render_density(thresh, density), "密度计算",
            f"密度 density = {density:.2f}%。前景像素占比。"))

        # Step 7: 骨架提取 + 直径
        if self.extractor.diameter_method == 'enhanced':
            diameter_nm, skel = self.extractor.calculate_diameter_enhanced(thresh)
            method_name = "分水岭分割"
        else:
            diameter_nm, skel = self.extractor.calculate_diameter(thresh)
            method_name = "距离变换"

        # skel 是 boolean array → uint8
        skel_uint8 = (skel.astype(np.uint8)) * 255
        processed_bgr = processed if len(processed.shape) == 3 else cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
        skel_vis = overlay_skeleton(processed_bgr, skel_uint8)
        self.steps.append(encode_step(
            skel_vis, "骨架提取",
            f"形态学闭运算 + Zhang-Suen精确骨架化。直径={diameter_nm:.1f}nm（{method_name}）。"))

        # Step 8: 分支清理
        branch_cleanup = self.extractor._clean_branch_skeleton(skel)
        cleaned_skel_mask = branch_cleanup["cleaned_skeleton"]
        cleaned_skel_uint8 = (cleaned_skel_mask.astype(np.uint8)) * 255
        cleanup_vis = render_branch_cleanup(skel_uint8, cleaned_skel_uint8, branch_cleanup)
        self.steps.append(encode_step(
            cleanup_vis, "分支清理",
            f"移除{branch_cleanup['removed_short_component_count']}个短孤立组件，"
            f"修剪{branch_cleanup['removed_spur_count']}个端刺。"))

        # Step 9: HOF 对齐
        base_components = self.extractor._collect_components(skel)
        alignment_metrics = self.extractor.calculate_hof_skeleton_adaptive(
            skel, processed=processed, base_components=base_components,
        )
        hof = alignment_metrics["alignment"]
        mean_phi = alignment_metrics["mean_phi_deg"]
        n_br = alignment_metrics["n_branches"]
        hof_method = alignment_metrics.get("hof_method", "skeleton")
        rot_deg = alignment_metrics.get("rotation_correction_deg", 0)
        align_vis = render_alignment_field(processed, hof, mean_phi, n_br, hof_method, rot_deg)
        self.steps.append(encode_step(
            align_vis, "HOF 对齐分析",
            f"HOF={hof:.3f}  phi={mean_phi:.1f}deg  branches={n_br}  "
            f"method={hof_method}  rot={rot_deg}deg。"))

        # Step 10: 直径测量可视化
        dist = cv2.distanceTransform(
            cv2.morphologyEx(thresh, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))),
            cv2.DIST_L2, 5)
        diam_vis = render_diameter(thresh, dist, skel_uint8, diameter_nm, nm_per_pixel)
        self.steps.append(encode_step(
            diam_vis, "直径测量",
            f"平均直径={diameter_nm:.1f}nm（{method_name}）。"
            f"黄色圆=距离变换估计的管径截面。"))

        # Step 11: V3 曲率
        ordered_branches = self.extractor._prepare_curvature_v3_branches(
            cleaned_skel_uint8, apply_branch_cleanup=False)
        curvature_bundle = self.extractor.calculate_curvature_v3_bundle(
            cleaned_skel_uint8, ordered_branches=ordered_branches)
        curv_vis = render_curvature_v3(processed_bgr, cleaned_skel_uint8,
                                       ordered_branches, curvature_bundle, nm_per_pixel)
        curv_label = curvature_bundle.get('curvature_v3', 'Unknown')
        self.steps.append(encode_step(
            curv_vis, "V3 曲率分析",
            f"V3曲率={curv_label}  kappa={curvature_bundle.get('curvature_nm_v3', 0) * 1000.0:.3f} um^-1  "
            f"branches={curvature_bundle.get('curvature_v3_branch_count', 0)}。"
            f"颜色：绿=直(κ<0.8) 黄=波(0.8≤κ<4.0) 红=卷曲(κ≥4.0)。"))

        # Step 12: 波曲度 v2
        waviness_v2 = self.extractor.calculate_waviness_v2(
            cleaned_skel_uint8, ordered_branches=ordered_branches)
        wave_vis = render_waviness(processed_bgr, cleaned_skel_uint8, waviness_v2)
        self.steps.append(encode_step(
            wave_vis, "波曲度 v2 分析",
            f"ratio={waviness_v2.get('waviness_ratio_v2', 0):.4f}  "
            f"height={waviness_v2.get('waviness_height_nm_v2', 0):.1f}nm  "
            f"wavelength={waviness_v2.get('waviness_wavelength_nm_v2', 0):.1f}nm  "
            f"tortuosity={waviness_v2.get('tortuosity_v2', 1):.3f}  "
            f"branches={waviness_v2.get('waviness_branches_v2', 0)}。"))

        return self.steps

    # ── 仅后处理（给 CNTSegNet 用）──────────────────────

    def visualize_post_processing(self, binary_mask: np.ndarray,
                                  processed: np.ndarray = None,
                                  roi: np.ndarray = None) -> list:
        """
        仅骨架→特征部分（steps 6-12），给 CNTSegNet 推理后拼接用。
        binary_mask: CNTSegNet 输出的二值分割图 (0/255 uint8)
        processed: 可选，用于对齐可视化（如无则从 mask 构造灰度图）
        """
        post_steps = []

        # 确保 mask 是 uint8
        if binary_mask.dtype != np.uint8:
            thresh = (binary_mask > 0).astype(np.uint8) * 255
        else:
            thresh = binary_mask

        if processed is None:
            processed = thresh.copy()

        density = float(np.count_nonzero(thresh) / max(thresh.size, 1) * 100.0)

        # 密度
        post_steps.append(encode_step(
            render_density(thresh, density), "密度计算",
            f"密度 density = {density:.2f}%（基于CNTSegNet分割）。"))

        # 标定
        h, w = thresh.shape
        self.extractor._calibrate(w)
        nm_per_pixel = 1000.0 / max(self.extractor.px_per_um, 1e-9)

        # 骨架 + 直径
        if self.extractor.diameter_method == 'enhanced':
            diameter_nm, skel = self.extractor.calculate_diameter_enhanced(thresh)
            method_name = "分水岭分割"
        else:
            diameter_nm, skel = self.extractor.calculate_diameter(thresh)
            method_name = "距离变换"

        skel_uint8 = (skel.astype(np.uint8)) * 255
        processed_bgr = processed if len(processed.shape) == 3 else cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)

        post_steps.append(encode_step(
            overlay_skeleton(processed_bgr, skel_uint8), "骨架提取",
            f"基于CNTSegNet分割结果的骨架化。直径={diameter_nm:.1f}nm（{method_name}）。"))

        # 分支清理
        branch_cleanup = self.extractor._clean_branch_skeleton(skel)
        cleaned_skel_mask = branch_cleanup["cleaned_skeleton"]
        cleaned_skel_uint8 = (cleaned_skel_mask.astype(np.uint8)) * 255
        post_steps.append(encode_step(
            render_branch_cleanup(skel_uint8, cleaned_skel_uint8, branch_cleanup),
            "分支清理",
            f"移除{branch_cleanup['removed_short_component_count']}个短组件，"
            f"修剪{branch_cleanup['removed_spur_count']}个端刺。"))

        # HOF 对齐
        base_components = self.extractor._collect_components(skel)
        alignment_metrics = self.extractor.calculate_hof_skeleton_adaptive(
            skel, processed=processed, base_components=base_components)
        hof = alignment_metrics["alignment"]
        mean_phi = alignment_metrics["mean_phi_deg"]
        n_br = alignment_metrics["n_branches"]
        hof_method = alignment_metrics.get("hof_method", "skeleton")
        rot_deg = alignment_metrics.get("rotation_correction_deg", 0)
        post_steps.append(encode_step(
            render_alignment_field(processed, hof, mean_phi, n_br, hof_method, rot_deg),
            "HOF 对齐分析",
            f"HOF={hof:.3f}  phi={mean_phi:.1f}deg  branches={n_br}  "
            f"method={hof_method}  rot={rot_deg}deg。"))

        # 直径
        dist = cv2.distanceTransform(
            cv2.morphologyEx(thresh, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))),
            cv2.DIST_L2, 5)
        post_steps.append(encode_step(
            render_diameter(thresh, dist, skel_uint8, diameter_nm, nm_per_pixel),
            "直径测量",
            f"平均直径={diameter_nm:.1f}nm（{method_name}）。"))

        # V3 曲率
        ordered_branches = self.extractor._prepare_curvature_v3_branches(
            cleaned_skel_uint8, apply_branch_cleanup=False)
        curvature_bundle = self.extractor.calculate_curvature_v3_bundle(
            cleaned_skel_uint8, ordered_branches=ordered_branches)
        curv_vis = render_curvature_v3(processed_bgr, cleaned_skel_uint8,
                                       ordered_branches, curvature_bundle, nm_per_pixel)
        curv_label = curvature_bundle.get('curvature_v3', 'Unknown')
        post_steps.append(encode_step(
            curv_vis, "V3 曲率分析",
            f"V3曲率={curv_label}  kappa={curvature_bundle.get('curvature_nm_v3', 0):.4f} nm^-1  "
            f"branches={curvature_bundle.get('curvature_v3_branch_count', 0)}。"))

        # 波曲度 v2
        waviness_v2 = self.extractor.calculate_waviness_v2(
            cleaned_skel_uint8, ordered_branches=ordered_branches)
        wave_vis = render_waviness(processed_bgr, cleaned_skel_uint8, waviness_v2)
        post_steps.append(encode_step(
            wave_vis, "波曲度 v2 分析",
            f"ratio={waviness_v2.get('waviness_ratio_v2', 0):.4f}  "
            f"height={waviness_v2.get('waviness_height_nm_v2', 0):.1f}nm  "
            f"wavelength={waviness_v2.get('waviness_wavelength_nm_v2', 0):.1f}nm  "
            f"tortuosity={waviness_v2.get('tortuosity_v2', 1):.3f}  "
            f"branches={waviness_v2.get('waviness_branches_v2', 0)}。"))

        return post_steps
