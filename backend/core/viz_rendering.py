"""
共享渲染工具 — 算法可视化的图像生成函数。
AlgorithmVisualizer 和 CNTSegNetVisualizer 共用。
"""
import cv2
import numpy as np
import base64


# ── 编码 ──────────────────────────────────────────────

def encode_step(image: np.ndarray, name: str, description: str) -> dict:
    """将 numpy 图像编码为 base64 step dict。"""
    if image is None or image.size == 0:
        # 占位黑图
        image = np.zeros((100, 100), dtype=np.uint8)
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    _, buf = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return {
        'name': name,
        'image': base64.b64encode(buf).decode('utf-8'),
        'description': description,
    }


# ── 骨架叠加 ──────────────────────────────────────────

def overlay_skeleton(base: np.ndarray, skel: np.ndarray,
                    color=(0, 255, 0)) -> np.ndarray:
    """在底图上叠加骨架像素。"""
    if len(base.shape) == 2:
        base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    display = base.copy()
    mask = skel > 0
    if np.any(mask):
        display[mask] = color
    return display


# ── ROI 裁剪可视化 ────────────────────────────────────

def render_roi_diff(img_gray: np.ndarray, roi: np.ndarray) -> np.ndarray:
    """原始图像 vs ROI 裁剪的对比：上半=保留区域，下半(红色)=裁剪区域。"""
    display = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    h_full, w_full = display.shape[:2]
    h_roi = roi.shape[0]
    # 裁剪区域标注为淡红色半透明
    if h_roi < h_full:
        overlay = display[h_roi:].copy()
        overlay[:] = (overlay.astype(float) * 0.3).astype(np.uint8)
        overlay[:] = (overlay + np.array([60, 30, 30], dtype=np.uint8))
        display[h_roi:] = overlay
        cv2.line(display, (0, h_roi), (w_full, h_roi), (0, 0, 255), 2)
        cv2.putText(display, f"ROI: {w_full}x{h_roi} -> {w_full}x{h_full}",
                    (10, h_full - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return display


# ── 密度可视化 ────────────────────────────────────────

def render_density(thresh: np.ndarray, density: float) -> np.ndarray:
    """二值图 + 密度数值标注。"""
    display = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    display[thresh > 0] = [200, 200, 100]
    cv2.putText(display, f"Density = {density:.2f}%", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    return display


# ── 直径测量可视化 ────────────────────────────────────

def render_diameter(thresh: np.ndarray, dist_map: np.ndarray,
                    skel: np.ndarray, diameter_nm: float,
                    nm_per_pixel: float, sample_n: int = 100) -> np.ndarray:
    """距离变换 + 采样圆 + 直径统计。"""
    display = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    skel_points = np.argwhere(skel > 0)
    if len(skel_points) > 0:
        radii = dist_map[skel_points[:, 0], skel_points[:, 1]]
        valid = radii > 0
        if np.any(valid):
            pts = skel_points[valid]
            idx = np.random.choice(len(pts), min(sample_n, len(pts)), replace=False)
            for i in idx:
                y, x = pts[i]
                r = int(dist_map[y, x])
                if r > 0:
                    cv2.circle(display, (x, y), r, (0, 255, 255), 1)

            valid_radii = radii[valid] * 2
            avg = np.mean(valid_radii) * nm_per_pixel
            std = np.std(valid_radii) * nm_per_pixel
            med = np.median(valid_radii) * nm_per_pixel
            cv2.putText(display,
                        f"avg={avg:.1f}nm std={std:.1f}nm med={med:.1f}nm",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return display


# ── 分支清理可视化 ────────────────────────────────────

def render_branch_cleanup(skel_orig: np.ndarray, skel_cleaned: np.ndarray,
                          cleanup_info: dict) -> np.ndarray:
    """绿=保留骨架，红=移除的短组件，橙=修剪的端刺。"""
    display = np.zeros((*skel_orig.shape, 3), dtype=np.uint8)

    orig_mask = skel_orig > 0
    removed_short = cleanup_info.get('removed_short_mask')
    removed_spur = cleanup_info.get('removed_spur_mask')

    # 保留的骨架 — 绿色
    if removed_short is not None and removed_spur is not None:
        kept = orig_mask & ~removed_short.astype(bool) & ~removed_spur.astype(bool)
        display[kept] = [0, 220, 0]
        # 被移除的短组件 — 红色
        short_removed = orig_mask & removed_short.astype(bool)
        display[short_removed] = [0, 0, 220]
        # 被修剪的端刺 — 橙色
        spur_removed = orig_mask & removed_spur.astype(bool)
        display[spur_removed] = [0, 140, 255]
    else:
        display[orig_mask] = [0, 220, 0]

    n_short = cleanup_info.get('removed_short_component_count', 0)
    n_spur = cleanup_info.get('removed_spur_count', 0)
    label = f"short={n_short} spur={n_spur}"
    cv2.putText(display, label, (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    # 图例
    cv2.rectangle(display, (20, 50), (30, 60), (0, 220, 0), -1)
    cv2.putText(display, "kept", (35, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    cv2.rectangle(display, (20, 65), (30, 75), (0, 0, 220), -1)
    cv2.putText(display, "short", (35, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    cv2.rectangle(display, (20, 80), (30, 90), (0, 140, 255), -1)
    cv2.putText(display, "spur", (35, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    return display


# ── HOF 对齐方向场 ───────────────────────────────────

def render_alignment_field(processed: np.ndarray, alignment: float,
                          mean_phi_deg: float, n_branches: int,
                          hof_method: str = "skeleton",
                          rotation_correction_deg: int = 0) -> np.ndarray:
    """结构张量方向场箭头 + HOF 统计。"""
    Ix = cv2.Scharr(processed, cv2.CV_64F, 1, 0)
    Iy = cv2.Scharr(processed, cv2.CV_64F, 0, 1)
    sigma = 3.0
    ksize = int(sigma * 4) | 1
    Jxx = cv2.GaussianBlur(Ix * Ix, (ksize, ksize), sigma)
    Jxy = cv2.GaussianBlur(Ix * Iy, (ksize, ksize), sigma)
    Jyy = cv2.GaussianBlur(Iy * Iy, (ksize, ksize), sigma)
    theta = 0.5 * np.arctan2(2.0 * Jxy, Jxx - Jyy)

    display = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
    step = 20
    h, w = processed.shape
    for y in range(step // 2, h, step):
        for x in range(step // 2, w, step):
            angle = theta[y, x]
            dx = int(15 * np.cos(angle))
            dy = int(15 * np.sin(angle))
            cv2.arrowedLine(display, (x, y), (x + dx, y + dy), (255, 100, 0), 1, tipLength=0.3)

    cv2.putText(display, f"HOF={alignment:.3f} phi={mean_phi_deg:.1f}deg branches={n_branches}",
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(display, f"method={hof_method} rot={rotation_correction_deg}deg",
                (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    return display


# ── V3 曲率可视化 ─────────────────────────────────────

def render_curvature_v3(base_bgr: np.ndarray, skel: np.ndarray,
                        ordered_branches: list,
                        curvature_bundle: dict,
                        nm_per_pixel: float) -> np.ndarray:
    """按曲率值着色骨架 + V3 多指标摘要面板。"""
    display = base_bgr.copy()
    display = overlay_skeleton(display, skel, (120, 255, 160))

    # 曲率分类阈值
    LOW_THRESH = 0.05
    MED_THRESH = 0.15

    for branch in ordered_branches:
        coords = branch['coords']
        if 'curvatures_px' in branch and branch['curvatures_px'] is not None:
            curvs_px = branch['curvatures_px']
            for k, pt in enumerate(coords):
                curv_nm = curvs_px[k] * nm_per_pixel if k < len(curvs_px) else 0
                if curv_nm < LOW_THRESH:
                    color = (0, 200, 0)       # 绿 — 直
                elif curv_nm < MED_THRESH:
                    color = (0, 200, 200)     # 黄 — 波
                else:
                    color = (0, 0, 255)       # 红 — 卷曲
                y, x = int(pt[0]), int(pt[1])
                if 0 <= y < display.shape[0] and 0 <= x < display.shape[1]:
                    cv2.circle(display, (x, y), 2, color, -1)

    # 摘要面板
    curv_v3 = curvature_bundle.get('curvature_v3', 'N/A')
    curv_nm_v3 = curvature_bundle.get('curvature_nm_v3', 0)
    curv_nm_p75_len = curvature_bundle.get('curvature_nm_v3_p75_length', 0)
    curv_nm_p75_sqrt = curvature_bundle.get('curvature_nm_v3_p75_sqrt_length', 0)
    curv_nm_mean = curvature_bundle.get('curvature_nm_v3_mean_length', 0)
    n_branch = curvature_bundle.get('curvature_v3_branch_count', 0)

    panel_h, panel_w = 100, 400
    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    panel[:] = (40, 40, 40)
    lines = [
        f"V3 Curvature: {curv_v3}  kappa={curv_nm_v3:.4f} nm^-1",
        f"p75_length={curv_nm_p75_len:.4f}  p75_sqrt={curv_nm_p75_sqrt:.4f}",
        f"mean_length={curv_nm_mean:.4f}  branches={n_branch}",
        f"green<0.05  yellow<0.15  red>=0.15 nm^-1",
    ]
    for i, line in enumerate(lines):
        cv2.putText(panel, line, (8, 20 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)
    # 放到右下角（仅当图像足够大时）
    dh, dw = display.shape[:2]
    if dh >= panel_h + 10 and dw >= panel_w + 10:
        y0 = dh - panel_h - 10
        x0 = dw - panel_w - 10
        display[y0:y0 + panel_h, x0:x0 + panel_w] = panel
    return display


# ── 波曲度 v2 可视化 ─────────────────────────────────

def render_waviness(base_bgr: np.ndarray, skel: np.ndarray,
                    waviness_v2: dict) -> np.ndarray:
    """波曲度分析结果可视化。"""
    display = base_bgr.copy()
    display = overlay_skeleton(display, skel, (120, 255, 160))

    ratio = waviness_v2.get('waviness_ratio_v2', 0)
    height_nm = waviness_v2.get('waviness_height_nm_v2', 0)
    wavelength_nm = waviness_v2.get('waviness_wavelength_nm_v2', 0)
    n_branch = waviness_v2.get('waviness_branches_v2', 0)
    tort = waviness_v2.get('tortuosity_v2', 1.0)

    panel_h, panel_w = 90, 400
    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    panel[:] = (40, 40, 40)
    lines = [
        f"Waviness v2: ratio={ratio:.4f}  branches={n_branch}",
        f"height={height_nm:.1f}nm  wavelength={wavelength_nm:.1f}nm",
        f"tortuosity={tort:.3f}",
    ]
    for i, line in enumerate(lines):
        cv2.putText(panel, line, (8, 22 + i * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1)
    dh, dw = display.shape[:2]
    if dh >= panel_h + 10 and dw >= panel_w + 10:
        y0 = dh - panel_h - 10
        x0 = dw - panel_w - 10
        display[y0:y0 + panel_h, x0:x0 + panel_w] = panel
    return display
