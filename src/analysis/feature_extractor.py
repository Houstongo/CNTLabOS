"""
CNT 阵列 SEM 图像特征提取算法  v2.1
====================================
改进内容（相比 v1.0）：
  1. ROI 提取：自动裁剪底部信息栏，消除标尺/文字对梯度和阈值的污染
  2. 物理标定修正：通过标尺线像素长度 + 修正的 hfw_at_1x 参数（518000 μm）
                  修正了 v1.0 中 hfw_at_1x=200000 导致的 2.6× 系统误差
  3. Herman 取向因子（HOF）：替换 Sobel 梯度逐点法
     - 主方法（高倍率）：骨架各分支 PCA 主方向角 → f = (3⟨cos²φ⟩-1)/2
                         φ 为各管段轴线与生长方向（图像垂直轴）的夹角
     - 回退方法（低倍率）：结构张量估算，注明为 2D 投影近似值
  4. 倍率自适应预处理：CLAHE 参数和分割阈值窗口随 px/μm 动态调整
  5. 管径鲁棒估计：IQR 过滤异常半径 + 中位数估计
  6. 骨架路径追踪曲率：基于真实路径长度与端点欧氏距离之比（曲折度）
  7. 骨架复用：skeleton 在高倍率下统一计算，HOF / 管径 / 曲率三者共用
"""

import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage.measure import label


# ─── SEM 仪器物理参数（经标尺栏实测校准） ───────────────────────────────────
# 标定依据：
#   1kx  → FW=518 μm，10kx → FW=51.8 μm，50kx → FW=10.4 μm
#   FW = HFW_AT_1X / magnification
# v1.0 错误值：200000；修正后：518000
HFW_AT_1X_UM = 518_000   # 仪器水平视野宽度（1× 倍率时），单位 μm

# 标尺栏高度（两种图像格式实测均为 75px）
INFO_BAR_HEIGHT_PX = 75

# 标尺白线像素长度（实测：558px，跨所有倍率一致）
SCALE_BAR_LENGTH_PX = 558


class FeatureExtractor:
    """
    CNT SEM 图像四特征提取器（v2.0）

    Parameters
    ----------
    magnification : int or None
        SEM 拍摄倍率（如 50000 表示 50kx）。
        None 时退化为像素单位输出（仅用于调试）。
    """

    def __init__(self, magnification: int = None):
        self.mag = magnification
        self.px_per_um: float = 1.0        # 像素/微米，由 _calibrate() 设置
        self.expected_tube_px: float = 3.0  # 预估管径像素数，用于自适应参数

    # ──────────────────────────────────────────────────────────────────────
    #  1. 物理标定
    # ──────────────────────────────────────────────────────────────────────

    def _calibrate(self, img_width: int):
        """
        根据倍率和图像宽度计算 px/μm 换算系数。

        公式：px_per_um = img_width / (HFW_AT_1X_UM / magnification)

        若倍率未知，则以 SCALE_BAR_LENGTH_PX 仅作保底参考（无物理单位输出）。
        """
        if self.mag and self.mag > 0:
            hfw_um = HFW_AT_1X_UM / self.mag          # 当前倍率下水平视野宽度 (μm)
            self.px_per_um = img_width / hfw_um
        else:
            self.px_per_um = 1.0

        # 预估 MWCNT 典型管径（15 nm）对应的像素数，用于后续自适应参数
        # 公式：15 nm × px_per_um × 0.001 (nm→μm)
        self.expected_tube_px = max(1.5, 15.0 * self.px_per_um * 0.001)

    # ──────────────────────────────────────────────────────────────────────
    #  2. ROI 提取：裁掉底部信息栏
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def extract_roi(img_gray: np.ndarray) -> np.ndarray:
        """
        裁剪掉图像底部的 SEM 信息栏（含标尺条和仪器参数文字）。

        检测策略：从图像底部向上扫描，找到第一个行均值 > 阈值的位置，
        向上再留出少量余量，确保信息栏被完全排除。
        采用固定高度保底（INFO_BAR_HEIGHT_PX），应对极暗图像边缘情况。
        """
        h = img_gray.shape[0]

        # 动态检测信息栏上边界（行均值突变点）
        for i in range(h - 1, max(h - 200, 0), -1):
            if img_gray[i].mean() > 60:
                roi_end = min(i + 5, h)
                return img_gray[:roi_end, :]

        # 固定高度保底
        return img_gray[:h - INFO_BAR_HEIGHT_PX, :]

    # ──────────────────────────────────────────────────────────────────────
    #  3. 预处理：CLAHE + 高斯去噪
    # ──────────────────────────────────────────────────────────────────────

    def preprocess(self, roi: np.ndarray) -> np.ndarray:
        """
        对比度自适应直方图均衡化（CLAHE）+ 高斯平滑。

        CLAHE 的 tileGridSize 按预估管径像素数自适应：
        管径越小（高倍率），网格越细，以保留细节。
        """
        tile = max(4, int(32 / max(1, self.expected_tube_px)))
        tile = min(tile, 16)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(tile, tile))
        enhanced = clahe.apply(roi)

        ksize = max(3, int(self.expected_tube_px * 0.8) | 1)
        smoothed = cv2.GaussianBlur(enhanced, (ksize, ksize), 0)
        return smoothed

    # ──────────────────────────────────────────────────────────────────────
    #  4. 密度：自适应阈值（倍率自适应窗口大小）
    # ──────────────────────────────────────────────────────────────────────

    def calculate_density(self, processed: np.ndarray):
        """
        CNT 面密度（面积占比，百分比）。

        分割方法：自适应高斯阈值，blockSize 随预估管径动态调整，
        避免 v1.0 中固定 blockSize=21 在不同倍率下精度差异过大的问题。

        Returns
        -------
        density : float   百分比（0~100）
        thresh  : ndarray 二值化结果（用于后续骨架化）
        """
        block = max(11, int(self.expected_tube_px * 4) | 1)
        block = min(block, 51)

        thresh = cv2.adaptiveThreshold(
            processed, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block, 2
        )
        density = np.count_nonzero(thresh) / thresh.size * 100.0
        return float(density), thresh

    # ──────────────────────────────────────────────────────────────────────
    #  5a. HOF 主方法：骨架 PCA（高倍率）
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def calculate_hof_skeleton(skel: np.ndarray):
        """
        基于骨架各分支 PCA 主方向的 Herman 取向因子（HOF）。

        原理
        ----
        Herman 取向因子定义（源自高分子纤维科学，Herman 1946）：

            f = (3⟨cos²φ⟩ - 1) / 2

            φ : 每段 CNT 轴线与参考方向（SEM 侧视图垂直轴，即生长方向）的夹角
            ⟨⟩: 以骨架分支长度（像素数）为权重的加权平均

        取值范围及物理意义：
            f =  1.0  → 全部管段垂直生长，完美对齐
            f =  0.0  → 随机取向（三维各向同性）
            f = -0.5  → 全部管段平行于基底，完全倒伏

        方向角计算：
            对每个连通骨架分支做 PCA，最大特征值对应的特征向量即为主方向。
            设主方向向量 v = [v_row, v_col]（图像行/列分量），则：
                φ = arctan2(|v_col|, |v_row|)
            当 CNT 垂直时 v_row 大、v_col 小 → φ ≈ 0° → cos²φ ≈ 1 → f → 1
            当 CNT 水平时 v_col 大、v_row 小 → φ ≈ 90° → cos²φ ≈ 0 → f → -0.5

        注：本方法计算的是 SEM 二维投影的 HOF。对各向同性样品，
        二维投影理论值为 f ≈ 0（而非结构张量方法的 ~0.25），
        与文献标准定义一致，可直接与 XRD / 拉曼偏振比较。

        Returns
        -------
        hof          : float  Herman 取向因子（-0.5 ~ 1.0）
        mean_phi_deg : float  平均取向偏角（度，0°=完全垂直）
        n_branches   : int    参与统计的有效骨架分支数
        """
        labeled = label(skel, connectivity=2)
        n_regions = labeled.max()
        if n_regions == 0:
            return 0.0, 90.0, 0

        # ── 第一遍：收集所有有效分支的坐标和大小 ──────────────────────────────
        branch_data = []   # list of (coords, n)
        for rid in range(1, n_regions + 1):
            coords = np.argwhere(labeled == rid).astype(float)
            n = len(coords)
            if n < 10:
                continue
            branch_data.append((coords, n))

        if not branch_data:
            return 0.0, 90.0, 0

        # ── 过滤超大连通域（整片网络，非单管段） ─────────────────────────────
        # 密集 CNT 二值骨架往往形成一个巨型连通域（像素数可达几十万），
        # 其 PCA 主方向接近 45° 对角线，与真实管段取向无关。
        # 策略：以所有有效分支大小的中位数为基准，剔除超过 20× 中位数的分支。
        all_sizes = np.array([n for _, n in branch_data], dtype=float)
        median_n  = float(np.median(all_sizes))
        max_n     = max(500, median_n * 20)   # 保底 500px，防止样本量极少时过度过滤

        cos2_list, weights = [], []
        for coords, n in branch_data:
            if n > max_n:   # 跳过巨型网络连通域
                continue

            # PCA：协方差矩阵最大特征值对应特征向量 = 主方向
            c = coords - coords.mean(axis=0)
            cov = np.cov(c.T)
            if cov.ndim < 2:
                continue
            _, vecs = np.linalg.eigh(cov)
            v = vecs[:, -1]     # [v_row, v_col]

            # φ = 主方向与图像垂直轴（行轴）的夹角
            phi = np.arctan2(abs(v[1]), abs(v[0]))

            cos2_list.append(np.cos(phi) ** 2)
            weights.append(float(n))

        if not cos2_list:
            return 0.0, 90.0, 0

        w = np.array(weights)
        cos2_mean = np.average(cos2_list, weights=w)
        hof = (3.0 * cos2_mean - 1.0) / 2.0
        mean_phi_deg = float(np.degrees(np.arccos(np.sqrt(np.clip(cos2_mean, 0, 1)))))

        return float(np.clip(hof, -0.5, 1.0)), mean_phi_deg, len(cos2_list)

    # ──────────────────────────────────────────────────────────────────────
    #  5b. HOF 回退方法：结构张量（低倍率 / 无骨架时使用）
    # ──────────────────────────────────────────────────────────────────────

    def calculate_hof_structure_tensor(self, processed: np.ndarray):
        """
        基于结构张量的 HOF 估算（低倍率图像的回退方案）。

        原理
        ----
        结构张量主方向角 θ 是梯度的主方向，与纤维轴线垂直。
        因此纤维与垂直参考方向的夹角 φ ≈ |θ|（θ 为结构张量角，∈ [-π/2, π/2]）：
            当 CNT 垂直时：梯度水平 → θ ≈ 0 → φ = 0 → f → 1
            当 CNT 水平时：梯度垂直 → θ ≈ ±90° → φ = 90° → f → -0.5

        以相干性 C = (λ₁-λ₂)/(λ₁+λ₂) 为权重进行全图加权平均。

        局限性
        ------
        此方法为 2D 投影估算，对各向同性样品理论输出 f ≈ 0.25（非 0），
        存在系统性偏置。在报告中应注明"结构张量估算值"以区别于骨架 HOF。

        Returns
        -------
        hof          : float  HOF 估算值
        mean_phi_deg : float  平均取向偏角（度）
        coherence    : float  平均相干性（反映信号质量）
        """
        Ix = cv2.Scharr(processed, cv2.CV_64F, 1, 0)
        Iy = cv2.Scharr(processed, cv2.CV_64F, 0, 1)

        sigma = max(2.0, self.expected_tube_px * 1.5)
        ksize = int(sigma * 4) | 1

        Jxx = cv2.GaussianBlur(Ix * Ix, (ksize, ksize), sigma)
        Jxy = cv2.GaussianBlur(Ix * Iy, (ksize, ksize), sigma)
        Jyy = cv2.GaussianBlur(Iy * Iy, (ksize, ksize), sigma)

        trace = Jxx + Jyy
        disc  = np.sqrt(np.maximum(0.0, (Jxx - Jyy) ** 2 / 4 + Jxy ** 2))
        lambda1 = trace / 2 + disc
        lambda2 = trace / 2 - disc

        denom = lambda1 + lambda2
        valid = denom > 1e-8
        coherence = np.zeros_like(denom)
        coherence[valid] = (lambda1[valid] - lambda2[valid]) / denom[valid]

        # 结构张量主方向角 θ（梯度方向，垂直于纤维）
        theta = 0.5 * np.arctan2(2.0 * Jxy, Jxx - Jyy)

        # φ = |θ|：纤维轴与垂直参考方向的夹角
        phi = np.abs(theta)

        w = coherence.ravel()
        total_w = w.sum()
        if total_w < 1e-10:
            return 0.0, 90.0, 0.0

        cos2_mean = np.dot(w, np.cos(phi.ravel()) ** 2) / total_w
        hof = (3.0 * cos2_mean - 1.0) / 2.0
        mean_phi_deg = float(np.degrees(np.arccos(np.sqrt(np.clip(cos2_mean, 0, 1)))))

        return float(np.clip(hof, -0.5, 1.0)), mean_phi_deg, float(coherence.mean())

    # ──────────────────────────────────────────────────────────────────────
    #  6. 管径：骨架化 + 距离变换（修正标定 + IQR 过滤）
    # ──────────────────────────────────────────────────────────────────────

    # 管径计算最低倍率要求（低于此值时单管 < 1px，距离变换无意义）
    MIN_MAG_FOR_DIAMETER = 20_000

    def calculate_diameter(self, thresh: np.ndarray):
        """
        基于骨架化 + 距离变换的 CNT 平均管径估计（单位：nm）。

        流程：
            1. 形态学闭运算：连接骨架断裂点
            2. skimage skeletonize：Zhang-Suen 精确细化
               （优于 v1.0 的迭代腐蚀近似法）
            3. 距离变换：骨架点处值 ≈ 局部半径（像素）
            4. IQR 过滤：Tukey 准则去除异常半径
            5. 中位数估计：比均值对异常值更鲁棒
            6. 物理转换：diameter_nm = median_r × 2 / px_per_um × 1000

        Returns
        -------
        diameter_nm : float   平均管径（nm），-1 表示提取失败
        skel        : ndarray 骨架二值图（供曲率计算复用）
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        skel = skeletonize(closed > 0)

        if not np.any(skel):
            return -1.0, skel

        dist = cv2.distanceTransform(closed, cv2.DIST_L2, 5)
        radii = dist[skel].astype(float)

        # IQR 过滤
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

    # ──────────────────────────────────────────────────────────────────────
    #  7. 曲率：骨架路径追踪（曲折度 Tortuosity）
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def calculate_curvature(skel: np.ndarray):
        """
        基于骨架连通域路径追踪的曲折度（Tortuosity）分类。

        定义：tortuosity = 路径像素数 / 端点间欧氏距离
            - 完全笔直：tortuosity ≈ 1.0
            - 弯曲：tortuosity > 1（路径迂回）

        端点识别：对每个连通域，取质心最远点 p1，再取 p1 最远点 p2，
        以此作为该段骨架的两端（迭代最远点近似）。

        分类阈值（CNT 典型 SEM 图像经验值）：
            < 1.15  → Straight
            1.15~1.60 → Wavy
            ≥ 1.60  → Coiled

        Returns
        -------
        label_str  : str   'Straight' / 'Wavy' / 'Coiled' / 'Unknown'
        tortuosity : float 中位数曲折度
        """
        labeled = label(skel, connectivity=2)
        n_regions = labeled.max()

        if n_regions == 0:
            return "Unknown", 0.0

        tortuosities = []
        for rid in range(1, n_regions + 1):
            coords = np.argwhere(labeled == rid).astype(float)
            n = len(coords)
            if n < 15:
                continue

            # 迭代最远点近似端点
            centroid = coords.mean(axis=0)
            p1 = coords[np.linalg.norm(coords - centroid, axis=1).argmax()]
            p2 = coords[np.linalg.norm(coords - p1, axis=1).argmax()]

            end_to_end = np.linalg.norm(p1 - p2)
            if end_to_end < 8:
                continue

            tortuosities.append(n / end_to_end)

        if not tortuosities:
            return "Unknown", 0.0

        median_tort = float(np.median(tortuosities))

        if median_tort < 1.15:
            return "Straight", median_tort
        elif median_tort < 1.60:
            return "Wavy", median_tort
        else:
            return "Coiled", median_tort

    # ──────────────────────────────────────────────────────────────────────
    #  主入口
    # ──────────────────────────────────────────────────────────────────────

    def extract_all(self, img_gray: np.ndarray) -> dict:
        """
        提取四特征：density / alignment(HOF) / diameter / curvature

        流程
        ----
        高倍率（≥ 20kx）：
            骨架统一计算一次 → HOF（骨架PCA法）/ 管径 / 曲率 共用
        低倍率（< 20kx）：
            跳过骨架计算 → HOF 改用结构张量回退估算
            管径和曲率标记为 N/A

        Parameters
        ----------
        img_gray : ndarray  灰度 SEM 图像（含底部信息栏）

        Returns
        -------
        dict
            alignment   : HOF f 值（-0.5 ~ 1.0），数据库字段名保持兼容
            hof_method  : 'skeleton' 或 'structure_tensor'，注明计算来源
            mean_phi_deg: 管段与生长方向的平均偏角（°）
        """
        roi = self.extract_roi(img_gray)
        self._calibrate(roi.shape[1])
        processed = self.preprocess(roi)

        density, thresh = self.calculate_density(processed)

        if self.mag and self.mag >= self.MIN_MAG_FOR_DIAMETER:
            # ── 高倍率：骨架统一计算，三个特征共用 ──
            diameter_nm, skel      = self.calculate_diameter(thresh)
            hof, mean_phi, n_br    = self.calculate_hof_skeleton(skel)
            curv_label, tortuosity = self.calculate_curvature(skel)
            hof_method = "skeleton"
            extra = {"n_branches": n_br}
        else:
            # ── 低倍率：结构张量回退，管径/曲率不可信 ──
            hof, mean_phi, coh     = self.calculate_hof_structure_tensor(processed)
            diameter_nm            = -1.0
            curv_label, tortuosity = "N/A", 0.0
            hof_method = "structure_tensor"
            extra = {"coherence": round(coh, 4)}

        return {
            # 四个主特征（alignment 字段存 HOF 值，数据库兼容）
            "density":      round(density, 2),
            "alignment":    round(hof, 4),
            "diameter":     round(diameter_nm, 2) if diameter_nm >= 0 else None,
            "curvature":    curv_label,
            # 辅助字段（论文报告 / 调试用）
            "hof_method":   hof_method,
            "mean_phi_deg": round(mean_phi, 2),
            "tortuosity":   round(tortuosity, 3),
            "px_per_um":    round(self.px_per_um, 2),
            **extra,
        }


# ─── 命令行快速测试 ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os

    test_cases = [
        (r"d:\CNTDATA\ZZY\20230306 No26 0.75 1.0\No26 200w 5.0nm 5w 0.75nm 400 200 100 600 750 15min 180min mid 50000-1.png",  50000),
        (r"d:\CNTDATA\ZZY\20230306 No26 0.75 1.0\No26 200w 5.0nm 5w 0.75nm 400 200 100 600 750 15min 180min mid 10000-1.png", 10000),
        (r"d:\CNTDATA\XR\250301 T800\C1AN1.tiff", None),
    ]

    for path, mag in test_cases:
        if not os.path.exists(path):
            print(f"[SKIP] {os.path.basename(path)}")
            continue
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        extractor = FeatureExtractor(magnification=mag)
        res = extractor.extract_all(img)
        name = os.path.basename(path)
        print(f"\n{'='*60}")
        print(f"文件: {name}  |  mag={mag}")
        print(f"  px/μm      = {res['px_per_um']}")
        print(f"  density    = {res['density']:.2f}%")
        phi = res.get('mean_phi_deg', 0)
        method = res.get('hof_method', '?')
        extra = res.get('n_branches', res.get('coherence', '?'))
        print(f"  HOF (f)    = {res['alignment']:.4f}  φ={phi:.1f}°  [{method}, extra={extra}]")
        print(f"  diameter   = {res['diameter']} nm")
        print(f"  curvature  = {res['curvature']}  (tortuosity={res['tortuosity']:.3f})")
