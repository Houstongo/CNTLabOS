"""
CNT 闃靛垪 SEM 鍥惧儚鐗瑰緛鎻愬彇绠楁硶  v2.2
====================================
鏀硅繘鍐呭锛堢浉姣?v1.0锛夛細
  1. ROI 鎻愬彇锛氳嚜鍔ㄨ鍓簳閮ㄤ俊鎭爮锛屾秷闄ゆ爣灏?鏂囧瓧瀵规搴﹀拰闃堝€肩殑姹℃煋
  2. 鐗╃悊鏍囧畾淇锛氶€氳繃鏍囧昂绾垮儚绱犻暱搴?+ 淇鐨?hfw_at_1x 鍙傛暟锛?18000 渭m锛?                  淇浜?v1.0 涓?hfw_at_1x=200000 瀵艰嚧鐨?2.6脳 绯荤粺璇樊
  3. Herman 鍙栧悜鍥犲瓙锛圚OF锛夛細鏇挎崲 Sobel 姊害閫愮偣娉?     - 涓绘柟娉曪紙楂樺€嶇巼锛夛細楠ㄦ灦鍚勫垎鏀?PCA 涓绘柟鍚戣 鈫?f = (3鉄╟os虏蠁鉄?1)/2
                         蠁 涓哄悇绠℃杞寸嚎涓庣敓闀挎柟鍚戯紙鍥惧儚鍨傜洿杞达級鐨勫す瑙?     - 鍥為€€鏂规硶锛堜綆鍊嶇巼锛夛細缁撴瀯寮犻噺浼扮畻锛屾敞鏄庝负 2D 鎶曞奖杩戜技鍊?  4. 鍊嶇巼鑷€傚簲棰勫鐞嗭細CLAHE 鍙傛暟鍜屽垎鍓查槇鍊肩獥鍙ｉ殢 px/渭m 鍔ㄦ€佽皟鏁?  5. 绠″緞椴佹浼拌锛欼QR 杩囨护寮傚父鍗婂緞 + 涓綅鏁颁及璁?  6. 楠ㄦ灦璺緞杩借釜鏇茬巼锛氬熀浜庣湡瀹炶矾寰勯暱搴︿笌绔偣娆ф皬璺濈涔嬫瘮锛堟洸鎶樺害锛?  7. 楠ㄦ灦澶嶇敤锛歴keleton 鍦ㄩ珮鍊嶇巼涓嬬粺涓€璁＄畻锛孒OF / 绠″緞 / 鏇茬巼涓夎€呭叡鐢?
v2.2 鏂板锛?  8. 澧炲己鍒嗘按宀垎鍓茬寰勭畻娉曪細鑷€傚簲绉嶅瓙鐐?+ 鏈€灏忚窛绂荤害鏉?+ 鍖哄煙澶у皬杩囨护
     瑙ｅ喅瀵嗛泦鍖哄煙闂繍绠楄繛鎺ョ浉閭籆NT瀵艰嚧鐨勭洿寰勭郴缁熸€у亸澶ч棶棰?"""

import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage.measure import label
import time


# 鈹€鈹€鈹€ SEM 浠櫒鐗╃悊鍙傛暟锛堢粡鏍囧昂鏍忓疄娴嬫牎鍑嗭級 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
# 鏍囧畾渚濇嵁锛?#   1kx  鈫?FW=518 渭m锛?0kx 鈫?FW=51.8 渭m锛?0kx 鈫?FW=10.4 渭m
#   FW = HFW_AT_1X / magnification
# v1.0 閿欒鍊硷細200000锛涗慨姝ｅ悗锛?18000
HFW_AT_1X_UM = 518_000   # 浠櫒姘村钩瑙嗛噹瀹藉害锛?脳 鍊嶇巼鏃讹級锛屽崟浣?渭m

# 鏍囧昂鏍忛珮搴︼紙涓ょ鍥惧儚鏍煎紡瀹炴祴鍧囦负 75px锛?INFO_BAR_HEIGHT_PX = 75

# 鏍囧昂鐧界嚎鍍忕礌闀垮害锛堝疄娴嬶細558px锛岃法鎵€鏈夊€嶇巼涓€鑷达級
SCALE_BAR_LENGTH_PX = 558
INFO_BAR_HEIGHT_PX = 75


class FeatureExtractor:
    """
    CNT SEM 鍥惧儚鍥涚壒寰佹彁鍙栧櫒锛坴2.2锛?
    Parameters
    ----------
    magnification : int or None
        SEM 鎷嶆憚鍊嶇巼锛堝 50000 琛ㄧず 50kx锛夈€?        None 鏃堕€€鍖栦负鍍忕礌鍗曚綅杈撳嚭锛堜粎鐢ㄤ簬璋冭瘯锛夈€?    diameter_method : str, optional
        绠″緞璁＄畻鏂规硶锛?standard' (闂繍绠楋紝榛樿) 鎴?'enhanced' (鍒嗘按宀垎鍓?
    """

    def __init__(
        self,
        magnification: int = None,
        diameter_method: str = "standard",
        speed_profile: str = "accurate",
    ):
        self.mag = magnification
        self.px_per_um: float = 1.0        # 鍍忕礌/寰背锛岀敱 _calibrate() 璁剧疆
        self.expected_tube_px: float = 3.0  # 棰勪及绠″緞鍍忕礌鏁帮紝鐢ㄤ簬鑷€傚簲鍙傛暟
        self.diameter_method = diameter_method
        self.speed_profile = speed_profile if speed_profile in {"accurate", "fast"} else "accurate"

    FAST_ALIGNMENT_BRANCH_LIMIT = 120
    FAST_CURVATURE_BRANCH_LIMIT = 120
    FAST_WAVINESS_BRANCH_LIMIT = 80
    FAST_MAX_POINTS_PER_BRANCH = 2000
    FAST_CURVATURE_SAMPLE_STEP = 4
    FAST_WAVINESS_MIN_WAVELENGTH_FACTOR = 10.0
    FAST_WAVINESS_MAX_RATIO = 5.0
    V2_MIN_BRANCH_POINTS = 8
    V2_MIN_BRANCH_LENGTH_FACTOR = 2.0

    @staticmethod
    def _collect_components(skel: np.ndarray):
        """
        Collect full connected skeleton components without filtering/downsampling.
        """
        labeled = label(skel, connectivity=2)
        if labeled.max() == 0:
            return []

        from scipy import ndimage as ndi

        objects = ndi.find_objects(labeled)
        components = []
        for rid, slc in enumerate(objects, start=1):
            if slc is None:
                continue
            comp = labeled[slc] == rid
            n_points = int(comp.sum())
            if n_points <= 0:
                continue

            coords = np.argwhere(comp).astype(float)
            coords[:, 0] += float(slc[0].start)
            coords[:, 1] += float(slc[1].start)
            components.append((coords, n_points))
        return components

    @staticmethod
    def _collect_branches_from_components(
        components,
        min_points: int = 10,
        max_branches: int = None,
        max_points_per_branch: int = None,
    ):
        """
        Build branch list from precomputed full components with exact same rules
        as the legacy _collect_branches implementation.
        """
        branches = []
        for coords_full, n_points_full in components:
            if n_points_full < min_points:
                continue

            coords = coords_full
            n_points = int(n_points_full)
            if max_points_per_branch and n_points > max_points_per_branch:
                step = int(np.ceil(n_points / max_points_per_branch))
                coords = coords[::max(1, step)]
                n_points = len(coords)

            branches.append((coords, n_points))

        branches.sort(key=lambda item: item[1], reverse=True)
        if max_branches is not None and len(branches) > max_branches:
            branches = branches[:max_branches]
        return branches

    @staticmethod
    def _collect_branches(
        skel: np.ndarray,
        min_points: int = 10,
        max_branches: int = None,
        max_points_per_branch: int = None,
    ):
        components = FeatureExtractor._collect_components(skel)
        return FeatureExtractor._collect_branches_from_components(
            components,
            min_points=min_points,
            max_branches=max_branches,
            max_points_per_branch=max_points_per_branch,
        )

    @staticmethod
    def _neighbor_count_map(skel: np.ndarray) -> np.ndarray:
        from scipy.ndimage import convolve

        kernel = np.ones((3, 3), dtype=np.uint8)
        kernel[1, 1] = 0
        return convolve((skel > 0).astype(np.uint8), kernel, mode="constant", cval=0)

    @staticmethod
    def _path_length(coords: np.ndarray) -> float:
        if coords.shape[0] < 2:
            return 0.0
        deltas = np.diff(coords.astype(float), axis=0)
        return float(np.sum(np.hypot(deltas[:, 0], deltas[:, 1])))

    @staticmethod
    def _trace_ordered_component_path(component_mask: np.ndarray) -> np.ndarray:
        points = np.argwhere(component_mask > 0)
        if points.size == 0:
            return np.zeros((0, 2), dtype=float)

        offsets = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]
        point_set = {tuple(int(v) for v in point) for point in points}
        adjacency = {}
        for point in point_set:
            y, x = point
            neighbors = []
            for dy, dx in offsets:
                candidate = (y + dy, x + dx)
                if candidate in point_set:
                    neighbors.append(candidate)
            adjacency[point] = neighbors

        endpoints = sorted(point for point, neighbors in adjacency.items() if len(neighbors) <= 1)
        start = endpoints[0] if endpoints else min(point_set)

        path = []
        visited = set()
        current = start
        previous = None

        while current is not None and current not in visited:
            path.append(current)
            visited.add(current)

            candidates = [neighbor for neighbor in adjacency[current] if neighbor != previous and neighbor not in visited]
            if not candidates:
                candidates = [neighbor for neighbor in adjacency[current] if neighbor not in visited]
            if not candidates:
                break

            if previous is None or len(candidates) == 1:
                next_point = sorted(candidates)[0]
            else:
                prev_vec = np.asarray(current, dtype=float) - np.asarray(previous, dtype=float)

                def direction_score(candidate):
                    cand_vec = np.asarray(candidate, dtype=float) - np.asarray(current, dtype=float)
                    return float(np.dot(prev_vec, cand_vec))

                next_point = max(candidates, key=direction_score)

            previous, current = current, next_point

        if len(visited) < len(point_set):
            remaining = sorted(point_set - visited)
            path.extend(remaining)

        return np.asarray(path, dtype=float)

    def _smooth_path_coords(self, coords: np.ndarray, window: int = None) -> np.ndarray:
        if coords.shape[0] < 5:
            return coords.astype(float)

        if window is None:
            base = 3 if self.speed_profile == "fast" else 5
            dynamic = int(round(self.expected_tube_px * 1.5)) | 1
            window = max(base, dynamic)

        window = max(3, min(window, coords.shape[0]))
        if window % 2 == 0:
            window -= 1
        if window < 3:
            return coords.astype(float)

        kernel = np.ones(window, dtype=float) / window
        pad = window // 2
        padded = np.pad(coords.astype(float), ((pad, pad), (0, 0)), mode="edge")
        smoothed = np.column_stack([
            np.convolve(padded[:, 0], kernel, mode="valid"),
            np.convolve(padded[:, 1], kernel, mode="valid"),
        ])
        smoothed[0] = coords[0]
        smoothed[-1] = coords[-1]
        return smoothed

    def _collect_ordered_branches_v2(
        self,
        skel: np.ndarray,
        min_points: int = None,
        max_branches: int = None,
        max_points_per_branch: int = None,
    ):
        skel_mask = (skel > 0).astype(np.uint8)
        if not np.any(skel_mask):
            return []

        if min_points is None:
            min_points = max(self.V2_MIN_BRANCH_POINTS, int(round(self.expected_tube_px * 2.0)))

        min_length_px = max(4.0, float(self.expected_tube_px * self.V2_MIN_BRANCH_LENGTH_FACTOR))
        neighbor_count = self._neighbor_count_map(skel_mask)
        junction_mask = (skel_mask > 0) & (neighbor_count >= 3)
        branch_mask = (skel_mask > 0) & np.logical_not(junction_mask)
        labeled = label(branch_mask, connectivity=2)
        if labeled.max() == 0:
            return []

        branches = []
        for branch_id in range(1, int(labeled.max()) + 1):
            component_mask = labeled == branch_id
            point_count = int(np.count_nonzero(component_mask))
            if point_count < min_points:
                continue

            ordered = self._trace_ordered_component_path(component_mask)
            if ordered.shape[0] < min_points:
                continue

            path_length_px = self._path_length(ordered)
            if path_length_px < min_length_px:
                continue

            ordered = self._smooth_path_coords(ordered)
            if max_points_per_branch and ordered.shape[0] > max_points_per_branch:
                sample_idx = np.linspace(0, ordered.shape[0] - 1, num=max_points_per_branch, dtype=int)
                ordered = ordered[sample_idx]
                path_length_px = self._path_length(ordered)

            branches.append({
                "coords": ordered,
                "n_points": int(ordered.shape[0]),
                "path_length_px": float(path_length_px),
            })

        branches.sort(key=lambda item: (item["path_length_px"], item["n_points"]), reverse=True)
        if max_branches is not None and len(branches) > max_branches:
            branches = branches[:max_branches]
        return branches

    @staticmethod
    def _sample_ordered_coords(coords: np.ndarray, sample_step: int) -> np.ndarray:
        if coords.shape[0] <= 2:
            return coords
        step = max(1, int(sample_step))
        sampled = coords[::step]
        if not np.array_equal(sampled[-1], coords[-1]):
            sampled = np.vstack([sampled, coords[-1]])
        return sampled

    # 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    #  1. 鐗╃悊鏍囧畾
    # 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

    def _calibrate(self, img_width: int):
        """
        鏍规嵁鍊嶇巼鍜屽浘鍍忓搴﹁绠?px/渭m 鎹㈢畻绯绘暟銆?
        鍏紡锛歱x_per_um = img_width / (HFW_AT_1X_UM / magnification)

        鑻ュ€嶇巼鏈煡锛屽垯浠?SCALE_BAR_LENGTH_PX 浠呬綔淇濆簳鍙傝€冿紙鏃犵墿鐞嗗崟浣嶈緭鍑猴級銆?        """
        if self.mag and self.mag > 0:
            hfw_um = HFW_AT_1X_UM / self.mag          # 褰撳墠鍊嶇巼涓嬫按骞宠閲庡搴?(渭m)
            self.px_per_um = img_width / hfw_um
        else:
            self.px_per_um = 1.0

        # 棰勪及 MWCNT 鍏稿瀷绠″緞锛?5 nm锛夊搴旂殑鍍忕礌鏁帮紝鐢ㄤ簬鍚庣画鑷€傚簲鍙傛暟
        # 鍏紡锛?5 nm 脳 px_per_um 脳 0.001 (nm鈫捨糾)
        self.expected_tube_px = max(1.5, 15.0 * self.px_per_um * 0.001)

    # 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    #  2. ROI 鎻愬彇锛氳鎺夊簳閮ㄤ俊鎭爮
    # 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

    @staticmethod
    def extract_roi(img_gray: np.ndarray) -> np.ndarray:
        """
        Crop out SEM footer / metadata bar at the bottom.
        """
        h, w = img_gray.shape

        row_means = np.mean(img_gray, axis=1)
        row_stds = np.std(img_gray, axis=1)
        row_grad_x = np.abs(cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3))
        row_grad_strength = np.mean(row_grad_x, axis=1)

        row_means_norm = (row_means - row_means.min()) / (row_means.max() - row_means.min() + 1e-8)
        row_stds_norm = (row_stds - row_stds.min()) / (row_stds.max() - row_stds.min() + 1e-8)
        row_grad_norm = (row_grad_strength - row_grad_strength.min()) / (row_grad_strength.max() - row_grad_strength.min() + 1e-8)

        combined_score = 0.5 * row_means_norm + 0.3 * row_stds_norm + 0.2 * row_grad_norm

        best_cutoff = h - INFO_BAR_HEIGHT_PX
        max_change = 0.0
        for i in range(h - 1, max(h - 200, 0), -1):
            if i > 10:
                change = combined_score[i] - combined_score[i - 10]
                if change > max_change:
                    max_change = change
                    best_cutoff = i

        tail_start = max(h - 220, 0)
        black_ratio = np.mean(img_gray < 15, axis=1)
        tail_mask = black_ratio[tail_start:] > 0.75
        run = 0
        footer_start = None
        for idx, is_dark in enumerate(tail_mask):
            run = run + 1 if is_dark else 0
            if run >= 5:
                footer_start = tail_start + idx - run + 1
                break
        if footer_start is not None:
            best_cutoff = min(best_cutoff, footer_start)

        min_cut = min(INFO_BAR_HEIGHT_PX, max(1, int(h * 0.08)))
        roi_end = min(best_cutoff + 10, h - min_cut)
        roi_end = max(1, min(roi_end, h))
        return img_gray[:roi_end, :]

    # 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    #  3. 棰勫鐞嗭細CLAHE + 楂樻柉鍘诲櫔
    # 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

    def preprocess(self, roi: np.ndarray) -> np.ndarray:
        """
        瀵规瘮搴﹁嚜閫傚簲鐩存柟鍥惧潎琛″寲锛圕LAHE锛? 楂樻柉骞虫粦銆?
        CLAHE 鐨?tileGridSize 鎸夐浼扮寰勫儚绱犳暟鑷€傚簲锛?        绠″緞瓒婂皬锛堥珮鍊嶇巼锛夛紝缃戞牸瓒婄粏锛屼互淇濈暀缁嗚妭銆?        """
        tile = max(4, int(32 / max(1, self.expected_tube_px)))
        tile = min(tile, 16)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(tile, tile))
        enhanced = clahe.apply(roi)

        ksize = max(3, int(self.expected_tube_px * 0.8) | 1)
        smoothed = cv2.GaussianBlur(enhanced, (ksize, ksize), 0)
        return smoothed

    # 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    #  4. 瀵嗗害锛氳嚜閫傚簲闃堝€硷紙鍊嶇巼鑷€傚簲绐楀彛澶у皬锛?    # 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

    def calculate_density(self, processed: np.ndarray):
        """
        CNT 闈㈠瘑搴︼紙闈㈢Н鍗犳瘮锛岀櫨鍒嗘瘮锛夈€?
        鍒嗗壊鏂规硶锛氳嚜閫傚簲楂樻柉闃堝€硷紝blockSize 闅忛浼扮寰勫姩鎬佽皟鏁达紝
        閬垮厤 v1.0 涓浐瀹?blockSize=21 鍦ㄤ笉鍚屽€嶇巼涓嬬簿搴﹀樊寮傝繃澶х殑闂銆?
        Returns
        -------
        density : float   鐧惧垎姣旓紙0~100锛?        thresh  : ndarray 浜屽€煎寲缁撴灉锛堢敤浜庡悗缁鏋跺寲锛?        """
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

    # 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    #  5a. HOF 涓绘柟娉曪細楠ㄦ灦 PCA锛堥珮鍊嶇巼锛?    # 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

    @staticmethod
    def calculate_hof_skeleton(
        skel: np.ndarray,
        max_branches: int = None,
        max_points_per_branch: int = None,
        components=None,
    ):
        """
        Skeleton-PCA Herman orientation factor.
        """
        if components is None:
            branch_data = FeatureExtractor._collect_branches(
                skel,
                min_points=10,
                max_branches=max_branches,
                max_points_per_branch=max_points_per_branch,
            )
        else:
            branch_data = FeatureExtractor._collect_branches_from_components(
                components,
                min_points=10,
                max_branches=max_branches,
                max_points_per_branch=max_points_per_branch,
            )
        if not branch_data:
            return 0.0, 90.0, 0

        all_sizes = np.array([n for _, n in branch_data], dtype=float)
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
            return 0.0, 90.0, 0

        w = np.array(weights)
        cos2_mean = np.average(cos2_list, weights=w)
        hof = (3.0 * cos2_mean - 1.0) / 2.0
        mean_phi_deg = float(np.degrees(np.arccos(np.sqrt(np.clip(cos2_mean, 0, 1)))))

        return float(np.clip(hof, -0.5, 1.0)), mean_phi_deg, len(cos2_list)

    # 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    #  5b. HOF 鍥為€€鏂规硶锛氱粨鏋勫紶閲忥紙浣庡€嶇巼 / 鏃犻鏋舵椂浣跨敤锛?    # 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

    def calculate_hof_structure_tensor(self, processed: np.ndarray):
        """
        鍩轰簬缁撴瀯寮犻噺鐨?HOF 浼扮畻锛堜綆鍊嶇巼鍥惧儚鐨勫洖閫€鏂规锛夈€?
        鍘熺悊
        ----
        缁撴瀯寮犻噺涓绘柟鍚戣 胃 鏄搴︾殑涓绘柟鍚戯紝涓庣氦缁磋酱绾垮瀭鐩淬€?        鍥犳绾ょ淮涓庡瀭鐩村弬鑰冩柟鍚戠殑澶硅 蠁 鈮?|胃|锛埼?涓虹粨鏋勫紶閲忚锛屸垐 [-蟺/2, 蟺/2]锛夛細
            褰?CNT 鍨傜洿鏃讹細姊害姘村钩 鈫?胃 鈮?0 鈫?蠁 = 0 鈫?f 鈫?1
            褰?CNT 姘村钩鏃讹細姊害鍨傜洿 鈫?胃 鈮?卤90掳 鈫?蠁 = 90掳 鈫?f 鈫?-0.5

        浠ョ浉骞叉€?C = (位鈧?位鈧?/(位鈧?位鈧? 涓烘潈閲嶈繘琛屽叏鍥惧姞鏉冨钩鍧囥€?
        灞€闄愭€?        ------
        姝ゆ柟娉曚负 2D 鎶曞奖浼扮畻锛屽鍚勫悜鍚屾€ф牱鍝佺悊璁鸿緭鍑?f 鈮?0.25锛堥潪 0锛夛紝
        瀛樺湪绯荤粺鎬у亸缃€傚湪鎶ュ憡涓簲娉ㄦ槑"缁撴瀯寮犻噺浼扮畻鍊?浠ュ尯鍒簬楠ㄦ灦 HOF銆?
        Returns
        -------
        hof          : float  HOF 浼扮畻鍊?        mean_phi_deg : float  骞冲潎鍙栧悜鍋忚锛堝害锛?        coherence    : float  骞冲潎鐩稿共鎬э紙鍙嶆槧淇″彿璐ㄩ噺锛?        """
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

        # 缁撴瀯寮犻噺涓绘柟鍚戣 胃锛堟搴︽柟鍚戯紝鍨傜洿浜庣氦缁达級
        theta = 0.5 * np.arctan2(2.0 * Jxy, Jxx - Jyy)

        # 蠁 = |胃|锛氱氦缁磋酱涓庡瀭鐩村弬鑰冩柟鍚戠殑澶硅
        phi = np.abs(theta)

        w = coherence.ravel()
        total_w = w.sum()
        if total_w < 1e-10:
            return 0.0, 90.0, 0.0

        cos2_mean = np.dot(w, np.cos(phi.ravel()) ** 2) / total_w
        hof = (3.0 * cos2_mean - 1.0) / 2.0
        mean_phi_deg = float(np.degrees(np.arccos(np.sqrt(np.clip(cos2_mean, 0, 1)))))

        return float(np.clip(hof, -0.5, 1.0)), mean_phi_deg, float(coherence.mean())

    def calculate_hof_skeleton_adaptive(
        self,
        skel: np.ndarray,
        processed: np.ndarray = None,
        base_components=None,
    ):
        """
        Adaptive HOF extractor.
        - accurate profile: evaluate skeleton orientation in 0° and 90°
        - fast profile: use capped skeleton-PCA with lightweight rotation correction
        """
        skeleton_opts = {}
        if self.speed_profile == "fast":
            skeleton_opts = {
                "max_branches": self.FAST_ALIGNMENT_BRANCH_LIMIT,
                "max_points_per_branch": self.FAST_MAX_POINTS_PER_BRANCH,
            }

        candidates = []
        rotations = ((0, skel), (90, np.rot90(skel)))
        for rotation_deg, candidate_skel in rotations:
            components = base_components if rotation_deg == 0 else None
            hof, mean_phi_deg, n_branches = self.calculate_hof_skeleton(
                candidate_skel,
                components=components,
                **skeleton_opts,
            )
            candidates.append({
                "rotation_correction_deg": rotation_deg,
                "alignment": hof,
                "mean_phi_deg": mean_phi_deg,
                "n_branches": n_branches,
            })

        best = max(
            candidates,
            key=lambda item: (
                item["alignment"],
                -item["mean_phi_deg"],
                item["n_branches"],
                -item["rotation_correction_deg"],
            ),
        )
        raw = candidates[0]
        if self.speed_profile == "fast" and best["n_branches"] == 0 and processed is not None:
            # Fallback only when skeleton route cannot produce any valid branches.
            hof, mean_phi_deg, coherence = self.calculate_hof_structure_tensor(processed)
            return {
                "alignment": float(hof),
                "mean_phi_deg": float(mean_phi_deg),
                "n_branches": 0,
                "rotation_correction_deg": 0,
                "alignment_raw": float(hof),
                "mean_phi_raw_deg": float(mean_phi_deg),
                "hof_method": "structure_tensor_fallback_no_skeleton",
                "coherence": float(coherence),
            }
        return {
            "alignment": float(best["alignment"]),
            "mean_phi_deg": float(best["mean_phi_deg"]),
            "n_branches": int(best["n_branches"]),
            "rotation_correction_deg": int(best["rotation_correction_deg"]),
            "alignment_raw": float(raw["alignment"]),
            "mean_phi_raw_deg": float(raw["mean_phi_deg"]),
            "hof_method": "skeleton_fast_dual_axis" if self.speed_profile == "fast" else "skeleton",
        }

    # 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    #  6. 绠″緞锛氶鏋跺寲 + 璺濈鍙樻崲锛堜慨姝ｆ爣瀹?+ IQR 杩囨护锛?    # 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

    # 绠″緞璁＄畻鏈€浣庡€嶇巼瑕佹眰锛堜綆浜庢鍊兼椂鍗曠 < 1px锛岃窛绂诲彉鎹㈡棤鎰忎箟锛?    MIN_MAG_FOR_DIAMETER = 20_000
    MIN_MAG_FOR_DIAMETER = 20_000
    MAX_ENHANCED_PIXELS = 6_000_000
    MAX_ENHANCED_DENSITY = 0.45
    MAX_ENHANCED_REGIONS = 4_000
    MAX_ENHANCED_MARKERS = 12_000

    def should_use_enhanced_diameter(
        self,
        thresh: np.ndarray,
        density_factor: float = None,
        n_regions: int = None,
        has_large_regions: bool = None,
        marker_count: int = None,
    ):
        """
        Guard enhanced watershed against the dense, large XR masks that turn the
        segmentation step into the performance bottleneck.
        """
        if density_factor is None:
            density_factor = float(np.count_nonzero(thresh) / max(1, thresh.size))

        if (
            self.mag
            and self.mag <= self.MIN_MAG_FOR_DIAMETER
            and thresh.size >= self.MAX_ENHANCED_PIXELS
            and density_factor >= self.MAX_ENHANCED_DENSITY
        ):
            return False, "large_dense_low_mag"

        if n_regions is not None and n_regions > self.MAX_ENHANCED_REGIONS:
            return False, "too_many_regions"

        if has_large_regions is False:
            return False, "simple_regions"

        if marker_count is not None and marker_count > self.MAX_ENHANCED_MARKERS:
            return False, "too_many_markers"

        return True, "ok"

    def calculate_diameter(self, thresh: np.ndarray):
        """
        鍩轰簬楠ㄦ灦鍖?+ 璺濈鍙樻崲鐨?CNT 骞冲潎绠″緞浼拌锛堝崟浣嶏細nm锛夈€?
        娴佺▼锛?            1. 褰㈡€佸闂繍绠楋細杩炴帴楠ㄦ灦鏂鐐?            2. skimage skeletonize锛歓hang-Suen 绮剧‘缁嗗寲
               锛堜紭浜?v1.0 鐨勮凯浠ｈ厫铓€杩戜技娉曪級
            3. 璺濈鍙樻崲锛氶鏋剁偣澶勫€?鈮?灞€閮ㄥ崐寰勶紙鍍忕礌锛?            4. IQR 杩囨护锛歍ukey 鍑嗗垯鍘婚櫎寮傚父鍗婂緞
            5. 涓綅鏁颁及璁★細姣斿潎鍊煎寮傚父鍊兼洿椴佹
            6. 鐗╃悊杞崲锛歞iameter_nm = median_r 脳 2 / px_per_um 脳 1000

        Returns
        -------
        diameter_nm : float   骞冲潎绠″緞锛坣m锛夛紝-1 琛ㄧず鎻愬彇澶辫触
        skel        : ndarray 楠ㄦ灦浜屽€煎浘锛堜緵鏇茬巼璁＄畻澶嶇敤锛?        """
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        skel = skeletonize(closed > 0)

        if not np.any(skel):
            return -1.0, skel

        dist = cv2.distanceTransform(closed, cv2.DIST_L2, 5)
        radii = dist[skel].astype(float)

        # IQR 杩囨护
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

    def calculate_diameter_enhanced(self, thresh: np.ndarray):
        """
        Enhanced watershed-based diameter estimation for dense connected masks.
        Falls back to standard diameter when complexity is too high.
        """
        from scipy import ndimage as ndi
        from scipy.ndimage import maximum_filter
        from skimage.segmentation import watershed

        density_factor = np.count_nonzero(thresh) / max(1, thresh.size)
        dist = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)

        labeled = label(thresh, connectivity=2)
        n_regions = int(labeled.max())
        has_large_regions = False
        if n_regions > 0:
            region_sizes = np.bincount(labeled.ravel())[1:]
            if len(region_sizes):
                median_size = float(np.median(region_sizes))
                has_large_regions = bool(region_sizes.max() > median_size * 10)

        use_enhanced, _ = self.should_use_enhanced_diameter(
            thresh,
            density_factor=density_factor,
            n_regions=n_regions,
            has_large_regions=has_large_regions,
        )
        if not use_enhanced:
            return self.calculate_diameter(thresh)

        max_dist = float(dist.max())
        if max_dist <= 0:
            return -1.0, np.zeros_like(thresh, dtype=bool)

        threshold_base = max_dist * (0.3 + 0.05 * density_factor)
        min_distance = max(2, int(round(self.expected_tube_px * 0.8)))

        local_max = (maximum_filter(dist, size=min_distance + 1) == dist) & (dist > threshold_base)
        markers, _ = ndi.label(local_max)
        marker_count = int(markers.max())
        if marker_count == 0:
            return self.calculate_diameter(thresh)

        use_enhanced, _ = self.should_use_enhanced_diameter(
            thresh,
            density_factor=density_factor,
            n_regions=n_regions,
            has_large_regions=True,
            marker_count=marker_count,
        )
        if not use_enhanced:
            return self.calculate_diameter(thresh)

        labels = watershed(-dist, markers, mask=thresh)
        label_ids = np.unique(labels)
        if len(label_ids) - 1 > self.MAX_ENHANCED_MARKERS:
            return self.calculate_diameter(thresh)

        diameters = []
        expected_area = (self.expected_tube_px * 2) ** 2
        for label_id in label_ids:
            if label_id == 0:
                continue
            cnt_mask = labels == label_id
            region_size = int(cnt_mask.sum())
            if region_size < 10 or region_size > expected_area * 3:
                continue
            region_radius = float(np.median(dist[cnt_mask]))
            diameters.append(region_radius * 2)

        if len(diameters) == 0:
            return self.calculate_diameter(thresh)

        diameters = np.asarray(diameters, dtype=float)
        q1, q3 = np.percentile(diameters, [25, 75])
        iqr = q3 - q1
        filtered = diameters[(diameters >= q1 - 1.5 * iqr) & (diameters <= q3 + 1.5 * iqr)]
        if len(filtered) == 0:
            filtered = diameters

        median_diameter_px = float(np.median(filtered))
        diameter_nm = (median_diameter_px / self.px_per_um) * 1000.0
        skel = skeletonize(thresh > 0)
        return float(diameter_nm), skel

    # 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    #  7. 鏇茬巼锛氬熀浜庨鏋剁殑涓夌偣娉曪紙魏 = 1/R锛?    # 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

    def calculate_curvature(
        self,
        skel: np.ndarray,
        max_branches: int = None,
        max_points_per_branch: int = None,
        sample_step: int = None,
        base_components=None,
    ):
        """
        Curvature classification from skeleton centerlines.
        """
        if max_branches is None and self.speed_profile == "fast":
            max_branches = self.FAST_CURVATURE_BRANCH_LIMIT
        if max_points_per_branch is None and self.speed_profile == "fast":
            max_points_per_branch = self.FAST_MAX_POINTS_PER_BRANCH
        if sample_step is None:
            sample_step = self.FAST_CURVATURE_SAMPLE_STEP if self.speed_profile == "fast" else 2

        if base_components is None:
            branch_data = self._collect_branches(
                skel,
                min_points=15,
                max_branches=max_branches,
                max_points_per_branch=max_points_per_branch,
            )
        else:
            branch_data = self._collect_branches_from_components(
                base_components,
                min_points=15,
                max_branches=max_branches,
                max_points_per_branch=max_points_per_branch,
            )
        if not branch_data:
            return "Unknown", 0.0

        all_curvatures = []
        for coords, _ in branch_data:
            sampled_coords = coords[::max(1, int(sample_step))]
            curvature_values = []
            m = len(sampled_coords)
            for i in range(1, m - 1):
                p_prev = sampled_coords[i - 1]
                p_curr = sampled_coords[i]
                p_next = sampled_coords[i + 1]

                ab = p_prev - p_curr
                bc = p_curr - p_next
                ca = p_next - p_prev
                a = np.linalg.norm(bc)
                b = np.linalg.norm(ca)
                c = np.linalg.norm(ab)

                if a > 2 and b > 2 and c > 2:
                    s = (a + b + c) / 2
                    area = np.sqrt(max(0.0, s * (s - a) * (s - b) * (s - c)))
                    curvature = 4 * area / (a * b * c)
                    curvature_values.append(curvature)

            if curvature_values:
                median_curv_px = float(np.median(curvature_values))
                px_per_nm = 1.0 / 15.0
                curvature_nm = median_curv_px / px_per_nm
                all_curvatures.append(curvature_nm)

        if not all_curvatures:
            return "Unknown", 0.0

        median_curvature = float(np.median(all_curvatures))
        if median_curvature < 0.05:
            label = "Straight"
        elif median_curvature < 0.15:
            label = "Wavy"
        else:
            label = "Coiled"

        return label, median_curvature

    def calculate_curvature_v2(
        self,
        skel: np.ndarray,
        max_branches: int = None,
        max_points_per_branch: int = None,
        sample_step: int = None,
        ordered_branches=None,
    ):
        if max_branches is None and self.speed_profile == "fast":
            max_branches = self.FAST_CURVATURE_BRANCH_LIMIT
        if max_points_per_branch is None and self.speed_profile == "fast":
            max_points_per_branch = self.FAST_MAX_POINTS_PER_BRANCH
        if sample_step is None:
            sample_step = 2 if self.speed_profile == "fast" else 1

        branches = ordered_branches
        if branches is None:
            branches = self._collect_ordered_branches_v2(
                skel,
                min_points=15,
                max_branches=max_branches,
                max_points_per_branch=max_points_per_branch,
            )
        if not branches:
            return "Unknown", 0.0

        px_per_nm = max(self.px_per_um / 1000.0, 1e-6)
        branch_curvatures = []
        weights = []

        for branch in branches:
            sampled_coords = self._sample_ordered_coords(branch["coords"], sample_step=sample_step)
            if sampled_coords.shape[0] < 3:
                continue

            curvature_values_px = []
            for idx in range(1, sampled_coords.shape[0] - 1):
                p_prev = sampled_coords[idx - 1]
                p_curr = sampled_coords[idx]
                p_next = sampled_coords[idx + 1]

                ab = p_curr - p_prev
                bc = p_next - p_curr
                ca = p_next - p_prev

                a = float(np.linalg.norm(ab))
                b = float(np.linalg.norm(bc))
                c = float(np.linalg.norm(ca))
                if min(a, b, c) <= 1e-6:
                    continue

                cross = abs(ab[0] * bc[1] - ab[1] * bc[0])
                curvature_px = (2.0 * cross) / max(a * b * c, 1e-6)
                if np.isfinite(curvature_px) and curvature_px > 0:
                    curvature_values_px.append(curvature_px)

            if curvature_values_px:
                branch_curvatures.append(float(np.median(curvature_values_px)) * px_per_nm)
                weights.append(max(branch["path_length_px"], 1.0))
            else:
                branch_curvatures.append(0.0)
                weights.append(max(branch["path_length_px"], 1.0))

        if not branch_curvatures:
            return "Unknown", 0.0

        median_curvature_nm = float(np.average(branch_curvatures, weights=np.asarray(weights, dtype=float)))
        if median_curvature_nm < 5e-4:
            label_name = "Straight"
        elif median_curvature_nm < 2.5e-3:
            label_name = "Wavy"
        else:
            label_name = "Coiled"
        return label_name, median_curvature_nm

    @staticmethod
    def _smooth_signal(values: np.ndarray, window: int = 5) -> np.ndarray:
        """Simple moving average smoothing used for waviness extraction."""
        if values.size < 3:
            return values
        window = max(3, min(window, values.size))
        if window % 2 == 0:
            window -= 1
        if window < 3:
            return values
        kernel = np.ones(window, dtype=float) / window
        return np.convolve(values, kernel, mode="same")

    def _calculate_component_waviness(self, coords: np.ndarray, fast_mode: bool = False):
        """Estimate wave height / wavelength for one connected skeleton component."""
        if coords.shape[0] < 20:
            return None
        # Guardrail for dense components: keep complexity bounded on very large skeletons.
        if coords.shape[0] > 120000:
            step = int(np.ceil(coords.shape[0] / 120000))
            coords = coords[::step]

        centered = coords - coords.mean(axis=0)
        cov = np.cov(centered.T)
        if cov.ndim < 2:
            return None

        eigvals, eigvecs = np.linalg.eigh(cov)
        axis = eigvecs[:, np.argmax(eigvals)]
        axis_norm = np.linalg.norm(axis)
        if axis_norm == 0:
            return None
        axis = axis / axis_norm
        normal = np.array([-axis[1], axis[0]])

        longitudinal = centered @ axis
        lateral = centered @ normal

        order = np.argsort(longitudinal)
        longitudinal = longitudinal[order]
        lateral = lateral[order]

        bins = np.round(longitudinal - longitudinal.min()).astype(np.int32)
        if bins.size == 0:
            return None
        # Vectorized grouped mean by longitudinal bin (O(n) via bincount).
        sums_s = np.bincount(bins, weights=longitudinal)
        sums_d = np.bincount(bins, weights=lateral)
        counts = np.bincount(bins)
        valid = counts > 0
        if np.count_nonzero(valid) < 12:
            return None
        s_vals = sums_s[valid] / counts[valid]
        d_vals = sums_d[valid] / counts[valid]

        linear = np.polyfit(s_vals, d_vals, deg=1)
        detrended = d_vals - np.polyval(linear, s_vals)
        smoothed = self._smooth_signal(detrended, window=5)

        if np.ptp(smoothed) < 2.0:
            return None

        extrema = []
        min_spacing = max(3.0, self.expected_tube_px * 2.0)
        for idx in range(1, smoothed.size - 1):
            prev_val = smoothed[idx - 1]
            curr_val = smoothed[idx]
            next_val = smoothed[idx + 1]
            kind = None
            if curr_val >= prev_val and curr_val > next_val:
                kind = "peak"
            elif curr_val <= prev_val and curr_val < next_val:
                kind = "trough"
            if kind is None:
                continue

            if extrema and kind == extrema[-1]["kind"]:
                if (kind == "peak" and curr_val > extrema[-1]["value"]) or (
                    kind == "trough" and curr_val < extrema[-1]["value"]
                ):
                    extrema[-1] = {"kind": kind, "s": s_vals[idx], "value": curr_val}
                continue

            if extrema and abs(s_vals[idx] - extrema[-1]["s"]) < min_spacing:
                more_extreme = (
                    (kind == "peak" and curr_val > extrema[-1]["value"]) or
                    (kind == "trough" and curr_val < extrema[-1]["value"])
                )
                if more_extreme:
                    extrema[-1] = {"kind": kind, "s": s_vals[idx], "value": curr_val}
                continue

            extrema.append({"kind": kind, "s": s_vals[idx], "value": curr_val})

        if len(extrema) < 3:
            return None

        waves = []
        for idx in range(len(extrema) - 2):
            first = extrema[idx]
            middle = extrema[idx + 1]
            last = extrema[idx + 2]
            if first["kind"] != last["kind"] or first["kind"] == middle["kind"]:
                continue

            wavelength_px = float(last["s"] - first["s"])
            min_wavelength_px = min_spacing * 1.5
            if fast_mode:
                min_wavelength_px = max(
                    min_wavelength_px * 1.5,
                    self.expected_tube_px * self.FAST_WAVINESS_MIN_WAVELENGTH_FACTOR,
                )
            if wavelength_px <= min_wavelength_px:
                continue

            height_px = 0.5 * (
                abs(middle["value"] - first["value"]) +
                abs(middle["value"] - last["value"])
            )
            if height_px <= 0:
                continue

            ratio = height_px / wavelength_px
            if fast_mode and ratio > self.FAST_WAVINESS_MAX_RATIO:
                ratio = self.FAST_WAVINESS_MAX_RATIO
                height_px = ratio * wavelength_px

            waves.append({
                "ratio": ratio,
                "height_px": height_px,
                "wavelength_px": wavelength_px,
            })

        if not waves:
            return None

        path_length = float(np.sum(np.hypot(np.diff(s_vals), np.diff(smoothed))))
        span = float(np.hypot(s_vals[-1] - s_vals[0], smoothed[-1] - smoothed[0]))
        tortuosity = path_length / span if span > 0 else 1.0

        weights = np.array([wave["wavelength_px"] for wave in waves], dtype=float)
        return {
            "ratio": float(np.average([wave["ratio"] for wave in waves], weights=weights)),
            "height_px": float(np.average([wave["height_px"] for wave in waves], weights=weights)),
            "wavelength_px": float(np.average([wave["wavelength_px"] for wave in waves], weights=weights)),
            "weight": float(weights.sum()),
            "tortuosity": float(max(tortuosity, 1.0)),
        }

    def calculate_waviness(
        self,
        skel: np.ndarray,
        max_branches: int = None,
        max_points_per_branch: int = None,
        base_components=None,
    ):
        """
        Estimate waviness from skeleton centerlines using wave height / wavelength.

        Returns a length-weighted aggregate over connected skeleton components.
        """
        if max_branches is None and self.speed_profile == "fast":
            max_branches = self.FAST_WAVINESS_BRANCH_LIMIT
        if max_points_per_branch is None and self.speed_profile == "fast":
            max_points_per_branch = self.FAST_MAX_POINTS_PER_BRANCH

        if base_components is None:
            branch_data = self._collect_branches(
                skel,
                min_points=20,
                max_branches=max_branches,
                max_points_per_branch=max_points_per_branch,
            )
        else:
            branch_data = self._collect_branches_from_components(
                base_components,
                min_points=20,
                max_branches=max_branches,
                max_points_per_branch=max_points_per_branch,
            )
        if not branch_data:
            return {
                "waviness_ratio": 0.0,
                "waviness_height_nm": 0.0,
                "waviness_wavelength_nm": 0.0,
                "waviness_branches": 0,
                "tortuosity": 1.0,
            }

        branch_metrics = []
        for coords, _ in branch_data:
            metrics = self._calculate_component_waviness(
                coords,
                fast_mode=(self.speed_profile == "fast"),
            )
            if metrics is not None:
                branch_metrics.append(metrics)

        if not branch_metrics:
            return {
                "waviness_ratio": 0.0,
                "waviness_height_nm": 0.0,
                "waviness_wavelength_nm": 0.0,
                "waviness_branches": 0,
                "tortuosity": 1.0,
            }

        px_per_nm = max(self.px_per_um / 1000.0, 1e-6)
        weights = np.array([metric["weight"] for metric in branch_metrics], dtype=float)
        height_px = float(np.average([metric["height_px"] for metric in branch_metrics], weights=weights))
        wavelength_px = float(np.average([metric["wavelength_px"] for metric in branch_metrics], weights=weights))

        return {
            "waviness_ratio": float(np.average([metric["ratio"] for metric in branch_metrics], weights=weights)),
            "waviness_height_nm": height_px / px_per_nm,
            "waviness_wavelength_nm": wavelength_px / px_per_nm,
            "waviness_branches": len(branch_metrics),
            "tortuosity": float(np.average([metric["tortuosity"] for metric in branch_metrics], weights=weights)),
        }

    def calculate_waviness_v2(
        self,
        skel: np.ndarray,
        max_branches: int = None,
        max_points_per_branch: int = None,
        ordered_branches=None,
    ):
        if max_branches is None and self.speed_profile == "fast":
            max_branches = self.FAST_WAVINESS_BRANCH_LIMIT
        if max_points_per_branch is None and self.speed_profile == "fast":
            max_points_per_branch = self.FAST_MAX_POINTS_PER_BRANCH

        branches = ordered_branches
        if branches is None:
            branches = self._collect_ordered_branches_v2(
                skel,
                min_points=20,
                max_branches=max_branches,
                max_points_per_branch=max_points_per_branch,
            )
        if not branches:
            return {
                "waviness_ratio_v2": 0.0,
                "waviness_height_nm_v2": 0.0,
                "waviness_wavelength_nm_v2": 0.0,
                "waviness_branches_v2": 0,
                "tortuosity_v2": 1.0,
            }

        branch_metrics = []
        for branch in branches:
            metrics = self._calculate_component_waviness(
                branch["coords"],
                fast_mode=(self.speed_profile == "fast"),
            )
            if metrics is None:
                continue
            metrics["weight"] = max(branch["path_length_px"], metrics["weight"])
            branch_metrics.append(metrics)

        if not branch_metrics:
            return {
                "waviness_ratio_v2": 0.0,
                "waviness_height_nm_v2": 0.0,
                "waviness_wavelength_nm_v2": 0.0,
                "waviness_branches_v2": 0,
                "tortuosity_v2": 1.0,
            }

        px_per_nm = max(self.px_per_um / 1000.0, 1e-6)
        weights = np.array([metric["weight"] for metric in branch_metrics], dtype=float)
        height_px = float(np.average([metric["height_px"] for metric in branch_metrics], weights=weights))
        wavelength_px = float(np.average([metric["wavelength_px"] for metric in branch_metrics], weights=weights))

        return {
            "waviness_ratio_v2": float(np.average([metric["ratio"] for metric in branch_metrics], weights=weights)),
            "waviness_height_nm_v2": height_px / px_per_nm,
            "waviness_wavelength_nm_v2": wavelength_px / px_per_nm,
            "waviness_branches_v2": len(branch_metrics),
            "tortuosity_v2": float(np.average([metric["tortuosity"] for metric in branch_metrics], weights=weights)),
        }

    def extract_all(
        self,
        img_gray: np.ndarray,
        progress_callback=None,
        external_binary_mask: np.ndarray = None,
    ) -> dict:
        """
        鎻愬彇鍥涚壒寰侊細density / alignment(HOF) / diameter / curvature

        娴佺▼
        ----
        楂樺€嶇巼锛堚墺 20kx锛夛細
            楠ㄦ灦缁熶竴璁＄畻涓€娆?鈫?HOF锛堥鏋禤CA娉曪級/ 绠″緞 / 鏇茬巼 鍏辩敤
        浣庡€嶇巼锛? 20kx锛夛細
            璺宠繃楠ㄦ灦璁＄畻 鈫?HOF 鏀圭敤缁撴瀯寮犻噺鍥為€€浼扮畻
            绠″緞鍜屾洸鐜囨爣璁颁负 N/A

        Parameters
        ----------
        img_gray : ndarray  鐏板害 SEM 鍥惧儚锛堝惈搴曢儴淇℃伅鏍忥級

        Returns
        -------
        dict
            alignment   : HOF f 鍊硷紙-0.5 ~ 1.0锛夛紝鏁版嵁搴撳瓧娈靛悕淇濇寔鍏煎
            hof_method  : 'skeleton' 鎴?'structure_tensor'锛屾敞鏄庤绠楁潵婧?            mean_phi_deg: 绠℃涓庣敓闀挎柟鍚戠殑骞冲潎鍋忚锛埪帮級
        """
        started_at = time.perf_counter()

        def emit_progress(step_name: str, **payload):
            if progress_callback is None:
                return
            progress_callback(
                step_name,
                round(time.perf_counter() - started_at, 3),
                payload,
            )

        roi = self.extract_roi(img_gray)
        emit_progress("roi", roi_shape=tuple(int(v) for v in roi.shape))
        self._calibrate(roi.shape[1])
        processed = self.preprocess(roi)
        emit_progress("preprocess", px_per_um=round(self.px_per_um, 3))

        if external_binary_mask is not None:
            mask = np.asarray(external_binary_mask)
            if mask.shape != roi.shape:
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    (roi.shape[1], roi.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            thresh = (mask > 0).astype(np.uint8) * 255
            density = float(np.count_nonzero(thresh) / max(thresh.size, 1) * 100.0)
            emit_progress("density", density=round(density, 3), source="external_mask")
        else:
            density, thresh = self.calculate_density(processed)
            emit_progress("density", density=round(density, 3), source="threshold")

        # 鈹€鈹€ 楠ㄦ灦缁熶竴璁＄畻锛欻OF/鏇茬巼鍏辩敤锛堝€嶇巼鑷€傚簲璋冩暣锛?鈹€鈹€
        # 鏍规嵁diameter_method閫夋嫨绠″緞璁＄畻鏂规硶
        if self.diameter_method == 'enhanced':
            diameter_nm, skel = self.calculate_diameter_enhanced(thresh)
            diameter_method_name = "enhanced_watershed"
        else:
            diameter_nm, skel = self.calculate_diameter(thresh)
            diameter_method_name = "standard"
        emit_progress(
            "diameter",
            diameter=round(diameter_nm, 3) if diameter_nm >= 0 else None,
            diameter_method=diameter_method_name,
        )

        base_components = self._collect_components(skel)

        alignment_metrics = self.calculate_hof_skeleton_adaptive(
            skel,
            processed=processed,
            base_components=base_components,
        )
        hof = alignment_metrics["alignment"]
        mean_phi = alignment_metrics["mean_phi_deg"]
        n_br = alignment_metrics["n_branches"]
        emit_progress(
            "alignment",
            alignment=round(hof, 4),
            n_branches=int(n_br),
            rotation_correction_deg=int(alignment_metrics["rotation_correction_deg"]),
        )
        curv_label, curvature_nm = self.calculate_curvature(skel, base_components=base_components)
        ordered_branches_v2 = self._collect_ordered_branches_v2(
            skel,
            min_points=15,
            max_branches=self.FAST_CURVATURE_BRANCH_LIMIT if self.speed_profile == "fast" else None,
            max_points_per_branch=self.FAST_MAX_POINTS_PER_BRANCH if self.speed_profile == "fast" else None,
        )
        curv_label_v2, curvature_nm_v2 = self.calculate_curvature_v2(
            skel,
            ordered_branches=ordered_branches_v2,
        )
        emit_progress(
            "curvature",
            curvature=curv_label,
            curvature_nm=round(curvature_nm, 4),
            curvature_v2=curv_label_v2,
            curvature_nm_v2=round(curvature_nm_v2, 6),
        )
        waviness = self.calculate_waviness(skel, base_components=base_components)
        waviness_v2 = self.calculate_waviness_v2(
            skel,
            ordered_branches=ordered_branches_v2,
        )
        emit_progress(
            "waviness",
            waviness_ratio=(
                round(waviness["waviness_ratio"], 4)
                if waviness["waviness_ratio"] is not None else None
            ),
            waviness_branches=int(waviness["waviness_branches"]),
            waviness_ratio_v2=(
                round(waviness_v2["waviness_ratio_v2"], 4)
                if waviness_v2["waviness_ratio_v2"] is not None else None
            ),
            waviness_branches_v2=int(waviness_v2["waviness_branches_v2"]),
        )
        hof_method = alignment_metrics.get("hof_method", "skeleton")
        tortuosity = waviness["tortuosity"]
        tortuosity_v2 = waviness_v2["tortuosity_v2"]

        if self.mag and self.mag < self.MIN_MAG_FOR_DIAMETER:
            # 浣庡€嶇巼锛氱寰?鏇茬巼鏍囪涓轰笉鍙俊
            diameter_nm = -1.0
            curv_label, tortuosity = "N/A", 0.0
            curvature_nm = 0.0
            curv_label_v2, tortuosity_v2 = "N/A", 0.0
            curvature_nm_v2 = 0.0
            diameter_method_name = "N/A"
            waviness = {
                "waviness_ratio": None,
                "waviness_height_nm": None,
                "waviness_wavelength_nm": None,
                "waviness_branches": 0,
                "tortuosity": 0.0,
            }
            waviness_v2 = {
                "waviness_ratio_v2": None,
                "waviness_height_nm_v2": None,
                "waviness_wavelength_nm_v2": None,
                "waviness_branches_v2": 0,
                "tortuosity_v2": 0.0,
            }

        extra = {
            "n_branches": n_br,
            "diameter_method": diameter_method_name,
            "rotation_correction_deg": alignment_metrics["rotation_correction_deg"],
            "alignment_raw": round(alignment_metrics["alignment_raw"], 4),
            "mean_phi_raw_deg": round(alignment_metrics["mean_phi_raw_deg"], 2),
            "speed_profile": self.speed_profile,
        }

        result = {
            # 鍥涗釜涓荤壒寰侊紙alignment 瀛楁瀛?HOF 鍊硷紝鏁版嵁搴撳吋瀹癸級
            "density":      round(density, 2),
            "alignment":    round(hof, 4),
            "diameter":     round(diameter_nm, 2) if diameter_nm >= 0 else None,
            "curvature":    curv_label,
            "curvature_v2": curv_label_v2,
            # 杈呭姪瀛楁锛堣鏂囨姤鍛?/ 璋冭瘯鐢級
            "hof_method":   hof_method,
            "mean_phi_deg": round(mean_phi, 2),
            "curvature_nm": round(curvature_nm, 4),  # 鐪熸鐨勬洸鐜囷紙nm鈦宦癸級
            "curvature_nm_v2": round(curvature_nm_v2, 6),
            "tortuosity":   round(tortuosity, 3),
            "tortuosity_v2": round(tortuosity_v2, 3),
            "waviness_ratio": round(waviness["waviness_ratio"], 4) if waviness["waviness_ratio"] is not None else None,
            "waviness_height_nm": round(waviness["waviness_height_nm"], 2) if waviness["waviness_height_nm"] is not None else None,
            "waviness_wavelength_nm": round(waviness["waviness_wavelength_nm"], 2) if waviness["waviness_wavelength_nm"] is not None else None,
            "waviness_branches": waviness["waviness_branches"],
            "waviness_ratio_v2": round(waviness_v2["waviness_ratio_v2"], 4) if waviness_v2["waviness_ratio_v2"] is not None else None,
            "waviness_height_nm_v2": round(waviness_v2["waviness_height_nm_v2"], 2) if waviness_v2["waviness_height_nm_v2"] is not None else None,
            "waviness_wavelength_nm_v2": round(waviness_v2["waviness_wavelength_nm_v2"], 2) if waviness_v2["waviness_wavelength_nm_v2"] is not None else None,
            "waviness_branches_v2": waviness_v2["waviness_branches_v2"],
            "px_per_um":    round(self.px_per_um, 2),
            **extra,
        }
        emit_progress("done", total_elapsed_s=round(time.perf_counter() - started_at, 3))
        return result


# 鈹€鈹€鈹€ 鍛戒护琛屽揩閫熸祴璇?鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
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
        print(f"鏂囦欢: {name}  |  mag={mag}")
        print(f"  px/渭m      = {res['px_per_um']}")
        print(f"  density    = {res['density']:.2f}%")
        phi = res.get('mean_phi_deg', 0)
        method = res.get('hof_method', '?')
        extra = res.get('n_branches', res.get('coherence', '?'))
        print(f"  HOF (f)    = {res['alignment']:.4f}  蠁={phi:.1f}掳  [{method}, extra={extra}]")
        print(f"  diameter   = {res['diameter']} nm")
        print(f"  curvature  = {res['curvature']}  (tortuosity={res['tortuosity']:.3f})")






