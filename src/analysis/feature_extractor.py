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
    V3_MIN_BRANCH_POINTS = 6
    V3_MIN_BRANCH_LENGTH_FACTOR = 1.0
    V3_BRANCH_QUANTILE = 75.0
    V3_TRIM_PERCENT = 10.0
    BRANCH_CLEANUP_ENABLED = True
    BRANCH_CLEANUP_ISOLATED_MIN_LENGTH_FACTOR = 2.0
    BRANCH_CLEANUP_ISOLATED_MIN_POINTS_FACTOR = 1.2
    BRANCH_CLEANUP_SPUR_LENGTH_FACTOR = 3.0
    BRANCH_CLEANUP_MAX_ITERATIONS = 12
    BRANCH_CLEANUP_NEIGHBOR_OFFSETS = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]
    LEGACY_STRAIGHT_CURVATURE_PX = 0.05 / 15.0
    LEGACY_WAVY_CURVATURE_PX = 0.15 / 15.0

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
    def _iter_labeled_component_points(labeled: np.ndarray, min_points: int = 1):
        rows, cols = np.nonzero(labeled)
        if rows.size == 0:
            return

        point_labels = labeled[rows, cols].astype(np.int64, copy=False)
        positive = point_labels > 0
        if not np.any(positive):
            return

        rows = rows[positive]
        cols = cols[positive]
        point_labels = point_labels[positive]
        order = np.argsort(point_labels, kind="mergesort")
        rows = rows[order]
        cols = cols[order]
        point_labels = point_labels[order]
        unique_labels, start_idx, counts = np.unique(
            point_labels,
            return_index=True,
            return_counts=True,
        )
        min_points = max(1, int(min_points))
        for label_id, start, count in zip(unique_labels, start_idx, counts):
            if int(count) < min_points:
                continue
            end = int(start + count)
            yield int(label_id), np.column_stack((rows[start:end], cols[start:end]))

    @staticmethod
    def _trace_ordered_component_points(points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=int)
        if points.size == 0:
            return np.zeros((0, 2), dtype=float)
        if points.shape[0] == 1:
            return points.astype(float)

        y = points[:, 0]
        x = points[:, 1]
        y0 = int(y.min())
        x0 = int(x.min())
        local_y = y - y0
        local_x = x - x0
        box_h = int(local_y.max()) + 1
        box_w = int(local_x.max()) + 1

        local_mask = np.zeros((box_h, box_w), dtype=np.uint8)
        local_mask[local_y, local_x] = 1
        padded = np.pad(local_mask, 1, mode="constant")
        neighbor_count = (
            padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:] +
            padded[1:-1, :-2] + padded[1:-1, 2:] +
            padded[2:, :-2] + padded[2:, 1:-1] + padded[2:, 2:]
        )
        degrees = neighbor_count[local_y, local_x]

        endpoint_idx = np.flatnonzero(degrees <= 1)
        if endpoint_idx.size > 0:
            endpoint_points = points[endpoint_idx]
            endpoint_order = np.lexsort((endpoint_points[:, 1], endpoint_points[:, 0]))
            start_idx = int(endpoint_idx[endpoint_order[0]])
        else:
            start_idx = int(np.lexsort((points[:, 1], points[:, 0]))[0])

        padded_index = np.full((box_h + 2, box_w + 2), -1, dtype=np.int32)
        point_idx = np.arange(points.shape[0], dtype=np.int32)
        padded_index[local_y + 1, local_x + 1] = point_idx

        ordered_idx = np.empty(points.shape[0], dtype=np.int32)
        visited = np.zeros(points.shape[0], dtype=bool)
        current_idx = start_idx
        previous_idx = -1
        count = 0

        while current_idx >= 0 and not visited[current_idx]:
            ordered_idx[count] = current_idx
            count += 1
            visited[current_idx] = True

            cy = int(local_y[current_idx]) + 1
            cx = int(local_x[current_idx]) + 1
            candidates = []
            for dy, dx in FeatureExtractor.BRANCH_CLEANUP_NEIGHBOR_OFFSETS:
                neighbor_idx = int(padded_index[cy + dy, cx + dx])
                if neighbor_idx < 0 or neighbor_idx == previous_idx or visited[neighbor_idx]:
                    continue
                candidates.append(neighbor_idx)
            if not candidates:
                for dy, dx in FeatureExtractor.BRANCH_CLEANUP_NEIGHBOR_OFFSETS:
                    neighbor_idx = int(padded_index[cy + dy, cx + dx])
                    if neighbor_idx >= 0 and not visited[neighbor_idx]:
                        candidates.append(neighbor_idx)
                if not candidates:
                    break

            if previous_idx < 0 or len(candidates) == 1:
                candidate_points = points[np.asarray(candidates, dtype=int)]
                candidate_order = np.lexsort((candidate_points[:, 1], candidate_points[:, 0]))
                next_idx = int(candidates[int(candidate_order[0])])
            else:
                prev_vec = points[current_idx].astype(float) - points[previous_idx].astype(float)
                best_score = None
                next_idx = candidates[0]
                for candidate_idx in candidates:
                    cand_vec = points[candidate_idx].astype(float) - points[current_idx].astype(float)
                    score = float(np.dot(prev_vec, cand_vec))
                    if best_score is None or score > best_score:
                        best_score = score
                        next_idx = candidate_idx

            previous_idx, current_idx = current_idx, int(next_idx)

        if count < points.shape[0]:
            remaining_idx = np.flatnonzero(np.logical_not(visited))
            if remaining_idx.size > 0:
                remaining_points = points[remaining_idx]
                remaining_order = np.lexsort((remaining_points[:, 1], remaining_points[:, 0]))
                tail = remaining_idx[remaining_order]
                ordered_idx[count:count + tail.size] = tail
                count += int(tail.size)

        return points[ordered_idx[:count]].astype(float)

    @staticmethod
    def _trace_ordered_component_path(component_mask: np.ndarray) -> np.ndarray:
        points = np.argwhere(component_mask > 0)
        return FeatureExtractor._trace_ordered_component_points(points)

    @classmethod
    def _iter_skeleton_neighbors(cls, point, skeleton_mask: np.ndarray):
        y, x = point
        neighbors = []
        h, w = skeleton_mask.shape
        for dy, dx in cls.BRANCH_CLEANUP_NEIGHBOR_OFFSETS:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and skeleton_mask[ny, nx] > 0:
                neighbors.append((ny, nx))
        return neighbors

    @classmethod
    def _neighbor_group_count(cls, point, skeleton_mask: np.ndarray) -> int:
        y, x = point
        local = np.zeros((3, 3), dtype=np.uint8)
        h, w = skeleton_mask.shape
        for dy, dx in cls.BRANCH_CLEANUP_NEIGHBOR_OFFSETS:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and skeleton_mask[ny, nx] > 0:
                local[dy + 1, dx + 1] = 1
        return int(cv2.connectedComponents(local, connectivity=4)[0] - 1)

    @classmethod
    def _trace_spur_from_endpoint(cls, start, skeleton_mask: np.ndarray, neighbor_count: np.ndarray):
        path = [start]
        prev = None
        current = start
        while True:
            candidates = [pt for pt in cls._iter_skeleton_neighbors(current, skeleton_mask) if pt != prev]
            if not candidates:
                break
            if len(candidates) == 1:
                nxt = candidates[0]
            else:
                prev_vec = None if prev is None else (current[0] - prev[0], current[1] - prev[1])

                def candidate_score(candidate):
                    group_count = cls._neighbor_group_count(candidate, skeleton_mask)
                    degree_count = int(neighbor_count[candidate])
                    if prev_vec is None:
                        direction_score = 0.0
                    else:
                        cand_vec = (candidate[0] - current[0], candidate[1] - current[1])
                        direction_score = float(prev_vec[0] * cand_vec[0] + prev_vec[1] * cand_vec[1])
                    return (group_count, degree_count, direction_score)

                nxt = max(candidates, key=candidate_score)
            path.append(nxt)
            if cls._neighbor_group_count(nxt, skeleton_mask) >= 3:
                break
            prev, current = current, nxt
        return path

    def _remove_short_isolated_skeleton_components(
        self,
        skeleton: np.ndarray,
        min_length_factor: float = None,
        min_points_factor: float = None,
    ):
        skeleton_mask = (skeleton > 0).astype(np.uint8)
        if not np.any(skeleton_mask):
            return {
                "cleaned_skeleton": skeleton_mask.astype(bool),
                "removed_mask": np.zeros_like(skeleton_mask, dtype=bool),
                "removed_component_count": 0,
                "removed_pixel_count": 0,
                "min_length_px": 0.0,
                "min_points": 0,
            }

        if min_length_factor is None:
            min_length_factor = self.BRANCH_CLEANUP_ISOLATED_MIN_LENGTH_FACTOR
        if min_points_factor is None:
            min_points_factor = self.BRANCH_CLEANUP_ISOLATED_MIN_POINTS_FACTOR

        labeled = cv2.connectedComponents(skeleton_mask, connectivity=8)[1]
        cleaned = skeleton_mask.copy()
        removed_mask = np.zeros_like(skeleton_mask, dtype=np.uint8)

        min_length_px = max(3.0, float(self.expected_tube_px * min_length_factor))
        min_points = max(4, int(round(self.expected_tube_px * min_points_factor)))
        removed_components = 0

        for _, component_points in self._iter_labeled_component_points(labeled):
            point_count = int(component_points.shape[0])
            if point_count < min_points:
                cleaned[component_points[:, 0], component_points[:, 1]] = 0
                removed_mask[component_points[:, 0], component_points[:, 1]] = 1
                removed_components += 1
                continue
            ordered = self._trace_ordered_component_points(component_points)
            path_length = self._path_length(ordered)
            if point_count < min_points or path_length < min_length_px:
                cleaned[component_points[:, 0], component_points[:, 1]] = 0
                removed_mask[component_points[:, 0], component_points[:, 1]] = 1
                removed_components += 1

        return {
            "cleaned_skeleton": (cleaned > 0),
            "removed_mask": (removed_mask > 0),
            "removed_component_count": int(removed_components),
            "removed_pixel_count": int(np.count_nonzero(removed_mask)),
            "min_length_px": float(min_length_px),
            "min_points": int(min_points),
        }

    def _prune_terminal_spurs(
        self,
        skeleton: np.ndarray,
        spur_factor: float = None,
        max_iterations: int = None,
    ):
        skeleton_mask = (skeleton > 0).astype(np.uint8)
        if not np.any(skeleton_mask):
            return {
                "cleaned_skeleton": skeleton_mask.astype(bool),
                "removed_spur_mask": np.zeros_like(skeleton_mask, dtype=bool),
                "removed_spur_count": 0,
                "removed_pixel_count": 0,
                "spur_length_limit_px": 0.0,
                "iterations": 0,
            }

        if spur_factor is None:
            spur_factor = self.BRANCH_CLEANUP_SPUR_LENGTH_FACTOR
        if max_iterations is None:
            max_iterations = self.BRANCH_CLEANUP_MAX_ITERATIONS

        removed_mask = np.zeros_like(skeleton_mask, dtype=np.uint8)
        total_removed_spurs = 0
        iteration = 0
        spur_length_limit_px = max(3.0, float(self.expected_tube_px * spur_factor))
        kernel = np.ones((3, 3), dtype=np.uint8)
        height, width = skeleton_mask.shape

        while iteration < max_iterations:
            iteration += 1
            neighbor_count = self._neighbor_count_map(skeleton_mask)
            junction_mask = (skeleton_mask > 0) & (neighbor_count >= 3)
            branch_mask = (skeleton_mask > 0) & np.logical_not(junction_mask)
            if not np.any(branch_mask):
                break

            junction_labels = label(junction_mask.astype(np.uint8), connectivity=2)
            branch_labels = label(branch_mask.astype(np.uint8), connectivity=2)
            valid_junction_labels = set()
            for junction_label, junction_points in self._iter_labeled_component_points(junction_labels):
                for point_row, point_col in junction_points:
                    if self._neighbor_group_count((int(point_row), int(point_col)), skeleton_mask) >= 3:
                        valid_junction_labels.add(int(junction_label))
                        break
            to_remove = []

            for _, component_points in self._iter_labeled_component_points(branch_labels):
                point_rows = component_points[:, 0]
                point_cols = component_points[:, 1]
                if not np.any(neighbor_count[point_rows, point_cols] <= 1):
                    continue

                y0 = max(int(point_rows.min()) - 1, 0)
                y1 = min(int(point_rows.max()) + 2, height)
                x0 = max(int(point_cols.min()) - 1, 0)
                x1 = min(int(point_cols.max()) + 2, width)
                local_component = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
                local_component[point_rows - y0, point_cols - x0] = 1
                touched = cv2.dilate(local_component, kernel, iterations=1).astype(bool)
                touched_labels = np.unique(junction_labels[y0:y1, x0:x1][touched])
                touched_labels = touched_labels[touched_labels > 0]
                if touched_labels.size != 1 or int(touched_labels[0]) not in valid_junction_labels:
                    continue

                ordered = self._trace_ordered_component_points(component_points)
                length_px = self._path_length(ordered)
                if length_px >= spur_length_limit_px:
                    continue
                total_removed_spurs += 1
                to_remove.append(component_points)

            if not to_remove:
                break

            for component_points in to_remove:
                point_rows = component_points[:, 0]
                point_cols = component_points[:, 1]
                skeleton_mask[point_rows, point_cols] = 0
                removed_mask[point_rows, point_cols] = 1
            # Re-thin the local topology after spur removal so junction halos
            # collapse back onto the surviving trunk before the next iteration.
            skeleton_mask = skeletonize(skeleton_mask > 0).astype(np.uint8)

        return {
            "cleaned_skeleton": (skeleton_mask > 0),
            "removed_spur_mask": (removed_mask > 0),
            "removed_spur_count": int(total_removed_spurs),
            "removed_pixel_count": int(np.count_nonzero(removed_mask)),
            "spur_length_limit_px": float(spur_length_limit_px),
            "iterations": int(iteration),
        }

    def _clean_branch_skeleton(self, skeleton: np.ndarray):
        skeleton_mask = (skeleton > 0)
        if not self.BRANCH_CLEANUP_ENABLED:
            return {
                "branch_cleanup_enabled": False,
                "cleaned_skeleton": skeleton_mask.astype(bool),
                "removed_short_mask": np.zeros_like(skeleton_mask, dtype=bool),
                "removed_spur_mask": np.zeros_like(skeleton_mask, dtype=bool),
                "removed_short_component_count": 0,
                "removed_short_pixel_count": 0,
                "removed_spur_count": 0,
                "removed_spur_pixel_count": 0,
                "isolated_min_length_px": 0.0,
                "isolated_min_points": 0,
                "spur_length_limit_px": 0.0,
                "spur_iterations": 0,
            }

        isolated = self._remove_short_isolated_skeleton_components(skeleton_mask)
        pruned = self._prune_terminal_spurs(isolated["cleaned_skeleton"])
        return {
            "branch_cleanup_enabled": True,
            "cleaned_skeleton": pruned["cleaned_skeleton"],
            "removed_short_mask": isolated["removed_mask"],
            "removed_spur_mask": pruned["removed_spur_mask"],
            "removed_short_component_count": int(isolated["removed_component_count"]),
            "removed_short_pixel_count": int(isolated["removed_pixel_count"]),
            "removed_spur_count": int(pruned["removed_spur_count"]),
            "removed_spur_pixel_count": int(pruned["removed_pixel_count"]),
            "isolated_min_length_px": float(isolated["min_length_px"]),
            "isolated_min_points": int(isolated["min_points"]),
            "spur_length_limit_px": float(pruned["spur_length_limit_px"]),
            "spur_iterations": int(pruned["iterations"]),
        }

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
        min_length_factor: float = None,
        max_branches: int = None,
        max_points_per_branch: int = None,
    ):
        skel_mask = (skel > 0).astype(np.uint8)
        if not np.any(skel_mask):
            return []

        if min_points is None:
            min_points = max(self.V2_MIN_BRANCH_POINTS, int(round(self.expected_tube_px * 2.0)))

        if min_length_factor is None:
            min_length_factor = self.V2_MIN_BRANCH_LENGTH_FACTOR
        min_length_px = max(4.0, float(self.expected_tube_px * min_length_factor))
        neighbor_count = self._neighbor_count_map(skel_mask)
        junction_mask = (skel_mask > 0) & (neighbor_count >= 3)
        branch_mask = (skel_mask > 0) & np.logical_not(junction_mask)
        labeled = label(branch_mask, connectivity=2)
        if labeled.max() == 0:
            return []

        branches = []
        for _, component_points in self._iter_labeled_component_points(labeled, min_points=min_points):
            point_count = int(component_points.shape[0])
            ordered = self._trace_ordered_component_points(component_points)
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

    def _filter_ordered_branches(
        self,
        branches,
        min_points: int = None,
        min_length_factor: float = None,
        max_branches: int = None,
    ):
        if not branches:
            return []

        if min_points is None:
            min_points = max(self.V2_MIN_BRANCH_POINTS, int(round(self.expected_tube_px * 2.0)))
        if min_length_factor is None:
            min_length_factor = self.V2_MIN_BRANCH_LENGTH_FACTOR

        min_length_px = max(4.0, float(self.expected_tube_px * min_length_factor))
        filtered = [
            branch
            for branch in branches
            if int(branch.get("n_points", 0)) >= min_points
            and float(branch.get("path_length_px", 0.0)) >= min_length_px
        ]

        if max_branches is not None and len(filtered) > max_branches:
            filtered = filtered[:max_branches]
        return filtered

    def _prepare_curvature_branch_sets(
        self,
        skel: np.ndarray,
        v2_min_points: int = 15,
        apply_branch_cleanup: bool = True,
    ):
        branch_source = self._clean_branch_skeleton(skel)["cleaned_skeleton"] if apply_branch_cleanup else (skel > 0)
        max_points_per_branch = self.FAST_MAX_POINTS_PER_BRANCH if self.speed_profile == "fast" else None
        max_branches = self.FAST_CURVATURE_BRANCH_LIMIT if self.speed_profile == "fast" else None

        relaxed_branches = self._collect_ordered_branches_v2(
            branch_source,
            min_points=max(self.V3_MIN_BRANCH_POINTS, int(round(self.expected_tube_px * 1.5))),
            min_length_factor=self.V3_MIN_BRANCH_LENGTH_FACTOR,
            max_points_per_branch=max_points_per_branch,
        )

        ordered_branches_v2 = self._filter_ordered_branches(
            relaxed_branches,
            min_points=v2_min_points,
            min_length_factor=self.V2_MIN_BRANCH_LENGTH_FACTOR,
            max_branches=max_branches,
        )
        ordered_branches_v3 = self._filter_ordered_branches(
            relaxed_branches,
            min_points=max(self.V3_MIN_BRANCH_POINTS, int(round(self.expected_tube_px * 1.5))),
            min_length_factor=self.V3_MIN_BRANCH_LENGTH_FACTOR,
            max_branches=max_branches,
        )
        return ordered_branches_v2, ordered_branches_v3

    @staticmethod
    def _sample_ordered_coords(coords: np.ndarray, sample_step: int) -> np.ndarray:
        if coords.shape[0] <= 2:
            return coords
        step = max(1, int(sample_step))
        sampled = coords[::step]
        if not np.array_equal(sampled[-1], coords[-1]):
            sampled = np.vstack([sampled, coords[-1]])
        return sampled

    @staticmethod
    def _compute_point_curvatures_px(sampled_coords: np.ndarray) -> np.ndarray:
        if sampled_coords.shape[0] < 3:
            return np.empty((0,), dtype=float)

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
            if np.isfinite(curvature_px) and curvature_px >= 0:
                curvature_values_px.append(curvature_px)

        if not curvature_values_px:
            return np.empty((0,), dtype=float)
        return np.asarray(curvature_values_px, dtype=float)

    @staticmethod
    def _classify_curvature_nm(curvature_nm: float, straight_threshold: float, wavy_threshold: float) -> str:
        if curvature_nm < straight_threshold:
            return "Straight"
        if curvature_nm < wavy_threshold:
            return "Wavy"
        return "Coiled"

    def _classify_v2_curvature_nm(self, curvature_nm: float) -> str:
        return self._classify_curvature_nm(curvature_nm, straight_threshold=5e-4, wavy_threshold=2.5e-3)

    def _classify_v3_curvature_nm(self, curvature_nm: float) -> str:
        return self._classify_curvature_nm(curvature_nm, straight_threshold=8e-4, wavy_threshold=4e-3)

    @staticmethod
    def _trimmed_mean(values: np.ndarray, trim_percent: float) -> float:
        values = np.asarray(values, dtype=float)
        if values.size == 0:
            return 0.0
        if values.size < 5:
            return float(np.mean(values))
        low, high = np.percentile(values, [trim_percent, 100.0 - trim_percent])
        trimmed = values[(values >= low) & (values <= high)]
        return float(np.mean(trimmed if trimmed.size > 0 else values))

    def _prepare_curvature_v3_branches(
        self,
        skel: np.ndarray,
        apply_branch_cleanup: bool = True,
    ):
        branch_source = self._clean_branch_skeleton(skel)["cleaned_skeleton"] if apply_branch_cleanup else (skel > 0)
        max_points_per_branch = self.FAST_MAX_POINTS_PER_BRANCH if self.speed_profile == "fast" else None
        max_branches = self.FAST_CURVATURE_BRANCH_LIMIT if self.speed_profile == "fast" else None
        return self._collect_ordered_branches_v2(
            branch_source,
            min_points=max(self.V3_MIN_BRANCH_POINTS, int(round(self.expected_tube_px * 1.5))),
            min_length_factor=self.V3_MIN_BRANCH_LENGTH_FACTOR,
            max_branches=max_branches,
            max_points_per_branch=max_points_per_branch,
        )

    def _cache_branch_curvature_stats_nm(self, branches, sample_step: int = 1):
        px_per_nm = max(self.px_per_um / 1000.0, 1e-6)
        sample_step = max(1, int(sample_step))
        for branch in branches:
            if (
                branch.get("_curvature_stats_nm") is not None
                and int(branch.get("_curvature_stats_sample_step", 0)) == sample_step
            ):
                continue
            coords = np.asarray(branch["coords"], dtype=float)
            sampled_coords = self._sample_ordered_coords(coords, sample_step=sample_step)
            point_curvature_px = self._compute_point_curvatures_px(sampled_coords)
            curvature_nm = point_curvature_px * px_per_nm if point_curvature_px.size > 0 else np.empty((0,), dtype=float)
            branch["_curvature_distribution_nm"] = curvature_nm
            if curvature_nm.size > 0:
                branch["_curvature_stats_nm"] = {
                    "p50": float(np.median(curvature_nm)),
                    "p75": float(np.percentile(curvature_nm, self.V3_BRANCH_QUANTILE)),
                    "mean": float(np.mean(curvature_nm)),
                    "trimmed_mean": self._trimmed_mean(curvature_nm, self.V3_TRIM_PERCENT),
                }
            else:
                branch["_curvature_stats_nm"] = {}
            branch["_curvature_stats_sample_step"] = sample_step
        return branches

    @staticmethod
    def _aggregate_cached_branch_curvature_nm(branches, branch_stat: str, weight_mode: str) -> float:
        branch_curvatures = []
        weights = []
        for branch in branches:
            cached_stats = branch.get("_curvature_stats_nm") or {}
            branch_curvature_nm = float(cached_stats.get(branch_stat, 0.0))
            if branch_curvature_nm <= 0:
                continue
            branch_curvatures.append(branch_curvature_nm)
            path_length_px = float(branch.get("path_length_px", 0.0))
            if weight_mode == "sqrt_length":
                weights.append(np.sqrt(max(path_length_px, 1.0)))
            elif weight_mode == "length":
                weights.append(max(path_length_px, 1.0))
            else:
                raise ValueError(f"Unsupported weight_mode: {weight_mode}")
        if not branch_curvatures:
            return 0.0
        return float(np.average(np.asarray(branch_curvatures, dtype=float), weights=np.asarray(weights, dtype=float)))

    @staticmethod
    def _empty_curvature_v3_bundle():
        return {
            "curvature_v3": "Unknown",
            "curvature_nm_v3": 0.0,
            "curvature_nm_v3_sqrt_length": 0.0,
            "curvature_nm_v3_length": 0.0,
            "curvature_nm_v3_p50_sqrt_length": 0.0,
            "curvature_nm_v3_p50_length": 0.0,
            "curvature_nm_v3_p75_sqrt_length": 0.0,
            "curvature_nm_v3_p75_length": 0.0,
            "curvature_nm_v3_mean_sqrt_length": 0.0,
            "curvature_nm_v3_mean_length": 0.0,
            "curvature_nm_v3_trimmed_mean_sqrt_length": 0.0,
            "curvature_nm_v3_trimmed_mean_length": 0.0,
            "curvature_v2": "Unknown",
            "curvature_nm_v2": 0.0,
            "curvature_v3_branch_count": 0,
        }

    def calculate_curvature_v3_bundle(
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
                self._clean_branch_skeleton(skel)["cleaned_skeleton"],
                min_points=max(self.V3_MIN_BRANCH_POINTS, int(round(self.expected_tube_px * 1.5))),
                min_length_factor=self.V3_MIN_BRANCH_LENGTH_FACTOR,
                max_branches=max_branches,
                max_points_per_branch=max_points_per_branch,
            )
        if not branches:
            return self._empty_curvature_v3_bundle()

        self._cache_branch_curvature_stats_nm(branches, sample_step=sample_step)
        bundle = self._empty_curvature_v3_bundle()
        bundle["curvature_v3_branch_count"] = int(len(branches))

        stat_to_key = {
            "p50": "p50",
            "p75": "p75",
            "mean": "mean",
            "trimmed_mean": "trimmed_mean",
        }
        for stat_name, stat_suffix in stat_to_key.items():
            bundle[f"curvature_nm_v3_{stat_suffix}_length"] = self._aggregate_cached_branch_curvature_nm(
                branches,
                branch_stat=stat_name,
                weight_mode="length",
            )
            bundle[f"curvature_nm_v3_{stat_suffix}_sqrt_length"] = self._aggregate_cached_branch_curvature_nm(
                branches,
                branch_stat=stat_name,
                weight_mode="sqrt_length",
            )

        bundle["curvature_nm_v3_sqrt_length"] = float(bundle["curvature_nm_v3_p75_sqrt_length"])
        bundle["curvature_nm_v3_length"] = float(bundle["curvature_nm_v3_p75_length"])
        bundle["curvature_nm_v3"] = float(bundle["curvature_nm_v3_p75_sqrt_length"])
        bundle["curvature_v3"] = self._classify_v3_curvature_nm(bundle["curvature_nm_v3"])
        bundle["curvature_nm_v2"] = float(bundle["curvature_nm_v3_p50_length"])
        bundle["curvature_v2"] = self._classify_v2_curvature_nm(bundle["curvature_nm_v2"])
        return bundle

    @staticmethod
    def _coords_to_pixel_indices(coords: np.ndarray, image_shape) -> np.ndarray:
        if coords is None:
            return np.zeros((0, 2), dtype=int)
        coords = np.asarray(coords, dtype=float)
        if coords.size == 0:
            return np.zeros((0, 2), dtype=int)

        height, width = int(image_shape[0]), int(image_shape[1])
        if height <= 0 or width <= 0:
            return np.zeros((0, 2), dtype=int)

        y_coords = np.clip(np.rint(coords[:, 0]).astype(int), 0, height - 1)
        x_coords = np.clip(np.rint(coords[:, 1]).astype(int), 0, width - 1)
        pixels = np.column_stack([y_coords, x_coords])
        if pixels.shape[0] <= 1:
            return pixels
        return np.unique(pixels, axis=0)

    @staticmethod
    def _sample_map_values(coords: np.ndarray, value_map: np.ndarray) -> np.ndarray:
        if value_map is None:
            return np.empty((0,), dtype=float)
        pixel_indices = FeatureExtractor._coords_to_pixel_indices(coords, value_map.shape)
        if pixel_indices.size == 0:
            return np.empty((0,), dtype=float)
        return value_map[pixel_indices[:, 0], pixel_indices[:, 1]].astype(float)

    @staticmethod
    def _candidate_border_distance(coords: np.ndarray, image_shape) -> float:
        pixel_indices = FeatureExtractor._coords_to_pixel_indices(coords, image_shape)
        if pixel_indices.size == 0:
            return 0.0

        height, width = int(image_shape[0]), int(image_shape[1])
        distances = np.minimum.reduce(
            [
                pixel_indices[:, 0],
                pixel_indices[:, 1],
                (height - 1) - pixel_indices[:, 0],
                (width - 1) - pixel_indices[:, 1],
            ]
        )
        return float(np.min(distances))

    @staticmethod
    def _candidate_nearest_junction_distance(
        coords: np.ndarray,
        junction_distance_map: np.ndarray,
    ) -> float:
        values = FeatureExtractor._sample_map_values(coords, junction_distance_map)
        if values.size == 0:
            return 0.0
        return float(np.min(values))

    @staticmethod
    def _candidate_width_stats(coords: np.ndarray, distance_map: np.ndarray):
        sampled_radii = FeatureExtractor._sample_map_values(coords, distance_map)
        if sampled_radii.size == 0:
            return None

        valid_radii = sampled_radii[sampled_radii > 0]
        if valid_radii.size == 0:
            return None

        widths_px = valid_radii * 2.0
        mean_width_px = float(np.mean(widths_px))
        width_std_px = float(np.std(widths_px))
        valid_fraction = float(valid_radii.size / max(sampled_radii.size, 1))
        return {
            "mean_width_px": mean_width_px,
            "width_std_px": width_std_px,
            "width_cv": float(width_std_px / max(mean_width_px, 1e-6)),
            "valid_width_fraction": valid_fraction,
        }

    @staticmethod
    def _candidate_pixel_set(coords: np.ndarray, image_shape):
        pixel_indices = FeatureExtractor._coords_to_pixel_indices(coords, image_shape)
        return {tuple(int(v) for v in pixel) for pixel in pixel_indices}

    @staticmethod
    def _candidate_overlap_ratio(candidate_a, candidate_b) -> float:
        pixels_a = candidate_a.get("_pixel_set")
        pixels_b = candidate_b.get("_pixel_set")
        if pixels_a is None:
            pixels_a = FeatureExtractor._candidate_pixel_set(
                candidate_a.get("coords"),
                candidate_a.get("image_shape"),
            )
        if pixels_b is None:
            pixels_b = FeatureExtractor._candidate_pixel_set(
                candidate_b.get("coords"),
                candidate_b.get("image_shape"),
            )
        if not pixels_a or not pixels_b:
            return 0.0
        overlap = len(pixels_a & pixels_b)
        return float(overlap / max(1, min(len(pixels_a), len(pixels_b))))

    def _score_kappa_lite_branch_candidate(
        self,
        branch,
        image_shape,
        junction_distance_map: np.ndarray,
        distance_map: np.ndarray,
        border_margin_px: float = None,
        junction_margin_px: float = None,
    ):
        coords = branch.get("coords")
        if coords is None or np.asarray(coords).size == 0:
            return None

        if border_margin_px is None:
            border_margin_px = max(4.0, self.expected_tube_px * 2.0)
        if junction_margin_px is None:
            junction_margin_px = max(3.0, self.expected_tube_px * 1.5)

        border_distance_px = self._candidate_border_distance(coords, image_shape)
        if border_distance_px < float(border_margin_px):
            return None

        junction_distance_px = self._candidate_nearest_junction_distance(coords, junction_distance_map)
        if junction_distance_px < float(junction_margin_px):
            return None

        width_stats = self._candidate_width_stats(coords, distance_map)
        if width_stats is None:
            return None
        if width_stats["valid_width_fraction"] < 0.6 or width_stats["mean_width_px"] <= 0:
            return None

        length_scale = max(20.0, self.expected_tube_px * 10.0)
        junction_scale = max(6.0, self.expected_tube_px * 2.5)

        length_score = float(np.tanh(float(branch.get("path_length_px", 0.0)) / max(length_scale, 1e-6)))
        junction_distance_score = float(np.tanh(junction_distance_px / max(junction_scale, 1e-6)))
        width_consistency_score = float(1.0 / (1.0 + width_stats["width_cv"]))
        segment_score = float(
            0.5 * length_score +
            0.3 * junction_distance_score +
            0.2 * width_consistency_score
        )

        candidate = dict(branch)
        candidate.update(
            {
                "image_shape": tuple(int(v) for v in image_shape[:2]),
                "score": segment_score,
                "length_score": length_score,
                "junction_distance_score": junction_distance_score,
                "width_consistency_score": width_consistency_score,
                "border_distance_px": float(border_distance_px),
                "junction_distance_px": float(junction_distance_px),
                "mean_width_px": float(width_stats["mean_width_px"]),
                "width_cv": float(width_stats["width_cv"]),
                "_width_stats": width_stats,
                "_pixel_set": self._candidate_pixel_set(coords, image_shape),
            }
        )
        return candidate

    def _select_kappa_lite_segments(
        self,
        scored_candidates,
        top_k: int = 10,
        max_overlap_ratio: float = 0.35,
    ):
        ranked = sorted(
            scored_candidates,
            key=lambda item: (item.get("score", 0.0), item.get("path_length_px", 0.0)),
            reverse=True,
        )

        selected = []
        for candidate in ranked:
            overlap_ratio = 0.0
            if selected:
                overlap_ratio = max(
                    self._candidate_overlap_ratio(candidate, existing)
                    for existing in selected
                )
            if overlap_ratio >= max_overlap_ratio:
                continue

            record = dict(candidate)
            record["segment_id"] = len(selected) + 1
            record["suppressed_overlap_ratio"] = float(overlap_ratio)
            selected.append(record)
            if len(selected) >= max(1, int(top_k)):
                break
        return selected

    def _compute_kappa_lite_segment_metrics(
        self,
        segment,
        distance_map: np.ndarray,
        sample_step: int = None,
    ) -> dict:
        coords = np.asarray(segment.get("coords"), dtype=float)
        path_length_px = float(segment.get("path_length_px", self._path_length(coords)))
        span_px = 0.0
        if coords.shape[0] >= 2:
            span_px = float(np.linalg.norm(coords[-1] - coords[0]))
        ld_ratio = float(path_length_px / max(span_px, 1e-6)) if path_length_px > 0 else 0.0

        if sample_step is None:
            sample_step = 2 if self.speed_profile == "fast" else 1
        sampled_coords = self._sample_ordered_coords(coords, sample_step=max(1, int(sample_step)))
        curvature_values_px = self._compute_point_curvatures_px(sampled_coords)

        px_per_nm = max(self.px_per_um / 1000.0, 1e-6)
        width_stats = segment.get("_width_stats")
        if width_stats is None:
            width_stats = self._candidate_width_stats(coords, distance_map)

        mean_width_px = float(width_stats["mean_width_px"]) if width_stats is not None else 0.0
        width_cv = float(width_stats["width_cv"]) if width_stats is not None else 0.0
        mean_curvature_nm = 0.0
        p90_curvature_nm = 0.0
        if curvature_values_px.size > 0:
            mean_curvature_nm = float(np.mean(curvature_values_px) * px_per_nm)
            p90_curvature_nm = float(np.percentile(curvature_values_px, 90) * px_per_nm)

        return {
            "segment_id": int(segment.get("segment_id", 0)),
            "score": float(segment.get("score", 0.0)),
            "length_score": float(segment.get("length_score", 0.0)),
            "junction_distance_score": float(segment.get("junction_distance_score", 0.0)),
            "width_consistency_score": float(segment.get("width_consistency_score", 0.0)),
            "path_length_px": float(path_length_px),
            "path_length_nm": float(path_length_px / px_per_nm),
            "span_px": float(span_px),
            "span_nm": float(span_px / px_per_nm),
            "ld_ratio": float(ld_ratio),
            "mean_curvature_nm": mean_curvature_nm,
            "p90_curvature_nm": p90_curvature_nm,
            "mean_width_nm": float(mean_width_px / px_per_nm),
            "width_cv": width_cv,
            "border_distance_px": float(segment.get("border_distance_px", 0.0)),
            "junction_distance_px": float(segment.get("junction_distance_px", 0.0)),
            "n_points": int(coords.shape[0]),
            "coords": coords,
        }

    def extract_kappa_lite_segments(
        self,
        img_gray: np.ndarray,
        external_binary_mask: np.ndarray = None,
        top_k: int = 10,
        max_branches: int = None,
        max_points_per_branch: int = None,
    ) -> dict:
        roi = self.extract_roi(img_gray)
        self._calibrate(roi.shape[1])
        processed = self.preprocess(roi)

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
        else:
            density, thresh = self.calculate_density(processed)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        closed_mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        skeleton = skeletonize(closed_mask > 0)
        distance_map = cv2.distanceTransform(closed_mask, cv2.DIST_L2, 5)

        if max_branches is None and self.speed_profile == "fast":
            max_branches = self.FAST_CURVATURE_BRANCH_LIMIT
        if max_points_per_branch is None and self.speed_profile == "fast":
            max_points_per_branch = self.FAST_MAX_POINTS_PER_BRANCH

        ordered_branches = self._collect_ordered_branches_v2(
            skeleton,
            min_points=15,
            max_branches=max_branches,
            max_points_per_branch=max_points_per_branch,
        )

        skeleton_mask = (skeleton > 0).astype(np.uint8)
        neighbor_count = self._neighbor_count_map(skeleton_mask)
        junction_mask = (skeleton_mask > 0) & (neighbor_count >= 3)
        if np.any(junction_mask):
            junction_distance_map = cv2.distanceTransform(
                np.logical_not(junction_mask).astype(np.uint8),
                cv2.DIST_L2,
                5,
            )
        else:
            junction_distance_map = np.full(
                skeleton.shape,
                float(max(skeleton.shape)) if skeleton.size else 0.0,
                dtype=np.float32,
            )

        candidate_segments = []
        for branch in ordered_branches:
            scored = self._score_kappa_lite_branch_candidate(
                branch,
                image_shape=roi.shape,
                junction_distance_map=junction_distance_map,
                distance_map=distance_map,
            )
            if scored is not None:
                candidate_segments.append(scored)

        selected_candidates = self._select_kappa_lite_segments(
            candidate_segments,
            top_k=top_k,
        )
        selected_segments = [
            self._compute_kappa_lite_segment_metrics(segment, distance_map=distance_map)
            for segment in selected_candidates
        ]

        return {
            "roi": roi,
            "processed": processed,
            "mask": thresh,
            "closed_mask": closed_mask,
            "skeleton": skeleton,
            "junction_mask": junction_mask,
            "ordered_branches": ordered_branches,
            "candidate_segments": candidate_segments,
            "selected_segments": selected_segments,
            "density": float(density),
            "px_per_um": float(self.px_per_um),
            "magnification": int(self.mag) if self.mag else None,
            "top_k": int(top_k),
        }

    @staticmethod
    def _normalize_vector(vector: np.ndarray) -> np.ndarray:
        vec = np.asarray(vector, dtype=float)
        norm = float(np.linalg.norm(vec))
        if norm <= 1e-8:
            return np.zeros((2,), dtype=float)
        return vec / norm

    def _estimate_endpoint_tangent(
        self,
        coords: np.ndarray,
        at_start: bool,
        lookahead_points: int = 6,
    ) -> np.ndarray:
        coords = np.asarray(coords, dtype=float)
        if coords.shape[0] < 2:
            return np.zeros((2,), dtype=float)

        step = max(1, min(int(lookahead_points), coords.shape[0] - 1))
        if at_start:
            vec = coords[step] - coords[0]
        else:
            vec = coords[-step - 1] - coords[-1]
        return self._normalize_vector(vec)

    def _branch_intensity_stats(self, coords: np.ndarray, image_map: np.ndarray):
        values = self._sample_map_values(coords, image_map)
        if values.size == 0:
            return {"mean": 0.0, "std": 0.0}
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }

    def _build_branch_graph_v1(
        self,
        skeleton: np.ndarray,
        processed: np.ndarray,
        width_distance_map: np.ndarray,
        min_length_factor: float = 2.0,
        min_points: int = None,
        junction_link_radius: int = 4,
    ) -> dict:
        skeleton_mask = (skeleton > 0).astype(np.uint8)
        if min_points is None:
            min_points = max(self.V3_MIN_BRANCH_POINTS, int(round(self.expected_tube_px * 1.5)))

        branches = self._collect_ordered_branches_v2(
            skeleton,
            min_points=min_points,
            min_length_factor=min_length_factor,
        )
        neighbor_count = self._neighbor_count_map(skeleton_mask)
        junction_mask = (skeleton_mask > 0) & (neighbor_count >= 3)
        endpoint_mask = (skeleton_mask > 0) & (neighbor_count <= 1)

        junction_labels = label(junction_mask.astype(np.uint8), connectivity=2)
        if junction_link_radius > 0 and np.any(junction_mask):
            dilated = cv2.dilate(
                junction_labels.astype(np.uint16),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (junction_link_radius * 2 + 1, junction_link_radius * 2 + 1)),
            )
        else:
            dilated = junction_labels

        nodes: dict[str, dict] = {}
        edge_records: dict[str, dict] = {}

        def ensure_node(node_id: str, kind: str, coord: np.ndarray):
            if node_id not in nodes:
                nodes[node_id] = {
                    "node_id": node_id,
                    "kind": kind,
                    "coord": np.asarray(coord, dtype=float),
                    "branch_ids": [],
                }
            else:
                nodes[node_id]["coord"] = 0.5 * (nodes[node_id]["coord"] + np.asarray(coord, dtype=float))
            return nodes[node_id]

        for branch_idx, branch in enumerate(branches, start=1):
            coords = np.asarray(branch["coords"], dtype=float)
            start_px = self._coords_to_pixel_indices(coords[:1], skeleton.shape)
            end_px = self._coords_to_pixel_indices(coords[-1:], skeleton.shape)
            if start_px.size == 0 or end_px.size == 0:
                continue
            start_y, start_x = [int(v) for v in start_px[0]]
            end_y, end_x = [int(v) for v in end_px[0]]

            start_junction = int(dilated[start_y, start_x]) if dilated.size else 0
            end_junction = int(dilated[end_y, end_x]) if dilated.size else 0
            start_node_id = f"junction_{start_junction}" if start_junction > 0 else f"endpoint_{branch_idx}_start"
            end_node_id = f"junction_{end_junction}" if end_junction > 0 else f"endpoint_{branch_idx}_end"

            ensure_node(start_node_id, "junction" if start_junction > 0 else "endpoint", coords[0])
            ensure_node(end_node_id, "junction" if end_junction > 0 else "endpoint", coords[-1])

            width_stats = self._candidate_width_stats(coords, width_distance_map)
            intensity_stats = self._branch_intensity_stats(coords, processed)
            branch_id = f"branch_{branch_idx}"
            edge_records[branch_id] = {
                "branch_id": branch_id,
                "coords": coords,
                "node_start": start_node_id,
                "node_end": end_node_id,
                "length_px": float(branch.get("path_length_px", self._path_length(coords))),
                "start_tangent": self._estimate_endpoint_tangent(coords, at_start=True),
                "end_tangent": self._estimate_endpoint_tangent(coords, at_start=False),
                "mean_width_px": float(width_stats["mean_width_px"]) if width_stats is not None else 0.0,
                "width_cv": float(width_stats["width_cv"]) if width_stats is not None else 0.0,
                "intensity_mean": float(intensity_stats["mean"]),
                "intensity_std": float(intensity_stats["std"]),
            }
            nodes[start_node_id]["branch_ids"].append(branch_id)
            nodes[end_node_id]["branch_ids"].append(branch_id)

        for node in nodes.values():
            node["degree"] = len(node["branch_ids"])

        return {
            "nodes": nodes,
            "branches": edge_records,
            "junction_mask": junction_mask,
            "endpoint_mask": endpoint_mask,
            "distance_map": width_distance_map,
            "processed": processed,
        }

    def _branch_orientation_from_node(self, branch: dict, node_id: str) -> dict:
        if node_id == branch["node_start"]:
            return {
                "node_to_interior": np.asarray(branch["start_tangent"], dtype=float),
                "coords_forward": np.asarray(branch["coords"], dtype=float),
                "other_node": branch["node_end"],
                "arrive_outward": np.asarray(branch["end_tangent"], dtype=float),
            }
        return {
            "node_to_interior": np.asarray(branch["end_tangent"], dtype=float),
            "coords_forward": np.asarray(branch["coords"][::-1], dtype=float),
            "other_node": branch["node_start"],
            "arrive_outward": np.asarray(branch["start_tangent"], dtype=float),
        }

    def _score_branch_graph_transition_v1(
        self,
        current_branch: dict,
        current_node_id: str,
        candidate_branch: dict,
        angle_soft_deg: float = 45.0,
        angle_hard_deg: float = 70.0,
    ):
        current_oriented = self._branch_orientation_from_node(current_branch, current_node_id)
        candidate_oriented = self._branch_orientation_from_node(candidate_branch, current_node_id)

        prev_out = self._normalize_vector(current_oriented["node_to_interior"])
        cand_out = self._normalize_vector(candidate_oriented["node_to_interior"])
        if np.linalg.norm(prev_out) <= 1e-8 or np.linalg.norm(cand_out) <= 1e-8:
            return None

        straight_similarity = float(np.clip(-np.dot(prev_out, cand_out), -1.0, 1.0))
        turning_angle_deg = float(np.degrees(np.arccos(straight_similarity)))
        if turning_angle_deg > angle_hard_deg:
            return None

        if turning_angle_deg <= angle_soft_deg:
            angle_cost = turning_angle_deg / max(angle_soft_deg, 1e-6)
        else:
            angle_cost = 1.0 + (turning_angle_deg - angle_soft_deg) / max(angle_hard_deg - angle_soft_deg, 1e-6)
        width_cost = abs(float(current_branch.get("mean_width_px", 0.0)) - float(candidate_branch.get("mean_width_px", 0.0))) / max(
            max(float(current_branch.get("mean_width_px", 0.0)), float(candidate_branch.get("mean_width_px", 0.0)), 1e-6),
            1e-6,
        )
        intensity_cost = abs(float(current_branch.get("intensity_mean", 0.0)) - float(candidate_branch.get("intensity_mean", 0.0))) / 255.0
        short_penalty = 1.0 / max(float(candidate_branch.get("length_px", 0.0)), 1.0)
        extension_bonus = min(float(candidate_branch.get("length_px", 0.0)) / max(self.expected_tube_px * 20.0, 1e-6), 1.0)
        total_cost = float(0.62 * angle_cost + 0.14 * width_cost + 0.09 * intensity_cost + 0.04 * short_penalty - 0.11 * extension_bonus)
        return {
            "total_cost": total_cost,
            "turning_angle_deg": turning_angle_deg,
            "angle_cost": float(angle_cost),
            "width_cost": float(width_cost),
            "intensity_cost": float(intensity_cost),
            "short_penalty": float(short_penalty),
            "extension_bonus": float(extension_bonus),
            "candidate_other_node": candidate_oriented["other_node"],
            "candidate_coords_forward": candidate_oriented["coords_forward"],
        }

    def _compute_reconstructed_path_metrics_v1(
        self,
        path_id: int,
        path_coords: np.ndarray,
        branch_ids: list[str],
        transition_costs: list[float],
        distance_map: np.ndarray,
    ) -> dict:
        coords = np.asarray(path_coords, dtype=float)
        path_length_px = self._path_length(coords)
        span_px = float(np.linalg.norm(coords[-1] - coords[0])) if coords.shape[0] >= 2 else 0.0
        ld_ratio = float(path_length_px / max(span_px, 1e-6)) if path_length_px > 0 else 0.0
        sample_step = 2 if self.speed_profile == "fast" else 1
        sampled = self._sample_ordered_coords(coords, sample_step=sample_step)
        curvature_values_px = self._compute_point_curvatures_px(sampled)
        px_per_nm = max(self.px_per_um / 1000.0, 1e-6)
        width_stats = self._candidate_width_stats(coords, distance_map)
        mean_curvature_nm = float(np.mean(curvature_values_px) * px_per_nm) if curvature_values_px.size > 0 else 0.0
        p90_curvature_nm = float(np.percentile(curvature_values_px, 90) * px_per_nm) if curvature_values_px.size > 0 else 0.0
        mean_width_nm = float(width_stats["mean_width_px"] / px_per_nm) if width_stats is not None else 0.0
        total_turning_angle_deg = float(np.sum(np.abs(np.diff(np.unwrap(np.arctan2(np.diff(coords[:, 0], prepend=coords[0, 0]), np.diff(coords[:, 1], prepend=coords[0, 1]))))))) if coords.shape[0] >= 3 else 0.0
        confidence = float(1.0 / (1.0 + np.mean(transition_costs))) if transition_costs else 1.0
        return {
            "path_id": int(path_id),
            "branch_ids": list(branch_ids),
            "coords": coords,
            "confidence": confidence,
            "path_length_px": float(path_length_px),
            "path_length_nm": float(path_length_px / px_per_nm),
            "span_px": float(span_px),
            "span_nm": float(span_px / px_per_nm),
            "ld_ratio": float(ld_ratio),
            "mean_curvature_nm": mean_curvature_nm,
            "p90_curvature_nm": p90_curvature_nm,
            "total_turning_angle_deg": float(np.degrees(total_turning_angle_deg)),
            "mean_width_nm": mean_width_nm,
        }

    def trace_branch_graph_paths_v1(
        self,
        skeleton: np.ndarray,
        processed: np.ndarray,
        width_distance_map: np.ndarray,
        min_length_factor: float = 2.0,
        angle_limit_deg: float = 45.0,
        angle_hard_deg: float = 70.0,
        beam_width: int = 2,
        max_paths: int = 40,
    ) -> dict:
        graph = self._build_branch_graph_v1(
            skeleton,
            processed=processed,
            width_distance_map=width_distance_map,
            min_length_factor=min_length_factor,
            junction_link_radius=4,
        )
        nodes = graph["nodes"]
        branches = graph["branches"]
        distance_map = graph["distance_map"]

        visited_branches: set[str] = set()
        reconstructed_paths = []

        def beam_rank(state: dict) -> float:
            total_cost = float(sum(state["transition_costs"]))
            total_length = float(sum(branches[branch_id]["length_px"] for branch_id in state["branch_ids"]))
            return total_length / max(1.0 + total_cost, 1e-6)

        def endpoint_start_order():
            endpoint_nodes = [node for node in nodes.values() if node["kind"] == "endpoint"]
            endpoint_nodes.sort(key=lambda node: max((branches[bid]["length_px"] for bid in node["branch_ids"]), default=0.0), reverse=True)
            for node in endpoint_nodes:
                for branch_id in node["branch_ids"]:
                    if branch_id not in visited_branches:
                        yield node["node_id"], branch_id
            remaining = sorted(
                (branch for branch in branches.values() if branch["branch_id"] not in visited_branches),
                key=lambda item: item["length_px"],
                reverse=True,
            )
            for branch in remaining:
                yield branch["node_start"], branch["branch_id"]

        for start_node_id, start_branch_id in endpoint_start_order():
            if start_branch_id in visited_branches:
                continue
            branch = branches[start_branch_id]
            oriented = self._branch_orientation_from_node(branch, start_node_id)
            beams = [
                {
                    "branch_ids": [start_branch_id],
                    "used_branches": {start_branch_id},
                    "coords": np.asarray(oriented["coords_forward"], dtype=float),
                    "transition_costs": [],
                    "current_branch_id": start_branch_id,
                    "current_node_id": oriented["other_node"],
                }
            ]

            while True:
                expanded = []
                all_stopped = True
                for state in beams:
                    current_node_id = state["current_node_id"]
                    current_branch = branches[state["current_branch_id"]]
                    if current_node_id not in nodes or nodes[current_node_id]["kind"] != "junction":
                        expanded.append(state)
                        continue

                    candidate_ids = [
                        bid for bid in nodes[current_node_id]["branch_ids"]
                        if bid != current_branch["branch_id"]
                        and bid not in state["used_branches"]
                        and bid not in visited_branches
                    ]
                    if not candidate_ids:
                        expanded.append(state)
                        continue

                    scored_candidates = []
                    for candidate_id in candidate_ids:
                        scored = self._score_branch_graph_transition_v1(
                            current_branch=current_branch,
                            current_node_id=current_node_id,
                            candidate_branch=branches[candidate_id],
                            angle_soft_deg=angle_limit_deg,
                            angle_hard_deg=angle_hard_deg,
                        )
                        if scored is not None:
                            scored_candidates.append((candidate_id, scored))

                    if not scored_candidates:
                        expanded.append(state)
                        continue

                    all_stopped = False
                    scored_candidates.sort(key=lambda item: item[1]["total_cost"])
                    for candidate_id, best in scored_candidates[: max(1, int(beam_width))]:
                        candidate_branch = branches[candidate_id]
                        next_coords = np.asarray(best["candidate_coords_forward"], dtype=float)
                        new_coords = state["coords"]
                        if next_coords.shape[0] > 1:
                            new_coords = np.vstack([state["coords"], next_coords[1:]])
                        expanded.append(
                            {
                                "branch_ids": state["branch_ids"] + [candidate_id],
                                "used_branches": set(state["used_branches"]) | {candidate_id},
                                "coords": new_coords,
                                "transition_costs": state["transition_costs"] + [float(best["total_cost"])],
                                "current_branch_id": candidate_id,
                                "current_node_id": best["candidate_other_node"],
                            }
                        )

                expanded.sort(key=beam_rank, reverse=True)
                beams = expanded[: max(1, int(beam_width))]
                if all_stopped:
                    break

            if not beams:
                continue
            best_state = max(beams, key=beam_rank)
            visited_branches.update(best_state["branch_ids"])
            reconstructed_paths.append(
                self._compute_reconstructed_path_metrics_v1(
                    path_id=len(reconstructed_paths) + 1,
                    path_coords=best_state["coords"],
                    branch_ids=best_state["branch_ids"],
                    transition_costs=best_state["transition_costs"],
                    distance_map=distance_map,
                )
            )
            if len(reconstructed_paths) >= max(1, int(max_paths)):
                break

        reconstructed_paths.sort(key=lambda item: (item["confidence"], item["path_length_px"]), reverse=True)
        for idx, path in enumerate(reconstructed_paths, start=1):
            path["path_id"] = idx
        graph["reconstructed_paths"] = reconstructed_paths
        return graph

    def extract_branch_graph_paths_v1(
        self,
        img_gray: np.ndarray,
        external_binary_mask: np.ndarray = None,
        min_length_factor: float = 2.0,
        angle_limit_deg: float = 45.0,
        angle_hard_deg: float = 70.0,
        beam_width: int = 2,
        max_paths: int = 40,
    ) -> dict:
        roi = self.extract_roi(img_gray)
        self._calibrate(roi.shape[1])
        processed = self.preprocess(roi)

        if external_binary_mask is not None:
            mask = np.asarray(external_binary_mask)
            if mask.shape != roi.shape:
                mask = cv2.resize(mask.astype(np.uint8), (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)
            thresh = (mask > 0).astype(np.uint8) * 255
        else:
            _, thresh = self.calculate_density(processed)

        _, skeleton = self.calculate_diameter(thresh)
        width_distance_map = cv2.distanceTransform((thresh > 0).astype(np.uint8), cv2.DIST_L2, 5)
        graph = self.trace_branch_graph_paths_v1(
            skeleton=skeleton,
            processed=processed,
            width_distance_map=width_distance_map,
            min_length_factor=min_length_factor,
            angle_limit_deg=angle_limit_deg,
            angle_hard_deg=angle_hard_deg,
            beam_width=beam_width,
            max_paths=max_paths,
        )
        graph.update(
            {
                "roi": roi,
                "processed": processed,
                "mask": thresh,
                "skeleton": skeleton,
                "px_per_um": float(self.px_per_um),
                "magnification": int(self.mag) if self.mag else None,
                "angle_limit_deg": float(angle_limit_deg),
                "angle_hard_deg": float(angle_hard_deg),
                "beam_width": int(beam_width),
                "min_length_factor": float(min_length_factor),
            }
        )
        return graph

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
                px_per_nm = max(self.px_per_um / 1000.0, 1e-6)
                curvature_nm = median_curv_px * px_per_nm
                all_curvatures.append(curvature_nm)

        if not all_curvatures:
            return "Unknown", 0.0

        median_curvature = float(np.median(all_curvatures))
        straight_threshold_nm = self.LEGACY_STRAIGHT_CURVATURE_PX * max(self.px_per_um / 1000.0, 1e-6)
        wavy_threshold_nm = self.LEGACY_WAVY_CURVATURE_PX * max(self.px_per_um / 1000.0, 1e-6)
        if median_curvature < straight_threshold_nm:
            label = "Straight"
        elif median_curvature < wavy_threshold_nm:
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
            branch_source = self._clean_branch_skeleton(skel)["cleaned_skeleton"]
            branches = self._collect_ordered_branches_v2(
                branch_source,
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
            curvature_values_px = self._compute_point_curvatures_px(sampled_coords)

            if curvature_values_px.size > 0:
                branch_curvatures.append(float(np.median(curvature_values_px)) * px_per_nm)
                weights.append(max(branch["path_length_px"], 1.0))

        if not branch_curvatures:
            return "Unknown", 0.0

        median_curvature_nm = float(np.average(branch_curvatures, weights=np.asarray(weights, dtype=float)))
        label_name = self._classify_v2_curvature_nm(median_curvature_nm)
        return label_name, median_curvature_nm

    def calculate_curvature_v3(
        self,
        skel: np.ndarray,
        max_branches: int = None,
        max_points_per_branch: int = None,
        sample_step: int = None,
        ordered_branches=None,
    ):
        bundle = self.calculate_curvature_v3_bundle(
            skel,
            max_branches=max_branches,
            max_points_per_branch=max_points_per_branch,
            sample_step=sample_step,
            ordered_branches=ordered_branches,
        )
        return bundle["curvature_v3"], float(bundle["curvature_nm_v3"])

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
            branch_source = self._clean_branch_skeleton(skel)["cleaned_skeleton"]
            branches = self._collect_ordered_branches_v2(
                branch_source,
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
        branch_cleanup = self._clean_branch_skeleton(skel)
        cleaned_branch_skeleton = branch_cleanup["cleaned_skeleton"]

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
        ordered_branches_v3 = self._prepare_curvature_v3_branches(
            cleaned_branch_skeleton,
            apply_branch_cleanup=False,
        )
        curvature_v3_bundle = self.calculate_curvature_v3_bundle(
            cleaned_branch_skeleton,
            ordered_branches=ordered_branches_v3,
        )
        curv_label = curvature_v3_bundle["curvature_v3"]
        curvature_nm = curvature_v3_bundle["curvature_nm_v3"]
        curv_label_v2 = curvature_v3_bundle["curvature_v2"]
        curvature_nm_v2 = curvature_v3_bundle["curvature_nm_v2"]
        curv_label_v3 = curvature_v3_bundle["curvature_v3"]
        curvature_nm_v3 = curvature_v3_bundle["curvature_nm_v3"]
        emit_progress(
            "curvature",
            curvature=curv_label,
            curvature_nm=round(curvature_nm, 4),
            curvature_v2=curv_label_v2,
            curvature_nm_v2=round(curvature_nm_v2, 6),
            curvature_v3=curv_label_v3,
            curvature_nm_v3=round(curvature_nm_v3, 6),
            curvature_nm_v3_p50_length=round(curvature_v3_bundle["curvature_nm_v3_p50_length"], 6),
            curvature_nm_v3_p50_sqrt_length=round(curvature_v3_bundle["curvature_nm_v3_p50_sqrt_length"], 6),
            curvature_nm_v3_mean_length=round(curvature_v3_bundle["curvature_nm_v3_mean_length"], 6),
            curvature_nm_v3_mean_sqrt_length=round(curvature_v3_bundle["curvature_nm_v3_mean_sqrt_length"], 6),
            curvature_nm_v3_trimmed_mean_length=round(curvature_v3_bundle["curvature_nm_v3_trimmed_mean_length"], 6),
            curvature_nm_v3_trimmed_mean_sqrt_length=round(curvature_v3_bundle["curvature_nm_v3_trimmed_mean_sqrt_length"], 6),
        )
        waviness_v3 = self.calculate_waviness_v2(
            cleaned_branch_skeleton,
            ordered_branches=ordered_branches_v3,
        )
        waviness = {
            "waviness_ratio": waviness_v3["waviness_ratio_v2"],
            "waviness_height_nm": waviness_v3["waviness_height_nm_v2"],
            "waviness_wavelength_nm": waviness_v3["waviness_wavelength_nm_v2"],
            "waviness_branches": waviness_v3["waviness_branches_v2"],
            "tortuosity": waviness_v3["tortuosity_v2"],
        }
        waviness_v2 = waviness_v3
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
            for key in list(curvature_v3_bundle.keys()):
                if key.startswith("curvature_nm_"):
                    curvature_v3_bundle[key] = 0.0
            curvature_v3_bundle["curvature_v2"] = "N/A"
            curvature_v3_bundle["curvature_v3"] = "N/A"
            curvature_v3_bundle["curvature_v3_branch_count"] = 0
            curv_label_v2, tortuosity_v2 = curvature_v3_bundle["curvature_v2"], None
            curvature_nm_v2 = curvature_v3_bundle["curvature_nm_v2"]
            curv_label_v3 = curvature_v3_bundle["curvature_v3"]
            curvature_nm_v3 = curvature_v3_bundle["curvature_nm_v3"]
            diameter_method_name = "N/A"
            waviness = {
                "waviness_ratio": None,
                "waviness_height_nm": None,
                "waviness_wavelength_nm": None,
                "waviness_branches": 0,
                "tortuosity": None,
            }
            waviness_v2 = {
                "waviness_ratio_v2": None,
                "waviness_height_nm_v2": None,
                "waviness_wavelength_nm_v2": None,
                "waviness_branches_v2": 0,
                "tortuosity_v2": None,
            }

        extra = {
            "n_branches": n_br,
            "diameter_method": diameter_method_name,
            "rotation_correction_deg": alignment_metrics["rotation_correction_deg"],
            "alignment_raw": round(alignment_metrics["alignment_raw"], 4),
            "mean_phi_raw_deg": round(alignment_metrics["mean_phi_raw_deg"], 2),
            "speed_profile": self.speed_profile,
            "branch_cleanup_enabled": bool(branch_cleanup["branch_cleanup_enabled"]),
            "removed_short_component_count": int(branch_cleanup["removed_short_component_count"]),
            "removed_short_pixel_count": int(branch_cleanup["removed_short_pixel_count"]),
            "removed_spur_count": int(branch_cleanup["removed_spur_count"]),
            "removed_spur_pixel_count": int(branch_cleanup["removed_spur_pixel_count"]),
            "isolated_min_length_px": round(branch_cleanup["isolated_min_length_px"], 4),
            "isolated_min_points": int(branch_cleanup["isolated_min_points"]),
            "spur_length_limit_px": round(branch_cleanup["spur_length_limit_px"], 4),
            "spur_iterations": int(branch_cleanup["spur_iterations"]),
        }

        result = {
            # 鍥涗釜涓荤壒寰侊紙alignment 瀛楁瀛?HOF 鍊硷紝鏁版嵁搴撳吋瀹癸級
            "density":      round(density, 2),
            "alignment":    round(hof, 4),
            "diameter":     round(diameter_nm, 2) if diameter_nm >= 0 else None,
            "curvature":    curv_label,
            "curvature_v2": curv_label_v2,
            "curvature_v3": curv_label_v3,
            # 杈呭姪瀛楁锛堣鏂囨姤鍛?/ 璋冭瘯鐢級
            "hof_method":   hof_method,
            "mean_phi_deg": round(mean_phi, 2),
            "curvature_nm": round(curvature_nm, 4),  # 鐪熸鐨勬洸鐜囷紙nm鈦宦癸級
            "curvature_nm_v2": round(curvature_nm_v2, 6),
            "curvature_nm_v3": round(curvature_nm_v3, 6),
            "curvature_nm_v3_sqrt_length": round(curvature_v3_bundle["curvature_nm_v3_sqrt_length"], 6),
            "curvature_nm_v3_length": round(curvature_v3_bundle["curvature_nm_v3_length"], 6),
            "curvature_nm_v3_p50_sqrt_length": round(curvature_v3_bundle["curvature_nm_v3_p50_sqrt_length"], 6),
            "curvature_nm_v3_p50_length": round(curvature_v3_bundle["curvature_nm_v3_p50_length"], 6),
            "curvature_nm_v3_p75_sqrt_length": round(curvature_v3_bundle["curvature_nm_v3_p75_sqrt_length"], 6),
            "curvature_nm_v3_p75_length": round(curvature_v3_bundle["curvature_nm_v3_p75_length"], 6),
            "curvature_nm_v3_mean_sqrt_length": round(curvature_v3_bundle["curvature_nm_v3_mean_sqrt_length"], 6),
            "curvature_nm_v3_mean_length": round(curvature_v3_bundle["curvature_nm_v3_mean_length"], 6),
            "curvature_nm_v3_trimmed_mean_sqrt_length": round(curvature_v3_bundle["curvature_nm_v3_trimmed_mean_sqrt_length"], 6),
            "curvature_nm_v3_trimmed_mean_length": round(curvature_v3_bundle["curvature_nm_v3_trimmed_mean_length"], 6),
            "curvature_v3_branch_count": int(curvature_v3_bundle["curvature_v3_branch_count"]),
            "tortuosity":   round(tortuosity, 3) if tortuosity is not None else None,
            "tortuosity_v2": round(tortuosity_v2, 3) if tortuosity_v2 is not None else None,
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






