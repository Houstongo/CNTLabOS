import unittest

import numpy as np

from src.analysis.feature_extractor import FeatureExtractor


class FeatureExtractorKappaLiteTests(unittest.TestCase):
    def _draw_polyline(self, points, shape):
        coords = []
        prev_y = None
        prev_x = None
        for y, x in points:
            curr_y = int(round(y))
            curr_x = int(round(x))
            if prev_y is None:
                coords.append((curr_y, curr_x))
            else:
                steps = max(abs(curr_y - prev_y), abs(curr_x - prev_x), 1)
                for step in range(steps + 1):
                    interp_y = int(round(prev_y + (curr_y - prev_y) * step / steps))
                    interp_x = int(round(prev_x + (curr_x - prev_x) * step / steps))
                    coords.append((interp_y, interp_x))
            prev_y = curr_y
            prev_x = curr_x
        return np.asarray(coords, dtype=float)

    def test_kappa_lite_scoring_prefers_longer_and_more_stable_branch(self):
        extractor = FeatureExtractor(magnification=100000)
        extractor.px_per_um = 100.0
        extractor.expected_tube_px = 3.0

        shape = (120, 120)
        stable_coords = self._draw_polyline([(50, x) for x in range(15, 95)], shape)
        noisy_points = [(75 + (1 if idx % 2 else -1), x) for idx, x in enumerate(range(20, 60))]
        noisy_coords = self._draw_polyline(noisy_points, shape)

        distance_map = np.full(shape, 2.5, dtype=np.float32)
        noisy_pixels = extractor._coords_to_pixel_indices(noisy_coords, shape)
        alternating_widths = np.where(np.arange(noisy_pixels.shape[0]) % 2 == 0, 1.0, 4.5)
        distance_map[noisy_pixels[:, 0], noisy_pixels[:, 1]] = alternating_widths
        junction_distance_map = np.full(shape, 20.0, dtype=np.float32)

        stable_branch = {
            "coords": stable_coords,
            "path_length_px": extractor._path_length(stable_coords),
            "n_points": int(stable_coords.shape[0]),
        }
        noisy_branch = {
            "coords": noisy_coords,
            "path_length_px": extractor._path_length(noisy_coords),
            "n_points": int(noisy_coords.shape[0]),
        }

        stable_score = extractor._score_kappa_lite_branch_candidate(
            stable_branch,
            image_shape=shape,
            junction_distance_map=junction_distance_map,
            distance_map=distance_map,
        )
        noisy_score = extractor._score_kappa_lite_branch_candidate(
            noisy_branch,
            image_shape=shape,
            junction_distance_map=junction_distance_map,
            distance_map=distance_map,
        )

        self.assertIsNotNone(stable_score)
        self.assertIsNotNone(noisy_score)
        self.assertGreater(stable_score["score"], noisy_score["score"])
        self.assertGreater(stable_score["width_consistency_score"], noisy_score["width_consistency_score"])

    def test_kappa_lite_selection_suppresses_overlapping_candidates(self):
        extractor = FeatureExtractor(magnification=100000)
        shape = (100, 100)

        shared_coords = self._draw_polyline([(40, x) for x in range(10, 70)], shape)
        distinct_coords = self._draw_polyline([(65, x) for x in range(10, 70)], shape)

        candidate_a = {
            "coords": shared_coords,
            "image_shape": shape,
            "path_length_px": extractor._path_length(shared_coords),
            "score": 0.95,
            "_pixel_set": extractor._candidate_pixel_set(shared_coords, shape),
        }
        candidate_b = {
            "coords": shared_coords.copy(),
            "image_shape": shape,
            "path_length_px": extractor._path_length(shared_coords),
            "score": 0.90,
            "_pixel_set": extractor._candidate_pixel_set(shared_coords, shape),
        }
        candidate_c = {
            "coords": distinct_coords,
            "image_shape": shape,
            "path_length_px": extractor._path_length(distinct_coords),
            "score": 0.70,
            "_pixel_set": extractor._candidate_pixel_set(distinct_coords, shape),
        }

        selected = extractor._select_kappa_lite_segments(
            [candidate_a, candidate_b, candidate_c],
            top_k=2,
            max_overlap_ratio=0.5,
        )

        self.assertEqual(len(selected), 2)
        self.assertAlmostEqual(selected[0]["score"], 0.95)
        self.assertAlmostEqual(selected[1]["score"], 0.70)

    def test_kappa_lite_geometry_reports_ld_ratio_near_one_for_straight_segment(self):
        extractor = FeatureExtractor(magnification=100000)
        extractor.px_per_um = 100.0
        shape = (120, 120)
        coords = self._draw_polyline([(60, x) for x in range(15, 95)], shape)
        distance_map = np.full(shape, 2.0, dtype=np.float32)

        metrics = extractor._compute_kappa_lite_segment_metrics(
            {
                "segment_id": 1,
                "coords": coords,
                "path_length_px": extractor._path_length(coords),
                "score": 1.0,
            },
            distance_map=distance_map,
        )

        self.assertAlmostEqual(metrics["ld_ratio"], 1.0, delta=0.05)
        self.assertLess(metrics["mean_curvature_nm"], 0.01)

    def test_kappa_lite_geometry_reports_ld_ratio_above_one_for_wavy_segment(self):
        extractor = FeatureExtractor(magnification=100000)
        extractor.px_per_um = 100.0
        x_vals = np.arange(10, 90)
        y_vals = 60 + 10 * np.sin(2 * np.pi * (x_vals - x_vals.min()) / 18.0)
        coords = self._draw_polyline(list(zip(y_vals, x_vals)), (120, 120))
        distance_map = np.full((120, 120), 2.0, dtype=np.float32)

        metrics = extractor._compute_kappa_lite_segment_metrics(
            {
                "segment_id": 1,
                "coords": coords,
                "path_length_px": extractor._path_length(coords),
                "score": 1.0,
            },
            distance_map=distance_map,
        )

        self.assertGreater(metrics["ld_ratio"], 1.1)
        self.assertGreater(metrics["mean_curvature_nm"], 0.001)


if __name__ == "__main__":
    unittest.main()
