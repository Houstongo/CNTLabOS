import unittest

import numpy as np

from src.analysis.feature_extractor import FeatureExtractor


class FeatureExtractorCurvatureV2Tests(unittest.TestCase):
    def _draw_polyline(self, points, shape):
        skel = np.zeros(shape, dtype=bool)
        prev_x = None
        prev_y = None
        for x, y in points:
            curr_x = int(round(x))
            curr_y = int(round(y))
            if prev_x is None:
                skel[curr_y, curr_x] = True
            else:
                steps = max(abs(curr_x - prev_x), abs(curr_y - prev_y), 1)
                for step in range(steps + 1):
                    interp_x = int(round(prev_x + (curr_x - prev_x) * step / steps))
                    interp_y = int(round(prev_y + (curr_y - prev_y) * step / steps))
                    skel[interp_y, interp_x] = True
            prev_x = curr_x
            prev_y = curr_y
        return skel

    def _make_straight_skeleton(self):
        return self._draw_polyline([(x, 60) for x in range(10, 130)], (140, 160))

    def _make_wave_skeleton(self, amplitude_px=14, wavelength_px=18, cycles=4):
        x_vals = np.arange(10, wavelength_px * cycles + 10)
        phase = 2 * np.pi * (x_vals - x_vals.min()) / wavelength_px
        y_vals = 70 + amplitude_px * np.sin(phase)
        points = list(zip(x_vals, y_vals))
        return self._draw_polyline(points, (160, int(x_vals.max()) + 20))

    def test_calculate_curvature_v2_returns_near_zero_for_straight_centerline(self):
        extractor = FeatureExtractor(magnification=50000)
        extractor.px_per_um = 100.0
        skel = self._make_straight_skeleton()

        label, curvature_nm_v2 = extractor.calculate_curvature_v2(skel)

        self.assertEqual(label, "Straight")
        self.assertLess(curvature_nm_v2, 0.05)

    def test_calculate_curvature_v2_is_positive_for_wave_centerline(self):
        extractor = FeatureExtractor(magnification=50000)
        extractor.px_per_um = 100.0
        skel = self._make_wave_skeleton()

        label, curvature_nm_v2 = extractor.calculate_curvature_v2(skel)

        self.assertIn(label, {"Wavy", "Coiled"})
        self.assertGreater(curvature_nm_v2, 0.001)

    def test_calculate_curvature_v2_uses_calibrated_px_per_um(self):
        skel = self._make_wave_skeleton()

        coarse = FeatureExtractor(magnification=50000)
        coarse.px_per_um = 50.0
        _, coarse_curvature = coarse.calculate_curvature_v2(skel)

        fine = FeatureExtractor(magnification=50000)
        fine.px_per_um = 200.0
        _, fine_curvature = fine.calculate_curvature_v2(skel)

        self.assertGreater(coarse_curvature, 0.0)
        self.assertGreater(fine_curvature, 0.0)
        self.assertGreater(fine_curvature, coarse_curvature)

    def test_calculate_curvature_v2_skips_invalid_branches_instead_of_zero_weighting(self):
        extractor = FeatureExtractor(magnification=50000)
        extractor.px_per_um = 100.0
        skel = self._make_wave_skeleton()

        branches = extractor._collect_ordered_branches_v2(
            skel,
            min_points=max(extractor.V3_MIN_BRANCH_POINTS, int(round(extractor.expected_tube_px * 1.5))),
            min_length_factor=extractor.V3_MIN_BRANCH_LENGTH_FACTOR,
        )
        self.assertTrue(branches)

        _, baseline_curvature = extractor.calculate_curvature_v2(skel, ordered_branches=branches)
        invalid_branch = {
            "coords": np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=float),
            "n_points": 2,
            "path_length_px": 20.0,
        }
        _, mixed_curvature = extractor.calculate_curvature_v2(skel, ordered_branches=branches + [invalid_branch])

        self.assertAlmostEqual(mixed_curvature, baseline_curvature, places=9)

    def test_calculate_curvature_legacy_uses_calibrated_px_per_um(self):
        skel = self._make_wave_skeleton()

        coarse = FeatureExtractor(magnification=50000)
        coarse.px_per_um = 50.0
        _, coarse_curvature = coarse.calculate_curvature(skel)

        fine = FeatureExtractor(magnification=50000)
        fine.px_per_um = 200.0
        _, fine_curvature = fine.calculate_curvature(skel)

        self.assertGreater(coarse_curvature, 0.0)
        self.assertGreater(fine_curvature, coarse_curvature)

    def test_calculate_curvature_v3_returns_near_zero_for_straight_centerline(self):
        extractor = FeatureExtractor(magnification=50000)
        extractor.px_per_um = 100.0
        skel = self._make_straight_skeleton()

        label, curvature_nm_v3 = extractor.calculate_curvature_v3(skel)

        self.assertEqual(label, "Straight")
        self.assertLess(curvature_nm_v3, 0.05)

    def test_calculate_curvature_v3_is_more_sensitive_than_v2_for_wave_centerline(self):
        extractor = FeatureExtractor(magnification=50000)
        extractor.px_per_um = 100.0
        skel = self._make_wave_skeleton()

        _, curvature_nm_v2 = extractor.calculate_curvature_v2(skel)
        label_v3, curvature_nm_v3 = extractor.calculate_curvature_v3(skel)

        self.assertIn(label_v3, {"Wavy", "Coiled"})
        self.assertGreater(curvature_nm_v3, 0.001)
        self.assertGreaterEqual(curvature_nm_v3, curvature_nm_v2)


if __name__ == "__main__":
    unittest.main()
