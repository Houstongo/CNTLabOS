import unittest

import numpy as np

from src.analysis.feature_extractor import FeatureExtractor


class FeatureExtractorWavinessTests(unittest.TestCase):
    def _make_wave_skeleton(self, amplitude_px=8, wavelength_px=40, cycles=2):
        width = wavelength_px * cycles + 20
        height = 120
        x_vals = np.arange(10, width - 10)
        phase = 2 * np.pi * (x_vals - x_vals.min()) / wavelength_px
        y_vals = 60 + amplitude_px * np.sin(phase)
        skel = np.zeros((height, width), dtype=bool)
        prev_x = None
        prev_y = None
        for x, y in zip(x_vals, y_vals):
            curr_x = int(x)
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

    def _make_branchy_wave_mesh(self):
        skel = np.zeros((220, 260), dtype=bool)
        x_vals = np.arange(20, 240)
        phase = 2 * np.pi * (x_vals - x_vals.min()) / 42.0
        y_top = 70 + 10 * np.sin(phase)
        y_bottom = 150 + 10 * np.sin(phase + 0.5)

        for x, yt, yb in zip(x_vals, y_top, y_bottom):
            yt_i = int(round(yt))
            yb_i = int(round(yb))
            skel[yt_i, x] = True
            skel[yb_i, x] = True
            if x % 12 == 0:
                y0, y1 = sorted((yt_i, yb_i))
                skel[y0:y1 + 1, x] = True
        return skel

    def test_calculate_waviness_detects_wave_ratio_from_skeleton(self):
        extractor = FeatureExtractor(magnification=50000)
        extractor.px_per_um = 100.0
        skel = self._make_wave_skeleton()

        metrics = extractor.calculate_waviness(skel)
        metrics_v2 = extractor.calculate_waviness_v2(skel)

        self.assertGreater(metrics["waviness_ratio"], 0.2)
        self.assertLess(metrics["waviness_ratio"], 0.6)
        self.assertGreater(metrics["waviness_height_nm"], 100.0)
        self.assertGreater(metrics["waviness_wavelength_nm"], 200.0)
        self.assertGreaterEqual(metrics["waviness_branches"], 1)
        self.assertGreater(metrics_v2["waviness_ratio_v2"], 0.2)
        self.assertLess(metrics_v2["waviness_ratio_v2"], 0.6)
        self.assertGreater(metrics_v2["waviness_height_nm_v2"], 100.0)
        self.assertGreater(metrics_v2["waviness_wavelength_nm_v2"], 200.0)
        self.assertGreaterEqual(metrics_v2["waviness_branches_v2"], 1)

    def test_calculate_waviness_returns_zero_for_straight_centerline(self):
        extractor = FeatureExtractor(magnification=50000)
        extractor.px_per_um = 100.0
        skel = np.zeros((80, 100), dtype=bool)
        skel[40, 10:90] = True

        metrics = extractor.calculate_waviness(skel)
        metrics_v2 = extractor.calculate_waviness_v2(skel)

        self.assertEqual(metrics["waviness_ratio"], 0.0)
        self.assertEqual(metrics["waviness_height_nm"], 0.0)
        self.assertEqual(metrics["waviness_wavelength_nm"], 0.0)
        self.assertEqual(metrics["waviness_branches"], 0)
        self.assertEqual(metrics_v2["waviness_ratio_v2"], 0.0)
        self.assertEqual(metrics_v2["waviness_height_nm_v2"], 0.0)
        self.assertEqual(metrics_v2["waviness_wavelength_nm_v2"], 0.0)
        self.assertEqual(metrics_v2["waviness_branches_v2"], 0)
        self.assertGreaterEqual(metrics_v2["tortuosity_v2"], 1.0)

    def test_fast_profile_bounds_branchy_mesh_waviness_ratio(self):
        extractor = FeatureExtractor(magnification=20000, speed_profile="fast")
        extractor.px_per_um = 120.0
        skel = self._make_branchy_wave_mesh()

        metrics = extractor.calculate_waviness(skel)
        metrics_v2 = extractor.calculate_waviness_v2(skel)

        self.assertGreaterEqual(metrics["waviness_ratio"], 0.0)
        self.assertLessEqual(metrics["waviness_ratio"], 5.0)
        self.assertGreaterEqual(metrics["waviness_branches"], 0)
        self.assertGreaterEqual(metrics_v2["waviness_ratio_v2"], 0.0)
        self.assertLessEqual(metrics_v2["waviness_ratio_v2"], 5.0)
        self.assertGreaterEqual(metrics_v2["waviness_branches_v2"], 0)


if __name__ == "__main__":
    unittest.main()
