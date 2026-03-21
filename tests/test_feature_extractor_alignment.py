import unittest

import cv2
import numpy as np

from src.analysis.feature_extractor import FeatureExtractor


class FeatureExtractorAlignmentTests(unittest.TestCase):
    @staticmethod
    def _make_stripe_image(rotate_90=False):
        img = np.zeros((220, 220), dtype=np.uint8)
        for x in range(35, 186, 22):
            cv2.line(img, (x, 20), (x, 190), 255, 5)
        if rotate_90:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        return img

    @staticmethod
    def _make_line_skeleton(horizontal=False):
        skel = np.zeros((160, 160), dtype=bool)
        for x in range(30, 130):
            y = 80
            if horizontal:
                skel[y, x] = True
            else:
                skel[x, y] = True
        return skel

    def test_calculate_hof_skeleton_adaptive_keeps_vertical_axis_without_rotation(self):
        extractor = FeatureExtractor(magnification=50000, diameter_method="enhanced")

        result = extractor.calculate_hof_skeleton_adaptive(self._make_line_skeleton(horizontal=False))

        self.assertEqual(result["rotation_correction_deg"], 0)
        self.assertAlmostEqual(result["alignment"], result["alignment_raw"], places=4)
        self.assertAlmostEqual(result["mean_phi_deg"], result["mean_phi_raw_deg"], places=2)

    def test_extract_all_recovers_horizontal_capture_with_90_degree_correction(self):
        extractor = FeatureExtractor(magnification=50000, diameter_method="enhanced")

        result = extractor.extract_all(self._make_stripe_image(rotate_90=True))

        self.assertEqual(result["rotation_correction_deg"], 90)
        self.assertLess(result["alignment_raw"], 0.0)
        self.assertGreater(result["alignment"], 0.5)
        self.assertLess(result["mean_phi_deg"], result["mean_phi_raw_deg"])

    def test_fast_profile_uses_skeleton_alignment_with_rotation_correction(self):
        extractor = FeatureExtractor(
            magnification=20000,
            diameter_method="standard",
            speed_profile="fast",
        )

        result = extractor.extract_all(self._make_stripe_image(rotate_90=True))

        self.assertTrue(result["hof_method"].startswith("skeleton_fast"))
        self.assertEqual(result["rotation_correction_deg"], 90)
        self.assertGreater(result["alignment"], 0.5)

    def test_should_use_enhanced_diameter_rejects_large_dense_low_mag_mask(self):
        extractor = FeatureExtractor(magnification=20000, diameter_method="enhanced")
        thresh = np.ones((2400, 3200), dtype=np.uint8) * 255

        use_enhanced, reason = extractor.should_use_enhanced_diameter(
            thresh,
            density_factor=0.55,
            n_regions=2400,
            has_large_regions=True,
        )

        self.assertFalse(use_enhanced)
        self.assertEqual(reason, "large_dense_low_mag")


if __name__ == "__main__":
    unittest.main()
