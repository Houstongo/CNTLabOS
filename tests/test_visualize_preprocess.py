import unittest

import numpy as np

from backend import main as api_main


class VisualizePreprocessTests(unittest.TestCase):
    def test_prepare_visualization_image_downscales_large_image(self):
        img = np.zeros((3000, 4000), dtype=np.uint8)
        out = api_main.prepare_visualization_image(img, max_side=1280)
        self.assertLessEqual(max(out.shape), 1280)
        self.assertEqual(out.dtype, np.uint8)

    def test_prepare_visualization_image_keeps_small_image(self):
        img = np.zeros((640, 960), dtype=np.uint8)
        out = api_main.prepare_visualization_image(img, max_side=1280)
        self.assertEqual(out.shape, img.shape)


if __name__ == "__main__":
    unittest.main()
