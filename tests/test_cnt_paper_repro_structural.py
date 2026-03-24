import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.cnt_paper_repro.structural import ridge_response_from_gray, soft_skeletonize


class CntPaperReproStructuralTests(unittest.TestCase):
    def test_soft_skeletonize_preserves_shape_and_is_finite(self):
        x = torch.rand(2, 1, 64, 64)
        skel = soft_skeletonize(x, iterations=8)
        self.assertEqual(tuple(skel.shape), tuple(x.shape))
        self.assertTrue(torch.isfinite(skel).all().item())

    def test_ridge_response_is_normalized_and_shape_stable(self):
        x = torch.rand(2, 1, 64, 64)
        ridge = ridge_response_from_gray(x)
        self.assertEqual(tuple(ridge.shape), tuple(x.shape))
        self.assertTrue(torch.isfinite(ridge).all().item())
        self.assertGreaterEqual(float(ridge.min()), 0.0)
        self.assertLessEqual(float(ridge.max()), 1.0)

    def test_ridge_response_highlights_simple_line(self):
        x = torch.zeros((1, 1, 64, 64), dtype=torch.float32)
        x[:, :, 32, 12:52] = 1.0
        ridge = ridge_response_from_gray(x)
        line_mean = float(ridge[:, :, 32, 16:48].mean())
        bg_mean = float(ridge[:, :, :8, :8].mean())
        self.assertGreater(line_mean, bg_mean)


if __name__ == "__main__":
    unittest.main()
