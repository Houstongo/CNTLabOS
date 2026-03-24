import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.cnt_paper_repro.losses import compute_phase_loss, orientation_histogram_from_map


class CntPaperReproLossTests(unittest.TestCase):
    def test_orientation_histogram_returns_expected_shape(self):
        x = torch.rand(2, 1, 64, 64)
        hist = orientation_histogram_from_map(x, bins=36)
        self.assertEqual(tuple(hist.shape), (2, 36))

    def test_phase_loss_is_finite(self):
        logits = torch.zeros((1, 1, 64, 64), requires_grad=True)
        gray = torch.zeros((1, 1, 64, 64))
        mask = torch.zeros((1, 1, 64, 64))
        loss, details = compute_phase_loss(
            {
                "dice_weight": 0.6,
                "orientation_weight": 1.0e-7,
                "orientation_bins": 36,
            },
            logits,
            mask,
            gray,
        )
        loss.backward()
        self.assertFalse(torch.isnan(loss.detach()).any().item())
        self.assertIn("orientation", details)


if __name__ == "__main__":
    unittest.main()
