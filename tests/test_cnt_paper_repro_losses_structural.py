import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.cnt_paper_repro.losses import cldice_loss_from_probs, compute_phase_loss, ridge_aux_loss_from_logits


class CntPaperReproStructuralLossTests(unittest.TestCase):
    def test_cldice_prefers_connected_prediction_over_broken_prediction(self):
        target = torch.zeros((1, 1, 64, 64), dtype=torch.float32)
        target[:, :, 32, 12:52] = 1.0

        connected = target.clone()
        broken = target.clone()
        broken[:, :, 32, 31:34] = 0.0

        loss_connected = cldice_loss_from_probs(connected, target)
        loss_broken = cldice_loss_from_probs(broken, target)
        self.assertLess(float(loss_connected), float(loss_broken))

    def test_ridge_aux_prefers_alignment_with_bright_line(self):
        gray = torch.zeros((1, 1, 64, 64), dtype=torch.float32)
        gray[:, :, 32, 12:52] = 1.0

        aligned_logits = torch.full((1, 1, 64, 64), -8.0, dtype=torch.float32)
        aligned_logits[:, :, 32, 12:52] = 8.0

        shifted_logits = torch.full((1, 1, 64, 64), -8.0, dtype=torch.float32)
        shifted_logits[:, :, 28, 12:52] = 8.0

        loss_aligned = ridge_aux_loss_from_logits(aligned_logits, gray)
        loss_shifted = ridge_aux_loss_from_logits(shifted_logits, gray)
        self.assertLess(float(loss_aligned), float(loss_shifted))

    def test_structural_phase_loss_is_finite_and_supports_backward(self):
        logits = torch.zeros((1, 1, 64, 64), requires_grad=True)
        gray = torch.zeros((1, 1, 64, 64))
        mask = torch.zeros((1, 1, 64, 64))
        loss, details = compute_phase_loss(
            {
                "dice_weight": 0.6,
                "orientation_weight": 1.0e-7,
                "orientation_bins": 36,
                "lambda_cl": 0.1,
                "lambda_ridge": 0.05,
            },
            logits,
            mask,
            gray,
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss).all().item())
        self.assertIn("cldice", details)
        self.assertIn("ridge", details)


if __name__ == "__main__":
    unittest.main()
