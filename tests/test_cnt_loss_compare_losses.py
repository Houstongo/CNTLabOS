import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.cnt_loss_compare.losses import orientation_guided_loss


class CntLossCompareLossTests(unittest.TestCase):
    def test_orientation_guided_loss_backward_is_finite_on_flat_inputs(self):
        logits = torch.zeros((1, 1, 8, 8), requires_grad=True)
        gray = torch.zeros((1, 1, 8, 8))

        loss = orientation_guided_loss(logits, gray)
        loss.backward()

        self.assertFalse(torch.isnan(loss.detach()).any().item())
        self.assertFalse(torch.isnan(logits.grad).any().item())
        self.assertFalse(torch.isinf(logits.grad).any().item())


if __name__ == "__main__":
    unittest.main()
