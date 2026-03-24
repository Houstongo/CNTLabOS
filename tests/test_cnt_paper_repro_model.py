import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.cnt_paper_repro.model import ResNet34UNet


class CntPaperReproModelTests(unittest.TestCase):
    def test_resnet34_unet_forward_shape(self):
        model = ResNet34UNet(in_channels=1, num_classes=1, encoder_weights=None)
        x = torch.randn(1, 1, 128, 128)
        y = model(x)
        self.assertEqual(tuple(y.shape), (1, 1, 128, 128))


if __name__ == "__main__":
    unittest.main()
