import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.cnt_paper_repro.patching import center_crop_or_pad, extract_patch_specs


class CntPaperReproPatchingTests(unittest.TestCase):
    def test_center_crop_or_pad_returns_fixed_patch(self):
        image = np.arange(100 * 120, dtype=np.uint8).reshape(100, 120)
        patch, spec = center_crop_or_pad(image, patch_size=64)
        self.assertEqual(patch.shape, (64, 64))
        self.assertEqual(spec.patch_size, 64)
        self.assertEqual(spec.height, 64)
        self.assertEqual(spec.width, 64)

    def test_extract_patch_specs_center_returns_single_patch(self):
        image = np.zeros((300, 400), dtype=np.uint8)
        specs = extract_patch_specs(image, patch_size=128, mode="center")
        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0].patch_size, 128)


if __name__ == "__main__":
    unittest.main()
