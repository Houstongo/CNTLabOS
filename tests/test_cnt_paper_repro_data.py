import sys
import unittest
from pathlib import Path
import shutil

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.cnt_paper_repro.data import CNTPatchDataset


class CntPaperReproDataTests(unittest.TestCase):
    def test_dataset_returns_expected_tensors(self):
        tmpdir = PROJECT_ROOT / "_tmp_cnt_paper_repro_data_case"
        if tmpdir.exists():
            shutil.rmtree(tmpdir, ignore_errors=True)
        tmpdir.mkdir(parents=True, exist_ok=True)
        try:
            root = tmpdir
            image_path = root / "patch.png"
            mask_path = root / "patch_mask.png"
            manifest_path = root / "manifest.csv"

            image = np.full((32, 32), 127, dtype=np.uint8)
            mask = np.zeros((32, 32), dtype=np.uint8)
            mask[:, 8:24] = 255
            image_path.write_bytes(cv2.imencode(".png", image)[1].tobytes())
            mask_path.write_bytes(cv2.imencode(".png", mask)[1].tobytes())
            manifest_path.write_text(
                "image_id,patch_index,patch_filename,patch_image_path,patch_mask_path,sample_id\n"
                f"1,1,patch.png,{image_path},{mask_path},sample-a\n",
                encoding="utf-8",
            )

            dataset = CNTPatchDataset(manifest_path)
            sample = dataset[0]
            self.assertEqual(tuple(sample["image"].shape), (1, 32, 32))
            self.assertEqual(tuple(sample["gray"].shape), (1, 32, 32))
            self.assertEqual(tuple(sample["mask"].shape), (1, 32, 32))
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
