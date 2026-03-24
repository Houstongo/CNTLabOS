import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.create_cnt_loss_compare_split import split_train_val


class CntLossCompareSplitTests(unittest.TestCase):
    def test_split_train_val_keeps_requested_val_count(self):
        rows = []
        for image_id in range(1, 11):
            rows.append(
                {
                    "image_id": str(image_id),
                    "sample_id": "NoA" if image_id <= 6 else "NoB",
                    "group_key": "G1" if image_id <= 6 else "G2",
                }
            )
        train_rows, val_rows = split_train_val(rows, val_count=2, seed=42)
        self.assertEqual(len(train_rows), 8)
        self.assertEqual(len(val_rows), 2)
        self.assertEqual(sorted(int(row["image_id"]) for row in train_rows + val_rows), list(range(1, 11)))


if __name__ == "__main__":
    unittest.main()
