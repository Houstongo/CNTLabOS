import unittest

from backend.core.data_manager import CNTADataParser


class ZZYPositionNormalizationTests(unittest.TestCase):
    def test_bpttom_filename_is_normalized_to_bottom(self):
        parser = CNTADataParser()
        row = parser.parse_zzy_filename(
            "No32 200w 5.0nm 5w 1.25nm-1 400 200 100 600 750 15min 180min bpttom 5000 0-1.png"
        )

        self.assertIsNotNone(row)
        self.assertEqual(row["position_label"], "bottom")


if __name__ == "__main__":
    unittest.main()
