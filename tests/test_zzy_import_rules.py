import unittest

from backend.core.data_manager import CNTADataParser


class ZZYImportRuleTests(unittest.TestCase):
    def setUp(self):
        self.parser = CNTADataParser()

    def test_mid_with_9000_magnification_is_included(self):
        row = self.parser.parse_zzy_filename(
            "No40 200w 5.0nm 5w 1.25nm 600 300 150 600 750 15min 180min mid 9000-1.png"
        )

        self.assertIsNotNone(row)
        self.assertTrue(self.parser.should_include_zzy_record(row))

    def test_top_with_10000_magnification_is_excluded(self):
        row = self.parser.parse_zzy_filename(
            "No40 200w 5.0nm 5w 1.25nm 600 300 150 600 750 15min 180min top 10000-1.png"
        )

        self.assertIsNotNone(row)
        self.assertFalse(self.parser.should_include_zzy_record(row))

    def test_mid_below_9000_magnification_is_excluded(self):
        row = self.parser.parse_zzy_filename(
            "No40 200w 5.0nm 5w 1.25nm 600 300 150 600 750 15min 180min mid 5000-1.png"
        )

        self.assertIsNotNone(row)
        self.assertFalse(self.parser.should_include_zzy_record(row))

    def test_mid_prefix_variants_are_included(self):
        self.assertTrue(self.parser.is_zzy_mid_position("mid"))
        self.assertTrue(self.parser.is_zzy_mid_position("midA"))
        self.assertTrue(self.parser.is_zzy_mid_position("mid10"))
        self.assertFalse(self.parser.is_zzy_mid_position("bottom"))


if __name__ == "__main__":
    unittest.main()
