import unittest
from pathlib import Path

from backend.core.sem_magnification import (
    build_zzy_filename_with_magnification,
    extract_magnification_from_png_metadata,
)


class SemMagnificationUtilsTests(unittest.TestCase):
    def _find_sample(self, pattern: str) -> Path:
        folder = Path(r"D:\CNTDATA\ZZY\20260319No40调流速间隔20s全样品·0.75-2.75 5w")
        matches = sorted(folder.glob(pattern))
        self.assertTrue(matches, f"No files matched pattern: {pattern}")
        return matches[0]

    def test_extracts_low_magnification_from_png_metadata(self):
        image_path = self._find_sample("*2.25nm*mid 500-6.png")

        self.assertEqual(extract_magnification_from_png_metadata(image_path), 500)

    def test_extracts_high_magnification_from_png_metadata(self):
        image_path = self._find_sample("*1.25nm*mid 100000-7.png")

        self.assertEqual(extract_magnification_from_png_metadata(image_path), 100000)

    def test_rewrites_only_the_magnification_segment(self):
        original = Path(
            r"D:\CNTDATA\ZZY\20260319No40调流速间隔20s全样品·0.75-2.75 5w\No40 200w 5.0nm 5w 1.25nm 600 300 150 600 750 15min 180min mid 5000-7.png"
        )

        renamed = build_zzy_filename_with_magnification(original, 100000)

        self.assertEqual(
            renamed.name,
            "No40 200w 5.0nm 5w 1.25nm 600 300 150 600 750 15min 180min mid 100000-7.png",
        )


if __name__ == "__main__":
    unittest.main()
