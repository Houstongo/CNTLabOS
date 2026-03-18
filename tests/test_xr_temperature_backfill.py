import unittest

from backend.core.calibrator import calibrator
from backend.core.data_manager import CNTADataParser


class XRTemperatureBackfillTests(unittest.TestCase):
    def setUp(self):
        self.parser = CNTADataParser()

    def test_xr_parser_uses_temperature_from_folder_name(self):
        data = self.parser.parse_xr_filename(
            "C4B2.tiff",
            folder_name="250309 T850 3h L250",
        )

        self.assertEqual(data["growth_temp"], 850.0)
        self.assertEqual(data["ar_flow"], 250.0)

    def test_xr_parser_uses_flow_and_catalyst_from_folder_name(self):
        data = self.parser.parse_xr_filename(
            "C6A1.tiff",
            folder_name="250524 T800 3H L200 0.5g",
        )

        self.assertEqual(data["ar_flow"], 200.0)
        self.assertEqual(data["catalyst_weight"], 0.5)

    def test_xr_parser_keeps_default_temperature_without_folder_context(self):
        data = self.parser.parse_xr_filename("C4B2.tiff")

        self.assertEqual(data["growth_temp"], 800.0)

    def test_calibrator_uses_backfilled_temperature_for_actual_temp(self):
        data = self.parser.parse_xr_filename(
            "C7A1.tiff",
            folder_name="250313 T750 3h L250",
        )
        data["membrane_pos_cm"] = 28.0

        calibrated = calibrator.calibrate(data)

        self.assertAlmostEqual(calibrated["actual_temp"], 789.0, places=1)


if __name__ == "__main__":
    unittest.main()
