import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.cnt_paper_repro.config import load_config


class CntPaperReproConfigTests(unittest.TestCase):
    def test_default_config_exposes_paper_defaults(self):
        config = load_config(PROJECT_ROOT / "experiments" / "cnt_paper_repro" / "configs" / "paper_100000x.yaml")
        self.assertEqual(config["data"]["patch_size"], 768)
        self.assertEqual(config["inference"]["threshold"], 0.7)
        self.assertEqual(config["model"]["name"], "ResNet34UNet")
        self.assertEqual(config["training"]["phases"][0]["name"], "dice")
        self.assertEqual(config["training"]["phases"][1]["name"], "dice_orientation")
        self.assertEqual(config["training"]["phases"][0]["lambda_cl"], 0.0)
        self.assertEqual(config["training"]["phases"][1]["lambda_ridge"], 0.0)

    def test_structural_configs_keep_invariants_and_expose_weights(self):
        config_dir = PROJECT_ROOT / "experiments" / "cnt_paper_repro" / "configs"
        baseline = load_config(config_dir / "paper_100000x.yaml")
        cldice = load_config(config_dir / "paper_100000x_cldice.yaml")
        cldice_ridge = load_config(config_dir / "paper_100000x_cldice_ridge.yaml")

        for other in [cldice, cldice_ridge]:
            self.assertEqual(other["data"]["source_dataset_root"], baseline["data"]["source_dataset_root"])
            self.assertEqual(other["data"]["patch_dataset_root"], baseline["data"]["patch_dataset_root"])
            self.assertEqual(other["data"]["patch_size"], baseline["data"]["patch_size"])
            self.assertEqual(other["inference"]["threshold"], baseline["inference"]["threshold"])
            self.assertEqual(other["model"]["name"], baseline["model"]["name"])
            self.assertEqual(other["seed"], baseline["seed"])

        self.assertGreater(cldice["training"]["phases"][1]["lambda_cl"], 0.0)
        self.assertEqual(cldice["training"]["phases"][1]["lambda_ridge"], 0.0)
        self.assertGreater(cldice_ridge["training"]["phases"][1]["lambda_cl"], 0.0)
        self.assertGreater(cldice_ridge["training"]["phases"][1]["lambda_ridge"], 0.0)


if __name__ == "__main__":
    unittest.main()
