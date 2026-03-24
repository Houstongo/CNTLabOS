import csv
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools import generate_wcntsegnet_weak_labels as weak_labels


class GenerateWCNTSegnetWeakLabelsTests(unittest.TestCase):
    def test_dataset_image_filename_matches_prepared_dataset_convention(self):
        row = {
            "image_id": "5713",
            "file_path": r"d:\CNTDATA\ZZY\No41 200w 5.0nm 5w 1.0nm mid 100000-2.png",
        }
        self.assertEqual(
            weak_labels.dataset_image_filename_from_row(row),
            "05713_No41 200w 5.0nm 5w 1.0nm mid 100000-2.png",
        )

    def test_derive_output_paths_uses_split_local_wcntsegnet_directories(self):
        dataset_root = Path(r"D:\CNTDATA\CNTA_ML_Project\experiments\cnt_loss_compare\datasets\demo")
        paths = weak_labels.derive_output_paths(dataset_root, "train", "05713_demo.png")
        self.assertEqual(paths["image_path"], dataset_root / "train" / "images" / "05713_demo.png")
        self.assertEqual(paths["mask_path"], dataset_root / "train" / "masks_wcntsegnet" / "05713_demo_mask.png")
        self.assertEqual(paths["overlay_path"], dataset_root / "previews_wcntsegnet" / "train" / "05713_demo_overlay.png")

    def test_load_split_manifest_reads_utf8_sig_csv(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_root = Path(temp_dir)
            manifests_dir = dataset_root / "manifests"
            manifests_dir.mkdir(parents=True, exist_ok=True)
            manifest_path = manifests_dir / "train_manifest.csv"
            with manifest_path.open("w", newline="", encoding="utf-8-sig") as fh:
                writer = csv.DictWriter(fh, fieldnames=["image_id", "file_path", "magnification"])
                writer.writeheader()
                writer.writerow({"image_id": "1", "file_path": r"d:\a.png", "magnification": "100000"})

            rows = weak_labels.load_split_manifest(dataset_root, "train")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["image_id"], "1")

    def test_build_manifest_row_includes_stats_and_output_paths(self):
        dataset_root = Path(r"D:\CNTDATA\CNTA_ML_Project\experiments\cnt_loss_compare\datasets\demo")
        manifest_row = {
            "image_id": "5713",
            "file_path": r"d:\CNTDATA\ZZY\No41 200w 5.0nm 5w 1.0nm mid 100000-2.png",
            "sample_id": "No41-100000-2",
            "magnification": "100000",
        }
        stats = {
            "roi_foreground_ratio_pct": 48.5,
            "full_foreground_ratio_pct": 43.1,
            "connected_components": 321,
        }
        row = weak_labels.build_manifest_row(
            dataset_root=dataset_root,
            split="test",
            manifest_row=manifest_row,
            image_filename="05713_demo.png",
            status="success",
            stats=stats,
        )
        self.assertEqual(row["split"], "test")
        self.assertEqual(row["status"], "success")
        self.assertEqual(row["mask_filename"], "05713_demo_mask.png")
        self.assertEqual(row["sample_id"], "No41-100000-2")
        self.assertEqual(row["connected_components"], 321)


if __name__ == "__main__":
    unittest.main()
