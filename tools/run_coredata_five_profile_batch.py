from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import generate_branch_selection_compare_panels as branch_panels  # noqa: E402


DEFAULT_INPUT_ROOT = Path(r"D:\CNTDATA\coredata")
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "reports"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
EXCLUDED_PATH_PARTS = {
    "rough_curvature_buckets_visual",
    "_review_sheets",
}
EXCLUDED_NAME_PATTERNS = (
    "quick_feature_panels_",
    "mask_skeleton_cleaning_panels_",
    "branch_selection_compare_panels_",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run resumable five-profile batch extraction with per-image panels.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--merge-only", action="store_true")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def slugify(text: str) -> str:
    safe = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_"}:
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe).strip("_") or "image"


def is_source_image(path: Path) -> bool:
    if not path.is_file():
        return False
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        return False
    if any(part in EXCLUDED_PATH_PARTS for part in path.parts):
        return False
    if any(pattern in part for part in path.parts for pattern in EXCLUDED_NAME_PATTERNS):
        return False
    return True


def infer_magnification(path: Path) -> int | None:
    for part in reversed(path.parts):
        digits = "".join(ch for ch in part if ch.isdigit())
        if digits in {"50000", "100000", "10000", "20000"}:
            return int(digits)
    name = path.name
    if "100000" in name:
        return 100000
    if "50000" in name:
        return 50000
    if "10000" in name:
        return 10000
    if "20000" in name:
        return 20000
    return None


def enumerate_source_images(input_root: Path) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for path in sorted(input_root.rglob("*")):
        if not is_source_image(path):
            continue
        magnification = infer_magnification(path)
        if magnification is None:
            continue
        rel_path = path.relative_to(input_root)
        image_id = slugify(str(rel_path.with_suffix("")))
        records.append(
            {
                "image_id": image_id,
                "file_path": str(path),
                "relative_path": str(rel_path),
                "magnification": magnification,
            }
        )
    return records


def write_manifest(manifest_path: Path, manifest_rows: List[Dict[str, object]]) -> None:
    ensure_dir(manifest_path.parent)
    with manifest_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["image_id", "relative_path", "file_path", "magnification"])
        writer.writeheader()
        writer.writerows(manifest_rows)


def load_manifest(manifest_path: Path) -> List[Dict[str, object]]:
    return list(csv.DictReader(manifest_path.open("r", encoding="utf-8", newline="")))


def process_one(record: Dict[str, object], output_dir: Path, resume: bool) -> Dict[str, object]:
    image_id = str(record["image_id"])
    file_path = Path(str(record["file_path"]))
    magnification = int(record["magnification"])
    item_dir = output_dir / "items" / image_id
    ensure_dir(item_dir)

    features_path = item_dir / "features.json"
    panel_path = item_dir / "comparison_panel.png"

    if resume and features_path.exists() and panel_path.exists():
        return json.loads(features_path.read_text(encoding="utf-8"))

    image = branch_panels.read_gray_image(file_path)
    common = branch_panels.prepare_common(image, magnification)
    analysis = branch_panels.analyze_profiles(common, magnification)

    branch_panels.render_panel(
        roi=common["roi"],
        mask=common["thresh"],
        accurate_v2_branches=analysis["accurate_v2_branches"],
        accurate_v3_branches=analysis["accurate_v3_branches"],
        fast_v2_branches=analysis["fast_v2_branches"],
        fast_v3_branches=analysis["fast_v3_branches"],
        metrics=analysis["metrics"],
        file_name=file_path.name,
        output_path=panel_path,
    )

    feature_record: Dict[str, object] = {
        "image_id": image_id,
        "file_name": file_path.name,
        "file_path": str(file_path),
        "relative_path": str(record["relative_path"]),
        "magnification": magnification,
        "panel_path": str(panel_path),
        **analysis["metrics"],
    }
    features_path.write_text(json.dumps(feature_record, ensure_ascii=False, indent=2), encoding="utf-8")
    return feature_record


def merge_outputs(output_dir: Path) -> List[Dict[str, object]]:
    items_dir = output_dir / "items"
    feature_paths = sorted(items_dir.glob("*/features.json"))
    records: List[Dict[str, object]] = []
    for path in feature_paths:
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue

    if records:
        summary_csv_path = output_dir / "summary.csv"
        with summary_csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)

    summary_json_path = output_dir / "summary.json"
    summary_json_path.write_text(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "count": len(records),
                "records": records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return records


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (DEFAULT_OUTPUT_ROOT / f"coredata_five_profile_batch_{timestamp}")
    ensure_dir(output_dir)

    manifest_path = output_dir / "manifest.csv"
    if manifest_path.exists():
        manifest_rows = load_manifest(manifest_path)
    else:
        manifest_rows = enumerate_source_images(args.input_root)
        if args.limit is not None:
            manifest_rows = manifest_rows[: max(0, int(args.limit))]
        write_manifest(manifest_path, manifest_rows)

    if args.merge_only:
        merge_outputs(output_dir)
        print(output_dir)
        return

    processed = 0
    for record in manifest_rows:
        process_one(record, output_dir=output_dir, resume=args.resume)
        processed += 1
        if processed % 10 == 0:
            print(f"processed {processed}/{len(manifest_rows)}")

    merge_outputs(output_dir)
    print(output_dir)


if __name__ == "__main__":
    main()
