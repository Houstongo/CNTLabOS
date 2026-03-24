"""Create fixed train/val manifests from the curated CNT loss-comparison dataset."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create deterministic train/val split manifests for CNT loss comparison.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--val-count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-name", default="repro_seed42")
    return parser.parse_args()


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def write_rows(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _dataset_image_filename(row: Dict[str, str]) -> str:
    return f"{int(row['image_id']):05d}_{Path(row['file_path']).name}"


def _enrich_row(row: Dict[str, str], dataset_root: Path, split_name: str) -> Dict[str, str]:
    enriched = dict(row)
    image_filename = _dataset_image_filename(row)
    mask_filename = f"{Path(image_filename).stem}_mask.png"
    enriched["split"] = split_name
    enriched["image_filename"] = image_filename
    enriched["image_path"] = str(dataset_root / split_name / "images" / image_filename)
    enriched["mask_filename"] = mask_filename
    enriched["mask_path"] = str(dataset_root / split_name / "masks_wcntsegnet" / mask_filename)
    return enriched


def _allocate_counts(groups: Dict[str, List[Dict[str, str]]], target: int) -> Dict[str, int]:
    total = sum(len(items) for items in groups.values())
    allocation: Dict[str, int] = {}
    remainders: List[Tuple[float, str]] = []
    for key, items in groups.items():
        ideal = len(items) * target / total
        base = min(int(math.floor(ideal)), len(items))
        allocation[key] = base
        remainders.append((ideal - base, key))
    assigned = sum(allocation.values())
    for _, key in sorted(remainders, reverse=True):
        if assigned >= target:
            break
        if allocation[key] < len(groups[key]):
            allocation[key] += 1
            assigned += 1
    return allocation


def split_train_val(rows: Sequence[Dict[str, str]], val_count: int, seed: int) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    groups: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row.get("group_key") or row.get("sample_id") or row["image_id"]].append(dict(row))

    rng = random.Random(seed)
    for items in groups.values():
        rng.shuffle(items)
        items.sort(key=lambda row: (row.get("sample_id", ""), int(row["image_id"])))

    val_alloc = _allocate_counts(groups, val_count)
    train_rows: List[Dict[str, str]] = []
    val_rows: List[Dict[str, str]] = []
    for key in sorted(groups.keys()):
        items = groups[key]
        val_n = val_alloc[key]
        val_rows.extend(items[:val_n])
        train_rows.extend(items[val_n:])
    train_rows.sort(key=lambda row: int(row["image_id"]))
    val_rows.sort(key=lambda row: int(row["image_id"]))
    return train_rows, val_rows


def main() -> None:
    args = parse_args()
    manifests_root = args.dataset_root / "manifests"
    train_rows = load_rows(manifests_root / "train_manifest.csv")
    test_rows = load_rows(manifests_root / "test_manifest.csv")
    reserve_rows = load_rows(manifests_root / "reserve_manifest.csv")
    train_rows, val_rows = split_train_val(train_rows, val_count=args.val_count, seed=args.seed)

    out_root = manifests_root / args.split_name
    train_rows = [_enrich_row(row, args.dataset_root, "train") for row in train_rows]
    val_rows = [_enrich_row(row, args.dataset_root, "train") for row in val_rows]
    for row in val_rows:
        row["split"] = "val"
    test_rows = [_enrich_row(row, args.dataset_root, "test") for row in test_rows]
    reserve_rows = [_enrich_row(row, args.dataset_root, "reserve") for row in reserve_rows]

    write_rows(out_root / "train_manifest.csv", train_rows)
    write_rows(out_root / "val_manifest.csv", val_rows)
    write_rows(out_root / "test_manifest.csv", test_rows)
    write_rows(out_root / "reserve_manifest.csv", reserve_rows)
    summary = {
        "dataset_root": str(args.dataset_root),
        "seed": args.seed,
        "train": len(train_rows),
        "val": len(val_rows),
        "test": len(test_rows),
        "reserve": len(reserve_rows),
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
