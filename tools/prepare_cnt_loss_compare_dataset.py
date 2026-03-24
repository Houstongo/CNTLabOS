"""
Prepare controlled CNT segmentation experiment datasets from the main image DB.

This script creates a reproducible image-only split for later weak-label
generation and training. It does not modify the database.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np


PROJECT_ROOT = Path(r"D:\CNTDATA\CNTA_ML_Project")
DB_PATH = PROJECT_ROOT / "database" / "cnta_experiments.sqlite"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "experiments" / "cnt_loss_compare" / "datasets"


GROUP_FIELDS = [
    "source",
    "magnification",
    "al2o3_power",
    "al2o3_thickness",
    "fe_power",
    "fe_thickness",
    "growth_temp",
    "growth_time",
    "ar_flow",
    "h2_flow",
    "c2h4_flow",
    "anneal_temp",
    "anneal_time",
]


@dataclass
class Candidate:
    image_id: int
    file_path: str
    source: str
    sample_id: str
    position_label: str
    magnification: int
    horizontal_pos: str | None
    vertical_pos: int | None
    repeat_id: int | None
    growth_temp: float | None
    growth_time: float | None
    ar_flow: float | None
    h2_flow: float | None
    c2h4_flow: float | None
    al2o3_power: float | None
    al2o3_thickness: float | None
    fe_power: float | None
    fe_thickness: float | None
    anneal_temp: float | None
    anneal_time: float | None

    @property
    def file_name(self) -> str:
        return Path(self.file_path).name

    @property
    def stem(self) -> str:
        return Path(self.file_path).stem

    @property
    def group_key(self) -> Tuple:
        return tuple(getattr(self, field) for field in GROUP_FIELDS)

    def as_row(self, split: str) -> Dict[str, object]:
        data = {field: getattr(self, field) for field in self.__dataclass_fields__}
        data["split"] = split
        data["group_key"] = "|".join("" if value is None else str(value) for value in self.group_key)
        return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare CNT loss-comparison dataset split from DB images.")
    parser.add_argument("--db-path", type=Path, default=DB_PATH)
    parser.add_argument("--source", default="ZZY")
    parser.add_argument("--position-label", default="mid")
    parser.add_argument("--magnification", type=int, required=True)
    parser.add_argument("--train-count", type=int, default=50)
    parser.add_argument("--test-count", type=int, default=50)
    parser.add_argument("--horizontal-pos", default=None)
    parser.add_argument("--vertical-pos", type=int, default=None)
    parser.add_argument("--exclude-image-ids", type=int, nargs="*", default=[])
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dataset-name", default=None)
    return parser.parse_args()


def fetch_candidates(args: argparse.Namespace) -> List[Candidate]:
    conn = sqlite3.connect(args.db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    clauses = [
        "source = ?",
        "is_deleted = 0",
        "magnification = ?",
    ]
    params: List[object] = [args.source, args.magnification]

    if args.position_label is not None:
        clauses.append("lower(coalesce(position_label, '')) = ?")
        params.append(args.position_label.lower())
    if args.horizontal_pos is not None:
        clauses.append("upper(coalesce(horizontal_pos, '')) = ?")
        params.append(args.horizontal_pos.upper())
    if args.vertical_pos is not None:
        clauses.append("vertical_pos = ?")
        params.append(args.vertical_pos)

    query = f"""
        select
            id as image_id,
            file_path,
            source,
            sample_id,
            position_label,
            magnification,
            horizontal_pos,
            vertical_pos,
            repeat_id,
            growth_temp,
            growth_time,
            ar_flow,
            h2_flow,
            c2h4_flow,
            al2o3_power,
            al2o3_thickness,
            fe_power,
            fe_thickness,
            anneal_temp,
            anneal_time
        from images
        where {" and ".join(clauses)}
        order by id
    """

    rows = cur.execute(query, params).fetchall()
    conn.close()

    excluded_ids = set(args.exclude_image_ids or [])
    candidates = [Candidate(**dict(row)) for row in rows if dict(row)["image_id"] not in excluded_ids]
    missing = [item.file_path for item in candidates if not Path(item.file_path).exists()]
    if missing:
        preview = "\n".join(missing[:5])
        raise FileNotFoundError(f"{len(missing)} candidate files are missing. Examples:\n{preview}")
    return candidates


def allocate_used_counts(groups: Dict[Tuple, List[Candidate]], target_used: int) -> Dict[Tuple, int]:
    total = sum(len(items) for items in groups.values())
    if target_used > total:
        raise ValueError(f"Requested {target_used} used items, but only {total} candidates are available.")

    base: Dict[Tuple, int] = {}
    remainders: List[Tuple[float, Tuple]] = []
    for key, items in groups.items():
        ideal = len(items) * target_used / total
        count = math.floor(ideal)
        base[key] = min(count, len(items))
        remainders.append((ideal - count, key))

    assigned = sum(base.values())
    for _, key in sorted(remainders, reverse=True):
        if assigned >= target_used:
            break
        if base[key] < len(groups[key]):
            base[key] += 1
            assigned += 1

    return base


def allocate_train_counts(used_counts: Dict[Tuple, int], target_train: int) -> Dict[Tuple, int]:
    total_used = sum(used_counts.values())
    if target_train > total_used:
        raise ValueError(f"Requested {target_train} training samples, but only {total_used} used samples exist.")

    base: Dict[Tuple, int] = {}
    remainders: List[Tuple[float, Tuple]] = []
    for key, count in used_counts.items():
        ideal = count * target_train / total_used
        train_count = math.floor(ideal)
        base[key] = min(train_count, count)
        remainders.append((ideal - train_count, key))

    assigned = sum(base.values())
    for _, key in sorted(remainders, reverse=True):
        if assigned >= target_train:
            break
        if base[key] < used_counts[key]:
            base[key] += 1
            assigned += 1

    if assigned != target_train:
        raise ValueError(f"Unable to allocate exactly {target_train} training samples; got {assigned}.")
    return base


def split_candidates(candidates: Sequence[Candidate], train_target: int, test_target: int) -> Dict[str, List[Candidate]]:
    target_used = train_target + test_target
    groups: Dict[Tuple, List[Candidate]] = defaultdict(list)
    for item in candidates:
        groups[item.group_key].append(item)

    for items in groups.values():
        items.sort(key=lambda item: (item.image_id, item.file_name))

    used_counts = allocate_used_counts(groups, target_used)
    train_counts = allocate_train_counts(used_counts, train_target)

    train_items: List[Candidate] = []
    test_items: List[Candidate] = []
    reserve_items: List[Candidate] = []

    for key, items in sorted(groups.items(), key=lambda pair: (-len(pair[1]), pair[0])):
        used_n = used_counts[key]
        train_n = train_counts[key]
        test_n = used_n - train_n
        used_items = items[:used_n]
        reserve_items.extend(items[used_n:])

        local_train: List[Candidate] = []
        local_test: List[Candidate] = []
        toggle_train = True
        for item in used_items:
            if toggle_train and len(local_train) < train_n:
                local_train.append(item)
            elif len(local_test) < test_n:
                local_test.append(item)
            else:
                local_train.append(item)
            toggle_train = not toggle_train

        if len(local_train) != train_n or len(local_test) != test_n:
            raise RuntimeError(f"Group split mismatch for {key}: train={len(local_train)} test={len(local_test)}")

        train_items.extend(local_train)
        test_items.extend(local_test)

    if len(train_items) != train_target or len(test_items) != test_target:
        raise RuntimeError(
            f"Final split mismatch: train={len(train_items)} test={len(test_items)} expected={train_target}/{test_target}"
        )

    reserve_items.sort(key=lambda item: (item.image_id, item.file_name))
    train_items.sort(key=lambda item: (item.image_id, item.file_name))
    test_items.sort(key=lambda item: (item.image_id, item.file_name))
    return {"train": train_items, "test": test_items, "reserve": reserve_items}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def copy_split_images(dataset_root: Path, split_name: str, items: Sequence[Candidate]) -> None:
    image_dir = dataset_root / split_name / "images"
    ensure_dir(image_dir)
    for item in items:
        dst_name = f"{item.image_id:05d}_{item.file_name}"
        shutil.copy2(item.file_path, image_dir / dst_name)


def build_contact_sheet(items: Sequence[Candidate], output_path: Path, thumb_size: Tuple[int, int] = (180, 120), cols: int = 5) -> None:
    if not items:
        return

    thumbs = []
    for item in items:
        image = cv2.imread(item.file_path)
        if image is None:
            continue
        thumb = cv2.resize(image, thumb_size, interpolation=cv2.INTER_AREA)
        label = f"{item.image_id} | {item.sample_id}"
        cv2.rectangle(thumb, (0, thumb_size[1] - 20), (thumb_size[0], thumb_size[1]), (0, 0, 0), -1)
        cv2.putText(thumb, label[:28], (4, thumb_size[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
        thumbs.append(thumb)

    if not thumbs:
        return

    rows = math.ceil(len(thumbs) / cols)
    sheet = 255 * np.ones((rows * thumb_size[1], cols * thumb_size[0], 3), dtype="uint8")
    for idx, thumb in enumerate(thumbs):
        row = idx // cols
        col = idx % cols
        y0 = row * thumb_size[1]
        x0 = col * thumb_size[0]
        sheet[y0:y0 + thumb_size[1], x0:x0 + thumb_size[0]] = thumb

    ensure_dir(output_path.parent)
    cv2.imwrite(str(output_path), sheet)


def summarize_split(items: Sequence[Candidate]) -> Dict[str, Dict[str, int]]:
    sample_counts: Dict[str, int] = defaultdict(int)
    combo_counts: Dict[str, int] = defaultdict(int)
    for item in items:
        sample_counts[item.sample_id] += 1
        combo_key = (
            f"Al={item.al2o3_power}/{item.al2o3_thickness}, "
            f"Fe={item.fe_power}/{item.fe_thickness}, "
            f"Flow={item.ar_flow}/{item.h2_flow}/{item.c2h4_flow}"
        )
        combo_counts[combo_key] += 1
    return {
        "sample_counts": dict(sorted(sample_counts.items())),
        "combo_counts": dict(sorted(combo_counts.items(), key=lambda pair: (-pair[1], pair[0]))),
    }


def write_summary(output_root: Path, args: argparse.Namespace, splits: Dict[str, List[Candidate]], candidate_count: int) -> None:
    summary = {
        "dataset_name": output_root.name,
        "selection": {
            "source": args.source,
            "position_label": args.position_label,
            "magnification": args.magnification,
            "horizontal_pos": args.horizontal_pos,
            "vertical_pos": args.vertical_pos,
            "exclude_image_ids": list(args.exclude_image_ids or []),
        },
        "counts": {
            "candidates": candidate_count,
            "train": len(splits["train"]),
            "test": len(splits["test"]),
            "reserve": len(splits["reserve"]),
        },
        "train": summarize_split(splits["train"]),
        "test": summarize_split(splits["test"]),
        "reserve": summarize_split(splits["reserve"]),
    }

    ensure_dir(output_root)
    with (output_root / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    lines = [
        f"# {output_root.name}",
        "",
        f"- source: `{args.source}`",
        f"- position_label: `{args.position_label}`",
        f"- magnification: `{args.magnification}`",
        f"- excluded image_ids: `{', '.join(str(x) for x in (args.exclude_image_ids or [])) or 'none'}`",
        f"- candidates: `{candidate_count}`",
        f"- train: `{len(splits['train'])}`",
        f"- test: `{len(splits['test'])}`",
        f"- reserve: `{len(splits['reserve'])}`",
        "",
        "## Train Sample Counts",
    ]
    for key, value in summary["train"]["sample_counts"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Test Sample Counts"])
    for key, value in summary["test"]["sample_counts"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Reserve Sample Counts"])
    for key, value in summary["reserve"]["sample_counts"].items():
        lines.append(f"- `{key}`: {value}")

    (output_root / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    candidates = fetch_candidates(args)
    total_needed = args.train_count + args.test_count
    if len(candidates) < total_needed:
        raise ValueError(f"Need at least {total_needed} candidates, found {len(candidates)}.")

    dataset_name = args.dataset_name or f"{args.source.lower()}_{args.position_label}_{args.magnification}_train{args.train_count}_test{args.test_count}"
    output_root = args.output_root / dataset_name

    splits = split_candidates(candidates, train_target=args.train_count, test_target=args.test_count)

    manifests_dir = output_root / "manifests"
    ensure_dir(manifests_dir)
    write_csv(manifests_dir / "all_candidates.csv", [item.as_row("candidate") for item in candidates])
    write_csv(manifests_dir / "train_manifest.csv", [item.as_row("train") for item in splits["train"]])
    write_csv(manifests_dir / "test_manifest.csv", [item.as_row("test") for item in splits["test"]])
    write_csv(manifests_dir / "reserve_manifest.csv", [item.as_row("reserve") for item in splits["reserve"]])

    for split_name in ("train", "test", "reserve"):
        copy_split_images(output_root, split_name, splits[split_name])

    preview_dir = output_root / "previews"
    build_contact_sheet(splits["train"], preview_dir / "train_contact_sheet.jpg")
    build_contact_sheet(splits["test"], preview_dir / "test_contact_sheet.jpg")
    build_contact_sheet(splits["reserve"], preview_dir / "reserve_contact_sheet.jpg", cols=5)

    write_summary(output_root, args, splits, candidate_count=len(candidates))

    print(f"Prepared dataset at: {output_root}")
    print(f"Candidates={len(candidates)} train={len(splits['train'])} test={len(splits['test'])} reserve={len(splits['reserve'])}")


if __name__ == "__main__":
    main()
