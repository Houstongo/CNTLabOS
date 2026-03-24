"""Prepare 768x768 patch assets for the CNT paper-reproduction line."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.cnt_paper_repro.patching import prepare_patch_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare patch dataset for CNT paper reproduction.")
    parser.add_argument("--source-manifest-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--patch-size", type=int, default=768)
    parser.add_argument("--mode", default="center")
    parser.add_argument("--stride", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = prepare_patch_dataset(
        source_manifest_dir=args.source_manifest_dir,
        output_root=args.output_root,
        patch_size=args.patch_size,
        mode=args.mode,
        stride=args.stride,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
