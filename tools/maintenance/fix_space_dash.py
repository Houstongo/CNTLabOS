#!/usr/bin/env python3
"""修复文件名中倍率与 repeat 之间多余的空格

格式: `10000 -1.png` → `10000-1.png`
"""

import os
import re
from pathlib import Path

ZZY_ROOT = Path(r"D:\CNTDATA\ZZY")


def fix_space_dash():
    """修复 `10000 -1.png` → `10000-1.png`"""
    count = 0
    pattern = re.compile(r"^(.+) (\d+)\s+-\d+\.png$")

    for f in os.listdir(ZZY_ROOT):
        if not f.lower().endswith(".png"):
            continue

        match = pattern.match(f)
        if match:
            old_path = ZZY_ROOT / f
            prefix = match.group(1)
            repeat_part = match.group(2)
            new_name = f"{prefix} {repeat_part}-{f.split('-')[-1]}"
            new_path = ZZY_ROOT / new_name

            if new_path.exists() and new_path != old_path:
                print(f"跳过（已存在）: {new_name}")
                continue

            os.rename(old_path, new_path)
            count += 1
            print(f"[{count}] {f} → {new_name}")

    print(f"\n共修复 {count} 个文件")


if __name__ == "__main__":
    fix_space_dash()
