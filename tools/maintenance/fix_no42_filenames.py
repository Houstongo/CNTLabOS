#!/usr/bin/env python3
"""修复 No42 文件名格式问题

1. 修复 `180minmid` → `180min mid`
2. 修复重复的倍率格式 `100000-1000001.png` → `100000-1.png`
"""

import os
import re
from pathlib import Path

ZZY_ROOT = Path(r"D:\CNTDATA\ZZY")


def fix_no42():
    """修复 No42 文件名"""
    count = 0

    # 匹配 No42 文件名并修复格式
    pattern = re.compile(r"^(No42 200w 10nm 5w [\d.]+nm \d+ \d+ \d+ \d+ \d+ \d+min \d+min)(mid)?(\s*)(\d+)-\d+-(\d+)\.png$")

    for f in os.listdir(ZZY_ROOT):
        if not f.startswith("No42 ") or not f.lower().endswith(".png"):
            continue

        # 修复 min 后面没有空格的情况
        if "minmid" in f or "minmid" in f.lower():
            old_path = ZZY_ROOT / f
            # 添加空格在 min 和 mid 之间
            new_name = f.replace("180minmid", "180min mid")
            new_path = ZZY_ROOT / new_name
            if new_path != old_path:
                os.rename(old_path, new_path)
                count += 1
                print(f"[{count}] {f} → {new_name}")
                continue

        # 修复重复倍率格式：100000-1000001.png → 100000-1.png
        match = re.search(r"(\d+)-(\d+)-(\d+)\.png$", f)
        if match:
            mag1, mag2, repeat = match.groups()
            if mag1 == mag2:  # 倍率重复
                old_path = ZZY_ROOT / f
                new_name = f"{f[:match.start()]}{mag1}-{repeat}.png"
                new_path = ZZY_ROOT / new_name
                if new_path != old_path and not new_path.exists():
                    os.rename(old_path, new_path)
                    count += 1
                    print(f"[{count}] {f} → {new_name}")

    print(f"\n共修复 {count} 个 No42 文件")


if __name__ == "__main__":
    fix_no42()
