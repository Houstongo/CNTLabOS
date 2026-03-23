#!/usr/bin/env python3
"""修复文件名格式问题

1. 修复 ` -` → `-`
2. 修复 No41 缺 position/mag 问题
"""

import os
import re
from pathlib import Path
import sys

# 添加父目录到路径
SCRIPT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from backend.core.sem_magnification import extract_magnification_from_png_metadata

ZZY_ROOT = Path(r"D:\CNTDATA\ZZY")


def fix_double_space_dash():
    """修复所有文件名中的 ` -` → `-`"""
    count = 0
    for f in os.listdir(ZZY_ROOT):
        if "  -" in f and f.lower().endswith(".png"):
            old_path = ZZY_ROOT / f
            new_name = f.replace("  -", " -")
            new_path = ZZY_ROOT / new_name
            if new_path.exists() and new_path != old_path:
                print(f"跳过（目标已存在）: {new_name}")
                continue
            os.rename(old_path, new_path)
            count += 1
            print(f"[{count}] {f} → {new_name}")
    print(f"\n修复 ` -` → `-`: 共 {count} 个文件")


def fix_no41_missing_position_mag():
    """修复 No41 缺 position/mag 的文件"""
    pattern = re.compile(r"^(No41 200w 5\.0nm \d+w [\d.]+nm \d+ \d+ \d+ \d+ \d+ \d+min \d+min)\s+-(\d+)\.png$")

    count = 0
    for f in os.listdir(ZZY_ROOT):
        match = pattern.match(f)
        if match:
            old_path = ZZY_ROOT / f
            prefix = match.group(1)
            repeat = match.group(2)

            # 从元数据获取倍率
            mag = extract_magnification_from_png_metadata(old_path)
            if not mag:
                print(f"跳过（无法提取倍率）: {f}")
                continue

            new_name = f"{prefix} mid {mag}-{repeat}.png"
            new_path = ZZY_ROOT / new_name
            if new_path.exists():
                print(f"跳过（已存在）: {new_name}")
                continue

            os.rename(old_path, new_path)
            count += 1
            print(f"[{count}] {f} → {new_name}")
    print(f"\n修复 No41 缺 position/mag: 共 {count} 个文件")


def main():
    print("=" * 60)
    print("修复文件名格式")
    print("=" * 60)

    print("\n[Step 1] 修复 ` -` → `-`")
    print("-" * 60)
    fix_double_space_dash()

    print("\n[Step 2] 修复 No41 缺 position/mag")
    print("-" * 60)
    fix_no41_missing_position_mag()

    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
