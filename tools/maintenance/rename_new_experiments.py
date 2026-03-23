#!/usr/bin/env python3
"""新实验数据批量重命名脚本

处理三组新实验数据：
1. Fe50w 组（149 张）→ No28
2. No41 氩气乙烯组（61 张）
3. No42 波动流速组（19 张，仅完整格式）
"""

import os
import re
import shutil
import sys
from pathlib import Path
from typing import Optional

# 添加父目录到路径以便导入 backend
SCRIPT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from backend.core.sem_magnification import extract_magnification_from_png_metadata

ZZY_ROOT = Path(r"D:\CNTDATA\ZZY")


def fix_fe50w_group():
    """Step 1: 重命名 Fe50w 组 → No28"""
    src_dir = ZZY_ROOT / "20260321 Fe50w"
    dst_dir = ZZY_ROOT
    count = 0

    for root, dirs, files in os.walk(src_dir):
        for f in files:
            if not f.lower().endswith(".png"):
                continue

            src_path = Path(root) / f
            # 添加 No28 前缀，修复 ` -` 为 `-`
            new_name = f"No28 {f.replace('  -', ' -')}"
            dst_path = dst_dir / new_name

            # 避免覆盖
            if dst_path.exists():
                print(f"跳过（已存在）: {new_name}")
                continue

            shutil.move(src_path, dst_path)
            count += 1
            print(f"[{count}] {f} → {new_name}")

    print(f"\nFe50w 组完成: 共 {count} 个文件")
    # 清空空目录
    for root, dirs, files in os.walk(src_dir, topdown=False):
        for d in dirs:
            try:
                os.rmdir(Path(root) / d)
            except OSError:
                pass
    try:
        os.rmdir(src_dir)
    except OSError:
        pass


def fix_no41_group():
    """Step 2: 修复 No41 组格式问题"""
    src_dir = ZZY_ROOT / "20260321 No41氩气乙烯先混和再通入"
    dst_dir = ZZY_ROOT
    count = 0

    for root, dirs, files in os.walk(src_dir):
        for f in files:
            if not f.lower().endswith(".png"):
                continue

            src_path = Path(root) / f
            base_name = f

            # 情况1: 修复 ` -` → `-`
            new_name = base_name.replace("  -", " -")

            # 情况2: position 和 mag 无空格，如 `bottom50000-1`
            new_name = re.sub(
                r"(top|mid|bottom|Top|Mid|Bottom|bpttom)(\d{4,6})-",
                r"\1 \2-",
                new_name,
            )

            # 情况3: mag+repeat 连写，如 `mid 1000001` → `mid 100000-1`
            new_name = re.sub(
                r"(top|mid|bottom|Top|Mid|Bottom|bpttom)\s+(\d{5,6})(\d)\.png$",
                r"\1 \2-\3.png",
                new_name,
            )

            # 情况4: 缺少 repeat，如 `mid 5000.png` → `mid 5000-1.png`
            if new_name.endswith(".png") and not re.search(r"\s+\d+-\d+\.png$", new_name):
                new_name = re.sub(r"\s+(\d+)\.png$", r" \1-1.png", new_name)

            # 情况5: 缺少 position/mag，如 `180min  -1.png`
            # 需要从元数据获取 mag，添加 mid position
            if re.search(r"\d+min\s+-\d+\.png$", new_name):
                mag = extract_magnification_from_png_metadata(src_path)
                if mag:
                    new_name = new_name.replace("  -", f" mid {mag}-")
                else:
                    print(f"跳过（无法提取倍率）: {f}")
                    continue

            dst_path = dst_dir / new_name
            if dst_path.exists():
                print(f"跳过（已存在）: {new_name}")
                continue

            shutil.move(src_path, dst_path)
            count += 1
            print(f"[{count}] {f} → {new_name}")

    print(f"\nNo41 组完成: 共 {count} 个文件")
    # 清空空目录
    for root, dirs, files in os.walk(src_dir, topdown=False):
        for d in dirs:
            try:
                os.rmdir(Path(root) / d)
            except OSError:
                pass
    try:
        os.rmdir(src_dir)
    except OSError:
        pass


def fix_no42_group():
    """Step 3: 重命名 No42 组完整格式文件"""
    src_dir = ZZY_ROOT / "20260321 No42 1英寸炉子 2xFlow 乙烯100到0波动半周期20s"
    dst_dir = ZZY_ROOT
    count = 0
    skipped = 0

    for root, dirs, files in os.walk(src_dir):
        for f in files:
            if not f.lower().endswith(".png"):
                continue

            src_path = Path(root) / f

            # 跳过简化格式（如 `2-2.75nm1.png`）
            if not re.match(r"\d+w\s+\d+nm", f):
                print(f"跳过（简化格式）: {f}")
                skipped += 1
                continue

            # 完整格式：添加 No42 前缀 + mid position + 从元数据获取 mag
            mag = extract_magnification_from_png_metadata(src_path)
            if not mag:
                print(f"跳过（无法提取倍率）: {f}")
                skipped += 1
                continue

            # 找到 repeat 号（最后一个数字）
            repeat_match = re.search(r"(\d+)\.png$", f)
            if not repeat_match:
                print(f"跳过（无 repeat）: {f}")
                skipped += 1
                continue
            repeat = repeat_match.group(1)

            # 去掉最后的 repeat 和 .png，然后添加标准格式
            base_without_repeat = re.sub(r"\d+\.png$", "", f)
            new_name = f"No42 {base_without_repeat}mid {mag}-{repeat}.png"

            dst_path = dst_dir / new_name
            if dst_path.exists():
                print(f"跳过（已存在）: {new_name}")
                continue

            shutil.move(src_path, dst_path)
            count += 1
            print(f"[{count}] {f} → {new_name}")

    print(f"\nNo42 组完成: {count} 个文件，跳过 {skipped} 个")
    # 清空空目录
    for root, dirs, files in os.walk(src_dir, topdown=False):
        for d in dirs:
            try:
                os.rmdir(Path(root) / d)
            except OSError:
                pass
    try:
        os.rmdir(src_dir)
    except OSError:
        pass


def main():
    print("=" * 60)
    print("新实验数据批量重命名脚本")
    print("=" * 60)

    print("\n[Step 1] 处理 Fe50w 组 → No28")
    print("-" * 60)
    fix_fe50w_group()

    print("\n[Step 2] 修复 No41 组格式")
    print("-" * 60)
    fix_no41_group()

    print("\n[Step 3] 重命名 No42 组完整格式")
    print("-" * 60)
    fix_no42_group()

    print("\n" + "=" * 60)
    print("全部完成！")
    print("=" * 60)
    print("\n下一步：")
    print("  python manage.py init-db")
    print("  python manage.py sync-mag")


if __name__ == "__main__":
    main()
