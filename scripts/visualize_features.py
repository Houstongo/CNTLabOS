#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CNT 阵列 SEM 图像特征可视化脚本"""

import os
import sys
import json
import argparse
import base64
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
# CNTSegNet 模型路径
VLMSAM_ROOT = PROJECT_ROOT.parent / "VLMSAM"
if VLMSAM_ROOT.exists():
    sys.path.insert(0, str(VLMSAM_ROOT))

import cv2
import numpy as np


class FeatureVisualizer:
    """特征可视化器 - 支持两种分割后端"""

    def __init__(self, backend="cntsegnet", output_dir="./vis_output",
                 device="cpu", checkpoint=None, tile_size=512, overlap=64,
                 seg_threshold=0.5, magnification=None):
        self.backend = backend
        self.output_dir = Path(output_dir)
        self.device = device
        self.checkpoint = checkpoint
        self.tile_size = tile_size
        self.overlap = overlap
        self.seg_threshold = seg_threshold
        self.magnification = magnification
        self._threshold_vis = None
        self._cntsegnet_vis = None

    def _get_threshold_vis(self, mag):
        if self._threshold_vis is None:
            from backend.core.algorithm_visualizer import AlgorithmVisualizer
            self._threshold_vis = AlgorithmVisualizer(magnification=mag)
        else:
            self._threshold_vis.mag = mag
        return self._threshold_vis

    def _get_cntsegnet_vis(self, mag):
        if self._cntsegnet_vis is None:
            from backend.core.cntsegnet_visualizer import CNTSegNetVisualizer
            self._cntsegnet_vis = CNTSegNetVisualizer(
                magnification=mag, device=self.device,
                checkpoint_path=self.checkpoint, tile_size=self.tile_size,
                overlap=self.overlap, seg_threshold=self.seg_threshold)
        else:
            self._cntsegnet_vis.mag = mag
        return self._cntsegnet_vis

    def visualize_single(self, img_path, save_steps=True, save_grid=True, grid_size=(3, 4)):
        """处理单张图片"""
        img_path = Path(img_path)
        if not img_path.exists():
            raise FileNotFoundError(f"图片不存在: {img_path}")

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"无法读取图片: {img_path}")

        mag = self.magnification or self._parse_magnification(img_path)
        img_name = img_path.stem
        output_subdir = self.output_dir / img_name
        output_subdir.mkdir(parents=True, exist_ok=True)

        results = {"image_path": str(img_path), "image_name": img_name,
                   "magnification": mag, "backend": self.backend, "outputs": []}

        if self.backend in ("threshold", "both"):
            r = self._process_backend(img, "threshold", mag, output_subdir,
                                       save_steps, save_grid, grid_size, img_name)
            results["threshold"] = r

        if self.backend in ("cntsegnet", "both"):
            r = self._process_backend(img, "cntsegnet", mag, output_subdir,
                                       save_steps, save_grid, grid_size, img_name)
            results["cntsegnet"] = r

        return results

    def _process_backend(self, img, backend, mag, output_dir, save_steps, save_grid, grid_size, img_name):
        """使用指定后端处理"""
        suffix = "" if backend == "threshold" else "_cntsegnet"

        if backend == "threshold":
            vis = self._get_threshold_vis(mag)
        else:
            vis = self._get_cntsegnet_vis(mag)

        steps = vis.visualize_extraction(img)
        result = {"backend": backend, "step_files": [], "features": {}}

        if save_steps:
            result["step_files"] = self._save_steps(steps, output_dir, suffix)

        if save_grid:
            grid_path = output_dir.parent / f"{img_name}{suffix}_grid.png"
            self._save_grid(steps, grid_path, grid_size)
            result["grid_file"] = str(grid_path)

        result["features"] = self._extract_features(steps)

        features_path = output_dir / f"features{suffix}.json"
        with open(features_path, "w", encoding="utf-8") as f:
            json.dump(result["features"], f, indent=2, ensure_ascii=False)
        result["features_file"] = str(features_path)

        return result

    # 步骤名称中英文映射
    STEP_NAME_MAP = {
        "原始图像": "original",
        "模型配置": "model_config",
        "CLAHE增强": "clahe",
        "高斯模糊": "gaussian_blur",
        "ROI提取": "roi_extract",
        "二值化": "binary",
        "Tile网格规划": "tile_grid",
        "分块推理热图": "inference_heatmap",
        "概率图融合": "prob_fusion",
        "密度计算": "density",
        "完整骨架": "skeleton_full",
        "最大骨架": "skeleton_max",
        "对齐方向场": "alignment",
        "直径测量": "diameter",
        "骨架追踪曲率": "curvature",
        "波曲度分析": "tortuosity",
    }

    def _save_steps(self, steps, output_dir, suffix=""):
        """保存步骤图片"""
        files = []
        for i, step in enumerate(steps):
            img_data = base64.b64decode(step["image"])
            img_array = np.frombuffer(img_data, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            step_num = f"{i + 1:02d}"
            # 使用英文文件名
            cn_name = step["name"]
            en_name = self.STEP_NAME_MAP.get(cn_name, f"step{i + 1}")
            filename = f"{step_num}_{en_name}{suffix}.png"
            filepath = output_dir / filename
            cv2.imwrite(str(filepath), img)
            files.append(str(filepath))
        return files

    def _save_grid(self, steps, output_path, grid_size=(3, 4)):
        """生成拼图"""
        rows, cols = grid_size
        n = min(len(steps), rows * cols)
        if n == 0:
            return

        images = []
        for i in range(n):
            img_data = base64.b64decode(steps[i]["image"])
            img_array = np.frombuffer(img_data, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            images.append(img)

        h, w = images[0].shape[:2]
        grid_img = np.ones((h * rows, w * cols, 3), dtype=np.uint8) * 255

        for idx, img in enumerate(images):
            if idx >= rows * cols:
                break
            row, col = idx // cols, idx % cols
            y1, x1 = row * h, col * w
            y2, x2 = y1 + h, x1 + w
            if img.shape[:2] != (h, w):
                img = cv2.resize(img, (w, h))
            grid_img[y1:y2, x1:x2] = img
            cv2.putText(grid_img, f"{idx + 1}.{steps[idx]['name']}", (x1 + 5, y1 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        cv2.imwrite(str(output_path), grid_img)

    def _extract_features(self, steps):
        """从步骤描述提取特征值"""
        features = {}
        for step in steps:
            desc = step.get("description", "")
            name = step.get("name", "")

            if "density" in name.lower() or "密度" in name:
                m = re.search(r"density\s*=\s*([\d.]+)", desc, re.I)
                if m:
                    features["density"] = float(m.group(1))

            if "curvature" in name.lower() or "曲率" in name:
                # 匹配 κ = X.XXX 或 κ = X.XXX nm 格式
                m = re.search(r"κ\s*=\s*([\d.]+)", desc)
                if not m:
                    m = re.search(r"curvature.*?=\s*([\d.]+)", desc, re.I)
                if m:
                    features["curvature"] = float(m.group(1))

            if "tortuosity" in name.lower() or "波曲度" in name:
                m = re.search(r"τ\s*=\s*([\d.]+)", desc)
                if not m:
                    m = re.search(r"tortuosity\s*=\s*([\d.]+)", desc, re.I)
                if m:
                    features["tortuosity"] = float(m.group(1))

            # 取向度
            if "alignment" in name.lower() or "对齐" in name or "方向" in name:
                # 匹配 alignment = X.XXX 格式，避免捕获末尾标点
                m = re.search(r"alignment\s*=\s*(\d+\.\d+)", desc, re.I)
                if m:
                    features["alignment"] = float(m.group(1))

            # 直径分布 - 提取均值、标准差、中位数、P10/P90
            if "diameter" in name.lower() or "直径" in name:
                # 平均直径
                m = re.search(r"平均直径\s*=\s*([\d.]+)\s*nm", desc)
                if not m:
                    m = re.search(r"avg.*?=\s*([\d.]+)\s*nm", desc, re.I)
                if m:
                    features["diameter_avg_nm"] = float(m.group(1))

                # 标准差
                m = re.search(r"标准差\s*=\s*([\d.]+)\s*nm", desc)
                if not m:
                    m = re.search(r"std\s*=\s*([\d.]+)\s*nm", desc, re.I)
                if m:
                    features["diameter_std_nm"] = float(m.group(1))

                # 中位数
                m = re.search(r"中位数\s*=\s*([\d.]+)\s*nm", desc)
                if not m:
                    m = re.search(r"median\s*=\s*([\d.]+)\s*nm", desc, re.I)
                if m:
                    features["diameter_median_nm"] = float(m.group(1))

                # P10/P90
                m = re.search(r"P10/P90\s*=\s*([\d.]+)/([\d.]+)\s*nm", desc)
                if m:
                    features["diameter_p10_nm"] = float(m.group(1))
                    features["diameter_p90_nm"] = float(m.group(2))
        return features

    def visualize_batch(self, input_dir, pattern="*.png", limit=None, save_steps=True, save_grid=True):
        """批量处理目录"""
        input_path = Path(input_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"目录不存在: {input_dir}")

        files = list(input_path.glob(pattern))
        if limit:
            files = files[:limit]

        if not files:
            print(f"未找到匹配 '{pattern}' 的文件")
            return []

        print(f"找到 {len(files)} 个文件待处理")
        results = []

        for i, img_path in enumerate(files):
            print(f"[{i + 1}/{len(files)}] 处理: {img_path.name}")
            try:
                result = self.visualize_single(str(img_path), save_steps, save_grid)
                results.append(result)
            except Exception as e:
                print(f"  错误: {e}")
                results.append({"image_path": str(img_path), "error": str(e)})

        summary_path = self.output_dir / "batch_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"批量汇总已保存: {summary_path}")
        return results

    def _parse_magnification(self, img_path):
        """从文件名解析倍率"""
        m = re.search(r"(\d{4,6})(?:-\d+)?\.(?:png|tif?f)$", img_path.name, re.I)
        return int(m.group(1)) if m else 50000


def main():
    parser = argparse.ArgumentParser(description="CNT 阵列 SEM 图像特征可视化")
    parser.add_argument("--input", "-i", default=None, help="输入文件或目录路径")
    parser.add_argument("--file-list", "-f", default=None, help="从文件读取输入列表")
    parser.add_argument("--output", "-o", default="./vis_output", help="输出目录")
    parser.add_argument("--backend", "-b", choices=["threshold", "cntsegnet", "both"],
                        default="cntsegnet", help="分割后端")
    parser.add_argument("--device", default="cpu", help="CNTSegNet 设备: cpu/cuda")
    parser.add_argument("--checkpoint", default=None, help="模型权重路径")
    parser.add_argument("--mag", type=int, default=None, help="倍率")
    parser.add_argument("--grid-size", default="3x4", help="拼图网格大小")
    parser.add_argument("--no-grid", action="store_true", help="禁用拼图")
    parser.add_argument("--no-steps", action="store_true", help="禁用步骤图片")
    parser.add_argument("--limit", type=int, default=None, help="批量处理数量限制")
    parser.add_argument("--pattern", default="*.png", help="文件匹配模式")
    parser.add_argument("--tile-size", type=int, default=512, help="分块大小")
    parser.add_argument("--overlap", type=int, default=64, help="分块重叠")
    parser.add_argument("--seg-threshold", type=float, default=0.5, help="分割阈值")
    args = parser.parse_args()

    grid_size = tuple(map(int, args.grid_size.lower().split("x")))

    visualizer = FeatureVisualizer(
        backend=args.backend, output_dir=args.output, device=args.device,
        checkpoint=args.checkpoint, tile_size=args.tile_size, overlap=args.overlap,
        seg_threshold=args.seg_threshold, magnification=args.mag)

    # 从文件列表读取输入
    if args.file_list:
        with open(args.file_list, 'r', encoding='utf-8') as f:
            files = [line.strip() for line in f if line.strip()]
        if args.limit:
            files = files[:args.limit]
        print(f"从列表读取 {len(files)} 个文件")
        results = []
        for i, img_path in enumerate(files):
            print(f"[{i + 1}/{len(files)}] {Path(img_path).name}")
            try:
                result = visualizer.visualize_single(
                    img_path, save_steps=not args.no_steps,
                    save_grid=not args.no_grid, grid_size=grid_size)
                results.append(result)
            except Exception as e:
                print(f"  Error: {e}")
                results.append({"image_path": img_path, "error": str(e)})
        success = sum(1 for r in results if "error" not in r)
        print(f"Done: {success}/{len(results)}")
        summary_path = Path(args.output) / "batch_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        return

    input_path = Path(args.input)

    if input_path.is_file():
        print(f"处理单文件: {input_path}")
        result = visualizer.visualize_single(
            str(input_path), save_steps=not args.no_steps,
            save_grid=not args.no_grid, grid_size=grid_size)
        print(f"输出目录: {args.output}/{input_path.stem}/")
        for backend in ["threshold", "cntsegnet"]:
            if backend in result:
                print(f"  {backend}: {len(result[backend].get('step_files', []))} 个文件")
                print(f"    特征: {result[backend].get('features', {})}")

    elif input_path.is_dir():
        print(f"批量处理: {input_path}")
        results = visualizer.visualize_batch(
            str(input_path), pattern=args.pattern, limit=args.limit,
            save_steps=not args.no_steps, save_grid=not args.no_grid)
        success = sum(1 for r in results if "error" not in r)
        print(f"完成: {success}/{len(results)} 成功")

    else:
        print(f"错误: 路径不存在: {args.input}")
        sys.exit(1)


if __name__ == "__main__":
    main()
