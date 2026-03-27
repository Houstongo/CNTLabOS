from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import cv2
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from experiments.cnt_paper_repro.config import load_config
from experiments.cnt_paper_repro.model import ResNet34UNet
from experiments.cnt_paper_repro.patching import extract_patch, extract_patch_specs
from src.analysis.feature_extractor import FeatureExtractor
from tools.generate_branch_selection_compare_panels import (
    build_mask_base,
    build_text_panel,
    draw_branches,
    estimate_diameter_p30_nm,
)


DEFAULT_SOURCE_DIRS = [
    Path(r"C:\Users\clearlove\Desktop\text10"),
    Path(r"C:\Users\clearlove\Desktop\text"),
]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "reports"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    config_path: Path
    checkpoint_path: Path
    mask_color: tuple[int, int, int]
    v2_color: tuple[int, int, int]
    v3_color: tuple[int, int, int]


BASELINE_SPEC = ModelSpec(
    key="baseline",
    label="Paper Repro Baseline",
    config_path=Path(
        r"D:\CNTDATA\CNTA_ML_Project\experiments\cnt_paper_repro\runs\cnt_paper_repro_100000x_center768_seed42\config_snapshot.yaml"
    ),
    checkpoint_path=Path(
        r"D:\CNTDATA\CNTA_ML_Project\experiments\cnt_paper_repro\runs\cnt_paper_repro_100000x_center768_seed42\best_model.pth"
    ),
    mask_color=(70, 120, 255),
    v2_color=(90, 255, 160),
    v3_color=(255, 200, 80),
)

EXPC_SPEC = ModelSpec(
    key="expc",
    label="Paper Repro Exp C",
    config_path=Path(
        r"D:\CNTDATA\CNTA_ML_Project\experiments\cnt_paper_repro\runs\cnt_paper_repro_100000x_center768_cldice_seed42\config_snapshot.yaml"
    ),
    checkpoint_path=Path(
        r"D:\CNTDATA\CNTA_ML_Project\experiments\cnt_paper_repro\runs\cnt_paper_repro_100000x_center768_cldice_seed42\best_model.pth"
    ),
    mask_color=(60, 200, 100),
    v2_color=(120, 255, 180),
    v3_color=(255, 170, 90),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate baseline vs Exp-C V2/V3 feature report for desktop CNT images.")
    parser.add_argument("--source-dir", type=Path, action="append", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=None)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_gray_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return image


def write_png(path: Path, image: np.ndarray) -> None:
    ensure_dir(path.parent)
    encoded = cv2.imencode(".png", image)[1]
    path.write_bytes(encoded.tobytes())


def infer_magnification(path: Path) -> int | None:
    match = re.search(r"(?<!\d)(50000|100000)(?!\d)", path.stem)
    if not match:
        return None
    return int(match.group(1))


def slugify(text: str) -> str:
    safe = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_"}:
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe).strip("_") or "image"


def collect_targets(source_dirs: List[Path]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for source_dir in source_dirs:
        if not source_dir.exists():
            continue
        for path in sorted(source_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
                continue
            magnification = infer_magnification(path)
            if magnification is None:
                continue
            records.append(
                {
                    "image_id": slugify(f"{source_dir.name}_{path.stem}"),
                    "source_dir": str(source_dir),
                    "file_name": path.name,
                    "file_path": str(path),
                    "magnification": magnification,
                }
            )
    return records


def load_model(spec: ModelSpec, device: torch.device) -> tuple[ResNet34UNet, dict]:
    config = load_config(spec.config_path)
    model = ResNet34UNet(
        in_channels=int(config["model"].get("in_channels", 1)),
        num_classes=int(config["model"].get("num_classes", 1)),
        encoder_weights=None,
    ).to(device)
    checkpoint = torch.load(spec.checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model, config


def predict_roi_mask(
    model: ResNet34UNet,
    roi_gray: np.ndarray,
    patch_size: int,
    stride: int,
    threshold: float,
    normalize_mean: float,
    normalize_std: float,
    device: torch.device,
    batch_size: int | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    specs = extract_patch_specs(roi_gray, patch_size=patch_size, mode="grid", stride=stride)
    accum = np.zeros(roi_gray.shape, dtype=np.float32)
    counts = np.zeros(roi_gray.shape, dtype=np.float32)
    if batch_size is None:
        batch_size = 8 if device.type == "cuda" else 2
    batch_size = max(1, int(batch_size))

    with torch.no_grad():
        for start in range(0, len(specs), batch_size):
            batch_specs = specs[start : start + batch_size]
            batch_patches = []
            for spec in batch_specs:
                patch = extract_patch(roi_gray, spec).astype(np.float32) / 255.0
                batch_patches.append(((patch - normalize_mean) / max(normalize_std, 1e-6)).astype(np.float32))

            tensor = torch.from_numpy(np.stack(batch_patches, axis=0)).unsqueeze(1).to(device)
            probs = torch.sigmoid(model(tensor)).detach().cpu().numpy()[:, 0]

            for spec, prob in zip(batch_specs, probs):
                valid_prob = prob[: spec.height, : spec.width]
                accum[spec.top : spec.top + spec.height, spec.left : spec.left + spec.width] += valid_prob
                counts[spec.top : spec.top + spec.height, spec.left : spec.left + spec.width] += 1.0

    prob_map = accum / np.maximum(counts, 1.0)
    mask = (prob_map >= threshold).astype(np.uint8) * 255
    return mask, prob_map, len(specs)


def analyze_mask_features(
    roi_gray: np.ndarray,
    magnification: int,
    mask: np.ndarray,
) -> Dict[str, Any]:
    extractor = FeatureExtractor(magnification=magnification, speed_profile="accurate")
    extractor._calibrate(roi_gray.shape[1])
    processed = extractor.preprocess(roi_gray)

    thresh = (np.asarray(mask) > 0).astype(np.uint8) * 255
    density = float(np.count_nonzero(thresh) / max(thresh.size, 1) * 100.0)
    diameter_nm, skel = extractor.calculate_diameter(thresh)
    base_components = extractor._collect_components(skel)
    ordered_branches_v2, ordered_branches_v3 = extractor._prepare_curvature_branch_sets(skel, v2_min_points=15)

    v2_label, v2_curvature_nm = extractor.calculate_curvature_v2(skel, ordered_branches=ordered_branches_v2)
    v3_label, v3_curvature_nm = extractor.calculate_curvature_v3(skel, ordered_branches=ordered_branches_v3)
    waviness_v2 = extractor.calculate_waviness_v2(skel, ordered_branches=ordered_branches_v2)
    alignment_metrics = extractor.calculate_hof_skeleton_adaptive(
        skel,
        processed=processed,
        base_components=base_components,
    )
    diameter_p30_nm = estimate_diameter_p30_nm(thresh, skel, extractor.px_per_um)

    return {
        "roi": roi_gray,
        "mask": thresh,
        "skeleton": skel,
        "diameter_nm": diameter_nm,
        "diameter_p30_nm": diameter_p30_nm,
        "density": density,
        "alignment": alignment_metrics["alignment"],
        "px_per_um": extractor.px_per_um,
        "ordered_branches_v2": ordered_branches_v2,
        "ordered_branches_v3": ordered_branches_v3,
        "v2_label": v2_label,
        "v2_curvature_nm": v2_curvature_nm,
        "v3_label": v3_label,
        "v3_curvature_nm": v3_curvature_nm,
        "waviness_ratio_v2": waviness_v2["waviness_ratio_v2"],
        "tortuosity_v2": waviness_v2["tortuosity_v2"],
    }


def build_mask_panel(mask: np.ndarray, branches: List[Dict[str, Any]], color: tuple[int, int, int]) -> np.ndarray:
    return draw_branches(build_mask_base(mask), branches, color)


def build_binary_mask_visual(mask: np.ndarray) -> np.ndarray:
    canvas = np.zeros((*mask.shape, 3), dtype=np.uint8)
    canvas[mask > 0] = (245, 245, 245)
    return canvas


def build_metrics_panel(file_name: str, magnification: int, baseline: Dict[str, Any], expc: Dict[str, Any], baseline_meta: Dict[str, Any], expc_meta: Dict[str, Any]) -> np.ndarray:
    lines = [
        f"file: {file_name}",
        f"magnification: {magnification}x",
        "",
        f"baseline patches: {baseline_meta['patch_count']}",
        f"expc patches: {expc_meta['patch_count']}",
        f"baseline threshold: {baseline_meta['threshold']:.2f}",
        f"expc threshold: {expc_meta['threshold']:.2f}",
        "",
        f"baseline density: {baseline['density']:.2f} %",
        f"baseline diameter p30: {baseline['diameter_p30_nm']}",
        f"baseline V2: {baseline['v2_curvature_nm']:.6f} ({baseline['v2_label']})",
        f"baseline V3: {baseline['v3_curvature_nm']:.6f} ({baseline['v3_label']})",
        f"baseline waviness_v2: {baseline['waviness_ratio_v2']}",
        f"baseline tortuosity_v2: {baseline['tortuosity_v2']}",
        f"baseline branches v2/v3: {len(baseline['ordered_branches_v2'])}/{len(baseline['ordered_branches_v3'])}",
        "",
        f"expc density: {expc['density']:.2f} %",
        f"expc diameter p30: {expc['diameter_p30_nm']}",
        f"expc V2: {expc['v2_curvature_nm']:.6f} ({expc['v2_label']})",
        f"expc V3: {expc['v3_curvature_nm']:.6f} ({expc['v3_label']})",
        f"expc waviness_v2: {expc['waviness_ratio_v2']}",
        f"expc tortuosity_v2: {expc['tortuosity_v2']}",
        f"expc branches v2/v3: {len(expc['ordered_branches_v2'])}/{len(expc['ordered_branches_v3'])}",
    ]
    return build_text_panel(lines)


def render_panel(
    output_path: Path,
    file_name: str,
    magnification: int,
    roi_gray: np.ndarray,
    baseline: Dict[str, Any],
    expc: Dict[str, Any],
    baseline_meta: Dict[str, Any],
    expc_meta: Dict[str, Any],
) -> None:
    metrics_panel = build_metrics_panel(file_name, magnification, baseline, expc, baseline_meta, expc_meta)

    panels = [
        ("Original ROI", cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR)),
        ("Baseline Mask", build_binary_mask_visual(baseline["mask"])),
        ("Exp C Mask", build_binary_mask_visual(expc["mask"])),
        ("Metrics", metrics_panel),
        (f"Baseline V2 ({len(baseline['ordered_branches_v2'])})", build_mask_panel(baseline["mask"], baseline["ordered_branches_v2"], BASELINE_SPEC.v2_color)),
        (f"Baseline V3 ({len(baseline['ordered_branches_v3'])})", build_mask_panel(baseline["mask"], baseline["ordered_branches_v3"], BASELINE_SPEC.v3_color)),
        (f"Exp C V2 ({len(expc['ordered_branches_v2'])})", build_mask_panel(expc["mask"], expc["ordered_branches_v2"], EXPC_SPEC.v2_color)),
        (f"Exp C V3 ({len(expc['ordered_branches_v3'])})", build_mask_panel(expc["mask"], expc["ordered_branches_v3"], EXPC_SPEC.v3_color)),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(22, 11), dpi=160, constrained_layout=True)
    for ax, (title, image) in zip(axes.flat, panels):
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else image, cmap=None if image.ndim == 3 else "gray")
        ax.set_title(title, fontsize=12)
        ax.axis("off")

    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    source_dirs = args.source_dir or DEFAULT_SOURCE_DIRS
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir or (DEFAULT_OUTPUT_ROOT / f"desktop_expc_baseline_v2v3_report_{timestamp}")
    ensure_dir(out_dir)

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model_bundle = {}
    for spec in (BASELINE_SPEC, EXPC_SPEC):
        model, config = load_model(spec, device)
        model_bundle[spec.key] = {
            "spec": spec,
            "model": model,
            "config": config,
        }

    records = collect_targets(source_dirs)
    if args.limit is not None:
        records = records[: max(0, int(args.limit))]

    manifest_rows: List[Dict[str, Any]] = []
    for record in records:
        path = Path(record["file_path"])
        item_dir = out_dir / "items" / str(record["image_id"])
        ensure_dir(item_dir)

        image_gray = read_gray_image(path)
        roi_gray = FeatureExtractor(magnification=int(record["magnification"])).extract_roi(image_gray)

        model_outputs: Dict[str, Dict[str, Any]] = {}
        feature_outputs: Dict[str, Dict[str, Any]] = {}
        for key, bundle in model_bundle.items():
            config = bundle["config"]
            mask, prob_map, patch_count = predict_roi_mask(
                model=bundle["model"],
                roi_gray=roi_gray,
                patch_size=int(config["data"].get("patch_size", 768)),
                stride=int(config["inference"].get("stride", config["data"].get("patch_size", 768) // 2)),
                threshold=float(config["inference"].get("threshold", 0.7)),
                normalize_mean=float(config["data"].get("normalize_mean", 0.5)),
                normalize_std=float(config["data"].get("normalize_std", 0.5)),
                device=device,
                batch_size=args.batch_size,
            )
            model_outputs[key] = {
                "mask": mask,
                "prob_map": prob_map,
                "patch_count": patch_count,
                "threshold": float(config["inference"].get("threshold", 0.7)),
            }
            feature_outputs[key] = analyze_mask_features(
                roi_gray=roi_gray,
                magnification=int(record["magnification"]),
                mask=mask,
            )
            write_png(item_dir / f"{key}_mask.png", mask.astype(np.uint8))
            write_png(item_dir / f"{key}_probability.png", np.clip(prob_map * 255.0, 0, 255).astype(np.uint8))

        panel_path = item_dir / "comparison_panel.png"
        render_panel(
            output_path=panel_path,
            file_name=path.name,
            magnification=int(record["magnification"]),
            roi_gray=roi_gray,
            baseline=feature_outputs["baseline"],
            expc=feature_outputs["expc"],
            baseline_meta=model_outputs["baseline"],
            expc_meta=model_outputs["expc"],
        )

        feature_record = {
            "image_id": record["image_id"],
            "source_dir": record["source_dir"],
            "file_name": record["file_name"],
            "file_path": record["file_path"],
            "magnification": record["magnification"],
            "panel_path": str(panel_path),
            "baseline_mask_path": str(item_dir / "baseline_mask.png"),
            "expc_mask_path": str(item_dir / "expc_mask.png"),
            "baseline_patch_count": model_outputs["baseline"]["patch_count"],
            "expc_patch_count": model_outputs["expc"]["patch_count"],
            "baseline_threshold": model_outputs["baseline"]["threshold"],
            "expc_threshold": model_outputs["expc"]["threshold"],
            "baseline_density": round(feature_outputs["baseline"]["density"], 4),
            "baseline_diameter_p30_nm": round(feature_outputs["baseline"]["diameter_p30_nm"], 2) if feature_outputs["baseline"]["diameter_p30_nm"] is not None else None,
            "baseline_curvature_nm_v2": round(feature_outputs["baseline"]["v2_curvature_nm"], 6),
            "baseline_curvature_nm_v3": round(feature_outputs["baseline"]["v3_curvature_nm"], 6),
            "baseline_curvature_v2_label": feature_outputs["baseline"]["v2_label"],
            "baseline_curvature_v3_label": feature_outputs["baseline"]["v3_label"],
            "baseline_waviness_ratio_v2": round(feature_outputs["baseline"]["waviness_ratio_v2"], 4) if feature_outputs["baseline"]["waviness_ratio_v2"] is not None else None,
            "baseline_tortuosity_v2": round(feature_outputs["baseline"]["tortuosity_v2"], 4),
            "baseline_v2_branch_count": len(feature_outputs["baseline"]["ordered_branches_v2"]),
            "baseline_v3_branch_count": len(feature_outputs["baseline"]["ordered_branches_v3"]),
            "expc_density": round(feature_outputs["expc"]["density"], 4),
            "expc_diameter_p30_nm": round(feature_outputs["expc"]["diameter_p30_nm"], 2) if feature_outputs["expc"]["diameter_p30_nm"] is not None else None,
            "expc_curvature_nm_v2": round(feature_outputs["expc"]["v2_curvature_nm"], 6),
            "expc_curvature_nm_v3": round(feature_outputs["expc"]["v3_curvature_nm"], 6),
            "expc_curvature_v2_label": feature_outputs["expc"]["v2_label"],
            "expc_curvature_v3_label": feature_outputs["expc"]["v3_label"],
            "expc_waviness_ratio_v2": round(feature_outputs["expc"]["waviness_ratio_v2"], 4) if feature_outputs["expc"]["waviness_ratio_v2"] is not None else None,
            "expc_tortuosity_v2": round(feature_outputs["expc"]["tortuosity_v2"], 4),
            "expc_v2_branch_count": len(feature_outputs["expc"]["ordered_branches_v2"]),
            "expc_v3_branch_count": len(feature_outputs["expc"]["ordered_branches_v3"]),
        }
        (item_dir / "features.json").write_text(json.dumps(feature_record, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest_rows.append(feature_record)

    if manifest_rows:
        with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(manifest_rows[0].keys()))
            writer.writeheader()
            writer.writerows(manifest_rows)

    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "output_dir": str(out_dir),
                "device": str(device),
                "count": len(manifest_rows),
                "source_dirs": [str(path) for path in source_dirs],
                "records": manifest_rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(out_dir)


if __name__ == "__main__":
    main()
