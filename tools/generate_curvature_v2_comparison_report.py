"""Generate an offline comparison report for the paper-repro checkpoint vs frozen WCNTSegNET results."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import cv2
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.cnt_paper_repro.config import load_config  # noqa: E402
from experiments.cnt_paper_repro.model import ResNet34UNet  # noqa: E402
from experiments.cnt_paper_repro.patching import extract_patch, extract_patch_specs  # noqa: E402
from src.analysis.feature_extractor import FeatureExtractor  # noqa: E402


DEFAULT_BASELINE_SUMMARY = PROJECT_ROOT / "reports" / "zzy_wcntsegnet_full_batch_20260324" / "summary.json"
DEFAULT_BASELINE_CSV = PROJECT_ROOT / "reports" / "zzy_wcntsegnet_full_batch_20260324" / "batch_features.csv"
DEFAULT_CONFIG = PROJECT_ROOT / "experiments" / "cnt_paper_repro" / "configs" / "paper_100000x_cldice.yaml"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "experiments" / "cnt_paper_repro" / "runs" / "cnt_paper_repro_100000x_center768_cldice_seed42" / "best_model.pth"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "reports"


@dataclass
class BaselineRecord:
    image_id: int
    sample_id: str
    file_path: str
    magnification: Optional[int]
    output_dir: Optional[str]
    summary_item: Dict[str, Any]
    csv_row: Dict[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate curvature/waviness comparison panels for the paper-repro checkpoint.")
    parser.add_argument("--baseline-summary", type=Path, default=DEFAULT_BASELINE_SUMMARY)
    parser.add_argument("--baseline-csv", type=Path, default=DEFAULT_BASELINE_CSV)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--patch-size", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None, help="Grid stride. Defaults to patch_size // 2.")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_gray_image(path: str | Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return image


def write_image(path: str | Path, image: np.ndarray) -> None:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    encoded = cv2.imencode(output_path.suffix or ".png", image)[1]
    output_path.write_bytes(encoded.tobytes())


def slugify(text: str) -> str:
    safe = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_"}:
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe).strip("_") or "image"


def safe_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_builtin(item) for item in value]
    if isinstance(value, tuple):
        return [to_builtin(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def format_metric(value: Any, pattern: str = "{:.4f}") -> str:
    numeric = safe_float(value)
    if numeric is None:
        if value in (None, ""):
            return "N/A"
        return str(value)
    return pattern.format(numeric)


def load_baseline_records(summary_path: Path, csv_path: Path, limit: Optional[int] = None) -> List[BaselineRecord]:
    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))

    csv_rows: Dict[int, Dict[str, str]] = {}
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                image_id = int(row["image_id"])
                csv_rows[image_id] = row

    records: List[BaselineRecord] = []
    for item in summary_payload.get("items", []):
        if item.get("status") != "success":
            continue
        image_id = int(item["image_id"])
        records.append(
            BaselineRecord(
                image_id=image_id,
                sample_id=item.get("sample_id") or f"image-{image_id}",
                file_path=item["file_path"],
                magnification=int(item["magnification"]) if item.get("magnification") is not None else None,
                output_dir=item.get("output_dir"),
                summary_item=item,
                csv_row=csv_rows.get(image_id, {}),
            )
        )

    if limit is not None:
        records = records[: max(limit, 0)]
    return records


def find_baseline_mask_path(output_dir: Optional[str]) -> Optional[Path]:
    if not output_dir:
        return None
    root = Path(output_dir)
    if not root.exists():
        return None

    candidates = [
        root / "wcntsegnet_mask.png",
        root / "mask_wcntsegnet.png",
        root / "threshold_mask.png",
    ]
    for path in candidates:
        if path.exists():
            return path

    for pattern in ("*wcntsegnet*mask*.png", "*threshold*mask*.png", "*mask*.png"):
        matches = sorted(root.glob(pattern))
        if matches:
            return matches[0]
    return None


def build_overlay(roi_gray: np.ndarray, mask: Optional[np.ndarray], color: tuple[int, int, int]) -> np.ndarray:
    canvas = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR)
    if mask is None:
        return canvas

    mask_u8 = (mask > 0).astype(np.uint8)
    if mask_u8.shape != roi_gray.shape:
        mask_u8 = cv2.resize(mask_u8, (roi_gray.shape[1], roi_gray.shape[0]), interpolation=cv2.INTER_NEAREST)

    overlay = np.zeros_like(canvas)
    overlay[mask_u8 > 0] = color
    blended = cv2.addWeighted(canvas, 0.76, overlay, 0.24, 0.0)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(blended, contours, -1, (255, 255, 255), 1)
    return blended


def resolve_device(config: Dict[str, Any], device_override: str) -> torch.device:
    if device_override and device_override.lower() != "auto":
        return torch.device(device_override)

    configured = str(config["training"].get("device", "auto"))
    if configured != "auto":
        return torch.device(configured)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(config: Dict[str, Any], checkpoint_path: Path, device: torch.device) -> ResNet34UNet:
    model = ResNet34UNet(
        in_channels=int(config["model"].get("in_channels", 1)),
        num_classes=int(config["model"].get("num_classes", 1)),
        encoder_weights=None,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict_roi_mask(
    model: ResNet34UNet,
    roi_gray: np.ndarray,
    patch_size: int,
    stride: int,
    threshold: float,
    normalize_mean: float,
    normalize_std: float,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, int]:
    specs = extract_patch_specs(roi_gray, patch_size=patch_size, mode="grid", stride=stride)
    accum = np.zeros(roi_gray.shape, dtype=np.float32)
    counts = np.zeros(roi_gray.shape, dtype=np.float32)

    with torch.no_grad():
        for spec in specs:
            patch = extract_patch(roi_gray, spec).astype(np.float32) / 255.0
            tensor = torch.from_numpy(((patch - normalize_mean) / max(normalize_std, 1e-6)).astype(np.float32))
            tensor = tensor.unsqueeze(0).unsqueeze(0).to(device)
            prob = torch.sigmoid(model(tensor))[0, 0].detach().cpu().numpy()
            valid_prob = prob[: spec.height, : spec.width]
            accum[spec.top : spec.top + spec.height, spec.left : spec.left + spec.width] += valid_prob
            counts[spec.top : spec.top + spec.height, spec.left : spec.left + spec.width] += 1.0

    prob_map = accum / np.maximum(counts, 1.0)
    mask = (prob_map >= threshold).astype(np.uint8) * 255
    return mask, prob_map, len(specs)


def collect_numeric_snapshot(features: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    keys = [
        "density",
        "alignment",
        "diameter",
        "curvature",
        "curvature_nm",
        "curvature_v2",
        "curvature_nm_v2",
        "tortuosity",
        "tortuosity_v2",
        "waviness_ratio",
        "waviness_ratio_v2",
        "waviness_height_nm",
        "waviness_height_nm_v2",
        "waviness_wavelength_nm",
        "waviness_wavelength_nm_v2",
        "waviness_branches",
        "waviness_branches_v2",
        "px_per_um",
        "n_branches",
        "rotation_correction_deg",
    ]
    snapshot: Dict[str, Any] = {}
    for key in keys:
        if key in features:
            snapshot[f"{prefix}{key}" if prefix else key] = to_builtin(features.get(key))
    return snapshot


def flatten_row(
    record: BaselineRecord,
    baseline_features: Dict[str, Any],
    new_features: Dict[str, Any],
    item_dir: Path,
    baseline_mask_path: Optional[Path],
    patch_count: int,
    inference_s: float,
    status: str,
    error: Optional[str],
) -> Dict[str, Any]:
    baseline_curvature = safe_float(baseline_features.get("curvature_nm") or record.csv_row.get("wcntsegnet_curvature_nm"))
    baseline_waviness = safe_float(baseline_features.get("waviness_ratio"))
    baseline_tortuosity = safe_float(baseline_features.get("tortuosity"))

    row = {
        "image_id": record.image_id,
        "sample_id": record.sample_id,
        "file_path": record.file_path,
        "magnification": record.magnification,
        "status": status,
        "error": error or "",
        "patch_count": patch_count,
        "inference_s": round(inference_s, 3),
        "item_dir": str(item_dir),
        "panel_path": str(item_dir / "comparison_panel.png"),
        "baseline_mask_path": str(baseline_mask_path) if baseline_mask_path else "",
        "new_mask_path": str(item_dir / "paper_repro_mask.png"),
        "baseline_curvature_label": baseline_features.get("curvature"),
        "baseline_curvature_nm": baseline_curvature,
        "baseline_waviness_ratio": baseline_waviness,
        "baseline_tortuosity": baseline_tortuosity,
        "new_curvature_label": new_features.get("curvature"),
        "new_curvature_nm": safe_float(new_features.get("curvature_nm")),
        "new_curvature_v2_label": new_features.get("curvature_v2"),
        "new_curvature_nm_v2": safe_float(new_features.get("curvature_nm_v2")),
        "new_waviness_ratio": safe_float(new_features.get("waviness_ratio")),
        "new_waviness_ratio_v2": safe_float(new_features.get("waviness_ratio_v2")),
        "new_tortuosity": safe_float(new_features.get("tortuosity")),
        "new_tortuosity_v2": safe_float(new_features.get("tortuosity_v2")),
        "delta_curvature_nm_legacy_vs_baseline": None,
        "delta_curvature_nm_v2_vs_baseline": None,
        "delta_waviness_ratio_legacy_vs_baseline": None,
        "delta_waviness_ratio_v2_vs_baseline": None,
        "delta_tortuosity_legacy_vs_baseline": None,
        "delta_tortuosity_v2_vs_baseline": None,
    }

    if baseline_curvature is not None:
        legacy_curvature = safe_float(new_features.get("curvature_nm"))
        v2_curvature = safe_float(new_features.get("curvature_nm_v2"))
        if legacy_curvature is not None:
            row["delta_curvature_nm_legacy_vs_baseline"] = legacy_curvature - baseline_curvature
        if v2_curvature is not None:
            row["delta_curvature_nm_v2_vs_baseline"] = v2_curvature - baseline_curvature

    if baseline_waviness is not None:
        legacy_waviness = safe_float(new_features.get("waviness_ratio"))
        v2_waviness = safe_float(new_features.get("waviness_ratio_v2"))
        if legacy_waviness is not None:
            row["delta_waviness_ratio_legacy_vs_baseline"] = legacy_waviness - baseline_waviness
        if v2_waviness is not None:
            row["delta_waviness_ratio_v2_vs_baseline"] = v2_waviness - baseline_waviness

    if baseline_tortuosity is not None:
        legacy_tortuosity = safe_float(new_features.get("tortuosity"))
        v2_tortuosity = safe_float(new_features.get("tortuosity_v2"))
        if legacy_tortuosity is not None:
            row["delta_tortuosity_legacy_vs_baseline"] = legacy_tortuosity - baseline_tortuosity
        if v2_tortuosity is not None:
            row["delta_tortuosity_v2_vs_baseline"] = v2_tortuosity - baseline_tortuosity

    row.update(collect_numeric_snapshot(baseline_features, prefix="baseline_"))
    row.update(collect_numeric_snapshot(new_features, prefix="new_"))
    return row


def render_panel(
    output_path: Path,
    record: BaselineRecord,
    roi_gray: np.ndarray,
    baseline_mask: Optional[np.ndarray],
    new_mask: np.ndarray,
    baseline_features: Dict[str, Any],
    new_features: Dict[str, Any],
    patch_count: int,
) -> None:
    fig = plt.figure(figsize=(16, 10), dpi=120)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.0], height_ratios=[1.0, 1.0])

    ax_original = fig.add_subplot(grid[0, 0])
    ax_original.imshow(roi_gray, cmap="gray")
    ax_original.set_title("ROI Original")
    ax_original.axis("off")

    ax_baseline = fig.add_subplot(grid[0, 1])
    ax_baseline.imshow(cv2.cvtColor(build_overlay(roi_gray, baseline_mask, (64, 96, 255)), cv2.COLOR_BGR2RGB))
    ax_baseline.set_title("Yesterday WCNTSegNET")
    ax_baseline.axis("off")

    ax_new = fig.add_subplot(grid[1, 0])
    ax_new.imshow(cv2.cvtColor(build_overlay(roi_gray, new_mask, (40, 170, 90)), cv2.COLOR_BGR2RGB))
    ax_new.set_title(f"Paper-Repro CLDice ({patch_count} patches)")
    ax_new.axis("off")

    ax_metrics = fig.add_subplot(grid[1, 1])
    ax_metrics.axis("off")

    header_lines = [
        f"image_id: {record.image_id}",
        f"sample_id: {record.sample_id}",
        f"magnification: {record.magnification or 'N/A'}",
        Path(record.file_path).name,
    ]

    y = 0.96
    ax_metrics.text(0.0, y, "Curvature / Waviness Comparison", fontsize=18, fontweight="bold", transform=ax_metrics.transAxes)
    y -= 0.10
    for line in header_lines:
        ax_metrics.text(0.0, y, line, fontsize=10.5, color="#334155", transform=ax_metrics.transAxes)
        y -= 0.055

    table_rows = [
        ["Metric", "Baseline WCNT", "New Legacy", "New V2 / Delta"],
        [
            "Curvature (nm^-1)",
            format_metric(baseline_features.get("curvature_nm")),
            format_metric(new_features.get("curvature_nm")),
            f"{format_metric(new_features.get('curvature_nm_v2'))} / {format_metric(safe_float(new_features.get('curvature_nm_v2')) - safe_float(baseline_features.get('curvature_nm')) if safe_float(new_features.get('curvature_nm_v2')) is not None and safe_float(baseline_features.get('curvature_nm')) is not None else None)}",
        ],
        [
            "Curvature label",
            str(baseline_features.get("curvature", "N/A")),
            str(new_features.get("curvature", "N/A")),
            str(new_features.get("curvature_v2", "N/A")),
        ],
        [
            "Waviness ratio",
            format_metric(baseline_features.get("waviness_ratio")),
            format_metric(new_features.get("waviness_ratio")),
            f"{format_metric(new_features.get('waviness_ratio_v2'))} / {format_metric(safe_float(new_features.get('waviness_ratio_v2')) - safe_float(baseline_features.get('waviness_ratio')) if safe_float(new_features.get('waviness_ratio_v2')) is not None and safe_float(baseline_features.get('waviness_ratio')) is not None else None)}",
        ],
        [
            "Tortuosity",
            format_metric(baseline_features.get("tortuosity")),
            format_metric(new_features.get("tortuosity")),
            f"{format_metric(new_features.get('tortuosity_v2'))} / {format_metric(safe_float(new_features.get('tortuosity_v2')) - safe_float(baseline_features.get('tortuosity')) if safe_float(new_features.get('tortuosity_v2')) is not None and safe_float(baseline_features.get('tortuosity')) is not None else None)}",
        ],
        [
            "Wavy branches",
            format_metric(baseline_features.get("waviness_branches"), "{:.0f}"),
            format_metric(new_features.get("waviness_branches"), "{:.0f}"),
            format_metric(new_features.get("waviness_branches_v2"), "{:.0f}"),
        ],
        [
            "Density (%)",
            format_metric(baseline_features.get("density"), "{:.2f}"),
            format_metric(new_features.get("density"), "{:.2f}"),
            "",
        ],
        [
            "Alignment",
            format_metric(baseline_features.get("alignment")),
            format_metric(new_features.get("alignment")),
            "",
        ],
    ]

    table = ax_metrics.table(
        cellText=table_rows[1:],
        colLabels=table_rows[0],
        cellLoc="center",
        colLoc="center",
        bbox=[0.0, 0.02, 0.98, 0.56],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1.0, 1.5)

    for col in range(4):
        table[(0, col)].set_facecolor("#0f766e")
        table[(0, col)].set_text_props(color="white", fontweight="bold")

    for row_index in range(1, len(table_rows)):
        table[(row_index, 0)].set_facecolor("#ecfeff")
        table[(row_index, 0)].set_text_props(color="#155e75", fontweight="bold")

    fig.suptitle(f"{record.sample_id} | {Path(record.file_path).name}", fontsize=15, y=0.98)
    fig.tight_layout()
    ensure_dir(output_path.parent)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    ensure_dir(path.parent)
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: to_builtin(value) for key, value in row.items()})


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = resolve_device(config, args.device)
    threshold = float(args.threshold if args.threshold is not None else config["inference"].get("threshold", 0.7))
    patch_size = int(args.patch_size if args.patch_size is not None else config["data"].get("patch_size", 768))
    stride = int(args.stride if args.stride is not None else max(1, patch_size // 2))
    normalize_mean = float(config["data"].get("normalize_mean", 0.5))
    normalize_std = float(config["data"].get("normalize_std", 0.5))

    if args.output_dir is not None:
        output_root = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_root = DEFAULT_OUTPUT_ROOT / f"curvature_v2_comparison_{timestamp}"

    ensure_dir(output_root)
    ensure_dir(output_root / "items")

    records = load_baseline_records(args.baseline_summary, args.baseline_csv, limit=args.limit)
    if not records:
        raise SystemExit("No successful baseline records were found.")

    model = load_model(config, args.checkpoint, device)
    comparison_rows: List[Dict[str, Any]] = []
    summary_items: List[Dict[str, Any]] = []

    for index, record in enumerate(records, start=1):
        item_dir = output_root / "items" / f"{record.image_id}_{slugify(Path(record.file_path).stem)}"
        ensure_dir(item_dir)
        baseline_features = dict(record.summary_item.get("wcntsegnet_features") or {})
        baseline_mask_path = find_baseline_mask_path(record.output_dir)
        patch_count = 0
        inference_s = 0.0

        print(f"[{index}/{len(records)}] image_id={record.image_id} sample={record.sample_id}")
        try:
            image_gray = read_gray_image(record.file_path)
            roi_gray = FeatureExtractor.extract_roi(image_gray)
            baseline_mask = read_gray_image(baseline_mask_path) if baseline_mask_path else None
            if baseline_mask is not None and baseline_mask.shape != roi_gray.shape:
                baseline_mask = cv2.resize(baseline_mask, (roi_gray.shape[1], roi_gray.shape[0]), interpolation=cv2.INTER_NEAREST)

            started_at = time.perf_counter()
            new_mask, prob_map, patch_count = predict_roi_mask(
                model=model,
                roi_gray=roi_gray,
                patch_size=patch_size,
                stride=stride,
                threshold=threshold,
                normalize_mean=normalize_mean,
                normalize_std=normalize_std,
                device=device,
            )
            inference_s = time.perf_counter() - started_at

            extractor = FeatureExtractor(magnification=record.magnification)
            new_features = extractor.extract_all(image_gray, external_binary_mask=new_mask)

            write_image(item_dir / "roi_original.png", roi_gray)
            if baseline_mask is not None:
                write_image(item_dir / "baseline_wcntsegnet_mask.png", baseline_mask.astype(np.uint8))
            write_image(item_dir / "paper_repro_mask.png", new_mask.astype(np.uint8))
            write_image(item_dir / "paper_repro_probability.png", np.clip(prob_map * 255.0, 0, 255).astype(np.uint8))
            render_panel(
                output_path=item_dir / "comparison_panel.png",
                record=record,
                roi_gray=roi_gray,
                baseline_mask=baseline_mask,
                new_mask=new_mask,
                baseline_features=baseline_features,
                new_features=new_features,
                patch_count=patch_count,
            )

            row = flatten_row(
                record=record,
                baseline_features=baseline_features,
                new_features=new_features,
                item_dir=item_dir,
                baseline_mask_path=baseline_mask_path,
                patch_count=patch_count,
                inference_s=inference_s,
                status="success",
                error=None,
            )
            comparison_rows.append(row)
            summary_items.append(
                {
                    "image_id": record.image_id,
                    "sample_id": record.sample_id,
                    "file_path": record.file_path,
                    "magnification": record.magnification,
                    "status": "success",
                    "artifacts": {
                        "item_dir": str(item_dir),
                        "panel_path": str(item_dir / "comparison_panel.png"),
                        "baseline_mask_path": str(baseline_mask_path) if baseline_mask_path else None,
                        "new_mask_path": str(item_dir / "paper_repro_mask.png"),
                        "probability_path": str(item_dir / "paper_repro_probability.png"),
                    },
                    "inference": {
                        "device": str(device),
                        "patch_size": patch_size,
                        "stride": stride,
                        "patch_count": patch_count,
                        "threshold": threshold,
                        "elapsed_s": round(inference_s, 3),
                    },
                    "baseline": collect_numeric_snapshot(baseline_features),
                    "new": collect_numeric_snapshot(new_features),
                }
            )
        except Exception as exc:
            row = flatten_row(
                record=record,
                baseline_features=baseline_features,
                new_features={},
                item_dir=item_dir,
                baseline_mask_path=baseline_mask_path,
                patch_count=patch_count,
                inference_s=inference_s,
                status="failed",
                error=str(exc),
            )
            comparison_rows.append(row)
            summary_items.append(
                {
                    "image_id": record.image_id,
                    "sample_id": record.sample_id,
                    "file_path": record.file_path,
                    "magnification": record.magnification,
                    "status": "failed",
                    "error": str(exc),
                    "artifacts": {
                        "item_dir": str(item_dir),
                        "baseline_mask_path": str(baseline_mask_path) if baseline_mask_path else None,
                    },
                    "baseline": collect_numeric_snapshot(baseline_features),
                }
            )

    write_csv(output_root / "comparison.csv", comparison_rows)

    summary_payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "baseline_summary": str(args.baseline_summary),
        "baseline_csv": str(args.baseline_csv),
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "output_root": str(output_root),
        "device": str(device),
        "patch_size": patch_size,
        "stride": stride,
        "threshold": threshold,
        "requested_count": len(records),
        "success_count": sum(1 for item in summary_items if item["status"] == "success"),
        "failed_count": sum(1 for item in summary_items if item["status"] != "success"),
        "items": to_builtin(summary_items),
    }
    (output_root / "summary.json").write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"OUTPUT_DIR={output_root}")
    print(f"SUMMARY_PATH={output_root / 'summary.json'}")
    print(f"CSV_PATH={output_root / 'comparison.csv'}")


if __name__ == "__main__":
    main()
