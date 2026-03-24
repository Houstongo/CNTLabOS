import argparse
import csv
import json
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


PROJECT_ROOT = Path(r"D:\CNTDATA\CNTA_ML_Project")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

VLMSAM_ROOT = PROJECT_ROOT.parent / "VLMSAM"
if str(VLMSAM_ROOT) not in sys.path:
    sys.path.insert(0, str(VLMSAM_ROOT))

from backend.core.batch_processor import (  # noqa: E402
    DEFAULT_CNTSEGNET_CHECKPOINT,
    _get_cntsegnet_segmenter,
)
from src.analysis.feature_extractor import FeatureExtractor  # noqa: E402
from skimage.morphology import skeletonize  # noqa: E402


DB_PATH = PROJECT_ROOT / "database" / "cnta_experiments.sqlite"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "reports"


@dataclass
class ImageRow:
    image_id: int
    sample_id: str
    file_path: str
    magnification: Optional[int]
    processed: int
    source: str
    metadata: Dict[str, object]


def read_gray_image(image_path: str) -> Optional[np.ndarray]:
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is not None:
        return image
    try:
        data = np.fromfile(image_path, dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    except Exception:
        return None


def write_image(image_path: Path, image: np.ndarray) -> None:
    image_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = image_path.suffix.lower() or ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise RuntimeError(f"failed to encode image for {image_path}")
    encoded.tofile(str(image_path))


def slugify(text: str) -> str:
    safe = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_"}:
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe).strip("_") or "image"


def query_sample_rows(samples: Iterable[str], include_all: bool) -> List[ImageRow]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    rows: List[ImageRow] = []
    try:
        for sample in samples:
            cursor.execute(
                """
                SELECT id, sample_id, file_path, magnification, processed, source
                     , membrane_id, growth_temp, actual_temp, membrane_pos_cm, growth_time
                     , ar_flow, h2_flow, c2h4_flow
                     , al2o3_power, al2o3_thickness, fe_power, fe_thickness
                     , anneal_temp, anneal_time, position_label
                     , horizontal_pos, vertical_pos, repeat_id, catalyst_weight
                FROM images
                WHERE sample_id LIKE ?
                  AND source = 'ZZY'
                  AND COALESCE(is_deleted, 0) = 0
                ORDER BY id DESC
                """,
                (f"{sample}%",),
            )
            sample_rows = [
                ImageRow(
                    image_id=int(row["id"]),
                    sample_id=row["sample_id"],
                    file_path=row["file_path"],
                    magnification=int(row["magnification"]) if row["magnification"] is not None else None,
                    processed=int(row["processed"] or 0),
                    source=row["source"] or "ZZY",
                    metadata={
                        "membrane_id": row["membrane_id"],
                        "growth_temp": row["growth_temp"],
                        "actual_temp": row["actual_temp"],
                        "membrane_pos_cm": row["membrane_pos_cm"],
                        "growth_time": row["growth_time"],
                        "ar_flow": row["ar_flow"],
                        "h2_flow": row["h2_flow"],
                        "c2h4_flow": row["c2h4_flow"],
                        "al2o3_power": row["al2o3_power"],
                        "al2o3_thickness": row["al2o3_thickness"],
                        "fe_power": row["fe_power"],
                        "fe_thickness": row["fe_thickness"],
                        "anneal_temp": row["anneal_temp"],
                        "anneal_time": row["anneal_time"],
                        "position_label": row["position_label"],
                        "horizontal_pos": row["horizontal_pos"],
                        "vertical_pos": row["vertical_pos"],
                        "repeat_id": row["repeat_id"],
                        "catalyst_weight": row["catalyst_weight"],
                    },
                )
                for row in cursor.fetchall()
            ]

            if include_all:
                rows.extend(sample_rows)
                continue

            preferred = next((row for row in sample_rows if row.magnification == 50000), None)
            if preferred is None and sample_rows:
                preferred = sample_rows[0]
            if preferred is not None:
                rows.append(preferred)
    finally:
        conn.close()

    return rows


def make_mask_overlay(roi_gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR)
    overlay[mask > 0] = (64, 96, 255)
    return overlay


def build_wcntsegnet_mask(img_gray: np.ndarray, magnification: Optional[int]):
    extractor = FeatureExtractor(magnification=magnification)
    roi = extractor.extract_roi(img_gray)
    extractor._calibrate(roi.shape[1])
    processed = extractor.preprocess(roi)
    _, wcntsegnet_mask = extractor.calculate_density(processed)
    features = extractor.extract_all(img_gray, external_binary_mask=wcntsegnet_mask)
    return roi, wcntsegnet_mask, features, extractor


def run_cntsegnet_features(
    img_gray: np.ndarray,
    magnification: Optional[int],
    device: str,
    checkpoint_path: str,
    tile_size: int,
    overlap: int,
    seg_threshold: float,
):
    extractor = FeatureExtractor(magnification=magnification)
    roi = extractor.extract_roi(img_gray)
    segmenter = _get_cntsegnet_segmenter(
        checkpoint_path=checkpoint_path,
        device=device,
        tile_size=tile_size,
        overlap=overlap,
        threshold=seg_threshold,
    )
    mask = segmenter.predict_mask(roi)
    features = extractor.extract_all(img_gray, external_binary_mask=mask)
    return roi, mask, features, extractor


def calculate_diameter_stats(mask_u8: np.ndarray, extractor: FeatureExtractor, representative_diameter_nm) -> dict:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
    skel = skeletonize(closed > 0)
    dist = cv2.distanceTransform(closed, cv2.DIST_L2, 5)
    diameters_px = dist[skel].astype(float) * 2.0
    diameters_px = diameters_px[diameters_px > 0]
    if len(diameters_px) == 0:
        return {
            "representative_nm": representative_diameter_nm,
            "mean_nm": None,
            "median_nm": None,
            "p10_nm": None,
            "p90_nm": None,
        }

    nm_per_pixel = 1000.0 / extractor.px_per_um
    diameters_nm = diameters_px * nm_per_pixel
    return {
        "representative_nm": representative_diameter_nm,
        "mean_nm": float(np.mean(diameters_nm)),
        "median_nm": float(np.median(diameters_nm)),
        "p10_nm": float(np.percentile(diameters_nm, 10)),
        "p90_nm": float(np.percentile(diameters_nm, 90)),
    }


def _format_metric(value, pattern: str = "{:.4f}") -> str:
    if value is None:
        return "N/A"
    return pattern.format(value)


def _format_range(p10, p90) -> str:
    if p10 is None or p90 is None:
        return "N/A"
    return f"{p10:.2f} - {p90:.2f}"


def save_combo_figure(
    output_path: Path,
    original_gray: np.ndarray,
    wcntsegnet_mask: np.ndarray,
    cntsegnet_mask: np.ndarray,
    row: ImageRow,
    wcntsegnet_features: dict,
    cntsegnet_features: dict,
    wcntsegnet_diameter_stats: dict,
    cntsegnet_diameter_stats: dict,
) -> None:
    fig = plt.figure(figsize=(16, 10), dpi=120)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.2, 1.0], height_ratios=[1.0, 1.0])

    ax_original = fig.add_subplot(grid[0, 0])
    ax_original.imshow(original_gray, cmap="gray")
    ax_original.set_title("Original SEM")
    ax_original.axis("off")

    ax_wcntsegnet = fig.add_subplot(grid[0, 1])
    ax_wcntsegnet.imshow(wcntsegnet_mask, cmap="gray", vmin=0, vmax=255)
    ax_wcntsegnet.set_title("WCNTSegNET Mask")
    ax_wcntsegnet.axis("off")

    ax_cntsegnet = fig.add_subplot(grid[1, 0])
    ax_cntsegnet.imshow(cntsegnet_mask, cmap="gray", vmin=0, vmax=255)
    ax_cntsegnet.set_title("CNTSegNet Mask")
    ax_cntsegnet.axis("off")

    ax_metrics = fig.add_subplot(grid[1, 1])
    ax_metrics.axis("off")

    meta_lines = [
        f"image_id: {row.image_id}",
        f"sample_id: {row.sample_id}",
        f"magnification: {row.magnification or 'N/A'}",
        f"processed flag: {row.processed}",
    ]

    y = 0.95
    ax_metrics.text(0.0, y, "WCNTSegNET vs CNTSegNet", fontsize=18, fontweight="bold", transform=ax_metrics.transAxes)
    y -= 0.10
    for line in meta_lines:
        ax_metrics.text(0.0, y, line, fontsize=11, color="#334155", transform=ax_metrics.transAxes)
        y -= 0.055

    ax_metrics.text(
        0.0,
        0.70,
        "直径同时显示当前系统代表值，以及均值/中位数/P10-P90，避免只看单一 diameter 误判。",
        fontsize=10,
        color="#475569",
        wrap=True,
        transform=ax_metrics.transAxes,
    )

    table_rows = [
        ["取向度", _format_metric(wcntsegnet_features.get("alignment")), _format_metric(cntsegnet_features.get("alignment"))],
        ["密度 (%)", _format_metric(wcntsegnet_features.get("density"), "{:.2f}"), _format_metric(cntsegnet_features.get("density"), "{:.2f}")],
        ["平均曲率 (nm^-1)", _format_metric(wcntsegnet_features.get("curvature_nm")), _format_metric(cntsegnet_features.get("curvature_nm"))],
        ["当前系统直径 (nm)", _format_metric(wcntsegnet_diameter_stats.get("representative_nm"), "{:.2f}"), _format_metric(cntsegnet_diameter_stats.get("representative_nm"), "{:.2f}")],
        ["直径均值 (nm)", _format_metric(wcntsegnet_diameter_stats.get("mean_nm"), "{:.2f}"), _format_metric(cntsegnet_diameter_stats.get("mean_nm"), "{:.2f}")],
        ["直径中位数 (nm)", _format_metric(wcntsegnet_diameter_stats.get("median_nm"), "{:.2f}"), _format_metric(cntsegnet_diameter_stats.get("median_nm"), "{:.2f}")],
        ["直径 P10-P90 (nm)", _format_range(wcntsegnet_diameter_stats.get("p10_nm"), wcntsegnet_diameter_stats.get("p90_nm")), _format_range(cntsegnet_diameter_stats.get("p10_nm"), cntsegnet_diameter_stats.get("p90_nm"))],
    ]

    table = ax_metrics.table(
        cellText=table_rows,
        colLabels=["指标", "WCNTSegNET", "CNTSegNet"],
        cellLoc="center",
        colLoc="center",
        bbox=[0.0, 0.03, 0.98, 0.60],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.5)

    for col in range(3):
        table[(0, col)].set_facecolor("#1d4ed8")
        table[(0, col)].set_text_props(color="white", fontweight="bold")

    for row_idx in range(1, len(table_rows) + 1):
        table[(row_idx, 0)].set_facecolor("#eff6ff")
        table[(row_idx, 0)].set_text_props(color="#1e3a8a", fontweight="bold")

    fig.suptitle(Path(row.file_path).name, fontsize=16, y=0.98)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def process_rows(
    rows: List[ImageRow],
    output_root: Path,
    device: str,
    checkpoint_path: str,
    tile_size: int,
    overlap: int,
    seg_threshold: float,
) -> List[dict]:
    output_root.mkdir(parents=True, exist_ok=True)
    summary: List[dict] = []

    for index, row in enumerate(rows, 1):
        print(f"[{index}/{len(rows)}] image_id={row.image_id} sample={row.sample_id}")
        item = {
            "image_id": row.image_id,
            "sample_id": row.sample_id,
            "file_path": row.file_path,
            "source": row.source,
            "magnification": row.magnification,
            "processed": row.processed,
            "metadata": row.metadata,
            "status": "pending",
        }

        try:
            img_gray = read_gray_image(row.file_path)
            if img_gray is None:
                raise RuntimeError("failed to read image")

            wcntsegnet_roi, wcntsegnet_mask, wcntsegnet_features, wcntsegnet_extractor = build_wcntsegnet_mask(
                img_gray=img_gray,
                magnification=row.magnification,
            )
            cntsegnet_roi, cntsegnet_mask, cntsegnet_features, cntsegnet_extractor = run_cntsegnet_features(
                img_gray=img_gray,
                magnification=row.magnification,
                device=device,
                checkpoint_path=checkpoint_path,
                tile_size=tile_size,
                overlap=overlap,
                seg_threshold=seg_threshold,
            )

            wcntsegnet_mask_u8 = wcntsegnet_mask.astype(np.uint8)
            cntsegnet_mask_u8 = (cntsegnet_mask > 0).astype(np.uint8) * 255
            wcntsegnet_diameter_stats = calculate_diameter_stats(
                wcntsegnet_mask_u8,
                wcntsegnet_extractor,
                wcntsegnet_features.get("diameter"),
            )
            cntsegnet_diameter_stats = calculate_diameter_stats(
                cntsegnet_mask_u8,
                cntsegnet_extractor,
                cntsegnet_features.get("diameter"),
            )

            sample_dir = output_root / f"{row.image_id}_{slugify(Path(row.file_path).stem)}"
            write_image(sample_dir / "original.png", img_gray)
            write_image(sample_dir / "wcntsegnet_mask.png", wcntsegnet_mask_u8)
            write_image(sample_dir / "cntsegnet_mask.png", cntsegnet_mask_u8)
            save_combo_figure(
                output_path=sample_dir / "comparison_4panel.png",
                original_gray=img_gray,
                wcntsegnet_mask=wcntsegnet_mask_u8,
                cntsegnet_mask=cntsegnet_mask_u8,
                row=row,
                wcntsegnet_features=wcntsegnet_features,
                cntsegnet_features=cntsegnet_features,
                wcntsegnet_diameter_stats=wcntsegnet_diameter_stats,
                cntsegnet_diameter_stats=cntsegnet_diameter_stats,
            )

            item.update(
                {
                    "status": "success",
                    "output_dir": str(sample_dir),
                    "wcntsegnet_features": wcntsegnet_features,
                    "cntsegnet_features": cntsegnet_features,
                    "wcntsegnet_diameter_stats": wcntsegnet_diameter_stats,
                    "cntsegnet_diameter_stats": cntsegnet_diameter_stats,
                    "roi_shape": {
                        "wcntsegnet": [int(wcntsegnet_roi.shape[0]), int(wcntsegnet_roi.shape[1])],
                        "cntsegnet": [int(cntsegnet_roi.shape[0]), int(cntsegnet_roi.shape[1])],
                    },
                }
            )
        except Exception as exc:
            item.update({"status": "failed", "error": str(exc)})
        summary.append(item)

    return summary


def write_batch_csv(summary: List[dict], output_path: Path) -> None:
    fieldnames = [
        "image_id",
        "sample_id",
        "file_path",
        "source",
        "magnification",
        "processed",
        "membrane_id",
        "growth_temp",
        "actual_temp",
        "membrane_pos_cm",
        "growth_time",
        "ar_flow",
        "h2_flow",
        "c2h4_flow",
        "al2o3_power",
        "al2o3_thickness",
        "fe_power",
        "fe_thickness",
        "anneal_temp",
        "anneal_time",
        "position_label",
        "horizontal_pos",
        "vertical_pos",
        "repeat_id",
        "catalyst_weight",
        "wcntsegnet_alignment",
        "wcntsegnet_density",
        "wcntsegnet_curvature_nm",
        "wcntsegnet_diameter_representative_nm",
        "wcntsegnet_diameter_mean_nm",
        "wcntsegnet_diameter_median_nm",
        "wcntsegnet_diameter_p10_nm",
        "wcntsegnet_diameter_p90_nm",
        "cntsegnet_alignment",
        "cntsegnet_density",
        "cntsegnet_curvature_nm",
        "cntsegnet_diameter_representative_nm",
        "cntsegnet_diameter_mean_nm",
        "cntsegnet_diameter_median_nm",
        "cntsegnet_diameter_p10_nm",
        "cntsegnet_diameter_p90_nm",
        "output_dir",
        "status",
        "error",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for item in summary:
            metadata = item.get("metadata", {})
            wcnt = item.get("wcntsegnet_features", {})
            cnt = item.get("cntsegnet_features", {})
            wcnt_d = item.get("wcntsegnet_diameter_stats", {})
            cnt_d = item.get("cntsegnet_diameter_stats", {})
            writer.writerow(
                {
                    "image_id": item.get("image_id"),
                    "sample_id": item.get("sample_id"),
                    "file_path": item.get("file_path"),
                    "source": item.get("source"),
                    "magnification": item.get("magnification"),
                    "processed": item.get("processed"),
                    "membrane_id": metadata.get("membrane_id"),
                    "growth_temp": metadata.get("growth_temp"),
                    "actual_temp": metadata.get("actual_temp"),
                    "membrane_pos_cm": metadata.get("membrane_pos_cm"),
                    "growth_time": metadata.get("growth_time"),
                    "ar_flow": metadata.get("ar_flow"),
                    "h2_flow": metadata.get("h2_flow"),
                    "c2h4_flow": metadata.get("c2h4_flow"),
                    "al2o3_power": metadata.get("al2o3_power"),
                    "al2o3_thickness": metadata.get("al2o3_thickness"),
                    "fe_power": metadata.get("fe_power"),
                    "fe_thickness": metadata.get("fe_thickness"),
                    "anneal_temp": metadata.get("anneal_temp"),
                    "anneal_time": metadata.get("anneal_time"),
                    "position_label": metadata.get("position_label"),
                    "horizontal_pos": metadata.get("horizontal_pos"),
                    "vertical_pos": metadata.get("vertical_pos"),
                    "repeat_id": metadata.get("repeat_id"),
                    "catalyst_weight": metadata.get("catalyst_weight"),
                    "wcntsegnet_alignment": wcnt.get("alignment"),
                    "wcntsegnet_density": wcnt.get("density"),
                    "wcntsegnet_curvature_nm": wcnt.get("curvature_nm"),
                    "wcntsegnet_diameter_representative_nm": wcnt_d.get("representative_nm"),
                    "wcntsegnet_diameter_mean_nm": wcnt_d.get("mean_nm"),
                    "wcntsegnet_diameter_median_nm": wcnt_d.get("median_nm"),
                    "wcntsegnet_diameter_p10_nm": wcnt_d.get("p10_nm"),
                    "wcntsegnet_diameter_p90_nm": wcnt_d.get("p90_nm"),
                    "cntsegnet_alignment": cnt.get("alignment"),
                    "cntsegnet_density": cnt.get("density"),
                    "cntsegnet_curvature_nm": cnt.get("curvature_nm"),
                    "cntsegnet_diameter_representative_nm": cnt_d.get("representative_nm"),
                    "cntsegnet_diameter_mean_nm": cnt_d.get("mean_nm"),
                    "cntsegnet_diameter_median_nm": cnt_d.get("median_nm"),
                    "cntsegnet_diameter_p10_nm": cnt_d.get("p10_nm"),
                    "cntsegnet_diameter_p90_nm": cnt_d.get("p90_nm"),
                    "output_dir": item.get("output_dir"),
                    "status": item.get("status"),
                    "error": item.get("error"),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CNTSegNet preview/batch outputs for ZZY DB images.")
    parser.add_argument("--samples", nargs="+", default=["No28", "No41", "No42"])
    parser.add_argument("--preview", action="store_true", help="Pick one preferred 50000x image per sample.")
    parser.add_argument("--output-dir", default=None, help="Output directory. Default goes under reports/.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CNTSEGNET_CHECKPOINT))
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--seg-threshold", type=float, default=0.5)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_root = Path(args.output_dir)
    else:
        suffix = "preview" if args.preview else "full"
        output_root = DEFAULT_OUTPUT_ROOT / f"zzy_cntsegnet_{suffix}_{timestamp}"

    rows = query_sample_rows(args.samples, include_all=not args.preview)
    if not rows:
        raise SystemExit("No matching ZZY rows found in database.")

    summary = process_rows(
        rows=rows,
        output_root=output_root,
        device=args.device,
        checkpoint_path=args.checkpoint,
        tile_size=args.tile_size,
        overlap=args.overlap,
        seg_threshold=args.seg_threshold,
    )

    summary_payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "db_path": str(DB_PATH),
        "mode": "preview" if args.preview else "full",
        "samples": args.samples,
        "requested_count": len(rows),
        "success_count": sum(1 for item in summary if item["status"] == "success"),
        "failed_count": sum(1 for item in summary if item["status"] != "success"),
        "items": summary,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.json").write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_batch_csv(summary, output_root / "batch_features.csv")
    print(f"OUTPUT_DIR={output_root}")
    print(f"SUMMARY_PATH={output_root / 'summary.json'}")
    print(f"CSV_PATH={output_root / 'batch_features.csv'}")


if __name__ == "__main__":
    main()
