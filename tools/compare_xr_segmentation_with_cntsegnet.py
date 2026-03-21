from __future__ import annotations

import argparse
import html
import sqlite3
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt


THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
VLMSAM_ROOT = PROJECT_ROOT.parent / "VLMSAM"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(VLMSAM_ROOT) not in sys.path:
    sys.path.insert(0, str(VLMSAM_ROOT))

from cntsegnet import CNTSegNet
from src.analysis.feature_extractor import FeatureExtractor


DB_PATH = PROJECT_ROOT / "database" / "cnta_experiments.sqlite"
CHECKPOINT_PATH = VLMSAM_ROOT / "checkpoints_512_v2" / "best_model.pth"


@dataclass
class Sample:
    image_id: int
    file_path: str
    magnification: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare CNTA threshold segmentation vs CNTSegNet on XR samples."
    )
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--min-mag", type=int, default=20000)
    parser.add_argument("--ids", type=int, nargs="*", default=None)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def fetch_samples(db_path: Path, limit: int, min_mag: int, ids: list[int] | None) -> list[Sample]:
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        if ids:
            placeholders = ",".join("?" for _ in ids)
            rows = cur.execute(
                f"""
                SELECT id, file_path, magnification
                FROM images
                WHERE id IN ({placeholders})
                ORDER BY id DESC
                """,
                ids,
            ).fetchall()
        else:
            rows = cur.execute(
                """
                SELECT id, file_path, magnification
                FROM images
                WHERE source = 'XR'
                  AND COALESCE(is_deleted, 0) = 0
                  AND COALESCE(magnification, 0) >= ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (min_mag, limit),
            ).fetchall()
    finally:
        conn.close()

    return [Sample(int(row[0]), str(row[1]), int(row[2] or 0)) for row in rows]


def load_model(checkpoint_path: Path, device: str) -> torch.nn.Module:
    model = CNTSegNet(num_classes=1)
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def tiled_predict_fast(
    model: torch.nn.Module,
    image_rgb: np.ndarray,
    device: str,
    tile_size: int,
    overlap: int,
) -> tuple[np.ndarray, int]:
    h, w = image_rgb.shape[:2]
    stride = max(1, tile_size - overlap)

    ys = list(range(0, max(h - tile_size + 1, 1), stride))
    xs = list(range(0, max(w - tile_size + 1, 1), stride))
    last_y = max(h - tile_size, 0)
    last_x = max(w - tile_size, 0)
    if not ys or ys[-1] != last_y:
        ys.append(last_y)
    if not xs or xs[-1] != last_x:
        xs.append(last_x)

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

    accum = np.zeros((h, w), dtype=np.float32)
    counts = np.zeros((h, w), dtype=np.float32)
    tiles = 0

    with torch.no_grad():
        for y in ys:
            for x in xs:
                tile = image_rgb[y : y + tile_size, x : x + tile_size]
                tile_h, tile_w = tile.shape[:2]

                if tile_h != tile_size or tile_w != tile_size:
                    padded = np.zeros((tile_size, tile_size, 3), dtype=np.float32)
                    padded[:tile_h, :tile_w] = tile
                    tile = padded

                tensor = torch.from_numpy(tile.transpose(2, 0, 1)).float()
                tensor = (tensor / 255.0 - torch.from_numpy(mean)) / torch.from_numpy(std)
                tensor = tensor.unsqueeze(0).to(device)

                pred = torch.sigmoid(model(tensor)).cpu().numpy()[0, 0]
                accum[y : y + tile_h, x : x + tile_w] += pred[:tile_h, :tile_w]
                counts[y : y + tile_h, x : x + tile_w] += 1.0
                tiles += 1

    return accum / np.maximum(counts, 1.0), tiles


def overlay_mask(gray: np.ndarray, mask: np.ndarray, color_bgr: tuple[int, int, int]) -> np.ndarray:
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    colored = np.zeros_like(base)
    colored[mask > 0] = color_bgr
    return cv2.addWeighted(base, 0.7, colored, 0.3, 0.0)


def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 1.0


def compute_diff_masks(traditional_mask: np.ndarray, cntsegnet_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t = traditional_mask.astype(bool)
    c = cntsegnet_mask.astype(bool)
    missed = np.logical_and(t, np.logical_not(c))
    over = np.logical_and(np.logical_not(t), c)
    return missed, over


def build_diff_heatmap(traditional_mask: np.ndarray, cntsegnet_mask: np.ndarray) -> np.ndarray:
    t = traditional_mask.astype(bool)
    c = cntsegnet_mask.astype(bool)
    missed, over = compute_diff_masks(traditional_mask, cntsegnet_mask)
    agree_fg = np.logical_and(t, c)
    agree_bg = np.logical_and(np.logical_not(t), np.logical_not(c))

    heat = np.zeros((traditional_mask.shape[0], traditional_mask.shape[1], 3), dtype=np.uint8)
    heat[agree_bg] = (20, 20, 20)
    heat[agree_fg] = (235, 235, 235)
    heat[missed] = (255, 80, 80)
    heat[over] = (80, 180, 255)
    return heat


def draw_mask_contours(roi: np.ndarray, traditional_mask: np.ndarray, cntsegnet_mask: np.ndarray) -> np.ndarray:
    canvas = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
    contours_t, _ = cv2.findContours((traditional_mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours_c, _ = cv2.findContours((cntsegnet_mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(canvas, contours_t, -1, (0, 255, 255), 1)
    cv2.drawContours(canvas, contours_c, -1, (0, 255, 0), 1)
    return canvas


def build_split_binary_view(traditional_mask: np.ndarray, cntsegnet_mask: np.ndarray) -> np.ndarray:
    h, w = traditional_mask.shape
    split = np.zeros((h, w), dtype=np.uint8)
    mid = w // 2
    split[:, :mid] = traditional_mask[:, :mid]
    split[:, mid:] = cntsegnet_mask[:, mid:]
    return split


def render_comparison(
    sample: Sample,
    roi: np.ndarray,
    traditional_mask: np.ndarray,
    cntsegnet_prob: np.ndarray,
    cntsegnet_mask: np.ndarray,
    metrics: dict[str, float],
    output_path: Path,
) -> None:
    diff_heatmap = build_diff_heatmap(traditional_mask, cntsegnet_mask)
    contour_overlay = draw_mask_contours(roi, traditional_mask, cntsegnet_mask)
    split_binary = build_split_binary_view(traditional_mask, cntsegnet_mask)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    axes[0, 0].imshow(roi, cmap="gray")
    axes[0, 0].set_title("ROI")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(traditional_mask, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axes[0, 1].set_title(f"Threshold binary mask\nfg={metrics['traditional_fg_ratio']:.2%}")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(cntsegnet_mask, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axes[0, 2].set_title(f"CNTSegNet binary mask\nfg={metrics['cntsegnet_fg_ratio']:.2%}")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(split_binary, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axes[1, 0].set_title(
        "Binary split compare\n"
        "Left: threshold | Right: CNTSegNet"
    )
    axes[1, 0].axis("off")
    axes[1, 0].axvline(x=split_binary.shape[1] / 2.0, color="red", linewidth=2, alpha=0.8)

    axes[1, 1].imshow(diff_heatmap)
    axes[1, 1].set_title(
        "Diff heatmap\n"
        f"missed={metrics['missed_ratio_roi']:.2%} over={metrics['over_ratio_roi']:.2%}\n"
        f"diff_area={metrics['diff_area_ratio']:.2%} IoU={metrics['mask_iou']:.3f}"
    )
    axes[1, 1].axis("off")
    axes[1, 1].text(
        0.02,
        0.02,
        "Red: missed\nBlue: over\nWhite: agree FG\nDark: agree BG",
        transform=axes[1, 1].transAxes,
        fontsize=9,
        color="white",
        va="bottom",
        ha="left",
        bbox={"facecolor": "black", "alpha": 0.5, "edgecolor": "none", "pad": 4},
    )

    axes[1, 2].imshow(cv2.cvtColor(contour_overlay, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title("Contour overlay\nYellow: threshold, Green: CNTSegNet")
    axes[1, 2].axis("off")

    fig.suptitle(f"XR #{sample.image_id} | {Path(sample.file_path).name} | {sample.magnification}x", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_report(report_dir: Path, rows: list[dict[str, object]]) -> Path:
    html_path = report_dir / "index.html"
    cards = []
    for row in rows:
        cards.append(
            f"""
            <div class="card">
              <h2>XR #{row['image_id']} | {html.escape(str(row['file_name']))}</h2>
              <p><strong>Magnification:</strong> {row['magnification']}x</p>
              <p><strong>Threshold fg:</strong> {row['traditional_fg_ratio']:.2%} |
                 <strong>CNTSegNet fg:</strong> {row['cntsegnet_fg_ratio']:.2%} |
                 <strong>Mask IoU:</strong> {row['mask_iou']:.3f}</p>
              <p><strong>Diff area:</strong> {row['diff_area_ratio']:.2%} |
                 <strong>Missed (ROI):</strong> {row['missed_ratio_roi']:.2%} |
                 <strong>Over (ROI):</strong> {row['over_ratio_roi']:.2%}</p>
              <p><strong>Missed vs threshold fg:</strong> {row['missed_ratio_vs_threshold_fg']:.2%} |
                 <strong>Over vs CNTSegNet fg:</strong> {row['over_ratio_vs_cntsegnet_fg']:.2%}</p>
              <p><strong>Inference:</strong> {row['inference_seconds']:.2f}s over {row['tiles']} tiles</p>
              <img src="{html.escape(row['image_name'])}" alt="{html.escape(str(row['file_name']))}">
              <p class="path">{html.escape(str(row['file_path']))}</p>
            </div>
            """
        )

    html_path.write_text(
        f"""
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
          <meta charset="utf-8">
          <title>XR Segmentation Comparison</title>
          <style>
            body {{ font-family: Arial, sans-serif; margin: 24px; background: #f4f6f8; }}
            .grid {{ display: grid; grid-template-columns: 1fr; gap: 24px; }}
            .card {{ background: white; padding: 20px; border-radius: 14px; box-shadow: 0 8px 24px rgba(0,0,0,0.08); }}
            img {{ width: 100%; border-radius: 10px; border: 1px solid #d9e2ec; }}
            .path {{ color: #52606d; word-break: break-all; }}
          </style>
        </head>
        <body>
          <h1>XR Segmentation Comparison</h1>
          <p>ROI + binary mask comparison + missed/over heatmap.</p>
          <div class="grid">
            {''.join(cards)}
          </div>
        </body>
        </html>
        """,
        encoding="utf-8",
    )
    return html_path


def main() -> int:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = PROJECT_ROOT / "reports" / f"xr_segmentation_compare_{timestamp}"
    report_dir.mkdir(parents=True, exist_ok=True)

    samples = fetch_samples(args.db, args.limit, args.min_mag, args.ids)
    if not samples:
        print("No XR samples found.")
        return 1

    print(f"Using device={args.device}")
    print(f"Loading model from {args.checkpoint}")
    model = load_model(args.checkpoint, args.device)

    rows: list[dict[str, object]] = []
    for sample in samples:
        image_path = Path(sample.file_path)
        print(f"\nProcessing XR #{sample.image_id}: {image_path.name}")
        gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"  skip: failed to read image {image_path}")
            continue

        extractor = FeatureExtractor(magnification=sample.magnification, diameter_method="standard")
        roi = extractor.extract_roi(gray)
        extractor._calibrate(roi.shape[1])
        processed = extractor.preprocess(roi)
        _, traditional_thresh = extractor.calculate_density(processed)
        traditional_mask = (traditional_thresh > 0).astype(np.uint8)

        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB).astype(np.float32)
        t0 = time.perf_counter()
        cntsegnet_prob, tiles = tiled_predict_fast(
            model,
            roi_rgb,
            device=args.device,
            tile_size=args.tile_size,
            overlap=args.overlap,
        )
        inference_seconds = time.perf_counter() - t0
        cntsegnet_mask = (cntsegnet_prob >= args.threshold).astype(np.uint8)
        missed_mask, over_mask = compute_diff_masks(traditional_mask, cntsegnet_mask)
        diff_pixels = np.logical_or(missed_mask, over_mask).sum()
        threshold_fg_pixels = max(int((traditional_mask > 0).sum()), 1)
        cntsegnet_fg_pixels = max(int((cntsegnet_mask > 0).sum()), 1)

        metrics = {
            "traditional_fg_ratio": float(traditional_mask.mean()),
            "cntsegnet_fg_ratio": float(cntsegnet_mask.mean()),
            "cntsegnet_prob_mean": float(cntsegnet_prob.mean()),
            "mask_iou": compute_iou(traditional_mask, cntsegnet_mask),
            "diff_area_ratio": float(diff_pixels / traditional_mask.size),
            "missed_ratio_roi": float(missed_mask.mean()),
            "over_ratio_roi": float(over_mask.mean()),
            "missed_ratio_vs_threshold_fg": float(missed_mask.sum() / threshold_fg_pixels),
            "over_ratio_vs_cntsegnet_fg": float(over_mask.sum() / cntsegnet_fg_pixels),
            "inference_seconds": float(inference_seconds),
            "tiles": int(tiles),
        }

        image_name = f"xr_{sample.image_id}_{image_path.stem}.png".replace(" ", "_")
        out_path = report_dir / image_name
        render_comparison(
            sample,
            roi,
            traditional_mask,
            cntsegnet_prob,
            cntsegnet_mask,
            metrics,
            out_path,
        )

        rows.append(
            {
                "image_id": sample.image_id,
                "file_name": image_path.name,
                "file_path": sample.file_path,
                "magnification": sample.magnification,
                "image_name": image_name,
                **metrics,
            }
        )
        print(
            "  "
            f"threshold_fg={metrics['traditional_fg_ratio']:.2%}, "
            f"cntsegnet_fg={metrics['cntsegnet_fg_ratio']:.2%}, "
            f"diff_area={metrics['diff_area_ratio']:.2%}, "
            f"mask_iou={metrics['mask_iou']:.3f}, "
            f"time={metrics['inference_seconds']:.2f}s"
        )

    report_path = build_report(report_dir, rows)
    print(f"\nReport directory: {report_dir}")
    print(f"Report HTML: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
