from __future__ import annotations

import math
import sqlite3
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DB_PATH = Path(r"D:\CNTDATA\CNTA_ML_Project\database\cnta_experiments.sqlite")
OUTPUT_DIR = Path(r"D:\CNTDATA\CNTA_ML_Project\reports\xr_rose_diagrams_20260324")


def apply_theme() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#CBD5E1",
            "axes.labelcolor": "#334155",
            "axes.titlecolor": "#0F172A",
            "axes.titlesize": 16,
            "axes.labelsize": 11,
            "xtick.color": "#475569",
            "ytick.color": "#475569",
            "grid.color": "#CBD5E1",
            "grid.alpha": 0.25,
            "font.family": "DejaVu Sans",
        }
    )


def load_representative_rows() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        """
        SELECT
            file_path,
            sample_id,
            actual_temp,
            ar_flow AS flow_rate,
            catalyst_weight AS catalyst_concentration,
            alignment,
            density,
            curvature
        FROM images
        WHERE source = 'XR' AND IFNULL(is_deleted, 0) = 0 AND alignment IS NOT NULL
        ORDER BY alignment ASC
        """,
        conn,
    )
    conn.close()

    if df.empty:
        raise RuntimeError("No XR rows with alignment values were found.")

    df["actual_temp"] = pd.to_numeric(df["actual_temp"], errors="coerce")
    df["flow_rate"] = pd.to_numeric(df["flow_rate"], errors="coerce")
    df["catalyst_concentration"] = pd.to_numeric(df["catalyst_concentration"], errors="coerce")
    df["alignment"] = pd.to_numeric(df["alignment"], errors="coerce")
    df["density"] = pd.to_numeric(df["density"], errors="coerce")
    df["curvature"] = pd.to_numeric(df["curvature"], errors="coerce")

    target_quantiles = [0.12, 0.24, 0.38, 0.54, 0.72, 0.88]
    density_center = float(df["density"].median()) if df["density"].notna().any() else 0.0
    curvature_center = float(df["curvature"].median()) if df["curvature"].notna().any() else 0.0

    chosen_rows = []
    used_paths: set[str] = set()
    for idx, q in enumerate(target_quantiles, start=1):
        target_alignment = float(df["alignment"].quantile(q))
        candidates = df.copy()
        candidates["score"] = (
            (candidates["alignment"] - target_alignment).abs()
            + 0.08 * (candidates["density"].fillna(density_center) - density_center).abs()
            + 0.08 * (candidates["curvature"].fillna(curvature_center) - curvature_center).abs()
        )
        candidates = candidates[~candidates["file_path"].isin(used_paths)].sort_values("score")
        row = candidates.iloc[0].copy()
        row["band"] = f"group_{idx:02d}"
        chosen_rows.append(row)
        used_paths.add(str(row["file_path"]))

    return pd.DataFrame(chosen_rows).sort_values("alignment").reset_index(drop=True)


def preprocess_image(img_gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img_gray)
    smoothed = cv2.GaussianBlur(enhanced, (3, 3), 0)
    return enhanced, smoothed


def compute_orientation_distribution(img_gray: np.ndarray) -> dict[str, np.ndarray | float]:
    enhanced, smoothed = preprocess_image(img_gray)

    _, thresh = cv2.threshold(smoothed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    ix = cv2.Scharr(smoothed, cv2.CV_64F, 1, 0)
    iy = cv2.Scharr(smoothed, cv2.CV_64F, 0, 1)
    sigma = 3.0
    ksize = int(sigma * 4) | 1
    jxx = cv2.GaussianBlur(ix * ix, (ksize, ksize), sigma)
    jxy = cv2.GaussianBlur(ix * iy, (ksize, ksize), sigma)
    jyy = cv2.GaussianBlur(iy * iy, (ksize, ksize), sigma)

    theta = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy)
    coherence = np.sqrt((jxx - jyy) ** 2 + 4.0 * (jxy ** 2)) / (jxx + jyy + 1e-8)
    grad_mag = np.sqrt(ix * ix + iy * iy)

    foreground = thresh > 0
    grad_gate = grad_mag > np.percentile(grad_mag[foreground], 45) if np.any(foreground) else grad_mag > 0
    valid = foreground & (coherence > 0.20) & grad_gate

    if not np.any(valid):
        valid = foreground

    # Structure tensor gives the dominant gradient direction; CNT axis direction is rotated by 90 deg.
    angles_deg = (np.degrees(theta[valid]) + 270.0) % 180.0
    weights = coherence[valid]
    if weights.size == 0:
        raise RuntimeError("No valid orientations could be extracted from the image.")

    bins_deg = np.arange(0.0, 190.0, 10.0)
    hist, edges = np.histogram(angles_deg, bins=bins_deg, weights=weights)
    hist = hist.astype(float)
    hist = hist / hist.sum() if hist.sum() > 0 else hist
    centers_deg = (edges[:-1] + edges[1:]) / 2.0
    dominant_angle_deg = float(centers_deg[int(np.argmax(hist))])

    overlay = cv2.cvtColor(smoothed, cv2.COLOR_GRAY2BGR)
    step = 24
    for y in range(step // 2, smoothed.shape[0], step):
        for x in range(step // 2, smoothed.shape[1], step):
            if not valid[y, x]:
                continue
            angle = theta[y, x] + (np.pi / 2.0)
            dx = 12 * math.cos(angle)
            dy = 12 * math.sin(angle)
            cv2.arrowedLine(
                overlay,
                (x, y),
                (int(x + dx), int(y + dy)),
                (64, 196, 255),
                1,
                tipLength=0.25,
            )

    return {
        "enhanced": enhanced,
        "smoothed": smoothed,
        "mask": thresh,
        "valid_mask": valid.astype(np.uint8) * 255,
        "overlay": overlay,
        "hist": hist,
        "centers_deg": centers_deg,
        "dominant_angle_deg": dominant_angle_deg,
    }


def add_rose_plot(ax: plt.Axes, centers_deg: np.ndarray, hist: np.ndarray, title: str) -> None:
    theta = np.radians(centers_deg)
    widths = np.radians(np.full_like(centers_deg, 10.0))

    # CNT orientation is axial, so mirror 0-180 deg into 180-360 deg.
    theta_full = np.concatenate([theta, theta + np.pi])
    hist_full = np.concatenate([hist, hist])
    width_full = np.concatenate([widths, widths])

    bars = ax.bar(
        theta_full,
        hist_full,
        width=width_full,
        bottom=0.0,
        color=plt.cm.viridis(hist_full / hist_full.max() if hist_full.max() > 0 else hist_full),
        edgecolor="white",
        linewidth=0.8,
        alpha=0.95,
    )
    _ = bars

    ax.set_theta_zero_location("E")
    ax.set_theta_direction(-1)
    ax.set_thetagrids(range(0, 360, 45), labels=[f"{v}°" for v in range(0, 360, 45)])
    ax.set_title(title, pad=20, fontweight="bold")
    ax.grid(alpha=0.28)
    ax.set_rlabel_position(90)


def save_single_report(row: pd.Series, result: dict[str, np.ndarray | float]) -> Path:
    fig = plt.figure(figsize=(12.8, 6.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1.0], wspace=0.18)

    ax_img = fig.add_subplot(gs[0, 0])
    ax_rose = fig.add_subplot(gs[0, 1], projection="polar")

    img = cv2.imread(str(row["file_path"]), cv2.IMREAD_GRAYSCALE)
    ax_img.imshow(img, cmap="gray")
    ax_img.set_title(
        f"{row['band'].replace('_', ' ').upper()} | alignment sample\n{Path(row['file_path']).name}",
        fontweight="bold",
        fontsize=15,
    )
    ax_img.axis("off")
    ax_img.text(
        0.02,
        0.02,
        (
            f"Alignment = {row['alignment']:.3f}\n"
            f"Temp = {row['actual_temp']:.1f} degC\n"
            f"Flow = {row['flow_rate']:.0f} sccm | Catalyst = {row['catalyst_concentration']:.1f} g"
        ),
        transform=ax_img.transAxes,
        fontsize=10,
        color="white",
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.35", fc=(0, 0, 0, 0.55), ec=(1, 1, 1, 0.15)),
    )

    add_rose_plot(
        ax_rose,
        result["centers_deg"],
        result["hist"],
        f"Fiber direction {result['dominant_angle_deg']:.0f} deg",
    )

    fig.suptitle("XR Orientation Distribution", fontsize=20, fontweight="bold", y=0.99)
    fig.subplots_adjust(top=0.82)
    out = OUTPUT_DIR / f"{row['band']}_xr_rose_diagram.png"
    fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def save_comparison_panel(rows: pd.DataFrame, results: dict[str, dict[str, np.ndarray | float]]) -> Path:
    fig = plt.figure(figsize=(22.0, 8.8))
    gs = fig.add_gridspec(2, 6, hspace=0.12, wspace=0.10)

    band_order = list(rows.sort_values("alignment")["band"])
    for idx, band in enumerate(band_order):
        row = rows[rows["band"] == band].iloc[0]
        result = results[band]
        img = cv2.imread(str(row["file_path"]), cv2.IMREAD_GRAYSCALE)

        ax_img = fig.add_subplot(gs[0, idx])
        ax_img.imshow(img, cmap="gray")
        ax_img.axis("off")
        ax_img.set_title(
            (
                f"{band.replace('_', ' ').upper()} | {row['alignment']:.3f}\n"
                f"{Path(row['file_path']).name}\n"
                f"{row['actual_temp']:.1f} degC | {row['flow_rate']:.0f} sccm | {row['catalyst_concentration']:.1f} g"
            ),
            fontsize=10,
            fontweight="bold",
            pad=10,
        )

        ax_rose = fig.add_subplot(gs[1, idx], projection="polar")
        add_rose_plot(
            ax_rose,
            result["centers_deg"],
            result["hist"],
            f"Fiber direction {result['dominant_angle_deg']:.0f} deg",
        )

    fig.suptitle("XR CNT Orientation Comparison by Rose Diagram", fontsize=21, fontweight="bold", y=0.995)
    fig.text(
        0.5,
        0.01,
        "Top row: SEM images. Bottom row: rose diagrams using the same structure-tensor pipeline as the current XR alignment analysis.",
        ha="center",
        fontsize=10,
        color="#64748B",
    )
    out = OUTPUT_DIR / "xr_rose_diagram_comparison_panel.png"
    fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def write_readme(rows: pd.DataFrame, outputs: list[Path]) -> Path:
    lines = [
        "# XR Rose Diagrams",
        "",
        "- Generated on: 2026-03-24",
        "- Method: structure tensor + axial orientation histogram",
        "- CNT axis direction = structure-tensor angle + 90 deg",
        "- Domain: 0-180 deg, mirrored to 360 deg for polar display",
        "",
        "## Selected samples",
    ]
    for _, row in rows.iterrows():
        lines.append(
            f"- {row['band']}: {Path(row['file_path']).name} | alignment={row['alignment']:.4f} | "
            f"temp={row['actual_temp']:.1f} degC | flow={row['flow_rate']:.0f} sccm | catalyst={row['catalyst_concentration']:.1f} g"
        )
        lines.append(f"  source={row['file_path']}")

    lines.extend(["", "## Output files"])
    lines.extend([f"- {path.name}" for path in outputs])

    readme = OUTPUT_DIR / "README.md"
    readme.write_text("\n".join(lines), encoding="utf-8")
    return readme


def main() -> None:
    apply_theme()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = load_representative_rows()
    results: dict[str, dict[str, np.ndarray | float]] = {}
    outputs: list[Path] = []

    for _, row in rows.iterrows():
        img = cv2.imread(str(row["file_path"]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {row['file_path']}")
        result = compute_orientation_distribution(img)
        results[str(row["band"])] = result
        outputs.append(save_single_report(row, result))

    outputs.append(save_comparison_panel(rows, results))
    outputs.append(write_readme(rows, outputs))

    print("XR rose diagrams generated:")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
