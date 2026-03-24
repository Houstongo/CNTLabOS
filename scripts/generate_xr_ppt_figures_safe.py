from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image


DB_PATH = Path(r"D:\CNTDATA\CNTA_ML_Project\database\cnta_experiments.sqlite")
OUTPUT_DIR = Path(r"D:\CNTDATA\CNTA_ML_Project\reports\xr_ppt_figures_20260324_ppt")

XR_CMAP = LinearSegmentedColormap.from_list(
    "xr_process",
    ["#10213A", "#2463EB", "#1DB5A6", "#F4B740", "#E66A26"],
)

FLOW_COLORS = {
    200.0: "#2463EB",
    250.0: "#1DB5A6",
    300.0: "#F97316",
}

FEATURE_META = [
    ("diameter", "Mean Diameter", "nm"),
    ("density", "Area Density", "%"),
    ("alignment", "Alignment", ""),
    ("curvature", "Curvature", ""),
]


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


def load_xr_data() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        """
        SELECT
            sample_id,
            file_path,
            actual_temp,
            ar_flow AS flow_rate,
            catalyst_weight AS catalyst_concentration,
            diameter,
            density,
            alignment,
            curvature,
            tortuosity
        FROM images
        WHERE source = 'XR' AND IFNULL(is_deleted, 0) = 0
        """,
        conn,
    )
    conn.close()

    df["run_name"] = df["file_path"].map(lambda p: os.path.basename(os.path.dirname(p)) if p else "")
    for col in [
        "actual_temp",
        "flow_rate",
        "catalyst_concentration",
        "diameter",
        "density",
        "alignment",
        "curvature",
        "tortuosity",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["actual_temp", "flow_rate", "catalyst_concentration"]).copy()
    df["temp_round"] = df["actual_temp"].round(1)
    return df


def save_fig(fig: plt.Figure, filename: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def add_badge(ax: plt.Axes, text: str, color: str = "#2463EB") -> None:
    ax.text(
        0.0,
        1.06,
        text,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        color=color,
        bbox=dict(boxstyle="round,pad=0.28", fc="#F8FBFF", ec="#BFDBFE"),
    )


def normalize_preview(img: Image.Image) -> np.ndarray:
    arr = np.asarray(img.convert("L"), dtype=np.float32)
    lo, hi = np.percentile(arr, [2, 98])
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((arr - lo) / (hi - lo), 0, 1)


def make_3d_process_space(df: pd.DataFrame) -> Path:
    data = df.dropna(subset=["diameter"]).copy()
    fig = plt.figure(figsize=(11.0, 8.2))
    ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(
        data["actual_temp"],
        data["flow_rate"],
        data["catalyst_concentration"],
        c=data["diameter"],
        cmap=XR_CMAP,
        s=60,
        alpha=0.95,
        edgecolor="white",
        linewidth=0.45,
    )
    ax.view_init(elev=24, azim=-58)
    ax.xaxis.pane.set_facecolor((0.96, 0.98, 1.0, 1.0))
    ax.yaxis.pane.set_facecolor((0.96, 0.98, 1.0, 1.0))
    ax.zaxis.pane.set_facecolor((0.99, 0.99, 1.0, 1.0))

    ax.set_title("XR Process Space vs Mean Diameter", pad=10, fontweight="bold")
    ax.set_xlabel("Actual Temperature (degC)", labelpad=12)
    ax.set_ylabel("Flow Rate (sccm)", labelpad=12)
    ax.set_zlabel("Catalyst (g)", labelpad=12)

    cbar = fig.colorbar(sc, ax=ax, pad=0.08, shrink=0.82)
    cbar.set_label("Mean Diameter (nm)")

    fig.text(
        0.11,
        0.955,
        f"N = {len(data)} | Temperature {data['actual_temp'].min():.0f}-{data['actual_temp'].max():.0f} degC",
        fontsize=10,
        color="#475569",
    )
    fig.subplots_adjust(top=0.90)
    return save_fig(fig, "01_xr_process_space_mean_diameter.png")


def make_temperature_trends(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 9.2), sharex=True)
    axes = axes.ravel()

    for ax, (feature, title, unit) in zip(axes, FEATURE_META):
        sub = df.dropna(subset=[feature]).copy()
        for flow, color in FLOW_COLORS.items():
            flow_sub = sub[sub["flow_rate"] == flow]
            if flow_sub.empty:
                continue
            ax.scatter(
                flow_sub["actual_temp"],
                flow_sub[feature],
                s=34,
                alpha=0.7,
                color=color,
                label=f"{int(flow)} sccm",
                edgecolors="white",
                linewidths=0.35,
            )

        trend = sub.groupby("temp_round", as_index=False)[feature].mean().sort_values("temp_round")
        ax.plot(trend["temp_round"], trend[feature], color="#0F172A", linewidth=2.4)
        ax.fill_between(trend["temp_round"], trend[feature], color="#0F172A", alpha=0.08)
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(f"{title}{f' ({unit})' if unit else ''}")
        ax.grid(True, linestyle="--")
        add_badge(ax, "Grouped by actual temperature")

    axes[0].legend(frameon=False, loc="best", title="Flow")
    for ax in axes[2:]:
        ax.set_xlabel("Actual Temperature (degC)")

    fig.suptitle("XR Morphology Trends Along Temperature", fontsize=20, y=0.98, fontweight="bold")
    fig.text(
        0.5,
        0.01,
        "Points are individual XR images. The black line is the mean trend at each measured temperature.",
        ha="center",
        fontsize=10,
        color="#64748B",
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return save_fig(fig, "02_xr_temperature_trends_four_features.png")


def make_flow_catalyst_heatmaps(df: pd.DataFrame) -> Path:
    target = "alignment"
    plot_df = df.dropna(subset=[target]).copy()
    flow_levels = sorted(v for v in plot_df["flow_rate"].dropna().unique())
    fig, axes = plt.subplots(1, len(flow_levels), figsize=(15.8, 8.8), sharey=True)
    if len(flow_levels) == 1:
        axes = [axes]

    vmin = plot_df[target].min()
    vmax = plot_df[target].max()
    counts_max = 1

    grouped_all = (
        plot_df.groupby(["flow_rate", "temp_round", "catalyst_concentration"], as_index=False)
        .agg(alignment=("alignment", "mean"), count=("alignment", "size"))
    )
    if not grouped_all.empty:
        counts_max = int(grouped_all["count"].max())

    last_scatter = None
    for ax, flow in zip(axes, flow_levels):
        sub = grouped_all[grouped_all["flow_rate"] == flow].copy()
        size_scale = 130 + (sub["count"] / counts_max) * 520
        last_scatter = ax.scatter(
            sub["catalyst_concentration"],
            sub["temp_round"],
            c=sub["alignment"],
            s=size_scale,
            cmap=XR_CMAP,
            vmin=vmin,
            vmax=vmax,
            alpha=0.92,
            edgecolors="white",
            linewidths=0.8,
        )
        ax.set_title(f"Flow {int(flow)} sccm", fontweight="bold")
        ax.set_xlabel("Catalyst (g)")
        if ax is axes[0]:
            ax.set_ylabel("Actual Temperature (degC)")
        ax.grid(True, linestyle="--")
        ax.set_xlim(plot_df["catalyst_concentration"].min() - 0.08, plot_df["catalyst_concentration"].max() + 0.08)
        ax.set_ylim(plot_df["temp_round"].min() - 8, plot_df["temp_round"].max() + 8)
        add_badge(ax, "Color = mean alignment | Size = sample count", color="#0F766E")

    if last_scatter is not None:
        cbar = fig.colorbar(last_scatter, ax=axes, fraction=0.024, pad=0.03)
        cbar.set_label("Alignment")

    fig.suptitle("XR Flow x Catalyst Process Map for Alignment", fontsize=20, y=0.99, fontweight="bold")
    fig.text(
        0.5,
        0.02,
        "Each bubble is a measured process window. Color shows mean alignment and bubble size shows sample count.",
        ha="center",
        fontsize=10,
        color="#64748B",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    return save_fig(fig, "03_xr_flow_catalyst_process_map_alignment.png")


def make_correlation_matrix(df: pd.DataFrame) -> Path:
    corr_cols = [
        "actual_temp",
        "flow_rate",
        "catalyst_concentration",
        "diameter",
        "density",
        "alignment",
        "curvature",
    ]
    labels = [
        "Actual Temp",
        "Flow",
        "Catalyst",
        "Diameter",
        "Density",
        "Alignment",
        "Curvature",
    ]
    corr = df[corr_cols].corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(10.8, 8.6))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_title("XR Parameter-Morphology Correlation", pad=16, fontweight="bold")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=28, ha="right")
    ax.set_yticklabels(labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            if j > i:
                continue
            val = corr.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=10, color="#0F172A")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson r")
    return save_fig(fig, "04_xr_parameter_morphology_correlation.png")


def make_representative_gallery(df: pd.DataFrame) -> Path:
    sub = df.dropna(subset=["actual_temp", "alignment", "diameter"]).sort_values("actual_temp").copy()
    if len(sub) < 6:
        chosen = sub.head(6)
    else:
        idx = np.linspace(0, len(sub) - 1, 6, dtype=int)
        chosen = sub.iloc[idx]

    fig, axes = plt.subplots(2, 3, figsize=(14.8, 8.8))
    axes = axes.ravel()

    for ax, (_, row) in zip(axes, chosen.iterrows()):
        try:
            img = Image.open(row["file_path"])
            arr = normalize_preview(img)
            ax.imshow(arr, cmap="gray")
        except Exception:
            ax.imshow(np.zeros((240, 240)), cmap="gray")
            ax.text(0.5, 0.5, "Image load failed", ha="center", va="center", color="white", transform=ax.transAxes)

        ax.set_title(f"{row['run_name']} / {row['sample_id']}", fontsize=10, pad=8, fontweight="bold")
        ax.text(
            0.02,
            0.02,
            (
                f"{row['actual_temp']:.0f} degC | {row['flow_rate']:.0f} sccm | {row['catalyst_concentration']:.1f} g\n"
                f"Alignment {row['alignment']:.3f} | Diameter {row['diameter']:.1f} nm"
            ),
            transform=ax.transAxes,
            fontsize=8.4,
            color="white",
            ha="left",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.28", fc=(0, 0, 0, 0.58), ec=(1, 1, 1, 0.18)),
        )
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("Representative XR SEM Gallery", fontsize=20, y=0.98, fontweight="bold")
    fig.text(
        0.5,
        0.02,
        "Samples are selected across the temperature range to connect process windows with image-level morphology.",
        ha="center",
        fontsize=10,
        color="#64748B",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    return save_fig(fig, "05_xr_representative_sem_gallery.png")


def write_summary(paths: list[Path], df: pd.DataFrame) -> Path:
    summary = OUTPUT_DIR / "README.md"
    lines = [
        "# XR PPT Figures",
        "",
        "- Generated on: 2026-03-24",
        f"- XR rows used: {len(df)}",
        f"- Actual temperature range: {df['actual_temp'].min():.1f} - {df['actual_temp'].max():.1f} degC",
        f"- Flow levels: {', '.join(str(int(v)) for v in sorted(df['flow_rate'].dropna().unique()))} sccm",
        f"- Catalyst levels: {', '.join(f'{v:.1f}' for v in sorted(df['catalyst_concentration'].dropna().unique()))} g",
        "",
        "## Files",
    ]
    lines.extend([f"- {p.name}" for p in paths])
    summary.write_text("\n".join(lines), encoding="utf-8")
    return summary


def main() -> None:
    apply_theme()
    df = load_xr_data()
    paths = [
        make_3d_process_space(df),
        make_temperature_trends(df),
        make_flow_catalyst_heatmaps(df),
        make_correlation_matrix(df),
        make_representative_gallery(df),
    ]
    summary = write_summary(paths, df)

    print("XR PPT figures generated:")
    for path in paths:
        print(path)
    print(summary)


if __name__ == "__main__":
    main()
