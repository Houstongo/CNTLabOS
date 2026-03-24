from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap


DB_PATH = Path(r"D:\CNTDATA\CNTA_ML_Project\database\cnta_experiments.sqlite")
OUTPUT_DIR = Path(r"D:\CNTDATA\CNTA_ML_Project\reports\xr_ppt_figures_20260324")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

for font_path in [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\msyhbd.ttc", r"C:\Windows\Fonts\simhei.ttf"]:
    if Path(font_path).exists():
        fm.fontManager.addfont(font_path)

plt.rcParams["font.family"] = "Microsoft YaHei"
CN_FONT = fm.FontProperties(fname=r"C:\Windows\Fonts\msyh.ttc")
CN_FONT_BOLD = fm.FontProperties(fname=r"C:\Windows\Fonts\msyhbd.ttc")


XR_CMAP = LinearSegmentedColormap.from_list(
    "xr_blue_orange",
    ["#0f172a", "#2563eb", "#14b8a6", "#f59e0b", "#ea580c"],
)

FLOW_COLORS = {
    200.0: "#2563eb",
    250.0: "#14b8a6",
    300.0: "#f97316",
}


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

    df["temp_round"] = df["actual_temp"].round(1)
    df["temp_bin"] = pd.cut(df["actual_temp"], bins=8, duplicates="drop")
    return df


def save_fig(fig: plt.Figure, filename: str) -> Path:
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def add_figure_badge(ax, text: str) -> None:
    ax.text(
        0.0,
        1.08,
        text,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        color="#2563eb",
        bbox=dict(boxstyle="round,pad=0.28", fc="#eff6ff", ec="#bfdbfe"),
        fontproperties=CN_FONT,
    )


def apply_axis_cn_font(ax) -> None:
    ax.title.set_fontproperties(CN_FONT_BOLD)
    ax.xaxis.label.set_fontproperties(CN_FONT)
    ax.yaxis.label.set_fontproperties(CN_FONT)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(CN_FONT)
    legend = ax.get_legend()
    if legend:
        for text in legend.get_texts():
            text.set_fontproperties(CN_FONT)
        if legend.get_title():
            legend.get_title().set_fontproperties(CN_FONT)


def make_3d_process_space(df: pd.DataFrame) -> Path:
    data = df.dropna(subset=["actual_temp", "flow_rate", "catalyst_concentration", "diameter"]).copy()
    fig = plt.figure(figsize=(10.8, 8.2))
    ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(
        data["actual_temp"],
        data["flow_rate"],
        data["catalyst_concentration"],
        c=data["diameter"],
        cmap=XR_CMAP,
        s=62,
        alpha=0.95,
        edgecolor="white",
        linewidth=0.5,
    )

    ax.set_title("XR组工艺空间与平均管径分布", fontsize=18, pad=22, fontproperties=CN_FONT_BOLD)
    ax.set_xlabel("实际温度 (°C)", labelpad=12, fontproperties=CN_FONT)
    ax.set_ylabel("流速 (sccm)", labelpad=12, fontproperties=CN_FONT)
    ax.set_zlabel("催化剂浓度 (g)", labelpad=12, fontproperties=CN_FONT)
    ax.view_init(elev=24, azim=-58)
    ax.xaxis.pane.set_facecolor((0.96, 0.98, 1.0, 1.0))
    ax.yaxis.pane.set_facecolor((0.96, 0.98, 1.0, 1.0))
    ax.zaxis.pane.set_facecolor((0.99, 0.99, 1.0, 1.0))

    cbar = fig.colorbar(sc, ax=ax, pad=0.08, shrink=0.82)
    cbar.set_label("平均管径 (nm)", fontsize=11, fontproperties=CN_FONT)
    for label in cbar.ax.get_yticklabels():
        label.set_fontproperties(CN_FONT)

    fig.text(
        0.12,
        0.92,
        f"XR样本点 {len(data)} 个 | 温度 {data['actual_temp'].min():.0f}-{data['actual_temp'].max():.0f}°C",
        fontsize=10,
        color="#475569",
        fontproperties=CN_FONT,
    )
    return save_fig(fig, "01_XR_3D工艺空间_平均管径.png")


def make_temperature_trends(df: pd.DataFrame) -> Path:
    feature_meta = [
        ("diameter", "平均管径", "nm"),
        ("density", "面填充密度", "%"),
        ("alignment", "垂直取向度", ""),
        ("curvature", "平均骨架曲率", ""),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.2), sharex=True)
    axes = axes.ravel()

    for ax, (feature, title, unit) in zip(axes, feature_meta):
        sub = df.dropna(subset=["actual_temp", feature]).copy()
        for flow, color in FLOW_COLORS.items():
            flow_sub = sub[sub["flow_rate"] == flow]
            if flow_sub.empty:
                continue
            ax.scatter(
                flow_sub["actual_temp"],
                flow_sub[feature],
                s=34,
                alpha=0.65,
                color=color,
                label=f"{int(flow)} sccm",
                edgecolors="white",
                linewidths=0.4,
            )

        trend = (
            sub.groupby("temp_round", as_index=False)[feature]
            .mean()
            .sort_values("temp_round")
        )
        ax.plot(trend["temp_round"], trend[feature], color="#0f172a", linewidth=2.4)
        ax.fill_between(
            trend["temp_round"],
            trend[feature],
            alpha=0.08,
            color="#0f172a",
        )
        ax.set_title(title, fontsize=14, fontproperties=CN_FONT_BOLD)
        ax.grid(alpha=0.18, linestyle="--")
        ax.set_ylabel(f"{title}{f' ({unit})' if unit else ''}")
        add_figure_badge(ax, "按实际温度展开")
        apply_axis_cn_font(ax)

    axes[0].legend(frameon=False, loc="best", title="流速", prop=CN_FONT)
    for ax in axes[2:]:
        ax.set_xlabel("实际温度 (°C)")
        apply_axis_cn_font(ax)

    fig.suptitle("XR组温度梯度下四个形貌特征的变化趋势", fontsize=20, y=0.98, fontproperties=CN_FONT_BOLD)
    fig.text(0.5, 0.01, "散点表示单个 XR 图像样本，黑色曲线为同温度样本均值趋势。", ha="center", fontsize=10, color="#64748b", fontproperties=CN_FONT)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return save_fig(fig, "02_XR_温度梯度_四特征趋势.png")


def make_flow_catalyst_heatmaps(df: pd.DataFrame) -> Path:
    target = "alignment"
    title = "垂直取向度"
    flow_levels = sorted(v for v in df["flow_rate"].dropna().unique())
    fig, axes = plt.subplots(1, len(flow_levels), figsize=(15.8, 9.2), sharey=True)
    if len(flow_levels) == 1:
        axes = [axes]

    vmin = df[target].min()
    vmax = df[target].max()

    for ax, flow in zip(axes, flow_levels):
        sub = df[(df["flow_rate"] == flow) & df[target].notna()].copy()
        pivot = pd.pivot_table(
            sub,
            index="temp_round",
            columns="catalyst_concentration",
            values=target,
            aggfunc="mean",
        ).sort_index()
        im = ax.imshow(
            pivot.values,
            aspect="auto",
            cmap=XR_CMAP,
            vmin=vmin,
            vmax=vmax,
            origin="lower",
        )
        ax.set_title(f"流速 {int(flow)} sccm", fontsize=14, fontproperties=CN_FONT_BOLD)
        ax.set_xlabel("催化剂浓度 (g)")
        if ax is axes[0]:
            ax.set_ylabel("实际温度 (°C)")
        else:
            ax.set_ylabel("")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{c:.1f}" for c in pivot.columns], rotation=0)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{i:.0f}" for i in pivot.index])
        add_figure_badge(ax, f"颜色 = {title}")
        if ax is axes[-1]:
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(title, fontsize=10, fontproperties=CN_FONT)
            for label in cbar.ax.get_yticklabels():
                label.set_fontproperties(CN_FONT)
        apply_axis_cn_font(ax)

    fig.suptitle("XR组流速 × 催化剂浓度分面热图", fontsize=20, y=0.99, fontproperties=CN_FONT_BOLD)
    fig.text(0.5, 0.02, "每个色块为对应工艺组合下样本平均取向度，适合直接用于工艺窗口展示。", ha="center", fontsize=10, color="#64748b", fontproperties=CN_FONT)
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    return save_fig(fig, "03_XR_流速x催化剂浓度_取向度热图.png")


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
    corr_df = df[corr_cols].copy()
    corr = corr_df.corr(numeric_only=True)
    labels = {
        "actual_temp": "实际温度",
        "flow_rate": "流速",
        "catalyst_concentration": "催化剂浓度",
        "diameter": "平均管径",
        "density": "面填充密度",
        "alignment": "垂直取向度",
        "curvature": "平均骨架曲率",
    }
    corr = corr.rename(index=labels, columns=labels)

    fig, ax = plt.subplots(figsize=(10.5, 8.5))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_title("XR组工艺参数与形貌特征相关性矩阵", fontsize=18, pad=16, fontproperties=CN_FONT_BOLD)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(list(corr.columns), rotation=28, ha="right")
    ax.set_yticklabels(list(corr.index))
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            if j > i:
                continue
            val = corr.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=10, color="black", fontproperties=CN_FONT)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson r", fontsize=10, fontproperties=CN_FONT)
    for label in cbar.ax.get_yticklabels():
        label.set_fontproperties(CN_FONT)
    apply_axis_cn_font(ax)
    return save_fig(fig, "04_XR_工艺参数_形貌特征_相关性矩阵.png")


def _normalize_preview(img: Image.Image) -> np.ndarray:
    arr = np.asarray(img.convert("L"), dtype=np.float32)
    lo, hi = np.percentile(arr, [2, 98])
    if hi <= lo:
        hi = lo + 1
    arr = np.clip((arr - lo) / (hi - lo), 0, 1)
    return arr


def make_representative_gallery(df: pd.DataFrame) -> Path:
    sub = df.dropna(subset=["actual_temp", "alignment", "diameter"]).sort_values("actual_temp").copy()
    if len(sub) < 6:
        chosen = sub.head(6)
    else:
        idx = np.linspace(0, len(sub) - 1, 6, dtype=int)
        chosen = sub.iloc[idx]

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.8))
    axes = axes.ravel()

    for ax, (_, row) in zip(axes, chosen.iterrows()):
        path = row["file_path"]
        try:
            img = Image.open(path)
            arr = _normalize_preview(img)
            ax.imshow(arr, cmap="gray")
        except Exception:
            ax.imshow(np.zeros((200, 200)), cmap="gray")
            ax.text(0.5, 0.5, "图像加载失败", ha="center", va="center", color="white", transform=ax.transAxes)

        ax.set_title(
            f"{row['run_name']}\n{row['sample_id']}",
            fontsize=10,
            pad=8,
            fontproperties=CN_FONT_BOLD,
        )
        ax.text(
            0.02,
            0.02,
            f"{row['actual_temp']:.0f}°C | {row['flow_rate']:.0f} sccm | {row['catalyst_concentration']:.1f} g\n"
            f"取向 {row['alignment']:.3f} | 管径 {row['diameter']:.1f} nm",
            transform=ax.transAxes,
            fontsize=8.6,
            color="white",
            ha="left",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.28", fc=(0, 0, 0, 0.55), ec=(1, 1, 1, 0.15)),
            fontproperties=CN_FONT,
        )
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("XR组代表性 SEM 图像带", fontsize=20, y=0.98, fontproperties=CN_FONT_BOLD)
    fig.text(0.5, 0.02, "从低到高温区均匀抽取代表样品，用于在 PPT 中把图像直觉和量化特征并列展示。", ha="center", fontsize=10, color="#64748b", fontproperties=CN_FONT)
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    return save_fig(fig, "05_XR_代表性SEM图像带.png")


def write_summary(paths: list[Path], df: pd.DataFrame) -> Path:
    summary = OUTPUT_DIR / "README.md"
    lines = [
        "# XR PPT 图输出",
        "",
        f"- 生成日期: 2026-03-24",
        f"- XR 样本总数: {len(df)}",
        f"- 实际温度范围: {df['actual_temp'].min():.1f} - {df['actual_temp'].max():.1f} °C",
        f"- 流速水平: {', '.join(str(int(v)) for v in sorted(df['flow_rate'].dropna().unique()))} sccm",
        f"- 催化剂浓度水平: {', '.join(f'{v:.1f}' for v in sorted(df['catalyst_concentration'].dropna().unique()))} g",
        "",
        "## 输出文件",
    ]
    lines.extend([f"- {p.name}" for p in paths])
    summary.write_text("\n".join(lines), encoding="utf-8")
    return summary


def main() -> None:
    plt.style.use("default")
    df = load_xr_data()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    paths = [
        make_3d_process_space(df),
        make_temperature_trends(df),
        make_flow_catalyst_heatmaps(df),
        make_correlation_matrix(df),
        make_representative_gallery(df),
    ]
    summary = write_summary(paths, df)

    print("XR PPT 图已生成:")
    for path in paths:
        print(path)
    print(summary)


if __name__ == "__main__":
    main()
