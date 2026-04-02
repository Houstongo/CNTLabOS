from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from sklearn.metrics import mean_squared_error, r2_score


matplotlib.use("Agg")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(r"D:\CNTDATA\CNTA_ML_Project")
SOURCE_DIR = PROJECT_ROOT / "reports" / "paper_section_4_4_data_bundle_20260402"
TARGET_DIR = Path(r"C:\Users\clearlove\Desktop\selected\4.4")
OOF_CSV = Path(r"C:\Users\clearlove\Desktop\all_targets_oof_predictions.csv")

TARGET_ORDER = ["alignment", "curvature", "waviness_ratio", "tortuosity"]
TARGET_LABELS = {
    "alignment": "取向度",
    "curvature": "有效平均曲率 / μm$^{-1}$",
    "waviness_ratio": "波曲度",
    "tortuosity": "迂曲度",
}
PANEL_TAGS = {
    "alignment": "（a） 取向度",
    "curvature": "（b） 有效平均曲率",
    "waviness_ratio": "（c） 波曲度",
    "tortuosity": "（d） 迂曲度",
}
PANEL_COLORS = {
    "alignment": "#2E86AB",
    "curvature": "#D97841",
    "waviness_ratio": "#3FA796",
    "tortuosity": "#C77DBB",
}
BOX_COLORS = ["#BCD8C0", "#95B6CF", "#DEBE8F"]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def add_panel_caption(ax: plt.Axes, text: str) -> None:
    ax.text(
        0.5,
        -0.21,
        text,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=12.5,
        fontweight="bold",
        color="#202020",
    )


def style_box_axes(ax: plt.Axes) -> None:
    ax.grid(axis="y", alpha=0.2, linestyle="--", color="#D7DEE7")
    for spine in ax.spines.values():
        spine.set_color("#7C8794")
        spine.set_linewidth(0.9)


def render_anneal_trend() -> None:
    df = pd.read_csv(SOURCE_DIR / "controlled_subset_raw.csv")
    metrics = ["alignment", "curvature", "waviness_ratio", "tortuosity"]
    label_map = {0.25: "15 min", 0.5: "30 min", 0.75: "45 min"}
    x = sorted(df["anneal_time"].dropna().unique())

    fig, axes = plt.subplots(2, 2, figsize=(10.6, 8.4))
    fig.subplots_adjust(left=0.07, right=0.985, top=0.965, bottom=0.1, hspace=0.26, wspace=0.14)
    axes = axes.flatten()

    for ax, metric in zip(axes, metrics):
        style_box_axes(ax)
        series_list = [df.loc[df["anneal_time"] == v, metric].dropna().to_numpy() for v in x]
        bp = ax.boxplot(
            series_list,
            patch_artist=True,
            widths=0.52,
            showfliers=False,
            medianprops={"color": "#303030", "linewidth": 1.5},
            whiskerprops={"color": "#5B6570", "linewidth": 1.1},
            capprops={"color": "#5B6570", "linewidth": 1.1},
            boxprops={"edgecolor": "#56606B", "linewidth": 1.05},
        )
        for patch, color in zip(bp["boxes"], BOX_COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(0.95)
        ax.set_xticks(range(1, len(x) + 1))
        ax.set_xticklabels([label_map.get(v, str(v)) for v in x])
        ax.set_xlabel("退火时间 / min")
        ax.set_ylabel(TARGET_LABELS[metric])
        add_panel_caption(ax, PANEL_TAGS[metric])

    fig.savefig(TARGET_DIR / "anneal_time_trend_thesis.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def render_feature_importance() -> None:
    importance_df = pd.read_csv(SOURCE_DIR / "feature_importance.csv")
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.8))
    fig.subplots_adjust(left=0.11, right=0.985, top=0.97, bottom=0.1, hspace=0.22, wspace=0.34)
    axes = axes.flatten()

    for ax, target in zip(axes, TARGET_ORDER):
        sub = importance_df[importance_df["target"] == target].copy()
        sub = sub.sort_values("importance_mean", ascending=False).head(8).sort_values("importance_mean", ascending=True)
        ax.barh(sub["feature_cn"], sub["importance_mean"], color=PANEL_COLORS[target], alpha=0.86)
        ax.set_xlabel("置换重要性")
        ax.grid(axis="x", alpha=0.18, linestyle="--", color="#D8DEE6")
        for spine in ax.spines.values():
            spine.set_color("#7C8794")
            spine.set_linewidth(0.9)
        add_panel_caption(ax, PANEL_TAGS[target])

    fig.savefig(TARGET_DIR / "feature_importance_panel_thesis.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def render_fe_response() -> None:
    df = pd.read_csv(SOURCE_DIR / "controlled_subset_raw.csv")
    metrics = ["curvature", "waviness_ratio", "tortuosity", "alignment"]
    cmaps = {
        "curvature": matplotlib.colormaps["magma"].copy(),
        "waviness_ratio": matplotlib.colormaps["viridis"].copy(),
        "tortuosity": matplotlib.colormaps["PuBu"].copy(),
        "alignment": matplotlib.colormaps["YlGn"].copy(),
    }
    for cmap in cmaps.values():
        cmap.set_bad("#F3F4F6")

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.8))
    fig.subplots_adjust(left=0.07, right=0.95, top=0.975, bottom=0.11, hspace=0.26, wspace=0.19)
    axes = axes.flatten()

    for ax, metric in zip(axes, metrics):
        pivot = df.pivot_table(index="fe_thickness", columns="fe_power", values=metric, aggfunc="mean")
        pivot = pivot.sort_index().sort_index(axis=1)
        masked = np.ma.masked_invalid(pivot.values.astype(float))
        im = ax.imshow(masked, cmap=cmaps[metric], aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{int(v)}" if float(v).is_integer() else f"{v}" for v in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{v:g}" for v in pivot.index])
        ax.set_xlabel("铁功率 / W")
        ax.set_ylabel("铁厚度 / nm")
        norm = im.norm
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if np.isfinite(val):
                    text = f"{val:.2f}" if metric != "alignment" else f"{val:.3f}"
                    rgba = im.cmap(norm(val))
                    luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                    ax.text(
                        j,
                        i,
                        text,
                        ha="center",
                        va="center",
                        fontsize=7.5,
                        color="white" if luminance < 0.45 else "#202020",
                    )
        cbar = fig.colorbar(im, ax=ax, shrink=0.88)
        cbar.ax.tick_params(labelsize=8)
        add_panel_caption(ax, PANEL_TAGS[metric])

    fig.savefig(TARGET_DIR / "fe_parameter_response_panel_thesis.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def panel_limits(sub: pd.DataFrame) -> tuple[float, float]:
    lo = float(min(sub["y_true"].min(), sub["y_predict"].min()))
    hi = float(max(sub["y_true"].max(), sub["y_predict"].max()))
    span = hi - lo
    pad = span * 0.08 if span > 0 else 0.1
    return lo - pad, hi + pad


def render_oof_scatter() -> None:
    df = pd.read_csv(OOF_CSV)
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.7))
    fig.subplots_adjust(left=0.07, right=0.985, top=0.97, bottom=0.095, hspace=0.24, wspace=0.14)
    axes = axes.flatten()

    for ax, target in zip(axes, TARGET_ORDER):
        sub = df[df["target"] == target].copy()
        y_true = sub["y_true"].to_numpy(dtype=float)
        y_pred = sub["y_predict"].to_numpy(dtype=float)
        lo, hi = panel_limits(sub)

        ax.set_facecolor("#FCFCFD")
        ax.grid(True, color="#E8ECF2", linestyle="--", linewidth=0.8, alpha=0.9)
        for spine in ax.spines.values():
            spine.set_color("#AAB4C3")
            spine.set_linewidth(0.9)
        ax.scatter(
            y_true,
            y_pred,
            s=40,
            color=PANEL_COLORS[target],
            alpha=0.84,
            edgecolors="white",
            linewidths=0.7,
            zorder=3,
        )
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.15, color="#5C6670", zorder=2)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("真实值")
        ax.set_ylabel("预测值")
        ax.text(
            0.04,
            0.96,
            (
                f"样本数 N = {len(sub)}\n"
                f"R² = {r2_score(y_true, y_pred):.3f}\n"
                f"RMSE = {np.sqrt(mean_squared_error(y_true, y_pred)):.3f}\n"
                f"MAE = {np.mean(np.abs(y_true - y_pred)):.3f}"
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            color="#2B2B2B",
            linespacing=1.35,
            bbox={
                "facecolor": "#FFFFFF",
                "edgecolor": "#D0D5DD",
                "alpha": 0.96,
                "boxstyle": "round,pad=0.34",
                "linewidth": 0.9,
            },
        )
        add_panel_caption(ax, PANEL_TAGS[target])

    fig.savefig(TARGET_DIR / "oof_prediction_scatter_style_A_thesis.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def render_spearman() -> None:
    df = pd.read_csv(SOURCE_DIR / "controlled_subset_raw.csv")
    cols = [
        "fe_power",
        "fe_thickness",
        "anneal_time",
        "fe_deposition_index",
        "alignment",
        "curvature",
        "waviness_ratio",
        "tortuosity",
    ]
    corr = df[cols].corr(method="spearman")
    labels = ["铁功率", "铁厚度", "退火时间", "铁沉积指数", "取向度", "有效平均曲率", "波曲度", "迂曲度"]

    fig, ax = plt.subplots(figsize=(8.8, 7.2))
    fig.subplots_adjust(left=0.11, right=0.94, top=0.96, bottom=0.12)
    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    im = ax.imshow(corr.values, cmap="coolwarm", norm=norm, aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=10)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = corr.values[i, j]
            color = "white" if abs(value) > 0.45 else "#202020"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8.6, color=color)
    cbar = fig.colorbar(im, ax=ax, shrink=0.92)
    cbar.set_label("秩相关系数", fontsize=10)
    fig.savefig(TARGET_DIR / "spearman_correlation_heatmap_thesis.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ensure_dir(TARGET_DIR)
    render_anneal_trend()
    render_feature_importance()
    render_fe_response()
    render_oof_scatter()
    render_spearman()


if __name__ == "__main__":
    main()
