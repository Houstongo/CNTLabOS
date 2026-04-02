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
TARGET_DIR = Path(r"C:\Users\clearlove\Desktop\selected\4.4\split_panels")
OOF_CSV = Path(r"C:\Users\clearlove\Desktop\all_targets_oof_predictions.csv")

TARGET_ORDER = ["alignment", "curvature", "waviness_ratio", "tortuosity"]
DISPLAY_NAMES = {
    "alignment": "取向度",
    "curvature": "有效平均曲率 / μm^-1",
    "waviness_ratio": "波曲度",
    "tortuosity": "迂曲度",
}
FEATURE_COLORS = {
    "alignment": "#4c78a8",
    "curvature": "#e15759",
    "waviness_ratio": "#59a14f",
    "tortuosity": "#f28e2b",
}
SCATTER_COLORS = {
    "alignment": "#2E86AB",
    "curvature": "#D97841",
    "waviness_ratio": "#3FA796",
    "tortuosity": "#C77DBB",
}
BOX_COLORS = ["#b8d8be", "#8fb3c9", "#d9b382"]
RESPONSE_ORDER = ["curvature", "waviness_ratio", "tortuosity", "alignment"]
RESPONSE_CMAPS = {
    "curvature": "magma",
    "waviness_ratio": "viridis",
    "tortuosity": "PuBu",
    "alignment": "YlGn",
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_anneal_single(df: pd.DataFrame, metric: str) -> Path:
    fig, ax = plt.subplots(figsize=(5.2, 4.1), constrained_layout=True)
    x = sorted(df["anneal_time"].dropna().unique())
    label_map = {0.25: "15 min", 0.5: "30 min", 0.75: "45 min"}
    series_list = [df.loc[df["anneal_time"] == v, metric].dropna().to_numpy() for v in x]
    bp = ax.boxplot(
        series_list,
        patch_artist=True,
        widths=0.52,
        showfliers=False,
        medianprops={"color": "#202020", "linewidth": 1.4},
        whiskerprops={"color": "#444444", "linewidth": 1.1},
        capprops={"color": "#444444", "linewidth": 1.1},
        boxprops={"edgecolor": "#444444", "linewidth": 1.1},
    )
    for patch, color in zip(bp["boxes"], BOX_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.9)
    ax.set_xticks(range(1, len(x) + 1))
    ax.set_xticklabels([label_map.get(v, str(v)) for v in x])
    ax.set_xlabel("退火时间 / min")
    ax.set_ylabel(DISPLAY_NAMES[metric])
    ax.set_title(DISPLAY_NAMES[metric], fontsize=12)
    ax.grid(axis="y", alpha=0.2, linestyle="--")
    out = TARGET_DIR / f"anneal_{metric}.png"
    fig.savefig(out, dpi=320, bbox_inches="tight")
    plt.close(fig)
    return out


def save_feature_importance_single(importance_df: pd.DataFrame, target: str) -> Path:
    fig, ax = plt.subplots(figsize=(5.3, 4.4), constrained_layout=True)
    sub = importance_df[importance_df["target"] == target].copy()
    sub = sub.sort_values("importance_mean", ascending=False).head(8).sort_values("importance_mean", ascending=True)
    ax.barh(sub["feature_cn"], sub["importance_mean"], color=FEATURE_COLORS[target], alpha=0.88)
    ax.set_title(DISPLAY_NAMES[target], fontsize=12)
    ax.set_xlabel("置换重要性")
    ax.grid(axis="x", alpha=0.18, linestyle="--")
    out = TARGET_DIR / f"feature_importance_{target}.png"
    fig.savefig(out, dpi=320, bbox_inches="tight")
    plt.close(fig)
    return out


def save_response_single(df: pd.DataFrame, metric: str) -> Path:
    cmap = matplotlib.colormaps[RESPONSE_CMAPS[metric]].copy()
    cmap.set_bad("#f1f1f1")
    pivot = df.pivot_table(index="fe_thickness", columns="fe_power", values=metric, aggfunc="mean")
    pivot = pivot.sort_index().sort_index(axis=1)
    masked = np.ma.masked_invalid(pivot.values.astype(float))

    fig, ax = plt.subplots(figsize=(5.3, 4.4), constrained_layout=True)
    im = ax.imshow(masked, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{int(v)}" if float(v).is_integer() else f"{v}" for v in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{v:g}" for v in pivot.index])
    ax.set_xlabel("铁功率 / W")
    ax.set_ylabel("铁厚度 / nm")
    ax.set_title(DISPLAY_NAMES[metric], fontsize=12)
    norm = im.norm
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if np.isfinite(val):
                text = f"{val:.2f}" if metric != "alignment" else f"{val:.3f}"
                rgba = im.cmap(norm(val))
                luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                ax.text(j, i, text, ha="center", va="center", fontsize=8, color="white" if luminance < 0.45 else "#202020")
    fig.colorbar(im, ax=ax, shrink=0.9)
    out = TARGET_DIR / f"response_{metric}.png"
    fig.savefig(out, dpi=320, bbox_inches="tight")
    plt.close(fig)
    return out


def panel_limits(sub: pd.DataFrame) -> tuple[float, float]:
    lo = float(min(sub["y_true"].min(), sub["y_predict"].min()))
    hi = float(max(sub["y_true"].max(), sub["y_predict"].max()))
    span = hi - lo
    pad = span * 0.08 if span > 0 else 0.1
    return lo - pad, hi + pad


def save_oof_single(df: pd.DataFrame, target: str) -> Path:
    sub = df[df["target"] == target].copy()
    y_true = sub["y_true"].to_numpy(dtype=float)
    y_pred = sub["y_predict"].to_numpy(dtype=float)
    lo, hi = panel_limits(sub)

    fig, ax = plt.subplots(figsize=(5.3, 4.4), constrained_layout=True)
    ax.set_facecolor("#FCFCFD")
    ax.grid(True, color="#E8ECF2", linestyle="--", linewidth=0.8, alpha=0.9)
    for spine in ax.spines.values():
        spine.set_color("#AAB4C3")
        spine.set_linewidth(0.9)
    ax.scatter(
        y_true,
        y_pred,
        s=42,
        color=SCATTER_COLORS[target],
        alpha=0.86,
        edgecolors="white",
        linewidths=0.7,
        zorder=3,
    )
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.15, color="#5C6670", zorder=2)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("真实值")
    ax.set_ylabel("预测值")
    ax.set_title(DISPLAY_NAMES[target], fontsize=12)
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
        bbox={"facecolor": "white", "alpha": 0.86, "edgecolor": "#cccccc", "boxstyle": "round,pad=0.24"},
    )
    out = TARGET_DIR / f"oof_{target}.png"
    fig.savefig(out, dpi=320, bbox_inches="tight")
    plt.close(fig)
    return out


def save_spearman_single(df: pd.DataFrame) -> Path:
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
    labels = ["铁功率", "铁厚度", "退火时间", "铁沉积指数", "取向度", "有效平均曲率", "波曲度", "迂曲度"]
    corr = df[cols].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(8.4, 6.8), constrained_layout=True)
    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    im = ax.imshow(corr.values, cmap="coolwarm", norm=norm, aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=10)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_title("工艺参数与关键形貌特征相关性热图", fontsize=14)
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = corr.values[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8.6, color="white" if abs(value) > 0.45 else "#202020")
    cbar = fig.colorbar(im, ax=ax, shrink=0.92)
    cbar.set_label("秩相关系数", fontsize=10)
    out = TARGET_DIR / "spearman_correlation_heatmap.png"
    fig.savefig(out, dpi=320, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    ensure_dir(TARGET_DIR)
    controlled = pd.read_csv(SOURCE_DIR / "controlled_subset_raw.csv")
    importance_df = pd.read_csv(SOURCE_DIR / "feature_importance.csv")
    oof_df = pd.read_csv(OOF_CSV)

    outputs: list[Path] = []
    for metric in TARGET_ORDER:
        outputs.append(save_anneal_single(controlled, metric))
    for target in TARGET_ORDER:
        outputs.append(save_feature_importance_single(importance_df, target))
    for metric in RESPONSE_ORDER:
        outputs.append(save_response_single(controlled, metric))
    for target in TARGET_ORDER:
        outputs.append(save_oof_single(oof_df, target))
    outputs.append(save_spearman_single(controlled))

    manifest = pd.DataFrame({"file": [p.name for p in outputs], "path": [str(p) for p in outputs]})
    manifest.to_csv(TARGET_DIR / "manifest.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
