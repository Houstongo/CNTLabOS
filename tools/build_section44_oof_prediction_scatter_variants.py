from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score


matplotlib.use("Agg")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(r"D:\CNTDATA\CNTA_ML_Project")
INPUT_CSV = Path(r"C:\Users\clearlove\Desktop\all_targets_oof_predictions.csv")
OUTPUT_DIR = PROJECT_ROOT / "reports" / "paper_section_4_4_data_bundle_20260402" / "oof_scatter_variants_20260402"

TARGET_ORDER = ["alignment", "curvature", "waviness_ratio", "tortuosity"]
TARGET_LABELS = {
    "alignment": "取向度",
    "curvature": "有效平均曲率 / μm$^{-1}$",
    "waviness_ratio": "波曲度",
    "tortuosity": "迂曲度",
}
STYLE_A_COLORS = {
    "alignment": "#2E86AB",
    "curvature": "#D97841",
    "waviness_ratio": "#3FA796",
    "tortuosity": "#C77DBB",
}
STYLE_B_COLORS = {
    "alignment": "#2B5C8A",
    "curvature": "#A85C32",
    "waviness_ratio": "#2C7A6B",
    "tortuosity": "#9D5C93",
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    needed = {"target", "y_true", "y_predict"}
    missing = needed.difference(df.columns)
    if missing:
        raise ValueError(f"CSV 缺少必要列: {sorted(missing)}")
    df = df[df["target"].isin(TARGET_ORDER)].copy()
    df["target"] = pd.Categorical(df["target"], categories=TARGET_ORDER, ordered=True)
    return df


def metric_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in TARGET_ORDER:
        sub = df[df["target"] == target].copy()
        y_true = sub["y_true"].to_numpy(dtype=float)
        y_pred = sub["y_predict"].to_numpy(dtype=float)
        rows.append(
            {
                "target": target,
                "target_cn": TARGET_LABELS[target],
                "n": int(len(sub)),
                "r2": float(r2_score(y_true, y_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                "mae": float(np.mean(np.abs(y_true - y_pred))),
            }
        )
    return pd.DataFrame(rows)


def panel_limits(sub: pd.DataFrame) -> tuple[float, float]:
    lo = float(min(sub["y_true"].min(), sub["y_predict"].min()))
    hi = float(max(sub["y_true"].max(), sub["y_predict"].max()))
    span = hi - lo
    pad = span * 0.08 if span > 0 else 0.1
    return lo - pad, hi + pad


def annotate_stats(ax: plt.Axes, summary_row: pd.Series, *, style_name: str) -> None:
    face = "#FFFFFF" if style_name == "A" else "#FAF8F5"
    edge = "#D0D5DD" if style_name == "A" else "#C7BBB0"
    text = (
        f"样本数 N = {int(summary_row['n'])}\n"
        f"R² = {summary_row['r2']:.3f}\n"
        f"RMSE = {summary_row['rmse']:.3f}\n"
        f"MAE = {summary_row['mae']:.3f}"
    )
    ax.text(
        0.04,
        0.96,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="#2B2B2B",
        linespacing=1.35,
        bbox={
            "facecolor": face,
            "edgecolor": edge,
            "alpha": 0.96,
            "boxstyle": "round,pad=0.34",
            "linewidth": 0.9,
        },
    )


def style_axes_a(ax: plt.Axes) -> None:
    ax.set_facecolor("#FCFCFD")
    ax.grid(True, color="#E8ECF2", linestyle="--", linewidth=0.8, alpha=0.9)
    for spine in ax.spines.values():
        spine.set_color("#AAB4C3")
        spine.set_linewidth(0.9)


def style_axes_b(ax: plt.Axes) -> None:
    ax.set_facecolor("#FFFEFC")
    ax.grid(True, axis="y", color="#E9E1D9", linestyle="-", linewidth=0.8, alpha=0.8)
    ax.grid(False, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#8B8178")
    ax.spines["bottom"].set_color("#8B8178")
    ax.spines["left"].set_linewidth(0.95)
    ax.spines["bottom"].set_linewidth(0.95)


def draw_style_a(df: pd.DataFrame, summary_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.8), constrained_layout=True)
    axes = axes.flatten()

    for ax, target in zip(axes, TARGET_ORDER):
        sub = df[df["target"] == target].copy()
        info = summary_df[summary_df["target"] == target].iloc[0]
        lo, hi = panel_limits(sub)
        color = STYLE_A_COLORS[target]

        style_axes_a(ax)
        ax.scatter(
            sub["y_true"],
            sub["y_predict"],
            s=42,
            color=color,
            alpha=0.86,
            edgecolors="white",
            linewidths=0.7,
            zorder=3,
        )
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.15, color="#5C6670", zorder=2)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("真实值", fontsize=10)
        ax.set_ylabel("预测值", fontsize=10)
        ax.set_title(TARGET_LABELS[target], fontsize=12, pad=8)
        annotate_stats(ax, info, style_name="A")

    fig.suptitle("四个关键形貌指标 OOF 预测散点图", fontsize=15, fontweight="bold")
    fig.savefig(out_path, dpi=360, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def draw_style_b(df: pd.DataFrame, summary_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.6), constrained_layout=True)
    fig.patch.set_facecolor("#FFFDF9")
    axes = axes.flatten()

    for ax, target in zip(axes, TARGET_ORDER):
        sub = df[df["target"] == target].copy()
        info = summary_df[summary_df["target"] == target].iloc[0]
        lo, hi = panel_limits(sub)
        color = STYLE_B_COLORS[target]
        abs_err = np.abs(sub["y_predict"] - sub["y_true"]).to_numpy(dtype=float)
        err_scale = abs_err / abs_err.max() if abs_err.max() > 0 else abs_err
        sizes = 30 + 42 * err_scale

        style_axes_b(ax)
        ax.scatter(
            sub["y_true"],
            sub["y_predict"],
            s=sizes,
            color=color,
            alpha=0.72,
            edgecolors="#FFF9F3",
            linewidths=0.8,
            zorder=3,
        )
        ax.plot([lo, hi], [lo, hi], linestyle=(0, (4, 2.2)), linewidth=1.2, color="#6B625A", zorder=2)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("真实值", fontsize=10)
        ax.set_ylabel("预测值", fontsize=10)
        ax.set_title(TARGET_LABELS[target], fontsize=12, pad=10, color="#3E352E")
        annotate_stats(ax, info, style_name="B")

    fig.suptitle("四个关键形貌指标 OOF 预测散点图", fontsize=15, fontweight="bold", color="#3E352E")
    fig.savefig(out_path, dpi=360, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    df = load_data(INPUT_CSV)
    summary_df = metric_summary(df)
    summary_df.to_csv(OUTPUT_DIR / "oof_prediction_summary.csv", index=False, encoding="utf-8-sig")
    draw_style_a(df, summary_df, OUTPUT_DIR / "oof_prediction_scatter_style_A.png")
    draw_style_b(df, summary_df, OUTPUT_DIR / "oof_prediction_scatter_style_B.png")


if __name__ == "__main__":
    main()
