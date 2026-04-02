from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_ROOT = PROJECT_ROOT / "reports"

REPORT_PREFIX = "zzy_feature_engineering_gt10000_"
VISUAL_PREFIX = "zzy_visual_report_"

CORE_FEATURES = [
    "density",
    "diameter",
    "alignment",
    "n_branches",
    "dk_bend_index",
    "curvature_nm_v3_trimmed_mean_sqrt_length",
    "tortuosity_v2",
    "junction_ratio",
]

GAS_PLOT_FEATURES = [
    "density",
    "diameter",
    "alignment",
    "dk_bend_index",
    "tortuosity_v2",
    "junction_ratio",
]

DK_RELATION_FEATURES = [
    "alignment",
    "tortuosity_v2",
    "waviness_ratio_v2",
    "junction_ratio",
    "density",
    "n_branches",
]

LABELS = {
    "density": "Density (%)",
    "diameter": "Diameter (nm)",
    "alignment": "Alignment",
    "n_branches": "Branch Count",
    "dk_bend_index": "d*k Bend Index",
    "curvature_nm_v3_trimmed_mean_sqrt_length": "Curvature Proxy (nm^-1)",
    "tortuosity_v2": "Tortuosity v2",
    "junction_ratio": "Junction Ratio",
    "waviness_ratio_v2": "Waviness Ratio v2",
}

GAS_ORDER = ["low", "mid", "high"]
MAG_ORDER = ["50k", "100k"]
GAS_COLORS = {
    "low": "#2563EB",
    "mid": "#F97316",
    "high": "#059669",
}

CMAP = LinearSegmentedColormap.from_list(
    "zzy_corr",
    ["#8B1E3F", "#F5F7FB", "#0F766E"],
)


def find_latest_engineering_dir() -> Path:
    candidates = [
        path
        for path in REPORTS_ROOT.iterdir()
        if path.is_dir()
        and path.name.startswith(REPORT_PREFIX)
        and (path / "engineered_dataset_active.csv").exists()
    ]
    if not candidates:
        raise FileNotFoundError("No ZZY feature engineering report directory found.")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def apply_theme() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#D6DEEB",
            "axes.labelcolor": "#334155",
            "axes.titlecolor": "#0F172A",
            "axes.titlesize": 15,
            "axes.labelsize": 11,
            "axes.grid": True,
            "grid.color": "#D8E1EC",
            "grid.alpha": 0.22,
            "grid.linestyle": "--",
            "xtick.color": "#475569",
            "ytick.color": "#475569",
            "font.family": "DejaVu Sans",
        }
    )


def save_fig(fig: plt.Figure, out_dir: Path, filename: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    fig.savefig(path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def load_dataset(report_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(report_dir / "engineered_dataset_active.csv")
    df["gas_level"] = pd.Categorical(df["gas_level"], categories=GAS_ORDER, ordered=True)
    df["magnification_bucket"] = pd.Categorical(
        df["magnification_bucket"], categories=MAG_ORDER, ordered=True
    )
    return df


def annotate_corr(ax: plt.Axes, corr: pd.DataFrame) -> None:
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            value = corr.iloc[i, j]
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=9,
                color="#0F172A",
            )


def make_corr_matrix(df: pd.DataFrame, columns: Iterable[str], title: str, subtitle: str, out_dir: Path, filename: str) -> Path:
    corr_df = df[list(columns)].corr(numeric_only=True)
    labels = [LABELS.get(col, col) for col in corr_df.columns]

    fig, ax = plt.subplots(figsize=(10.8, 8.9))
    im = ax.imshow(corr_df.values, cmap=CMAP, vmin=-1, vmax=1)
    ax.set_title(title, pad=16, fontweight="bold")
    ax.text(
        0.0,
        1.03,
        subtitle,
        transform=ax.transAxes,
        fontsize=10,
        color="#64748B",
    )
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=28, ha="right")
    ax.set_yticklabels(labels)
    annotate_corr(ax, corr_df)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson r")
    return save_fig(fig, out_dir, filename)


def make_gas_distribution_panel(df: pd.DataFrame, out_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(15.8, 9.6))
    axes = axes.ravel()

    for ax, feature in zip(axes, GAS_PLOT_FEATURES):
        groups = []
        positions = []
        colors = []
        for idx, level in enumerate(GAS_ORDER, start=1):
            values = df.loc[df["gas_level"] == level, feature].dropna().to_numpy()
            if len(values) == 0:
                continue
            groups.append(values)
            positions.append(idx)
            colors.append(GAS_COLORS[level])

        box = ax.boxplot(
            groups,
            positions=positions,
            widths=0.52,
            patch_artist=True,
            medianprops={"color": "#0F172A", "linewidth": 1.6},
            boxprops={"linewidth": 1.0, "edgecolor": "#CBD5E1"},
            whiskerprops={"color": "#94A3B8"},
            capprops={"color": "#94A3B8"},
        )
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.24)

        for pos, level in zip(positions, [lvl for lvl in GAS_ORDER if not df.loc[df["gas_level"] == lvl, feature].dropna().empty]):
            values = df.loc[df["gas_level"] == level, feature].dropna().to_numpy()
            jitter = np.linspace(-0.12, 0.12, len(values)) if len(values) > 1 else np.array([0.0])
            ax.scatter(
                np.full(len(values), pos) + jitter,
                values,
                s=22,
                alpha=0.78,
                color=GAS_COLORS[level],
                edgecolors="white",
                linewidths=0.35,
            )

        ax.set_title(LABELS.get(feature, feature), fontweight="bold")
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(["Low", "Mid", "High"])
        ax.set_ylabel(LABELS.get(feature, feature))

    fig.suptitle("ZZY Gas-Level Distribution Panel", fontsize=19, y=0.99, fontweight="bold")
    fig.text(
        0.5,
        0.015,
        "Three gas tiers represent total-flow levels under the same gas ratio rather than three independent composition variables.",
        ha="center",
        fontsize=10,
        color="#64748B",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    return save_fig(fig, out_dir, "03_zzy_gas_level_distribution_panel.png")


def add_trend_line(ax: plt.Axes, x: np.ndarray, y: np.ndarray, color: str) -> None:
    if len(x) < 2:
        return
    coef = np.polyfit(x, y, deg=1)
    xs = np.linspace(float(np.min(x)), float(np.max(x)), 80)
    ys = np.polyval(coef, xs)
    ax.plot(xs, ys, color=color, linewidth=2.1, alpha=0.95)


def make_dk_relationship_panel(df: pd.DataFrame, out_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(15.8, 9.6))
    axes = axes.ravel()

    for ax, feature in zip(axes, DK_RELATION_FEATURES):
        for level in GAS_ORDER:
            sub = df[df["gas_level"] == level]
            if sub.empty:
                continue
            ax.scatter(
                sub["dk_bend_index"],
                sub[feature],
                s=28,
                alpha=0.78,
                color=GAS_COLORS[level],
                label=level if feature == DK_RELATION_FEATURES[0] else None,
                edgecolors="white",
                linewidths=0.35,
            )
        valid = df[["dk_bend_index", feature]].dropna()
        add_trend_line(
            ax,
            valid["dk_bend_index"].to_numpy(dtype=float),
            valid[feature].to_numpy(dtype=float),
            color="#0F172A",
        )
        ax.set_title(f"d*k vs {LABELS.get(feature, feature)}", fontweight="bold")
        ax.set_xlabel("d*k Bend Index")
        ax.set_ylabel(LABELS.get(feature, feature))

    axes[0].legend(frameon=False, title="Gas level", loc="best")
    fig.suptitle("ZZY d*k Relationship Panel", fontsize=19, y=0.99, fontweight="bold")
    fig.text(
        0.5,
        0.015,
        "d*k is defined as diameter multiplied by the trimmed V3 curvature proxy.",
        ha="center",
        fontsize=10,
        color="#64748B",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    return save_fig(fig, out_dir, "04_zzy_dk_relationship_panel.png")


def make_distribution_panel(df: pd.DataFrame, out_dir: Path) -> Path:
    plot_features = [
        "density",
        "diameter",
        "alignment",
        "dk_bend_index",
        "tortuosity_v2",
        "junction_ratio",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15.8, 9.6))
    axes = axes.ravel()

    for ax, feature in zip(axes, plot_features):
        values = df[feature].dropna().to_numpy(dtype=float)
        ax.hist(values, bins=18, color="#0F766E", alpha=0.86, edgecolor="white", linewidth=0.8)
        ax.axvline(np.median(values), color="#F97316", linewidth=2.1, linestyle="--", label="Median")
        ax.axvline(np.mean(values), color="#1D4ED8", linewidth=2.1, linestyle="-", label="Mean")
        ax.set_title(f"{LABELS.get(feature, feature)} Distribution", fontweight="bold")
        ax.set_xlabel(LABELS.get(feature, feature))
        ax.set_ylabel("Count")
        if feature == plot_features[0]:
            ax.legend(frameon=False)

    fig.suptitle("ZZY Core Feature Distributions", fontsize=19, y=0.99, fontweight="bold")
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    return save_fig(fig, out_dir, "05_zzy_feature_distribution_panel.png")


def strongest_pairs(df: pd.DataFrame, columns: List[str], top_n: int = 6) -> List[str]:
    corr = df[columns].corr(numeric_only=True)
    pairs = []
    for i, left in enumerate(columns):
        for right in columns[i + 1 :]:
            value = corr.loc[left, right]
            pairs.append((abs(value), value, left, right))
    pairs.sort(reverse=True)
    lines = []
    for _, value, left, right in pairs[:top_n]:
        lines.append(f"- `{LABELS.get(left, left)}` vs `{LABELS.get(right, right)}`: `{value:.3f}`")
    return lines


def write_report(out_dir: Path, source_dir: Path, df: pd.DataFrame, figures: List[Path]) -> Path:
    dk_corr = df[["dk_bend_index", "alignment", "tortuosity_v2", "waviness_ratio_v2", "junction_ratio"]].corr(
        numeric_only=True
    )
    strongest = strongest_pairs(df, CORE_FEATURES)
    counts = df["gas_level"].value_counts().reindex(GAS_ORDER).fillna(0).astype(int).to_dict()
    mag_counts = (
        df["magnification_bucket"].value_counts().reindex(MAG_ORDER).fillna(0).astype(int).to_dict()
    )

    lines = [
        "# ZZY 可视化分析报告",
        "",
        f"- 生成时间: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- 来源工程化数据: `{source_dir}`",
        f"- 样本总数: `{len(df)}`",
        f"- 气体三档分布: `{counts}`",
        f"- 倍率分布: `{mag_counts}`",
        "",
        "## 关键观察",
        "",
        "- `d*k` 与 `alignment` 呈负相关，与 `tortuosity / waviness / junction_ratio` 呈正相关。",
        "- `d*k` 与倍率耦合很弱，适合做跨倍率弯曲强度代理。",
        "- 这批样本的最强混杂仍然是倍率，不是气体三档本身。",
        "",
        "## 核心相关性",
        "",
        *strongest,
        "",
        "## d*k 专项",
        "",
        f"- `d*k` vs `alignment`: `{dk_corr.loc['dk_bend_index', 'alignment']:.3f}`",
        f"- `d*k` vs `tortuosity_v2`: `{dk_corr.loc['dk_bend_index', 'tortuosity_v2']:.3f}`",
        f"- `d*k` vs `waviness_ratio_v2`: `{dk_corr.loc['dk_bend_index', 'waviness_ratio_v2']:.3f}`",
        f"- `d*k` vs `junction_ratio`: `{dk_corr.loc['dk_bend_index', 'junction_ratio']:.3f}`",
        "",
        "## 图件",
        "",
    ]

    captions = {
        "01_zzy_core_feature_correlation.png": "全量核心特征相关性矩阵",
        "02a_zzy_50k_correlation.png": "50k 子集相关性矩阵",
        "02b_zzy_100k_correlation.png": "100k 子集相关性矩阵",
        "03_zzy_gas_level_distribution_panel.png": "气体三档分布面板",
        "04_zzy_dk_relationship_panel.png": "d*k 关系专题图",
        "05_zzy_feature_distribution_panel.png": "核心特征分布图",
    }

    for figure in figures:
        lines.append(f"### {captions.get(figure.name, figure.name)}")
        lines.append("")
        lines.append(f"![{figure.name}]({figure.name})")
        lines.append("")

    path = out_dir / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    apply_theme()
    source_dir = find_latest_engineering_dir()
    df = load_dataset(source_dir)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = REPORTS_ROOT / f"{VISUAL_PREFIX}{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    figures = [
        make_corr_matrix(
            df,
            CORE_FEATURES,
            title="ZZY Core Feature Correlation Matrix",
            subtitle="Full cleaned dataset, Pearson correlation",
            out_dir=out_dir,
            filename="01_zzy_core_feature_correlation.png",
        ),
        make_corr_matrix(
            df[df["magnification_bucket"] == "50k"],
            CORE_FEATURES,
            title="ZZY 50k Correlation Matrix",
            subtitle="Within-magnification view to reduce magnification confounding",
            out_dir=out_dir,
            filename="02a_zzy_50k_correlation.png",
        ),
        make_corr_matrix(
            df[df["magnification_bucket"] == "100k"],
            CORE_FEATURES,
            title="ZZY 100k Correlation Matrix",
            subtitle="Within-magnification view to reduce magnification confounding",
            out_dir=out_dir,
            filename="02b_zzy_100k_correlation.png",
        ),
        make_gas_distribution_panel(df, out_dir),
        make_dk_relationship_panel(df, out_dir),
        make_distribution_panel(df, out_dir),
    ]
    report_path = write_report(out_dir, source_dir, df, figures)

    print(out_dir)
    print(report_path)


if __name__ == "__main__":
    main()
