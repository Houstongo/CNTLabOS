from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sqlite3
from matplotlib.colors import LinearSegmentedColormap


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_ROOT = PROJECT_ROOT / "reports"
DB_PATH = PROJECT_ROOT / "database" / "cnta_experiments.sqlite"

ENGINEERING_PREFIX = "zzy_feature_engineering_gt10000_"
OUTPUT_PREFIX = "zzy_process_visual_report_"

PROCESS_COLS = [
    "fe_thickness",
    "fe_power",
    "al2o3_thickness",
    "al2o3_power",
    "gas_total",
]

FEATURE_COLS = [
    "density",
    "diameter",
    "alignment",
    "dk_bend_index",
    "curvature_nm_v3_trimmed_mean_sqrt_length",
    "tortuosity_v2",
    "junction_ratio",
]

LABELS = {
    "fe_thickness": "Fe Thickness (nm)",
    "fe_power": "Fe Power (W)",
    "al2o3_thickness": "Al2O3 Thickness (nm)",
    "al2o3_power": "Al2O3 Power (W)",
    "gas_total": "Gas Total Flow",
    "density": "Density (%)",
    "diameter": "Diameter (nm)",
    "alignment": "Alignment",
    "dk_bend_index": "d*k Bend Index",
    "curvature_nm_v3_trimmed_mean_sqrt_length": "Curvature Proxy (nm^-1)",
    "tortuosity_v2": "Tortuosity v2",
    "junction_ratio": "Junction Ratio",
}

GAS_ORDER = ["low", "mid", "high"]
MAG_ORDER = ["50k", "100k"]
GAS_COLORS = {
    "low": "#2563EB",
    "mid": "#F97316",
    "high": "#059669",
}

CMAP = LinearSegmentedColormap.from_list(
    "process_corr",
    ["#8B1E3F", "#F7F9FC", "#0F766E"],
)


def find_latest_engineering_dir() -> Path:
    candidates = [
        path
        for path in REPORTS_ROOT.iterdir()
        if path.is_dir()
        and path.name.startswith(ENGINEERING_PREFIX)
        and (path / "engineered_dataset_active.csv").exists()
    ]
    if not candidates:
        raise FileNotFoundError("No engineered ZZY dataset found.")
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


def load_dataset(engineering_dir: Path) -> pd.DataFrame:
    eng = pd.read_csv(engineering_dir / "engineered_dataset_active.csv")
    conn = sqlite3.connect(DB_PATH)
    proc = pd.read_sql_query(
        """
        SELECT
            id AS image_id,
            fe_thickness,
            fe_power,
            al2o3_thickness,
            al2o3_power,
            actual_temp,
            membrane_pos_cm,
            growth_time,
            anneal_temp,
            anneal_time
        FROM images
        WHERE source='ZZY' AND COALESCE(is_deleted,0)=0 AND magnification>10000
        """,
        conn,
    )
    conn.close()
    df = eng.merge(proc, on="image_id", how="left")
    df["gas_total"] = df["c2h4_flow"] + df["ar_flow"] + df["h2_flow"]
    df["gas_level"] = pd.Categorical(df["gas_level"], categories=GAS_ORDER, ordered=True)
    df["magnification_bucket"] = pd.Categorical(
        df["magnification_bucket"], categories=MAG_ORDER, ordered=True
    )
    return df


def annotate_corr(ax: plt.Axes, corr: pd.DataFrame) -> None:
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            value = corr.iloc[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8.5, color="#0F172A")


def make_process_feature_corr(df: pd.DataFrame, out_dir: Path) -> Path:
    cols = PROCESS_COLS + FEATURE_COLS
    corr = df[cols].corr(numeric_only=True)
    labels = [LABELS.get(col, col) for col in cols]

    fig, ax = plt.subplots(figsize=(12.4, 9.8))
    im = ax.imshow(corr.values, cmap=CMAP, vmin=-1, vmax=1)
    ax.set_title("ZZY Process-to-Morphology Correlation", pad=16, fontweight="bold")
    ax.text(
        0.0,
        1.03,
        "Only process variables with real variation in this batch are included.",
        transform=ax.transAxes,
        fontsize=10,
        color="#64748B",
    )
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    annotate_corr(ax, corr)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson r")
    return save_fig(fig, out_dir, "01_zzy_process_feature_correlation.png")


def add_trend_line(ax: plt.Axes, x: np.ndarray, y: np.ndarray, color: str = "#0F172A") -> None:
    if len(x) < 2:
        return
    coef = np.polyfit(x, y, deg=1)
    xs = np.linspace(float(np.min(x)), float(np.max(x)), 80)
    ys = np.polyval(coef, xs)
    ax.plot(xs, ys, color=color, linewidth=2.0, alpha=0.92)


def make_fe_thickness_panel(df: pd.DataFrame, out_dir: Path) -> Path:
    features = ["dk_bend_index", "diameter", "alignment", "density"]
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 9.6))
    axes = axes.ravel()

    for ax, feature in zip(axes, features):
        for level in GAS_ORDER:
            sub = df[df["gas_level"] == level]
            if sub.empty:
                continue
            ax.scatter(
                sub["fe_thickness"],
                sub[feature],
                s=34,
                alpha=0.8,
                color=GAS_COLORS[level],
                label=level if feature == features[0] else None,
                edgecolors="white",
                linewidths=0.4,
            )
        valid = df[["fe_thickness", feature]].dropna()
        add_trend_line(ax, valid["fe_thickness"].to_numpy(float), valid[feature].to_numpy(float))
        ax.set_title(f"Fe Thickness vs {LABELS[feature]}", fontweight="bold")
        ax.set_xlabel(LABELS["fe_thickness"])
        ax.set_ylabel(LABELS[feature])

    axes[0].legend(frameon=False, title="Gas level", loc="best")
    fig.suptitle("ZZY Fe Thickness Relationship Panel", fontsize=19, y=0.99, fontweight="bold")
    fig.text(
        0.5,
        0.015,
        "Gas tiers are shown as colors; black lines show overall linear trends.",
        ha="center",
        fontsize=10,
        color="#64748B",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    return save_fig(fig, out_dir, "02_zzy_fe_thickness_panel.png")


def make_fe_power_panel(df: pd.DataFrame, out_dir: Path) -> Path:
    features = ["dk_bend_index", "diameter", "alignment", "junction_ratio"]
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 9.4))
    axes = axes.ravel()
    power_levels = sorted(df["fe_power"].dropna().unique().tolist())

    for ax, feature in zip(axes, features):
        groups = [df.loc[df["fe_power"] == p, feature].dropna().to_numpy() for p in power_levels]
        ax.boxplot(
            groups,
            positions=range(1, len(power_levels) + 1),
            widths=0.54,
            patch_artist=True,
            medianprops={"color": "#0F172A", "linewidth": 1.6},
            boxprops={"linewidth": 1.0, "edgecolor": "#CBD5E1", "facecolor": "#CFFAFE"},
            whiskerprops={"color": "#94A3B8"},
            capprops={"color": "#94A3B8"},
        )
        for idx, power in enumerate(power_levels, start=1):
            values = df.loc[df["fe_power"] == power, feature].dropna().to_numpy()
            jitter = np.linspace(-0.11, 0.11, len(values)) if len(values) > 1 else np.array([0.0])
            ax.scatter(
                np.full(len(values), idx) + jitter,
                values,
                s=24,
                color="#0EA5E9",
                alpha=0.78,
                edgecolors="white",
                linewidths=0.35,
            )
        ax.set_title(f"Fe Power vs {LABELS[feature]}", fontweight="bold")
        ax.set_xticks(range(1, len(power_levels) + 1))
        ax.set_xticklabels([f"{int(p)}W" for p in power_levels])
        ax.set_xlabel(LABELS["fe_power"])
        ax.set_ylabel(LABELS[feature])

    fig.suptitle("ZZY Fe Power Comparison Panel", fontsize=19, y=0.99, fontweight="bold")
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    return save_fig(fig, out_dir, "03_zzy_fe_power_panel.png")


def make_catalyst_recipe_panel(df: pd.DataFrame, out_dir: Path) -> Path:
    grouped = (
        df.groupby(["al2o3_thickness", "fe_thickness"], as_index=False)
        .agg(
            dk_bend_index=("dk_bend_index", "mean"),
            alignment=("alignment", "mean"),
            count=("image_id", "size"),
        )
    )

    fig, axes = plt.subplots(1, 2, figsize=(14.4, 6.8), sharey=True)
    metrics = ["dk_bend_index", "alignment"]
    titles = ["Catalyst Stack vs d*k", "Catalyst Stack vs Alignment"]

    for ax, metric, title in zip(axes, metrics, titles):
        sizes = 130 + (grouped["count"] / grouped["count"].max()) * 520
        sc = ax.scatter(
            grouped["fe_thickness"],
            grouped["al2o3_thickness"],
            c=grouped[metric],
            s=sizes,
            cmap="viridis",
            alpha=0.9,
            edgecolors="white",
            linewidths=0.8,
        )
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel(LABELS["fe_thickness"])
        ax.set_ylabel(LABELS["al2o3_thickness"])
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(LABELS[metric])

    fig.suptitle("ZZY Catalyst Recipe Map", fontsize=19, y=0.99, fontweight="bold")
    fig.text(
        0.5,
        0.02,
        "Bubble size indicates sample count for each Al2O3-thickness / Fe-thickness recipe window.",
        ha="center",
        fontsize=10,
        color="#64748B",
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    return save_fig(fig, out_dir, "04_zzy_catalyst_recipe_map.png")


def make_gas_process_panel(df: pd.DataFrame, out_dir: Path) -> Path:
    features = ["density", "alignment", "dk_bend_index", "tortuosity_v2"]
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 9.4))
    axes = axes.ravel()

    for ax, feature in zip(axes, features):
        groups = []
        colors = []
        positions = []
        for idx, level in enumerate(GAS_ORDER, start=1):
            values = df.loc[df["gas_level"] == level, feature].dropna().to_numpy()
            if len(values) == 0:
                continue
            groups.append(values)
            colors.append(GAS_COLORS[level])
            positions.append(idx)
        box = ax.boxplot(
            groups,
            positions=positions,
            widths=0.54,
            patch_artist=True,
            medianprops={"color": "#0F172A", "linewidth": 1.6},
            boxprops={"linewidth": 1.0, "edgecolor": "#CBD5E1"},
            whiskerprops={"color": "#94A3B8"},
            capprops={"color": "#94A3B8"},
        )
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.24)
        ax.set_title(f"Gas Tier vs {LABELS[feature]}", fontweight="bold")
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(["Low", "Mid", "High"])
        ax.set_xlabel("Gas total-flow tier")
        ax.set_ylabel(LABELS[feature])

    fig.suptitle("ZZY Gas-Level Process Panel", fontsize=19, y=0.99, fontweight="bold")
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    return save_fig(fig, out_dir, "05_zzy_gas_level_process_panel.png")


def top_process_correlations(df: pd.DataFrame) -> List[str]:
    corr = df[PROCESS_COLS + FEATURE_COLS].corr(numeric_only=True)
    rows = []
    for proc in PROCESS_COLS:
        for feat in FEATURE_COLS:
            rows.append((abs(corr.loc[proc, feat]), corr.loc[proc, feat], proc, feat))
    rows.sort(reverse=True)
    lines = []
    for _, value, proc, feat in rows[:8]:
        lines.append(f"- `{LABELS[proc]}` vs `{LABELS[feat]}`: `{value:.3f}`")
    return lines


def write_report(out_dir: Path, source_dir: Path, df: pd.DataFrame, figures: List[Path]) -> Path:
    variable_notes = []
    constant_cols = ["actual_temp", "membrane_pos_cm", "growth_time", "anneal_temp", "anneal_time"]
    for col in constant_cols:
        unique_values = sorted(df[col].dropna().unique().tolist())
        if len(unique_values) == 1:
            variable_notes.append(f"- `{col}` 在这批数据里是常数: `{unique_values[0]}`")

    lines = [
        "# ZZY 工艺参数关系可视化报告",
        "",
        f"- 生成时间: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- 来源工程化数据: `{source_dir}`",
        f"- 样本总数: `{len(df)}`",
        "",
        "## 可分析的工艺变量",
        "",
        "- 这批真正有变化的工艺参数主要是 `fe_thickness / fe_power / al2o3_thickness / al2o3_power / gas_total`。",
        "- `gas_total` 代表你确认过的三档总流量条件，不再拆成三个独立气体变量。",
        "",
        "## 当前口径下无法单独分析的变量",
        "",
        *variable_notes,
        "",
        "## 工艺参数相关性亮点",
        "",
        *top_process_correlations(df),
        "",
        "## 图件",
        "",
    ]

    captions = {
        "01_zzy_process_feature_correlation.png": "工艺参数与形貌特征相关性矩阵",
        "02_zzy_fe_thickness_panel.png": "Fe 厚度专题图",
        "03_zzy_fe_power_panel.png": "Fe 功率专题图",
        "04_zzy_catalyst_recipe_map.png": "催化剂配方窗口地图",
        "05_zzy_gas_level_process_panel.png": "气体三档工艺关系图",
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
    out_dir = REPORTS_ROOT / f"{OUTPUT_PREFIX}{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    figures = [
        make_process_feature_corr(df, out_dir),
        make_fe_thickness_panel(df, out_dir),
        make_fe_power_panel(df, out_dir),
        make_catalyst_recipe_panel(df, out_dir),
        make_gas_process_panel(df, out_dir),
    ]
    report_path = write_report(out_dir, source_dir, df, figures)

    print(out_dir)
    print(report_path)


if __name__ == "__main__":
    main()
