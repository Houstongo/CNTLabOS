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
OUTPUT_PREFIX = "zzy_50000_bend_process_report_"

PROCESS_COLS = [
    "fe_thickness",
    "fe_power",
    "al2o3_thickness",
    "al2o3_power",
    "gas_total",
]

BEND_FEATURE_COLS = [
    "dk_bend_index",
    "curvature_nm_v3_trimmed_mean_sqrt_length",
    "curvature_nm_v3",
    "tortuosity_v2",
    "waviness_ratio_v2",
    "alignment",
    "junction_ratio",
    "junctions_per_100um",
    "density",
    "diameter",
]

GAS_ORDER = ["low", "mid", "high"]
GAS_COLORS = {
    "low": "#2563EB",
    "mid": "#F97316",
    "high": "#059669",
}

CMAP = LinearSegmentedColormap.from_list(
    "zzy_50000_corr",
    ["#8B1E3F", "#F8FAFC", "#0F766E"],
)

LABELS = {
    "fe_thickness": "Fe Thickness (nm)",
    "fe_power": "Fe Power (W)",
    "al2o3_thickness": "Al2O3 Thickness (nm)",
    "al2o3_power": "Al2O3 Power (W)",
    "gas_total": "Gas Total Flow",
    "dk_bend_index": "d*k Bend Index",
    "curvature_nm_v3_trimmed_mean_sqrt_length": "Curvature Proxy (nm^-1)",
    "curvature_nm_v3": "Curvature Raw (nm^-1)",
    "tortuosity_v2": "Tortuosity v2",
    "waviness_ratio_v2": "Waviness Ratio v2",
    "alignment": "Alignment",
    "junction_ratio": "Junction Ratio",
    "junctions_per_100um": "Junctions / 100um",
    "density": "Density (%)",
    "diameter": "Diameter (nm)",
}


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
    eng = eng[eng["magnification"].eq(50000)].copy()

    conn = sqlite3.connect(DB_PATH)
    proc = pd.read_sql_query(
        """
        SELECT
            id AS image_id,
            fe_thickness,
            fe_power,
            al2o3_thickness,
            al2o3_power,
            c2h4_flow,
            ar_flow,
            h2_flow,
            actual_temp,
            membrane_pos_cm,
            growth_time,
            anneal_temp,
            anneal_time
        FROM images
        WHERE source='ZZY' AND COALESCE(is_deleted,0)=0 AND magnification=50000
        """,
        conn,
    )
    conn.close()

    df = eng.merge(proc, on="image_id", how="left", suffixes=("", "_db"))
    df["gas_total"] = df["c2h4_flow_db"].fillna(df["c2h4_flow"]) + df["ar_flow_db"].fillna(df["ar_flow"]) + df["h2_flow_db"].fillna(df["h2_flow"])
    df["gas_level"] = pd.Categorical(df["gas_level"], categories=GAS_ORDER, ordered=True)
    return df


def annotate_corr(ax: plt.Axes, corr: pd.DataFrame) -> None:
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            value = corr.iloc[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8.3, color="#0F172A")


def make_corr_heatmap(df: pd.DataFrame, cols: Iterable[str], title: str, subtitle: str, out_dir: Path, filename: str) -> Path:
    corr = df[list(cols)].corr(numeric_only=True)
    labels = [LABELS.get(col, col) for col in corr.columns]
    fig, ax = plt.subplots(figsize=(10.8, 8.8))
    im = ax.imshow(corr.values, cmap=CMAP, vmin=-1, vmax=1)
    ax.set_title(title, pad=16, fontweight="bold")
    ax.text(0.0, 1.03, subtitle, transform=ax.transAxes, fontsize=10, color="#64748B")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=28, ha="right")
    ax.set_yticklabels(labels)
    annotate_corr(ax, corr)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson r")
    corr.to_csv(out_dir / filename.replace(".png", ".csv"), encoding="utf-8-sig")
    return save_fig(fig, out_dir, filename)


def add_trend_line(ax: plt.Axes, x: np.ndarray, y: np.ndarray, color: str = "#0F172A") -> None:
    if len(x) < 2:
        return
    coef = np.polyfit(x, y, deg=1)
    xs = np.linspace(float(np.min(x)), float(np.max(x)), 80)
    ys = np.polyval(coef, xs)
    ax.plot(xs, ys, color=color, linewidth=2.0, alpha=0.92)


def make_fe_thickness_panel(df: pd.DataFrame, out_dir: Path) -> Path:
    features = ["dk_bend_index", "curvature_nm_v3_trimmed_mean_sqrt_length", "tortuosity_v2", "alignment"]
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
                linewidths=0.35,
            )
        valid = df[["fe_thickness", feature]].dropna()
        add_trend_line(ax, valid["fe_thickness"].to_numpy(float), valid[feature].to_numpy(float))
        ax.set_title(f"Fe Thickness vs {LABELS[feature]}", fontweight="bold")
        ax.set_xlabel(LABELS["fe_thickness"])
        ax.set_ylabel(LABELS[feature])

    axes[0].legend(frameon=False, title="Gas level", loc="best")
    fig.suptitle("ZZY 50000X Fe Thickness Panel", fontsize=19, y=0.99, fontweight="bold")
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    return save_fig(fig, out_dir, "03_zzy50000_fe_thickness_panel.png")


def make_fe_power_panel(df: pd.DataFrame, out_dir: Path) -> Path:
    features = ["dk_bend_index", "curvature_nm_v3_trimmed_mean_sqrt_length", "junction_ratio", "alignment"]
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 9.2))
    axes = axes.ravel()
    levels = sorted(df["fe_power"].dropna().unique().tolist())

    for ax, feature in zip(axes, features):
        groups = [df.loc[df["fe_power"] == lvl, feature].dropna().to_numpy() for lvl in levels]
        ax.boxplot(
            groups,
            positions=range(1, len(levels) + 1),
            widths=0.54,
            patch_artist=True,
            medianprops={"color": "#0F172A", "linewidth": 1.6},
            boxprops={"linewidth": 1.0, "edgecolor": "#CBD5E1", "facecolor": "#CFFAFE"},
            whiskerprops={"color": "#94A3B8"},
            capprops={"color": "#94A3B8"},
        )
        for idx, lvl in enumerate(levels, start=1):
            values = df.loc[df["fe_power"] == lvl, feature].dropna().to_numpy()
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
        ax.set_xticks(range(1, len(levels) + 1))
        ax.set_xticklabels([f"{int(v)}W" for v in levels])
        ax.set_xlabel(LABELS["fe_power"])
        ax.set_ylabel(LABELS[feature])

    fig.suptitle("ZZY 50000X Fe Power Panel", fontsize=19, y=0.99, fontweight="bold")
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    return save_fig(fig, out_dir, "04_zzy50000_fe_power_panel.png")


def make_gas_level_panel(df: pd.DataFrame, out_dir: Path) -> Path:
    features = ["dk_bend_index", "curvature_nm_v3_trimmed_mean_sqrt_length", "tortuosity_v2", "junction_ratio"]
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 9.2))
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

    fig.suptitle("ZZY 50000X Gas-Level Panel", fontsize=19, y=0.99, fontweight="bold")
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    return save_fig(fig, out_dir, "05_zzy50000_gas_level_panel.png")


def make_recipe_map(df: pd.DataFrame, out_dir: Path) -> Path:
    grouped = (
        df.groupby(["al2o3_thickness", "fe_thickness"], as_index=False)
        .agg(
            dk_bend_index=("dk_bend_index", "mean"),
            curvature_nm_v3_trimmed_mean_sqrt_length=("curvature_nm_v3_trimmed_mean_sqrt_length", "mean"),
            count=("image_id", "size"),
        )
    )
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 6.8), sharey=True)
    metrics = [
        ("dk_bend_index", "Catalyst Recipe vs d*k"),
        ("curvature_nm_v3_trimmed_mean_sqrt_length", "Catalyst Recipe vs Curvature"),
    ]
    for ax, (metric, title) in zip(axes, metrics):
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
    fig.suptitle("ZZY 50000X Catalyst Recipe Map", fontsize=19, y=0.99, fontweight="bold")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return save_fig(fig, out_dir, "06_zzy50000_catalyst_recipe_map.png")


def strongest_pairs(df: pd.DataFrame, left_cols: List[str], right_cols: List[str] | None = None, top_n: int = 10) -> List[str]:
    lines: List[str] = []
    if right_cols is None:
        corr = df[left_cols].corr(numeric_only=True)
        rows = []
        for i, left in enumerate(left_cols):
            for right in left_cols[i + 1 :]:
                value = corr.loc[left, right]
                rows.append((abs(value), value, left, right))
    else:
        corr = df[left_cols + right_cols].corr(numeric_only=True)
        rows = []
        for left in left_cols:
            for right in right_cols:
                value = corr.loc[left, right]
                rows.append((abs(value), value, left, right))
    rows.sort(reverse=True)
    for _, value, left, right in rows[:top_n]:
        lines.append(f"- `{LABELS.get(left, left)}` vs `{LABELS.get(right, right)}`: `{value:.3f}`")
    return lines


def write_report(out_dir: Path, source_dir: Path, df: pd.DataFrame, figures: List[Path]) -> Path:
    constant_cols = ["actual_temp", "membrane_pos_cm", "growth_time", "anneal_temp", "anneal_time"]
    constant_lines = []
    for col in constant_cols:
        values = sorted(df[col].dropna().unique().tolist())
        if len(values) == 1:
            constant_lines.append(f"- `{col}` 在 50000X 子集里是常数: `{values[0]}`")

    lines = [
        "# ZZY 50000X 弯曲特征与工艺相关性分析",
        "",
        f"- 生成时间: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- 来源工程化数据: `{source_dir}`",
        f"- 数据口径: `source='ZZY' AND is_deleted=0 AND magnification=50000`",
        f"- 样本总数: `{len(df)}`",
        f"- 组别覆盖: `{sorted(df['sample_no'].dropna().unique().tolist())}`",
        "",
        "## 分析范围",
        "",
        "- 弯曲相关特征: `d*k`, `curvature`, `tortuosity`, `waviness`, `alignment`, `junction_ratio`, `junctions_per_100um`, `density`, `diameter`。",
        "- 可分析工艺参数: `fe_thickness`, `fe_power`, `al2o3_thickness`, `al2o3_power`, `gas_total`。",
        "",
        "## 本子集里无法单独分析的工艺参数",
        "",
        *constant_lines,
        "",
        "## 最强弯曲特征相关性",
        "",
        *strongest_pairs(df, BEND_FEATURE_COLS, None, top_n=12),
        "",
        "## 最强工艺-弯曲相关性",
        "",
        *strongest_pairs(df, PROCESS_COLS, BEND_FEATURE_COLS, top_n=12),
        "",
        "## 图件",
        "",
    ]
    captions = {
        "01_zzy50000_bend_feature_correlation.png": "50000X 弯曲特征相关性矩阵",
        "02_zzy50000_process_bend_correlation.png": "50000X 工艺参数 vs 弯曲特征矩阵",
        "03_zzy50000_fe_thickness_panel.png": "Fe thickness 专题图",
        "04_zzy50000_fe_power_panel.png": "Fe power 专题图",
        "05_zzy50000_gas_level_panel.png": "Gas level 专题图",
        "06_zzy50000_catalyst_recipe_map.png": "催化剂配方窗口图",
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
        make_corr_heatmap(
            df,
            BEND_FEATURE_COLS,
            "ZZY 50000X Bend-Feature Correlation",
            "Pearson correlation within the 50000X subset only",
            out_dir,
            "01_zzy50000_bend_feature_correlation.png",
        ),
        make_corr_heatmap(
            df,
            PROCESS_COLS + BEND_FEATURE_COLS,
            "ZZY 50000X Process-to-Bend Correlation",
            "Only process variables with actual variation are included",
            out_dir,
            "02_zzy50000_process_bend_correlation.png",
        ),
        make_fe_thickness_panel(df, out_dir),
        make_fe_power_panel(df, out_dir),
        make_gas_level_panel(df, out_dir),
        make_recipe_map(df, out_dir),
    ]

    report_path = write_report(out_dir, source_dir, df, figures)
    print(out_dir)
    print(report_path)


if __name__ == "__main__":
    main()
