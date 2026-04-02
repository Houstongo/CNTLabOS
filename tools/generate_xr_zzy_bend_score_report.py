from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sqlite3
from matplotlib.colors import LinearSegmentedColormap


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_ROOT = PROJECT_ROOT / "reports"
DB_PATH = PROJECT_ROOT / "database" / "cnta_experiments.sqlite"

XR_PREFIX = "slice_standard_batch_"
ZZY_PREFIX = "zzy_feature_engineering_gt10000_"
OUTPUT_PREFIX = "xr_zzy_bend_score_report_"

FEATURES_HIGH = [
    "dk_bend_index",
    "curvature_proxy",
    "waviness_proxy",
    "tortuosity_proxy",
    "junction_ratio",
]
FEATURE_LOW = ["alignment"]
COMPONENT_COLUMNS = FEATURES_HIGH + ["alignment_inverse"]
DATASET_COLORS = {"XR": "#2563EB", "ZZY": "#F97316"}

CMAP = LinearSegmentedColormap.from_list(
    "bend_components",
    ["#F8FAFC", "#7C3AED", "#0F766E"],
)


def latest_xr_summary() -> Path:
    candidates = []
    for directory in REPORTS_ROOT.iterdir():
        summary = directory / "summary.csv"
        if directory.is_dir() and directory.name.startswith(XR_PREFIX) and summary.exists():
            candidates.append(summary)
    if not candidates:
        raise FileNotFoundError("No XR summary.csv found.")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def latest_zzy_engineering() -> Path:
    candidates = []
    for directory in REPORTS_ROOT.iterdir():
        dataset = directory / "engineered_dataset_active.csv"
        if directory.is_dir() and directory.name.startswith(ZZY_PREFIX) and dataset.exists():
            candidates.append(dataset)
    if not candidates:
        raise FileNotFoundError("No ZZY engineered_dataset_active.csv found.")
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


def load_xr(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["status"].astype(str).str.lower().eq("success")].copy()
    conn = sqlite3.connect(DB_PATH)
    meta = pd.read_sql_query(
        """
        SELECT id AS image_id, actual_temp, ar_flow, catalyst_weight
        FROM images
        WHERE source='XR' AND COALESCE(is_deleted,0)=0
        """,
        conn,
    )
    conn.close()
    df = df.merge(meta, on="image_id", how="left")
    for col in [
        "image_id",
        "magnification",
        "density",
        "alignment",
        "diameter_mean_nm",
        "l2_curvature_trimmed_mean_sqrt_length_nm",
        "l2_waviness_ratio_v2",
        "l2_tortuosity_v2",
        "junction_ratio",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    out = pd.DataFrame(
        {
            "dataset": "XR",
            "image_id": df["image_id"],
            "sample_key": df["sample_id"].fillna(df["file_name"]),
            "sample_group": df["sample_id"].astype(str).str.split("_").str[0],
            "file_name": df["file_name"],
            "file_path": df["file_path"],
            "magnification": df["magnification"],
            "density": df["density"],
            "alignment": df["alignment"],
            "dk_bend_index": df["diameter_mean_nm"] * df["l2_curvature_trimmed_mean_sqrt_length_nm"],
            "curvature_proxy": df["l2_curvature_trimmed_mean_sqrt_length_nm"],
            "waviness_proxy": df["l2_waviness_ratio_v2"],
            "tortuosity_proxy": df["l2_tortuosity_v2"],
            "junction_ratio": df["junction_ratio"],
            "process_hint": (
                "T="
                + df["actual_temp"].round(1).astype(str)
                + ", Flow="
                + df["ar_flow"].astype(str)
                + ", Cat="
                + df["catalyst_weight"].astype(str)
            ),
        }
    )
    return out


def load_zzy(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    conn = sqlite3.connect(DB_PATH)
    meta = pd.read_sql_query(
        """
        SELECT id AS image_id, fe_thickness
        FROM images
        WHERE source='ZZY' AND COALESCE(is_deleted,0)=0 AND magnification>10000
        """,
        conn,
    )
    conn.close()
    df = df.merge(meta, on="image_id", how="left")
    for col in [
        "image_id",
        "magnification",
        "density",
        "alignment",
        "dk_bend_index",
        "curvature_nm_v3_trimmed_mean_sqrt_length",
        "waviness_ratio_v2",
        "tortuosity_v2",
        "junction_ratio",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    out = pd.DataFrame(
        {
            "dataset": "ZZY",
            "image_id": df["image_id"],
            "sample_key": df["sample_id"].fillna(df["file_name"]),
            "sample_group": df["sample_no"],
            "file_name": df["file_name"],
            "file_path": df["file_path"],
            "magnification": df["magnification"],
            "density": df["density"],
            "alignment": df["alignment"],
            "dk_bend_index": df["dk_bend_index"],
            "curvature_proxy": df["curvature_nm_v3_trimmed_mean_sqrt_length"],
            "waviness_proxy": df["waviness_ratio_v2"],
            "tortuosity_proxy": df["tortuosity_v2"],
            "junction_ratio": df["junction_ratio"],
            "process_hint": (
                "Gas="
                + df["gas_condition"].astype(str)
                + ", Fe="
                + df["fe_thickness"].astype(str)
            ),
        }
    )
    return out


def percentile_score(series: pd.Series, ascending: bool = True) -> pd.Series:
    valid = pd.to_numeric(series, errors="coerce")
    ranked = valid.rank(method="average", pct=True, ascending=ascending)
    return ranked


def add_bend_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for dataset in out["dataset"].dropna().unique():
        mask = out["dataset"].eq(dataset)
        for col in FEATURES_HIGH:
            out.loc[mask, f"{col}_score"] = percentile_score(out.loc[mask, col], ascending=True)
        out.loc[mask, "alignment_inverse"] = 1.0 - percentile_score(
            out.loc[mask, "alignment"], ascending=True
        )

    score_cols = [f"{col}_score" for col in FEATURES_HIGH] + ["alignment_inverse"]
    out["bend_score"] = out[score_cols].mean(axis=1, skipna=True)
    out["bend_rank_dataset"] = out.groupby("dataset")["bend_score"].rank(
        method="min", ascending=False
    )
    out["bend_rank_overall"] = out["bend_score"].rank(method="min", ascending=False)
    return out


def top_rows(df: pd.DataFrame, dataset: str | None = None, n: int = 15) -> pd.DataFrame:
    sub = df if dataset is None else df[df["dataset"].eq(dataset)]
    cols = [
        "dataset",
        "bend_rank_overall",
        "bend_rank_dataset",
        "bend_score",
        "sample_key",
        "sample_group",
        "magnification",
        "dk_bend_index",
        "curvature_proxy",
        "waviness_proxy",
        "tortuosity_proxy",
        "junction_ratio",
        "alignment",
        "process_hint",
        "file_name",
    ]
    return sub.sort_values("bend_score", ascending=False)[cols].head(n).reset_index(drop=True)


def write_csvs(df: pd.DataFrame, out_dir: Path) -> None:
    df.sort_values("bend_score", ascending=False).to_csv(
        out_dir / "combined_bend_score_ranking.csv", index=False, encoding="utf-8-sig"
    )
    for dataset in ["XR", "ZZY"]:
        df[df["dataset"].eq(dataset)].sort_values("bend_score", ascending=False).to_csv(
            out_dir / f"{dataset.lower()}_bend_score_ranking.csv",
            index=False,
            encoding="utf-8-sig",
        )
    pd.concat(
        [top_rows(df, "XR", 20), top_rows(df, "ZZY", 20), top_rows(df, None, 20)],
        keys=["XR_top20", "ZZY_top20", "combined_top20"],
        names=["table", "row"],
    ).reset_index(level=0).to_csv(
        out_dir / "bend_score_top_tables.csv",
        index=False,
        encoding="utf-8-sig",
    )


def make_distribution_plot(df: pd.DataFrame, out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.8), constrained_layout=True)
    for ax, dataset in zip(axes, ["XR", "ZZY"]):
        sub = df[df["dataset"].eq(dataset)]
        ax.hist(
            sub["bend_score"].dropna(),
            bins=18,
            color=DATASET_COLORS[dataset],
            alpha=0.82,
            edgecolor="white",
            linewidth=0.8,
        )
        ax.axvline(sub["bend_score"].median(), color="#0F172A", linestyle="--", linewidth=1.8)
        ax.set_title(f"{dataset} Bend Score Distribution", fontweight="bold")
        ax.set_xlabel("Bend score")
        ax.set_ylabel("Count")
    fig.suptitle("XR / ZZY Bend Score Distributions", fontsize=18, y=1.02, fontweight="bold")
    return save_fig(fig, out_dir, "01_bend_score_distribution.png")


def add_trend_line(ax: plt.Axes, x: np.ndarray, y: np.ndarray, color: str) -> None:
    if len(x) < 2:
        return
    coef = np.polyfit(x, y, deg=1)
    xs = np.linspace(float(np.min(x)), float(np.max(x)), 80)
    ys = np.polyval(coef, xs)
    ax.plot(xs, ys, color=color, linewidth=2.0)


def make_score_vs_dk_plot(df: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8.8, 6.6))
    for dataset in ["XR", "ZZY"]:
        sub = df[df["dataset"].eq(dataset)].dropna(subset=["dk_bend_index", "bend_score"])
        ax.scatter(
            sub["dk_bend_index"],
            sub["bend_score"],
            s=34,
            alpha=0.8,
            color=DATASET_COLORS[dataset],
            label=dataset,
            edgecolors="white",
            linewidths=0.35,
        )
        add_trend_line(
            ax,
            sub["dk_bend_index"].to_numpy(dtype=float),
            sub["bend_score"].to_numpy(dtype=float),
            DATASET_COLORS[dataset],
        )
    ax.set_title("Bend Score vs d*k", fontweight="bold")
    ax.set_xlabel("d*k bend index")
    ax.set_ylabel("Unified bend score")
    ax.legend(frameon=False)
    return save_fig(fig, out_dir, "02_bend_score_vs_dk.png")


def make_top_bar_plot(df: pd.DataFrame, out_dir: Path, n: int = 20) -> Path:
    top = df.sort_values("bend_score", ascending=False).head(n).copy()
    top = top.iloc[::-1]
    labels = [f"{row.dataset}:{row.sample_key}" for row in top.itertuples()]
    colors = [DATASET_COLORS[row.dataset] for row in top.itertuples()]

    fig, ax = plt.subplots(figsize=(12.8, 8.4))
    ax.barh(labels, top["bend_score"], color=colors, alpha=0.86)
    ax.set_title(f"Top {n} Highest-Bend Samples", fontweight="bold")
    ax.set_xlabel("Bend score")
    ax.set_ylabel("Sample")
    return save_fig(fig, out_dir, "03_top_bend_score_samples.png")


def make_component_heatmap(df: pd.DataFrame, out_dir: Path, n: int = 20) -> Path:
    top = df.sort_values("bend_score", ascending=False).head(n).copy()
    labels = [f"{row.dataset}:{row.sample_key}" for row in top.itertuples()]
    component_cols = [
        "dk_bend_index_score",
        "curvature_proxy_score",
        "waviness_proxy_score",
        "tortuosity_proxy_score",
        "junction_ratio_score",
        "alignment_inverse",
    ]
    pretty = ["d*k", "curvature", "waviness", "tortuosity", "junction", "1-alignment"]
    data = top[component_cols].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(10.8, 8.8))
    im = ax.imshow(data, cmap=CMAP, vmin=0, vmax=1)
    ax.set_title("Top Bend Samples: Component Score Heatmap", fontweight="bold", pad=14)
    ax.set_xticks(range(len(pretty)))
    ax.set_xticklabels(pretty, rotation=20, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=7.5, color="#0F172A")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Percentile score")
    return save_fig(fig, out_dir, "04_component_score_heatmap.png")


def strongest_per_dataset(df: pd.DataFrame) -> Dict[str, List[str]]:
    result: Dict[str, List[str]] = {}
    for dataset in ["XR", "ZZY"]:
        sub = df[df["dataset"].eq(dataset)]
        corr = sub[
            ["bend_score", "dk_bend_index", "curvature_proxy", "waviness_proxy", "tortuosity_proxy", "junction_ratio", "alignment"]
        ].corr(numeric_only=True)
        lines = []
        for col in ["dk_bend_index", "curvature_proxy", "waviness_proxy", "tortuosity_proxy", "junction_ratio", "alignment"]:
            value = corr.loc["bend_score", col]
            lines.append(f"- `{col}`: `{value:.3f}`")
        result[dataset] = lines
    return result


def write_report(out_dir: Path, xr_source: Path, zzy_source: Path, df: pd.DataFrame, figures: Iterable[Path]) -> Path:
    xr_top = top_rows(df, "XR", 10)
    zzy_top = top_rows(df, "ZZY", 10)
    combined_top = top_rows(df, None, 15)
    strength = strongest_per_dataset(df)

    lines = [
        "# XR + ZZY Unified Bend Score Report",
        "",
        f"- 生成时间: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- XR 数据源: `{xr_source}`",
        f"- ZZY 数据源: `{zzy_source}`",
        f"- XR 样本数: `{int(df['dataset'].eq('XR').sum())}`",
        f"- ZZY 样本数: `{int(df['dataset'].eq('ZZY').sum())}`",
        "",
        "## 评分定义",
        "",
        "- 统一弯曲评分是数据集内百分位得分，不是跨流程的绝对物理量。",
        "- 组件包括: `高 d*k + 高 curvature + 高 waviness + 高 tortuosity + 高 junction_ratio + 低 alignment`。",
        "- 也就是说，这个分数更适合回答“在 XR 或 ZZY 自己体系里，谁最弯”。",
        "",
        "## 组件相关性",
        "",
        "### XR",
        "",
        *strength["XR"],
        "",
        "### ZZY",
        "",
        *strength["ZZY"],
        "",
        "## Top XR",
        "",
        "| rank | bend_score | sample | magnification | d*k | process |",
        "| --- | ---: | --- | ---: | ---: | --- |",
    ]

    for row in xr_top.itertuples():
        lines.append(
            f"| {int(row.bend_rank_dataset)} | {row.bend_score:.4f} | {row.sample_key} | {int(row.magnification) if pd.notna(row.magnification) else ''} | {row.dk_bend_index:.4f} | {row.process_hint} |"
        )

    lines += [
        "",
        "## Top ZZY",
        "",
        "| rank | bend_score | sample | magnification | d*k | process |",
        "| --- | ---: | --- | ---: | ---: | --- |",
    ]
    for row in zzy_top.itertuples():
        lines.append(
            f"| {int(row.bend_rank_dataset)} | {row.bend_score:.4f} | {row.sample_key} | {int(row.magnification) if pd.notna(row.magnification) else ''} | {row.dk_bend_index:.4f} | {row.process_hint} |"
        )

    lines += [
        "",
        "## Combined Top Samples",
        "",
        "| overall_rank | dataset | bend_score | sample | d*k |",
        "| --- | --- | ---: | --- | ---: |",
    ]
    for row in combined_top.itertuples():
        lines.append(
            f"| {int(row.bend_rank_overall)} | {row.dataset} | {row.bend_score:.4f} | {row.sample_key} | {row.dk_bend_index:.4f} |"
        )

    lines += ["", "## 图件", ""]
    captions = {
        "01_bend_score_distribution.png": "XR / ZZY bend score 分布",
        "02_bend_score_vs_dk.png": "bend score 与 d*k 的关系",
        "03_top_bend_score_samples.png": "最高弯曲样本排名图",
        "04_component_score_heatmap.png": "Top 样本组件得分热图",
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
    xr_source = latest_xr_summary()
    zzy_source = latest_zzy_engineering()
    xr = load_xr(xr_source)
    zzy = load_zzy(zzy_source)
    combined = add_bend_scores(pd.concat([xr, zzy], ignore_index=True))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = REPORTS_ROOT / f"{OUTPUT_PREFIX}{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    write_csvs(combined, out_dir)
    figures = [
        make_distribution_plot(combined, out_dir),
        make_score_vs_dk_plot(combined, out_dir),
        make_top_bar_plot(combined, out_dir),
        make_component_heatmap(combined, out_dir),
    ]
    report_path = write_report(out_dir, xr_source, zzy_source, combined, figures)
    print(out_dir)
    print(report_path)


if __name__ == "__main__":
    main()
