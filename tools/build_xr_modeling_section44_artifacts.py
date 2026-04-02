from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


matplotlib.use("Agg")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(r"D:\CNTDATA\CNTA_ML_Project")
SUMMARY_CSV = PROJECT_ROOT / "reports" / "slice_standard_batch_20260331_005741" / "summary.csv"
DB_PATH = PROJECT_ROOT / "database" / "cnta_experiments.sqlite"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "slice_standard_batch_20260331_005741" / "data_cleaning_review" / "xr_modeling_section44"

OUTPUT_FEATURES = [
    "density",
    "alignment",
    "diameter_mean_nm",
    "l2_curvature_trimmed_mean_sqrt_length_nm",
    "dk_bend_index",
    "l2_waviness_ratio_v2",
    "junction_count",
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_merged_df() -> pd.DataFrame:
    summary = pd.read_csv(SUMMARY_CSV)
    conn = sqlite3.connect(DB_PATH)
    meta = pd.read_sql_query(
        """
        SELECT
            id AS image_id,
            sample_id AS db_sample_id,
            growth_temp,
            actual_temp,
            growth_time,
            ar_flow,
            catalyst_weight,
            magnification
        FROM images
        WHERE source = 'XR' AND COALESCE(is_deleted, 0) = 0
        """,
        conn,
    )
    conn.close()
    df = summary.merge(meta, on="image_id", how="left")
    for col in [
        "growth_temp",
        "actual_temp",
        "growth_time",
        "ar_flow",
        "catalyst_weight",
        "magnification",
        *OUTPUT_FEATURES,
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["dk_bend_index"] = (
        pd.to_numeric(df["diameter_mean_nm"], errors="coerce")
        * pd.to_numeric(df["l2_curvature_trimmed_mean_sqrt_length_nm"], errors="coerce")
    )
    return df


def subset_definitions(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "full": df.copy(),
        "800C": df[df["growth_temp"].eq(800.0)].copy(),
        "1.0g": df[df["catalyst_weight"].eq(1.0)].copy(),
    }


def corr_columns(sub: pd.DataFrame) -> list[str]:
    cols = []
    for col in ["actual_temp", "growth_temp", "catalyst_weight"]:
        if col in sub.columns and sub[col].nunique(dropna=True) > 1:
            cols.append(col)
    cols.extend(OUTPUT_FEATURES)
    return cols


def draw_heatmap(ax: plt.Axes, corr: pd.DataFrame, title: str) -> None:
    data = corr.to_numpy(dtype=float)
    im = ax.imshow(data, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index, fontsize=8)
    ax.set_title(title, fontsize=12)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            if np.isnan(value):
                continue
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7, color="black")
    return im


def save_subset_heatmaps(subsets: dict[str, pd.DataFrame], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    last_im = None
    for ax, (name, sub) in zip(axes, subsets.items()):
        cols = corr_columns(sub)
        corr = sub[cols].corr(numeric_only=True)
        last_im = draw_heatmap(ax, corr, f"{name} subset")
        corr.to_csv(output_dir / f"corr_{name}.csv", encoding="utf-8-sig")
    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
        cbar.set_label("Pearson r")
    fig.suptitle("XR Section 4.4 Subset Correlation Heatmaps", fontsize=15)
    fig.savefig(output_dir / "xr_subset_correlation_heatmaps.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_key_scatter_plots(subsets: dict[str, pd.DataFrame], output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)

    plots = [
        ("1.0g", "actual_temp", "alignment", "1.0g: temp vs alignment"),
        ("1.0g", "actual_temp", "dk_bend_index", "1.0g: temp vs d*k"),
        ("1.0g", "actual_temp", "junction_count", "1.0g: temp vs junction_count"),
        ("800C", "catalyst_weight", "alignment", "800C: catalyst vs alignment"),
        ("800C", "catalyst_weight", "dk_bend_index", "800C: catalyst vs d*k"),
        ("800C", "catalyst_weight", "junction_count", "800C: catalyst vs junction_count"),
    ]

    for ax, (subset_name, xcol, ycol, title) in zip(axes.ravel(), plots):
        sub = subsets[subset_name][[xcol, ycol]].dropna()
        ax.scatter(sub[xcol], sub[ycol], s=28, alpha=0.75, color="#2878B5", edgecolors="none")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(xcol)
        ax.set_ylabel(ycol)

    fig.suptitle("XR Section 4.4 Key Scatter Plots", fontsize=15)
    fig.savefig(output_dir / "xr_key_scatter_plots.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_baseline_table(subsets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, sub in subsets.items():
        for target in OUTPUT_FEATURES:
            work = sub[["actual_temp", "growth_temp", "catalyst_weight", target]].copy()
            work[target] = pd.to_numeric(work[target], errors="coerce")
            work = work.dropna(subset=[target])
            if len(work) < 25:
                continue

            num_cols = [c for c in ["actual_temp", "catalyst_weight"] if work[c].nunique(dropna=True) > 1]
            cat_cols = [c for c in ["growth_temp"] if work[c].nunique(dropna=True) > 1]
            if not num_cols and not cat_cols:
                continue

            transformers = []
            if num_cols:
                transformers.append(
                    (
                        "num",
                        Pipeline(
                            [
                                ("imp", SimpleImputer(strategy="median")),
                                ("sc", StandardScaler()),
                            ]
                        ),
                        num_cols,
                    )
                )
            if cat_cols:
                transformers.append(
                    (
                        "cat",
                        Pipeline(
                            [
                                ("imp", SimpleImputer(strategy="most_frequent")),
                                ("oh", OneHotEncoder(handle_unknown="ignore")),
                            ]
                        ),
                        cat_cols,
                    )
                )

            model = Pipeline(
                [
                    ("pre", ColumnTransformer(transformers)),
                    ("ridge", Ridge(alpha=1.0)),
                ]
            )
            n_splits = min(5, max(3, len(work) // 20))
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            scores = cross_validate(
                model,
                work[num_cols + cat_cols],
                work[target],
                cv=cv,
                scoring={
                    "mae": "neg_mean_absolute_error",
                    "rmse": "neg_root_mean_squared_error",
                    "r2": "r2",
                },
                error_score="raise",
            )

            rows.append(
                {
                    "subset": name,
                    "target": target,
                    "n": int(len(work)),
                    "features": ",".join(num_cols + cat_cols),
                    "mae": round(float(-scores["test_mae"].mean()), 4),
                    "rmse": round(float(-scores["test_rmse"].mean()), 4),
                    "r2": round(float(scores["test_r2"].mean()), 4),
                }
            )
    return pd.DataFrame(rows).sort_values(["subset", "target"]).reset_index(drop=True)


def save_baseline_tables(table: pd.DataFrame, output_dir: Path) -> None:
    table.to_csv(output_dir / "xr_baseline_results_all.csv", index=False, encoding="utf-8-sig")

    for subset_name in table["subset"].unique():
        sub = table[table["subset"].eq(subset_name)].copy()
        sub.to_csv(output_dir / f"xr_baseline_results_{subset_name}.csv", index=False, encoding="utf-8-sig")

    md_lines = ["# XR Baseline Results", ""]
    for subset_name in ["full", "800C", "1.0g"]:
        sub = table[table["subset"].eq(subset_name)].copy()
        if sub.empty:
            continue
        md_lines.append(f"## {subset_name}")
        md_lines.append("")
        md_lines.append("| target | n | features | MAE | RMSE | R2 |")
        md_lines.append("| --- | ---: | --- | ---: | ---: | ---: |")
        for _, row in sub.iterrows():
            md_lines.append(
                f"| {row['target']} | {int(row['n'])} | {row['features']} | {row['mae']:.4f} | {row['rmse']:.4f} | {row['r2']:.4f} |"
            )
        md_lines.append("")
    (output_dir / "xr_baseline_results.md").write_text("\n".join(md_lines), encoding="utf-8")


def save_summary_json(subsets: dict[str, pd.DataFrame], table: pd.DataFrame, output_dir: Path) -> None:
    corr_summary = {}
    for name, sub in subsets.items():
        cols = corr_columns(sub)
        corr = sub[cols].corr(numeric_only=True)
        dk_pairs = []
        if "dk_bend_index" in corr.index:
            for col in corr.columns:
                if col == "dk_bend_index":
                    continue
                value = corr.loc["dk_bend_index", col]
                if pd.isna(value):
                    continue
                dk_pairs.append({"feature": col, "pearson": round(float(value), 6)})
            dk_pairs.sort(key=lambda item: abs(item["pearson"]), reverse=True)
        corr_summary[name] = dk_pairs[:6]

    payload = {
        "subsets": {
            name: {
                "rows": int(len(sub)),
                "sample_ids": int(sub["sample_id"].nunique()),
                "growth_temp_levels": sorted(
                    [float(v) for v in sub["growth_temp"].dropna().unique().tolist()]
                ),
                "catalyst_weight_levels": sorted(
                    [float(v) for v in sub["catalyst_weight"].dropna().unique().tolist()]
                ),
            }
            for name, sub in subsets.items()
        },
        "best_r2_per_subset": {
            name: (
                table[table["subset"].eq(name)]
                .sort_values("r2", ascending=False)
                .head(3)[["target", "r2"]]
                .to_dict(orient="records")
            )
            for name in table["subset"].unique()
        },
        "dk_top_correlations": corr_summary,
    }
    (output_dir / "xr_modeling_section44_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    md_lines = ["# XR Correlation Refresh Summary", ""]
    md_lines.append("- `d*k` 定义: `diameter_mean_nm * l2_curvature_trimmed_mean_sqrt_length_nm`")
    md_lines.append("")
    for subset_name, items in corr_summary.items():
        md_lines.append(f"## {subset_name}")
        md_lines.append("")
        if not items:
            md_lines.append("- 无可用 d*k 相关性结果")
            md_lines.append("")
            continue
        md_lines.append("| feature | pearson |")
        md_lines.append("| --- | ---: |")
        for item in items:
            md_lines.append(f"| {item['feature']} | {item['pearson']:.6f} |")
        md_lines.append("")
    (output_dir / "xr_correlation_refresh_summary.md").write_text(
        "\n".join(md_lines),
        encoding="utf-8",
    )


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    df = load_merged_df()
    subsets = subset_definitions(df)

    save_subset_heatmaps(subsets, OUTPUT_DIR)
    save_key_scatter_plots(subsets, OUTPUT_DIR)
    baseline_table = build_baseline_table(subsets)
    save_baseline_tables(baseline_table, OUTPUT_DIR)
    save_summary_json(subsets, baseline_table, OUTPUT_DIR)

    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
