from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


matplotlib.use("Agg")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(r"D:\CNTDATA\CNTA_ML_Project")
DB_PATH = PROJECT_ROOT / "database" / "cnta_experiments.sqlite"
MODEL_BASE_DIR = PROJECT_ROOT / "reports" / "zzy_50000_fe_time_model_20260402"
CONTROLLED_CSV = MODEL_BASE_DIR / "zzy_50000_fe_time_modeling_table.csv"
NOISEAWARE_BEST = MODEL_BASE_DIR / "best_l_target_specialized_noiseaware_experiment" / "best_results.csv"
CV_SUMMARY_JSON = MODEL_BASE_DIR / "noiseaware_cv_detail_all_targets" / "all_targets_cv_summary.json"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "paper_section_4_4_data_bundle_20260402"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def add_gas_level(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    def gas_level(row: pd.Series) -> str:
        if row["ar_flow"] == 600 and row["h2_flow"] == 300 and row["c2h4_flow"] == 150:
            return "high"
        if row["ar_flow"] == 400 and row["h2_flow"] == 200 and row["c2h4_flow"] == 100:
            return "mid"
        if row["ar_flow"] == 200 and row["h2_flow"] == 100 and row["c2h4_flow"] == 50:
            return "low"
        return "other"

    out["gas_level"] = out.apply(gas_level, axis=1)
    return out


def save_spearman_heatmap(df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
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

    fig, ax = plt.subplots(figsize=(8.2, 6.4), constrained_layout=True)
    im = ax.imshow(corr.values, cmap="YlGnBu", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(cols)
    ax.set_title("Spearman Correlation: Process vs Morphology")
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.9, label="Spearman rho")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return corr


def save_anneal_trend_plot(df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    metrics = ["alignment", "curvature", "waviness_ratio", "tortuosity"]
    trend = (
        df.groupby("anneal_time")[metrics]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    trend.columns = [
        "anneal_time"
    ] + [
        f"{metric}_{stat}"
        for metric in metrics
        for stat in ["mean", "std", "count"]
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.8), constrained_layout=True)
    axes = axes.flatten()
    x = sorted(df["anneal_time"].dropna().unique())
    label_map = {0.25: "15 min", 0.5: "30 min", 0.75: "45 min"}
    display_names = {
        "alignment": "取向度",
        "curvature": "有效平均曲率 / μm^-1",
        "waviness_ratio": "波曲度",
        "tortuosity": "迂曲度",
    }

    for ax, metric in zip(axes, metrics):
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
        colors = ["#b8d8be", "#8fb3c9", "#d9b382"]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.9)
        ax.set_xticks(range(1, len(x) + 1))
        ax.set_xticklabels([label_map.get(v, str(v)) for v in x])
        ax.set_xlabel("退火时间 / min")
        ax.set_ylabel(display_names[metric])
        ax.set_title(display_names[metric], fontsize=11)
        ax.grid(axis="y", alpha=0.2, linestyle="--")
    fig.suptitle("退火时间对关键形貌指标分布的影响", fontsize=14)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return trend


def save_response_heatmap(df: pd.DataFrame, metric: str, out_path: Path) -> pd.DataFrame:
    pivot = df.pivot_table(index="fe_thickness", columns="fe_power", values=metric, aggfunc="mean")
    pivot = pivot.sort_index().sort_index(axis=1)

    fig, ax = plt.subplots(figsize=(7.2, 5.4), constrained_layout=True)
    im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(v) for v in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(v) for v in pivot.index])
    ax.set_xlabel("Fe power (W)")
    ax.set_ylabel("Fe thickness (nm)")
    ax.set_title(f"Mean {metric} by Fe power and thickness")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.9, label=metric)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return pivot


def fetch_gas_level_summary() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    query = """
    SELECT
      CASE
        WHEN ar_flow=600 AND h2_flow=300 AND c2h4_flow=150 THEN 'high'
        WHEN ar_flow=400 AND h2_flow=200 AND c2h4_flow=100 THEN 'mid'
        WHEN ar_flow=200 AND h2_flow=100 AND c2h4_flow=50 THEN 'low'
        ELSE 'other'
      END as gas_level,
      COUNT(*) as n,
      AVG(curvature) as curvature_mean,
      AVG(waviness_ratio) as waviness_mean,
      AVG(tortuosity) as tortuosity_mean,
      AVG(alignment) as alignment_mean,
      AVG(density) as density_mean
    FROM images
    WHERE source='ZZY' AND COALESCE(is_deleted,0)=0 AND magnification=50000
      AND curvature IS NOT NULL AND alignment IS NOT NULL AND tortuosity IS NOT NULL
    GROUP BY gas_level
    ORDER BY CASE gas_level WHEN 'low' THEN 1 WHEN 'mid' THEN 2 WHEN 'high' THEN 3 ELSE 4 END
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def fetch_gas_gain_table() -> pd.DataFrame:
    # Reuse already established values from a direct model comparison by recomputing quickly.
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import r2_score
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    conn = sqlite3.connect(DB_PATH)
    query = """
    SELECT id as image_id, sample_id,
           CASE
             WHEN sample_id LIKE 'No49_%' THEN 'No49'
             ELSE substr(sample_id,1,4)
           END as sample_prefix,
           COALESCE(position_label,'') as position_label,
           repeat_id,
           fe_power, fe_thickness, anneal_time,
           ar_flow, h2_flow, c2h4_flow,
           curvature, waviness_ratio, tortuosity, alignment
    FROM images
    WHERE source='ZZY' AND COALESCE(is_deleted,0)=0 AND magnification=50000
      AND curvature IS NOT NULL AND waviness_ratio IS NOT NULL AND tortuosity IS NOT NULL AND alignment IS NOT NULL
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    df = add_gas_level(df)
    df = df[df["gas_level"].isin(["high", "mid", "low"])].copy()

    def make_group(row: pd.Series) -> str:
        pos = row["position_label"] if str(row["position_label"]).strip() else "na"
        rep = int(row["repeat_id"]) if pd.notna(row["repeat_id"]) else -1
        return f"{row['sample_prefix']}|{row['fe_power']}|{row['fe_thickness']}|{row['anneal_time']}|{pos}|{rep}"

    df["group_key"] = df.apply(make_group, axis=1)

    rows = []
    for target in ["curvature", "waviness_ratio", "tortuosity", "alignment"]:
        tdf = df.dropna(subset=[target]).copy()
        y = tdf[target].to_numpy(float)
        groups = tdf["group_key"].astype(str)
        cv = GroupKFold(n_splits=min(5, groups.nunique()))

        def build_pipe(include_gas: bool):
            num_cols = ["fe_power", "fe_thickness", "anneal_time"]
            cat_cols = ["gas_level"] if include_gas else []
            pre = ColumnTransformer(
                [
                    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
                    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
                ]
            )
            return Pipeline([("pre", pre), ("model", ExtraTreesRegressor(n_estimators=600, min_samples_leaf=2, random_state=42, n_jobs=1))]), num_cols + cat_cols

        base_pipe, base_cols = build_pipe(False)
        gas_pipe, gas_cols = build_pipe(True)
        base_pred = cross_val_predict(base_pipe, tdf[base_cols], y, cv=cv, groups=groups)
        gas_pred = cross_val_predict(gas_pipe, tdf[gas_cols], y, cv=cv, groups=groups)
        rows.append(
            {
                "target": target,
                "r2_without_gas": float(r2_score(y, base_pred)),
                "r2_with_gas": float(r2_score(y, gas_pred)),
                "r2_gain": float(r2_score(y, gas_pred) - r2_score(y, base_pred)),
            }
        )
    return pd.DataFrame(rows)


def flatten_cv_summary(cv_json: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall_rows = []
    fold_rows = []
    for target, payload in cv_json.items():
        overall = payload["overall"].copy()
        overall["target"] = target
        overall_rows.append(overall)
        for fold in payload["folds"]:
            row = fold.copy()
            row["target"] = target
            fold_rows.append(row)
    return pd.DataFrame(overall_rows), pd.DataFrame(fold_rows)


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    controlled = pd.read_csv(CONTROLLED_CSV)
    controlled.to_csv(OUTPUT_DIR / "controlled_subset_raw.csv", index=False, encoding="utf-8-sig")

    corr = save_spearman_heatmap(controlled, OUTPUT_DIR / "spearman_correlation_heatmap.png")
    corr.to_csv(OUTPUT_DIR / "spearman_correlation_matrix.csv", encoding="utf-8-sig")

    anneal_trend = save_anneal_trend_plot(controlled, OUTPUT_DIR / "anneal_time_trend.png")
    anneal_trend.to_csv(OUTPUT_DIR / "anneal_time_trend_summary.csv", index=False, encoding="utf-8-sig")

    curv_heat = save_response_heatmap(controlled, "curvature", OUTPUT_DIR / "fe_power_thickness_curvature_heatmap.png")
    curv_heat.to_csv(OUTPUT_DIR / "fe_power_thickness_curvature_mean.csv", encoding="utf-8-sig")

    align_heat = save_response_heatmap(controlled, "alignment", OUTPUT_DIR / "fe_power_thickness_alignment_heatmap.png")
    align_heat.to_csv(OUTPUT_DIR / "fe_power_thickness_alignment_mean.csv", encoding="utf-8-sig")

    gas_summary = fetch_gas_level_summary()
    gas_summary.to_csv(OUTPUT_DIR / "gas_level_summary_50000x.csv", index=False, encoding="utf-8-sig")

    gas_gain = fetch_gas_gain_table()
    gas_gain.to_csv(OUTPUT_DIR / "gas_level_model_gain_50000x.csv", index=False, encoding="utf-8-sig")

    model_best = pd.read_csv(NOISEAWARE_BEST)
    model_best.to_csv(OUTPUT_DIR / "final_model_best_results.csv", index=False, encoding="utf-8-sig")

    cv_json = json.loads(CV_SUMMARY_JSON.read_text(encoding="utf-8"))
    cv_overall, cv_folds = flatten_cv_summary(cv_json)
    cv_overall.to_csv(OUTPUT_DIR / "cv_overall_summary.csv", index=False, encoding="utf-8-sig")
    cv_folds.to_csv(OUTPUT_DIR / "cv_fold_summary.csv", index=False, encoding="utf-8-sig")

    summary = {
        "controlled_subset_n": int(len(controlled)),
        "controlled_subset_path": str(CONTROLLED_CSV),
        "output_dir": str(OUTPUT_DIR),
        "files": [
            "controlled_subset_raw.csv",
            "spearman_correlation_matrix.csv",
            "spearman_correlation_heatmap.png",
            "anneal_time_trend_summary.csv",
            "anneal_time_trend.png",
            "fe_power_thickness_curvature_mean.csv",
            "fe_power_thickness_curvature_heatmap.png",
            "fe_power_thickness_alignment_mean.csv",
            "fe_power_thickness_alignment_heatmap.png",
            "gas_level_summary_50000x.csv",
            "gas_level_model_gain_50000x.csv",
            "final_model_best_results.csv",
            "cv_overall_summary.csv",
            "cv_fold_summary.csv",
        ],
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Section 4.4 Data Bundle",
        "",
        f"- controlled subset rows: {len(controlled)}",
        "- core morphology targets: alignment, curvature, waviness_ratio, tortuosity",
        "- process variables: fe_power, fe_thickness, anneal_time",
        "",
        "## Included files",
    ]
    for file_name in summary["files"]:
        lines.append(f"- {file_name}")
    (OUTPUT_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
