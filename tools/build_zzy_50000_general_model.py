from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sqlite3
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


matplotlib.use("Agg")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(r"D:\CNTDATA\CNTA_ML_Project")
DB_PATH = PROJECT_ROOT / "database" / "cnta_experiments.sqlite"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "zzy_50000_general_model_20260402"

TARGETS = [
    "curvature",
    "waviness_ratio",
    "tortuosity",
    "alignment",
    "density",
    "diameter",
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def infer_gas_level(ar_flow: float | int | None, h2_flow: float | int | None, c2h4_flow: float | int | None) -> str:
    key = (int(ar_flow or 0), int(h2_flow or 0), int(c2h4_flow or 0))
    mapping = {
        (200, 100, 50): "low",
        (400, 200, 100): "mid",
        (600, 300, 150): "high",
    }
    return mapping.get(key, f"{key[0]}/{key[1]}/{key[2]}")


def infer_sample_prefix(file_path: str, sample_id: str | None) -> str:
    stem = Path(file_path).stem
    match = re.search(r"(No\d+)", stem)
    if match:
        return match.group(1)
    if sample_id:
        match = re.search(r"(No\d+)", sample_id)
        if match:
            return match.group(1)
    return "UNKNOWN"


def infer_group_key(file_path: str) -> str:
    stem = Path(file_path).stem.strip()
    stem = re.sub(r"\s+", " ", stem)

    # Standard "50000-1" / "50000 0-1" style repeat suffixes.
    stem = re.sub(r"( 50000(?: \d+)?)\s*-\d+$", r"\1", stem)
    # Coordinate-like suffixes such as "1.841-2".
    stem = re.sub(r"([+-]?\d+(?:\.\d+)?)-\d+$", r"\1", stem)
    # top2/mid2/bottom2 -> top/mid/bottom.
    stem = re.sub(r"\b(top|mid|bottom)(\d+)$", r"\1", stem, flags=re.IGNORECASE)
    return stem


def load_df() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    query = """
    SELECT
        id AS image_id,
        sample_id,
        file_path,
        source,
        magnification,
        al2o3_power,
        al2o3_thickness,
        fe_power,
        fe_thickness,
        ar_flow,
        h2_flow,
        c2h4_flow,
        anneal_temp,
        growth_temp,
        anneal_time,
        growth_time,
        curvature,
        curvature_p70,
        waviness_ratio,
        tortuosity,
        alignment,
        density,
        diameter,
        junction_count,
        junction_ratio,
        branch_count
    FROM images
    WHERE source='ZZY'
      AND COALESCE(is_deleted, 0)=0
      AND magnification=50000
      AND fe_power IS NOT NULL
      AND fe_thickness IS NOT NULL
      AND anneal_time IS NOT NULL
      AND curvature IS NOT NULL
      AND waviness_ratio IS NOT NULL
      AND tortuosity IS NOT NULL
      AND alignment IS NOT NULL
      AND density IS NOT NULL
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    numeric_cols = [
        "al2o3_power",
        "al2o3_thickness",
        "fe_power",
        "fe_thickness",
        "ar_flow",
        "h2_flow",
        "c2h4_flow",
        "anneal_temp",
        "growth_temp",
        "anneal_time",
        "growth_time",
        "curvature",
        "curvature_p70",
        "waviness_ratio",
        "tortuosity",
        "alignment",
        "density",
        "diameter",
        "junction_count",
        "junction_ratio",
        "branch_count",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["gas_level"] = [
        infer_gas_level(ar, h2, c2h4)
        for ar, h2, c2h4 in zip(df["ar_flow"], df["h2_flow"], df["c2h4_flow"])
    ]
    df["gas_tuple"] = [
        f"{int(ar)}/{int(h2)}/{int(c2h4)}"
        for ar, h2, c2h4 in zip(df["ar_flow"], df["h2_flow"], df["c2h4_flow"])
    ]
    df["sample_prefix"] = [
        infer_sample_prefix(path, sample_id)
        for path, sample_id in zip(df["file_path"], df["sample_id"])
    ]
    df["group_key"] = [infer_group_key(path) for path in df["file_path"]]
    df["fe_deposition_index"] = df["fe_power"] * df["fe_thickness"]
    return df


def build_model_specs() -> list[dict]:
    return [
        {
            "name": "ridge_main_no_anneal",
            "num_cols": ["fe_thickness"],
            "cat_cols": ["gas_level", "fe_power"],
            "estimator": Ridge(alpha=1.0),
        },
        {
            "name": "ridge_main_with_anneal",
            "num_cols": ["fe_thickness", "anneal_time"],
            "cat_cols": ["gas_level", "fe_power"],
            "estimator": Ridge(alpha=1.0),
        },
        {
            "name": "elastic_main_with_anneal",
            "num_cols": ["fe_thickness", "anneal_time"],
            "cat_cols": ["gas_level", "fe_power"],
            "estimator": ElasticNet(alpha=0.02, l1_ratio=0.2, max_iter=20000),
        },
        {
            "name": "rf_main_no_anneal",
            "num_cols": ["fe_thickness", "fe_deposition_index"],
            "cat_cols": ["gas_level", "fe_power"],
            "estimator": RandomForestRegressor(
                n_estimators=300,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=1,
            ),
        },
        {
            "name": "rf_main_with_anneal",
            "num_cols": ["fe_thickness", "anneal_time", "fe_deposition_index"],
            "cat_cols": ["gas_level", "fe_power"],
            "estimator": RandomForestRegressor(
                n_estimators=300,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=1,
            ),
        },
    ]


def build_pipeline(num_cols: list[str], cat_cols: list[str], estimator) -> Pipeline:
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
    return Pipeline(
        [
            ("pre", ColumnTransformer(transformers)),
            ("model", estimator),
        ]
    )


def evaluate_models(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray | dict]]]:
    rows: list[dict] = []
    best_predictions: dict[str, dict[str, np.ndarray | dict]] = {}
    model_specs = build_model_specs()

    for target in TARGETS:
        target_df = df.dropna(subset=[target]).copy()
        y = target_df[target].to_numpy(dtype=float)
        target_groups = target_df["group_key"].astype(str)
        n_splits = min(5, int(target_groups.nunique()))
        if n_splits < 2:
            raise ValueError(f"Not enough groups for CV on target {target}")
        cv = GroupKFold(n_splits=n_splits)

        best_row = None
        best_pred = None

        for spec in model_specs:
            feature_cols = spec["num_cols"] + spec["cat_cols"]
            x = target_df[feature_cols].copy()
            pipeline = build_pipeline(spec["num_cols"], spec["cat_cols"], spec["estimator"])
            y_pred = cross_val_predict(pipeline, x, y, cv=cv, groups=target_groups)
            row = {
                "target": target,
                "model": spec["name"],
                "n": int(len(target_df)),
                "feature_columns": ",".join(feature_cols),
                "group_count": int(target_groups.nunique()),
                "mae": float(mean_absolute_error(y, y_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
                "r2": float(r2_score(y, y_pred)),
            }
            rows.append(row)
            if best_row is None or row["r2"] > best_row["r2"]:
                best_row = row
                best_pred = y_pred

        assert best_row is not None and best_pred is not None
        best_predictions[target] = {
            "meta": best_row,
            "y_true": y,
            "y_pred": best_pred,
        }

    result_df = (
        pd.DataFrame(rows)
        .sort_values(["target", "r2"], ascending=[True, False])
        .reset_index(drop=True)
    )
    return result_df, best_predictions


def build_anneal_gain_table(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for target in sorted(results["target"].unique().tolist()):
        no_df = results[
            results["target"].eq(target)
            & results["model"].str.contains("no_anneal")
        ].sort_values("r2", ascending=False)
        with_df = results[
            results["target"].eq(target)
            & results["model"].str.contains("with_anneal")
        ].sort_values("r2", ascending=False)
        if no_df.empty or with_df.empty:
            continue
        best_no = no_df.iloc[0]
        best_with = with_df.iloc[0]
        rows.append(
            {
                "target": target,
                "best_no_anneal_model": best_no["model"],
                "best_no_anneal_r2": float(best_no["r2"]),
                "best_with_anneal_model": best_with["model"],
                "best_with_anneal_r2": float(best_with["r2"]),
                "r2_gain": float(best_with["r2"] - best_no["r2"]),
            }
        )
    return pd.DataFrame(rows).sort_values("r2_gain", ascending=False).reset_index(drop=True)


def save_prediction_scatter(best_predictions: dict[str, dict[str, np.ndarray | dict]], output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    for ax, target in zip(axes.ravel(), TARGETS):
        payload = best_predictions[target]
        y_true = np.asarray(payload["y_true"], dtype=float)
        y_pred = np.asarray(payload["y_pred"], dtype=float)
        meta = payload["meta"]
        ax.scatter(y_true, y_pred, s=32, alpha=0.8, color="#2878B5", edgecolors="none")
        vmin = float(min(np.min(y_true), np.min(y_pred)))
        vmax = float(max(np.max(y_true), np.max(y_pred)))
        ax.plot([vmin, vmax], [vmin, vmax], linestyle="--", color="#999999", linewidth=1)
        ax.set_title(f"{target}\n{meta['model']} | R2={meta['r2']:.3f}", fontsize=10)
        ax.set_xlabel("True")
        ax.set_ylabel("Pred")
    fig.suptitle("ZZY + 50000x General Modeling", fontsize=15)
    fig.savefig(output_dir / "zzy_50000_general_prediction_scatter.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_outputs(df: pd.DataFrame, results: pd.DataFrame, anneal_gain: pd.DataFrame, best_predictions: dict[str, dict[str, np.ndarray | dict]], output_dir: Path) -> None:
    keep_cols = [
        "image_id",
        "sample_id",
        "sample_prefix",
        "group_key",
        "file_path",
        "gas_level",
        "gas_tuple",
        "al2o3_power",
        "al2o3_thickness",
        "fe_power",
        "fe_thickness",
        "fe_deposition_index",
        "anneal_temp",
        "growth_temp",
        "anneal_time",
        "growth_time",
        "curvature",
        "curvature_p70",
        "waviness_ratio",
        "tortuosity",
        "alignment",
        "density",
        "diameter",
        "junction_count",
        "junction_ratio",
        "branch_count",
    ]
    df[keep_cols].to_csv(output_dir / "zzy_50000_general_modeling_table.csv", index=False, encoding="utf-8-sig")
    results.to_csv(output_dir / "baseline_results.csv", index=False, encoding="utf-8-sig")
    anneal_gain.to_csv(output_dir / "anneal_time_gain_comparison.csv", index=False, encoding="utf-8-sig")

    best_rows = []
    for target in TARGETS:
        best_rows.append(
            results[results["target"].eq(target)]
            .sort_values("r2", ascending=False)
            .iloc[0]
            .to_dict()
        )
    pd.DataFrame(best_rows).to_csv(output_dir / "best_results_by_target.csv", index=False, encoding="utf-8-sig")

    process_counts = (
        df.groupby(["gas_level", "anneal_time", "fe_power", "fe_thickness"])
        .size()
        .reset_index(name="n")
        .sort_values(["gas_level", "anneal_time", "fe_power", "fe_thickness"])
    )
    process_counts.to_csv(output_dir / "process_coverage_counts.csv", index=False, encoding="utf-8-sig")

    summary = {
        "db_path": str(DB_PATH),
        "row_count": int(len(df)),
        "sample_prefix_count": int(df["sample_prefix"].nunique()),
        "group_count": int(df["group_key"].nunique()),
        "gas_levels": sorted(df["gas_level"].dropna().unique().tolist()),
        "anneal_time_hours": sorted([float(x) for x in df["anneal_time"].dropna().unique().tolist()]),
        "recommended_main_inputs": ["gas_level", "fe_power", "fe_thickness", "anneal_time"],
        "note": "anneal_time is included as an auxiliary process variable in the broad ZZY+50000x model; interpret independently with caution because it is coupled with batch/sample coverage.",
        "best_results_by_target": {
            row["target"]: {
                "model": row["model"],
                "n": int(row["n"]),
                "group_count": int(row["group_count"]),
                "mae": round(float(row["mae"]), 6),
                "rmse": round(float(row["rmse"]), 6),
                "r2": round(float(row["r2"]), 6),
                "feature_columns": row["feature_columns"],
            }
            for row in best_rows
        },
        "anneal_gain": [
            {
                "target": row["target"],
                "best_no_anneal_model": row["best_no_anneal_model"],
                "best_no_anneal_r2": round(float(row["best_no_anneal_r2"]), 6),
                "best_with_anneal_model": row["best_with_anneal_model"],
                "best_with_anneal_r2": round(float(row["best_with_anneal_r2"]), 6),
                "r2_gain": round(float(row["r2_gain"]), 6),
            }
            for _, row in anneal_gain.iterrows()
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    md_lines = ["# ZZY + 50000x General Modeling", ""]
    md_lines.append(f"- rows: `{len(df)}`")
    md_lines.append(f"- sample prefixes: `{df['sample_prefix'].nunique()}`")
    md_lines.append(f"- grouped fields: `{df['group_key'].nunique()}`")
    md_lines.append("- main inputs: `gas_level`, `fe_power`, `fe_thickness`, `anneal_time`")
    md_lines.append("- CV: `GroupKFold` by normalized `group_key`")
    md_lines.append("")
    md_lines.append("| target | best model | features | MAE | RMSE | R2 |")
    md_lines.append("| --- | --- | --- | ---: | ---: | ---: |")
    for row in best_rows:
        md_lines.append(
            f"| {row['target']} | {row['model']} | {row['feature_columns']} | "
            f"{row['mae']:.4f} | {row['rmse']:.4f} | {row['r2']:.4f} |"
        )
    md_lines.append("")
    md_lines.append("## Anneal Time Gain")
    md_lines.append("")
    md_lines.append("| target | best no anneal R2 | best with anneal R2 | gain |")
    md_lines.append("| --- | ---: | ---: | ---: |")
    for _, row in anneal_gain.iterrows():
        md_lines.append(
            f"| {row['target']} | {row['best_no_anneal_r2']:.4f} | "
            f"{row['best_with_anneal_r2']:.4f} | {row['r2_gain']:.4f} |"
        )
    (output_dir / "summary.md").write_text("\n".join(md_lines), encoding="utf-8")


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    df = load_df()
    results, best_predictions = evaluate_models(df)
    anneal_gain = build_anneal_gain_table(results)
    save_prediction_scatter(best_predictions, OUTPUT_DIR)
    save_outputs(df, results, anneal_gain, best_predictions, OUTPUT_DIR)
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
