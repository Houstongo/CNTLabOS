from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
INPUT_DIR = PROJECT_ROOT / "reports" / "zzy_50000_length_threshold_reextract_20260402_033320"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "zzy_50000_length_threshold_reextract_20260402_033320" / "l0_l4_model_compare"
TARGETS = ["curvature", "waviness_ratio", "tortuosity", "alignment"]
L_LABELS = ["L0", "L1", "L2", "L3", "L4"]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_model_specs() -> list[dict]:
    return [
        {
            "name": "ridge_all_numeric",
            "num_cols": ["fe_power", "fe_thickness", "anneal_time"],
            "cat_cols": [],
            "estimator": Ridge(alpha=1.0),
        },
        {
            "name": "ridge_power_cat",
            "num_cols": ["fe_thickness", "anneal_time"],
            "cat_cols": ["fe_power"],
            "estimator": Ridge(alpha=1.0),
        },
        {
            "name": "elastic_power_cat",
            "num_cols": ["fe_thickness", "anneal_time"],
            "cat_cols": ["fe_power"],
            "estimator": ElasticNet(alpha=0.02, l1_ratio=0.2, max_iter=20000),
        },
        {
            "name": "rf_all_plus_index",
            "num_cols": ["fe_power", "fe_thickness", "anneal_time", "fe_deposition_index"],
            "cat_cols": [],
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
    return Pipeline([("pre", ColumnTransformer(transformers)), ("model", estimator)])


def load_l_table(label: str) -> pd.DataFrame:
    df = pd.read_csv(INPUT_DIR / f"{label.lower()}_modeling_table.csv")
    df["fe_deposition_index"] = pd.to_numeric(df["fe_deposition_index"], errors="coerce")
    df["curvature"] = pd.to_numeric(df["curvature_trimmed_mean_sqrt_length_nm"], errors="coerce") * 1000.0
    df["waviness_ratio"] = pd.to_numeric(df["waviness_ratio_v2"], errors="coerce")
    df["tortuosity"] = pd.to_numeric(df["tortuosity_v2"], errors="coerce")
    df["alignment"] = pd.to_numeric(df["alignment"], errors="coerce")
    return df


def evaluate_l_table(df: pd.DataFrame, l_label: str) -> tuple[pd.DataFrame, list[dict]]:
    rows = []
    best_rows = []
    model_specs = build_model_specs()

    for target in TARGETS:
        target_df = df.dropna(subset=[target]).copy()
        y = target_df[target].to_numpy(dtype=float)
        groups = target_df["group_key"].astype(str)
        n_splits = min(5, int(groups.nunique()))
        if n_splits < 2:
            raise ValueError(f"Not enough groups for CV on {l_label} {target}")
        cv = GroupKFold(n_splits=n_splits)

        best_row = None
        for spec in model_specs:
            feature_cols = spec["num_cols"] + spec["cat_cols"]
            x = target_df[feature_cols].copy()
            pipeline = build_pipeline(spec["num_cols"], spec["cat_cols"], spec["estimator"])
            y_pred = cross_val_predict(pipeline, x, y, cv=cv, groups=groups)
            row = {
                "L": l_label,
                "target": target,
                "model": spec["name"],
                "n": int(len(target_df)),
                "group_count": int(groups.nunique()),
                "feature_columns": ",".join(feature_cols),
                "mae": float(mean_absolute_error(y, y_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
                "r2": float(r2_score(y, y_pred)),
            }
            rows.append(row)
            if best_row is None or row["r2"] > best_row["r2"]:
                best_row = row

        assert best_row is not None
        best_rows.append(best_row)

    return pd.DataFrame(rows), best_rows


def save_heatmap(best_df: pd.DataFrame) -> None:
    pivot = best_df.pivot(index="target", columns="L", values="r2").reindex(index=TARGETS, columns=L_LABELS)
    fig, ax = plt.subplots(figsize=(8, 4.2), constrained_layout=True)
    im = ax.imshow(pivot.values, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(L_LABELS)))
    ax.set_xticklabels(L_LABELS)
    ax.set_yticks(range(len(TARGETS)))
    ax.set_yticklabels(TARGETS)
    ax.set_title("L0-L4 Best R2 by Target")
    for i in range(len(TARGETS)):
        for j in range(len(L_LABELS)):
            val = pivot.values[i, j]
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=9, color="black")
    fig.colorbar(im, ax=ax, shrink=0.85, label="R2")
    fig.savefig(OUTPUT_DIR / "l0_l4_best_r2_heatmap.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    all_results = []
    best_rows = []

    for label in L_LABELS:
        df = load_l_table(label)
        result_df, best_for_l = evaluate_l_table(df, label)
        all_results.append(result_df)
        best_rows.extend(best_for_l)

    results = pd.concat(all_results, ignore_index=True)
    best_df = pd.DataFrame(best_rows).sort_values(["target", "r2"], ascending=[True, False]).reset_index(drop=True)

    results.to_csv(OUTPUT_DIR / "all_baseline_results.csv", index=False, encoding="utf-8-sig")
    best_df.to_csv(OUTPUT_DIR / "best_results_by_l_and_target.csv", index=False, encoding="utf-8-sig")

    per_target = (
        best_df.sort_values(["target", "r2"], ascending=[True, False])
        .groupby("target", as_index=False)
        .first()
        .sort_values("target")
    )
    per_target.to_csv(OUTPUT_DIR / "best_l_per_target.csv", index=False, encoding="utf-8-sig")

    per_l = (
        best_df.groupby("L", as_index=False)["r2"]
        .mean()
        .rename(columns={"r2": "mean_best_r2"})
        .sort_values("mean_best_r2", ascending=False)
    )
    per_l.to_csv(OUTPUT_DIR / "mean_best_r2_by_l.csv", index=False, encoding="utf-8-sig")

    save_heatmap(best_df)

    summary = {
        "input_dir": str(INPUT_DIR),
        "output_dir": str(OUTPUT_DIR),
        "targets": TARGETS,
        "l_labels": L_LABELS,
        "best_l_per_target": {
            row["target"]: {
                "L": row["L"],
                "model": row["model"],
                "r2": round(float(row["r2"]), 6),
                "mae": round(float(row["mae"]), 6),
                "rmse": round(float(row["rmse"]), 6),
                "feature_columns": row["feature_columns"],
            }
            for _, row in per_target.iterrows()
        },
        "mean_best_r2_by_l": [
            {"L": row["L"], "mean_best_r2": round(float(row["mean_best_r2"]), 6)}
            for _, row in per_l.iterrows()
        ],
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    md_lines = ["# ZZY 50000x L0-L4 Modeling Compare", ""]
    md_lines.append("| target | best L | best model | R2 | MAE | RMSE |")
    md_lines.append("| --- | --- | --- | ---: | ---: | ---: |")
    for _, row in per_target.iterrows():
        md_lines.append(
            f"| {row['target']} | {row['L']} | {row['model']} | {row['r2']:.4f} | {row['mae']:.4f} | {row['rmse']:.4f} |"
        )
    md_lines.append("")
    md_lines.append("| L | mean(best R2 across targets) |")
    md_lines.append("| --- | ---: |")
    for _, row in per_l.iterrows():
        md_lines.append(f"| {row['L']} | {row['mean_best_r2']:.4f} |")
    (OUTPUT_DIR / "summary.md").write_text("\n".join(md_lines), encoding="utf-8")

    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
