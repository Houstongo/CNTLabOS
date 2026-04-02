from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(r"D:\CNTDATA\CNTA_ML_Project")
BASE_DIR = PROJECT_ROOT / "reports" / "zzy_50000_fe_time_model_20260402"
INPUT_CSV = BASE_DIR / "zzy_50000_fe_time_modeling_table.csv"
OUTPUT_DIR = BASE_DIR / "powerbin_interaction_experiment"
TARGETS = ["curvature", "waviness_ratio", "tortuosity", "alignment"]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    def fe_power_bin(value: float) -> str:
        if value <= 5:
            return "low"
        if value <= 20:
            return "mid"
        return "high"

    out["power_bin"] = out["fe_power"].astype(float).map(fe_power_bin)
    out["anneal_label"] = out["anneal_time"].map({0.25: "15min", 0.5: "30min", 0.75: "45min"})
    out["anneal_power_combo"] = out["anneal_label"].astype(str) + "_" + out["power_bin"].astype(str)
    out["anneal_time_x_thickness"] = out["anneal_time"] * out["fe_thickness"]
    out["anneal_time_x_power"] = out["anneal_time"] * out["fe_power"]
    out["deposition_time_index"] = out["fe_deposition_index"] * out["anneal_time"]
    out["thickness_sq"] = out["fe_thickness"] ** 2
    return out


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


def build_specs() -> list[dict]:
    return [
        {
            "name": "ridge_powerbin_interaction",
            "num_cols": [
                "fe_thickness",
                "anneal_time",
                "fe_deposition_index",
                "anneal_time_x_thickness",
                "deposition_time_index",
                "thickness_sq",
            ],
            "cat_cols": ["power_bin", "anneal_power_combo"],
            "estimator": Ridge(alpha=1.0),
        },
        {
            "name": "elastic_powerbin_interaction",
            "num_cols": [
                "fe_thickness",
                "anneal_time",
                "fe_deposition_index",
                "anneal_time_x_thickness",
                "deposition_time_index",
                "thickness_sq",
            ],
            "cat_cols": ["power_bin", "anneal_power_combo"],
            "estimator": ElasticNet(alpha=0.01, l1_ratio=0.15, max_iter=20000),
        },
        {
            "name": "rf_powerbin_interaction",
            "num_cols": [
                "fe_power",
                "fe_thickness",
                "anneal_time",
                "fe_deposition_index",
                "anneal_time_x_thickness",
                "anneal_time_x_power",
                "deposition_time_index",
                "thickness_sq",
            ],
            "cat_cols": ["power_bin"],
            "estimator": RandomForestRegressor(
                n_estimators=500,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=1,
            ),
        },
        {
            "name": "extratrees_powerbin_interaction",
            "num_cols": [
                "fe_power",
                "fe_thickness",
                "anneal_time",
                "fe_deposition_index",
                "anneal_time_x_thickness",
                "anneal_time_x_power",
                "deposition_time_index",
                "thickness_sq",
            ],
            "cat_cols": ["power_bin"],
            "estimator": ExtraTreesRegressor(
                n_estimators=500,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=1,
            ),
        },
    ]


def evaluate(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    specs = build_specs()
    for target in TARGETS:
        target_df = df.dropna(subset=[target]).copy()
        y = target_df[target].to_numpy(dtype=float)
        groups = target_df["group_key"].astype(str)
        n_splits = min(5, int(groups.nunique()))
        cv = GroupKFold(n_splits=n_splits)

        for spec in specs:
            feature_cols = spec["num_cols"] + spec["cat_cols"]
            x = target_df[feature_cols].copy()
            pipe = build_pipeline(spec["num_cols"], spec["cat_cols"], spec["estimator"])
            y_pred = cross_val_predict(pipe, x, y, cv=cv, groups=groups)
            rows.append(
                {
                    "target": target,
                    "model": spec["name"],
                    "n": int(len(target_df)),
                    "group_count": int(groups.nunique()),
                    "feature_columns": ",".join(feature_cols),
                    "mae": float(mean_absolute_error(y, y_pred)),
                    "rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
                    "r2": float(r2_score(y, y_pred)),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    df = pd.read_csv(INPUT_CSV, dtype={"image_id": str})
    df = add_features(df)
    results = evaluate(df)
    results.to_csv(OUTPUT_DIR / "results.csv", index=False, encoding="utf-8-sig")

    best = (
        results.sort_values(["target", "r2"], ascending=[True, False])
        .groupby("target", as_index=False)
        .first()
    )
    best.to_csv(OUTPUT_DIR / "best_results.csv", index=False, encoding="utf-8-sig")

    current = pd.read_csv(BASE_DIR / "best_results_by_target.csv")
    current = current[current["target"].isin(TARGETS)].copy()
    compare = current.rename(
        columns={"model": "current_model", "mae": "current_mae", "rmse": "current_rmse", "r2": "current_r2"}
    ).merge(
        best.rename(
            columns={"model": "new_model", "mae": "new_mae", "rmse": "new_rmse", "r2": "new_r2"}
        ),
        on="target",
        how="outer",
        suffixes=("", "_dup"),
    )
    dup_cols = [col for col in compare.columns if col.endswith("_dup")]
    if dup_cols:
        compare = compare.drop(columns=dup_cols)
    compare["r2_gain"] = compare["new_r2"] - compare["current_r2"]
    compare["mae_delta"] = compare["new_mae"] - compare["current_mae"]
    compare["rmse_delta"] = compare["new_rmse"] - compare["current_rmse"]
    compare.to_csv(OUTPUT_DIR / "vs_current.csv", index=False, encoding="utf-8-sig")

    summary = {
        "input_csv": str(INPUT_CSV),
        "row_count": int(len(df)),
        "targets": TARGETS,
        "power_bin_counts": {str(k): int(v) for k, v in df["power_bin"].value_counts().sort_index().items()},
        "best_results": best.to_dict(orient="records"),
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
