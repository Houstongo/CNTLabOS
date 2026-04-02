from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
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
OUTPUT_DIR = BASE_DIR / "target_specialized_experiment"
TARGETS = ["curvature", "waviness_ratio", "tortuosity"]


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
    out["power_sq"] = out["fe_power"] ** 2
    out["time_sq"] = out["anneal_time"] ** 2
    out["inv_thickness"] = 1.0 / out["fe_thickness"]
    out["power_x_thickness_sq"] = out["fe_power"] * out["thickness_sq"]
    return out


def build_preprocessor(num_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
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
    return ColumnTransformer(transformers)


def wrap_ttr(estimator, mode: str | None):
    if mode is None:
        return estimator
    if mode == "log1p":
        return TransformedTargetRegressor(
            regressor=estimator,
            func=np.log1p,
            inverse_func=np.expm1,
            check_inverse=False,
        )
    if mode == "tortuosity_delta":
        return TransformedTargetRegressor(
            regressor=estimator,
            func=lambda y: np.log1p(np.clip(y - 1.0, a_min=0.0, a_max=None)),
            inverse_func=lambda y: np.expm1(y) + 1.0,
            check_inverse=False,
        )
    raise ValueError(f"Unknown transform mode: {mode}")


def build_pipeline(num_cols: list[str], cat_cols: list[str], estimator, target_transform: str | None) -> Pipeline:
    pre = build_preprocessor(num_cols, cat_cols)
    model = wrap_ttr(estimator, target_transform)
    return Pipeline([("pre", pre), ("model", model)])


def build_target_specs() -> dict[str, list[dict]]:
    return {
        "curvature": [
            {
                "name": "rf_curvature_engineered",
                "num_cols": [
                    "fe_power",
                    "fe_thickness",
                    "anneal_time",
                    "fe_deposition_index",
                    "anneal_time_x_thickness",
                    "anneal_time_x_power",
                    "deposition_time_index",
                    "thickness_sq",
                    "inv_thickness",
                    "power_x_thickness_sq",
                ],
                "cat_cols": ["power_bin"],
                "estimator": RandomForestRegressor(
                    n_estimators=700,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=1,
                ),
                "target_transform": None,
            },
            {
                "name": "extratrees_curvature_engineered",
                "num_cols": [
                    "fe_power",
                    "fe_thickness",
                    "anneal_time",
                    "fe_deposition_index",
                    "anneal_time_x_thickness",
                    "anneal_time_x_power",
                    "deposition_time_index",
                    "thickness_sq",
                    "inv_thickness",
                    "power_x_thickness_sq",
                ],
                "cat_cols": ["power_bin"],
                "estimator": ExtraTreesRegressor(
                    n_estimators=700,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=1,
                ),
                "target_transform": None,
            },
            {
                "name": "elastic_curvature_combo",
                "num_cols": [
                    "fe_thickness",
                    "anneal_time",
                    "fe_deposition_index",
                    "anneal_time_x_thickness",
                    "thickness_sq",
                    "inv_thickness",
                ],
                "cat_cols": ["power_bin", "anneal_power_combo"],
                "estimator": ElasticNet(alpha=0.006, l1_ratio=0.2, max_iter=30000),
                "target_transform": "log1p",
            },
        ],
        "waviness_ratio": [
            {
                "name": "extratrees_waviness_engineered",
                "num_cols": [
                    "fe_power",
                    "fe_thickness",
                    "anneal_time",
                    "fe_deposition_index",
                    "anneal_time_x_thickness",
                    "anneal_time_x_power",
                    "deposition_time_index",
                    "thickness_sq",
                    "time_sq",
                ],
                "cat_cols": ["power_bin", "anneal_power_combo"],
                "estimator": ExtraTreesRegressor(
                    n_estimators=900,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=1,
                ),
                "target_transform": "log1p",
            },
            {
                "name": "rf_waviness_engineered",
                "num_cols": [
                    "fe_power",
                    "fe_thickness",
                    "anneal_time",
                    "fe_deposition_index",
                    "anneal_time_x_thickness",
                    "anneal_time_x_power",
                    "deposition_time_index",
                    "thickness_sq",
                    "time_sq",
                ],
                "cat_cols": ["power_bin"],
                "estimator": RandomForestRegressor(
                    n_estimators=900,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=1,
                ),
                "target_transform": "log1p",
            },
            {
                "name": "ridge_waviness_combo",
                "num_cols": [
                    "fe_thickness",
                    "anneal_time",
                    "fe_deposition_index",
                    "anneal_time_x_thickness",
                    "deposition_time_index",
                    "time_sq",
                ],
                "cat_cols": ["power_bin", "anneal_power_combo"],
                "estimator": Ridge(alpha=1.0),
                "target_transform": "log1p",
            },
        ],
        "tortuosity": [
            {
                "name": "extratrees_tortuosity_delta",
                "num_cols": [
                    "fe_power",
                    "fe_thickness",
                    "anneal_time",
                    "fe_deposition_index",
                    "anneal_time_x_thickness",
                    "anneal_time_x_power",
                    "deposition_time_index",
                    "thickness_sq",
                    "time_sq",
                ],
                "cat_cols": ["power_bin", "anneal_power_combo"],
                "estimator": ExtraTreesRegressor(
                    n_estimators=900,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=1,
                ),
                "target_transform": "tortuosity_delta",
            },
            {
                "name": "rf_tortuosity_delta",
                "num_cols": [
                    "fe_power",
                    "fe_thickness",
                    "anneal_time",
                    "fe_deposition_index",
                    "anneal_time_x_thickness",
                    "anneal_time_x_power",
                    "deposition_time_index",
                    "thickness_sq",
                    "time_sq",
                ],
                "cat_cols": ["power_bin"],
                "estimator": RandomForestRegressor(
                    n_estimators=900,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=1,
                ),
                "target_transform": "tortuosity_delta",
            },
            {
                "name": "ridge_tortuosity_combo",
                "num_cols": [
                    "fe_thickness",
                    "anneal_time",
                    "fe_deposition_index",
                    "anneal_time_x_thickness",
                    "deposition_time_index",
                    "time_sq",
                ],
                "cat_cols": ["power_bin", "anneal_power_combo"],
                "estimator": Ridge(alpha=1.0),
                "target_transform": "tortuosity_delta",
            },
        ],
    }


def evaluate(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    target_specs = build_target_specs()

    for target in TARGETS:
        tdf = df.dropna(subset=[target]).copy()
        y = tdf[target].to_numpy(dtype=float)
        groups = tdf["group_key"].astype(str)
        n_splits = min(5, int(groups.nunique()))
        cv = GroupKFold(n_splits=n_splits)

        for spec in target_specs[target]:
            feature_cols = spec["num_cols"] + spec["cat_cols"]
            x = tdf[feature_cols].copy()
            pipe = build_pipeline(spec["num_cols"], spec["cat_cols"], spec["estimator"], spec["target_transform"])
            y_pred = cross_val_predict(pipe, x, y, cv=cv, groups=groups)
            rows.append(
                {
                    "target": target,
                    "model": spec["name"],
                    "n": int(len(tdf)),
                    "group_count": int(groups.nunique()),
                    "feature_columns": ",".join(feature_cols),
                    "target_transform": spec["target_transform"] or "",
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
        "best_results": best.to_dict(orient="records"),
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
