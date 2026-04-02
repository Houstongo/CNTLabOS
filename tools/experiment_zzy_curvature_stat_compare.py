from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(r"D:\CNTDATA\CNTA_ML_Project")
INPUT_DIR = PROJECT_ROOT / "reports" / "zzy_50000_length_threshold_reextract_20260402_033320"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "zzy_50000_fe_time_model_20260402" / "curvature_stat_compare_experiment"
L_LABELS = ["L0", "L1", "L2", "L3", "L4"]
STAT_LABELS = ["mean", "trimmed_mean"]
WEIGHT_LABELS = ["uniform", "sqrt_length", "length"]


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
    raise ValueError(f"Unknown transform mode: {mode}")


def build_pipeline(num_cols: list[str], cat_cols: list[str], estimator, target_transform: str | None) -> Pipeline:
    pre = build_preprocessor(num_cols, cat_cols)
    model = wrap_ttr(estimator, target_transform)
    return Pipeline([("pre", pre), ("model", model)])


def build_specs() -> list[dict]:
    return [
        {
            "name": "rf_curvature_stat",
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
                n_estimators=1000,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=1,
            ),
            "target_transform": None,
        },
        {
            "name": "extratrees_curvature_stat",
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
            "cat_cols": ["power_bin", "anneal_power_combo"],
            "estimator": ExtraTreesRegressor(
                n_estimators=1000,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=1,
            ),
            "target_transform": None,
        },
        {
            "name": "elastic_curvature_stat",
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
    ]


def load_table(l_label: str) -> pd.DataFrame:
    table_path = INPUT_DIR / f"{l_label.lower()}_modeling_table.csv"
    df = pd.read_csv(table_path, dtype={"image_id": str})
    return add_features(df)


def evaluate_target(df: pd.DataFrame, target_col: str, model_name: str, spec: dict) -> dict:
    tdf = df.dropna(subset=[target_col]).copy()
    y = pd.to_numeric(tdf[target_col], errors="coerce").to_numpy(dtype=float) * 1000.0
    groups = tdf["group_key"].astype(str)
    n_splits = min(5, int(groups.nunique()))
    cv = GroupKFold(n_splits=n_splits)
    feature_cols = spec["num_cols"] + spec["cat_cols"]
    x = tdf[feature_cols].copy()
    pipe = build_pipeline(spec["num_cols"], spec["cat_cols"], spec["estimator"], spec["target_transform"])
    y_pred = cross_val_predict(pipe, x, y, cv=cv, groups=groups)
    return {
        "model": model_name,
        "n": int(len(tdf)),
        "group_count": int(groups.nunique()),
        "feature_columns": ",".join(feature_cols),
        "target_transform": spec["target_transform"] or "",
        "mae": float(mean_absolute_error(y, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
        "r2": float(r2_score(y, y_pred)),
    }


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    specs = build_specs()
    rows: list[dict] = []

    for l_label in L_LABELS:
        df = load_table(l_label)
        for stat in STAT_LABELS:
            for weight in WEIGHT_LABELS:
                target_col = f"curvature_{stat}_{weight}_nm"
                for spec in specs:
                    result = evaluate_target(df, target_col, spec["name"], spec)
                    rows.append(
                        {
                            "L": l_label,
                            "stat": stat,
                            "weight": weight,
                            "target_col": target_col,
                            **result,
                        }
                    )

    results = pd.DataFrame(rows).sort_values(["r2", "rmse", "mae"], ascending=[False, True, True]).reset_index(drop=True)
    results.to_csv(OUTPUT_DIR / "all_results.csv", index=False, encoding="utf-8-sig")

    best_by_combo = (
        results.sort_values(["L", "stat", "weight", "r2"], ascending=[True, True, True, False])
        .groupby(["L", "stat", "weight"], as_index=False)
        .first()
        .sort_values(["r2", "rmse"], ascending=[False, True])
        .reset_index(drop=True)
    )
    best_by_combo.to_csv(OUTPUT_DIR / "best_by_combo.csv", index=False, encoding="utf-8-sig")

    best_overall = best_by_combo.iloc[0].to_dict()

    best_by_l = (
        best_by_combo.sort_values(["L", "r2"], ascending=[True, False])
        .groupby("L", as_index=False)
        .first()
        .sort_values("r2", ascending=False)
        .reset_index(drop=True)
    )
    best_by_l.to_csv(OUTPUT_DIR / "best_by_l.csv", index=False, encoding="utf-8-sig")

    best_by_stat_weight = (
        best_by_combo.sort_values(["stat", "weight", "r2"], ascending=[True, True, False])
        .groupby(["stat", "weight"], as_index=False)
        .first()
        .sort_values("r2", ascending=False)
        .reset_index(drop=True)
    )
    best_by_stat_weight.to_csv(OUTPUT_DIR / "best_by_stat_weight.csv", index=False, encoding="utf-8-sig")

    summary = {
        "input_dir": str(INPUT_DIR),
        "output_dir": str(OUTPUT_DIR),
        "l_labels": L_LABELS,
        "stat_labels": STAT_LABELS,
        "weight_labels": WEIGHT_LABELS,
        "best_overall": {
            "L": best_overall["L"],
            "stat": best_overall["stat"],
            "weight": best_overall["weight"],
            "model": best_overall["model"],
            "r2": round(float(best_overall["r2"]), 6),
            "mae": round(float(best_overall["mae"]), 6),
            "rmse": round(float(best_overall["rmse"]), 6),
        },
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# Curvature Statistic Compare", ""]
    lines.append("| rank | L | stat | weight | model | R2 | MAE | RMSE |")
    lines.append("| --- | --- | --- | --- | --- | ---: | ---: | ---: |")
    for idx, (_, row) in enumerate(best_by_combo.head(15).iterrows(), start=1):
        lines.append(
            f"| {idx} | {row['L']} | {row['stat']} | {row['weight']} | {row['model']} | {row['r2']:.4f} | {row['mae']:.4f} | {row['rmse']:.4f} |"
        )
    (OUTPUT_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
