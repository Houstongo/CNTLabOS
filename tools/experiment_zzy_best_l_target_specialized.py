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
INPUT_DIR = PROJECT_ROOT / "reports" / "zzy_50000_length_threshold_reextract_20260402_033320"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "zzy_50000_fe_time_model_20260402" / "best_l_target_specialized_experiment"
CURRENT_BASE_DIR = PROJECT_ROOT / "reports" / "zzy_50000_fe_time_model_20260402"
BEST_L_BASELINE_DIR = INPUT_DIR / "l0_l4_model_compare"

TARGET_CONFIG = {
    "curvature": {
        "l_label": "L4",
        "target_col": "curvature_trimmed_mean_sqrt_length_nm",
        "scale": 1000.0,
        "output_name": "curvature",
    },
    "waviness_ratio": {
        "l_label": "L4",
        "target_col": "waviness_ratio_v2",
        "scale": 1.0,
        "output_name": "waviness_ratio",
    },
    "tortuosity": {
        "l_label": "L0",
        "target_col": "tortuosity_v2",
        "scale": 1.0,
        "output_name": "tortuosity",
    },
    "alignment": {
        "l_label": "L0",
        "target_col": "alignment",
        "scale": 1.0,
        "output_name": "alignment",
    },
}


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
                "name": "rf_curvature_bestl",
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
                    n_estimators=900,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=1,
                ),
                "target_transform": None,
            },
            {
                "name": "extratrees_curvature_bestl",
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
                "name": "elastic_curvature_bestl",
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
                "name": "extratrees_waviness_bestl",
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
                    n_estimators=1000,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=1,
                ),
                "target_transform": "log1p",
            },
            {
                "name": "rf_waviness_bestl",
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
                    n_estimators=1000,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=1,
                ),
                "target_transform": "log1p",
            },
            {
                "name": "ridge_waviness_bestl",
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
                "name": "extratrees_tortuosity_bestl",
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
                    n_estimators=1000,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=1,
                ),
                "target_transform": "tortuosity_delta",
            },
            {
                "name": "rf_tortuosity_bestl",
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
                    n_estimators=1000,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=1,
                ),
                "target_transform": "tortuosity_delta",
            },
            {
                "name": "ridge_tortuosity_bestl",
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
        "alignment": [
            {
                "name": "extratrees_alignment_bestl",
                "num_cols": [
                    "fe_power",
                    "fe_thickness",
                    "anneal_time",
                    "fe_deposition_index",
                    "anneal_time_x_thickness",
                    "anneal_time_x_power",
                    "deposition_time_index",
                    "thickness_sq",
                    "power_sq",
                    "time_sq",
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
                "name": "rf_alignment_bestl",
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
                    n_estimators=1000,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=1,
                ),
                "target_transform": None,
            },
            {
                "name": "ridge_alignment_bestl",
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
                "target_transform": None,
            },
        ],
    }


def load_target_dataframe(target: str) -> pd.DataFrame:
    cfg = TARGET_CONFIG[target]
    table_path = INPUT_DIR / f"{cfg['l_label'].lower()}_modeling_table.csv"
    df = pd.read_csv(table_path, dtype={"image_id": str})
    df = add_features(df)
    target_values = pd.to_numeric(df[cfg["target_col"]], errors="coerce") * float(cfg["scale"])
    df[cfg["output_name"]] = target_values
    df["source_L"] = cfg["l_label"]
    return df


def evaluate() -> pd.DataFrame:
    rows: list[dict] = []
    target_specs = build_target_specs()

    for target, specs in target_specs.items():
        df = load_target_dataframe(target)
        tdf = df.dropna(subset=[target]).copy()
        y = tdf[target].to_numpy(dtype=float)
        groups = tdf["group_key"].astype(str)
        n_splits = min(5, int(groups.nunique()))
        cv = GroupKFold(n_splits=n_splits)

        for spec in specs:
            feature_cols = spec["num_cols"] + spec["cat_cols"]
            x = tdf[feature_cols].copy()
            pipe = build_pipeline(spec["num_cols"], spec["cat_cols"], spec["estimator"], spec["target_transform"])
            y_pred = cross_val_predict(pipe, x, y, cv=cv, groups=groups)
            rows.append(
                {
                    "target": target,
                    "source_L": TARGET_CONFIG[target]["l_label"],
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


def compare_to_current(best: pd.DataFrame) -> pd.DataFrame:
    current = pd.read_csv(CURRENT_BASE_DIR / "best_results_by_target.csv")
    current = current[current["target"].isin(best["target"].unique())].copy()
    compare = current.rename(
        columns={"model": "current_model", "mae": "current_mae", "rmse": "current_rmse", "r2": "current_r2"}
    ).merge(
        best.rename(columns={"model": "new_model", "mae": "new_mae", "rmse": "new_rmse", "r2": "new_r2"}),
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
    return compare


def compare_to_best_l_baseline(best: pd.DataFrame) -> pd.DataFrame:
    baseline = pd.read_csv(BEST_L_BASELINE_DIR / "best_l_per_target.csv")
    compare = baseline.rename(
        columns={"L": "baseline_L", "model": "baseline_model", "mae": "baseline_mae", "rmse": "baseline_rmse", "r2": "baseline_r2"}
    ).merge(
        best.rename(columns={"model": "new_model", "mae": "new_mae", "rmse": "new_rmse", "r2": "new_r2"}),
        on="target",
        how="outer",
        suffixes=("", "_dup"),
    )
    dup_cols = [col for col in compare.columns if col.endswith("_dup")]
    if dup_cols:
        compare = compare.drop(columns=dup_cols)
    compare["r2_gain_vs_best_l_baseline"] = compare["new_r2"] - compare["baseline_r2"]
    compare["mae_delta_vs_best_l_baseline"] = compare["new_mae"] - compare["baseline_mae"]
    compare["rmse_delta_vs_best_l_baseline"] = compare["new_rmse"] - compare["baseline_rmse"]
    return compare


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    results = evaluate()
    results.to_csv(OUTPUT_DIR / "results.csv", index=False, encoding="utf-8-sig")

    best = (
        results.sort_values(["target", "r2"], ascending=[True, False])
        .groupby("target", as_index=False)
        .first()
        .sort_values("target")
    )
    best.to_csv(OUTPUT_DIR / "best_results.csv", index=False, encoding="utf-8-sig")

    vs_current = compare_to_current(best)
    vs_current.to_csv(OUTPUT_DIR / "vs_current.csv", index=False, encoding="utf-8-sig")

    vs_best_l = compare_to_best_l_baseline(best)
    vs_best_l.to_csv(OUTPUT_DIR / "vs_best_l_baseline.csv", index=False, encoding="utf-8-sig")

    summary = {
        "input_dir": str(INPUT_DIR),
        "output_dir": str(OUTPUT_DIR),
        "target_config": TARGET_CONFIG,
        "best_results": best.to_dict(orient="records"),
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
