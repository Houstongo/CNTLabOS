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
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(r"D:\CNTDATA\CNTA_ML_Project")
INPUT_DIR = PROJECT_ROOT / "reports" / "zzy_50000_length_threshold_reextract_20260402_033320"
OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "zzy_50000_fe_time_model_20260402"
    / "best_l_target_specialized_noiseaware_plus_structure_experiment"
)
PREV_DIR = (
    PROJECT_ROOT
    / "reports"
    / "zzy_50000_fe_time_model_20260402"
    / "best_l_target_specialized_noiseaware_experiment"
)

PROCESS_COLS = ["fe_power", "fe_thickness", "anneal_time"]

TARGET_CONFIG = {
    "curvature": {
        "l_label": "L4",
        "target_col": "curvature_trimmed_mean_uniform_nm",
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
    out["density_x_junction"] = out["density"] * out["junction_ratio"]
    out["branch_density_proxy"] = out["branch_count"] / np.maximum(out["density"], 1e-6)
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
    structure_cols = ["density", "junction_ratio", "branch_count", "density_x_junction", "branch_density_proxy"]
    return {
        "curvature": [
            {
                "name": "extratrees_curvature_noiseaware_structure",
                "num_cols": [
                    "fe_power", "fe_thickness", "anneal_time", "fe_deposition_index",
                    "anneal_time_x_thickness", "anneal_time_x_power", "deposition_time_index",
                    "thickness_sq", "inv_thickness", "power_x_thickness_sq", *structure_cols,
                ],
                "cat_cols": ["power_bin", "anneal_power_combo"],
                "estimator": ExtraTreesRegressor(n_estimators=1000, min_samples_leaf=2, random_state=42, n_jobs=1),
                "target_transform": None,
            },
            {
                "name": "rf_curvature_noiseaware_structure",
                "num_cols": [
                    "fe_power", "fe_thickness", "anneal_time", "fe_deposition_index",
                    "anneal_time_x_thickness", "anneal_time_x_power", "deposition_time_index",
                    "thickness_sq", "inv_thickness", "power_x_thickness_sq", *structure_cols,
                ],
                "cat_cols": ["power_bin"],
                "estimator": RandomForestRegressor(n_estimators=1000, min_samples_leaf=2, random_state=42, n_jobs=1),
                "target_transform": None,
            },
        ],
        "waviness_ratio": [
            {
                "name": "extratrees_waviness_noiseaware_structure",
                "num_cols": [
                    "fe_power", "fe_thickness", "anneal_time", "fe_deposition_index",
                    "anneal_time_x_thickness", "anneal_time_x_power", "deposition_time_index",
                    "thickness_sq", "time_sq", *structure_cols,
                ],
                "cat_cols": ["power_bin", "anneal_power_combo"],
                "estimator": ExtraTreesRegressor(n_estimators=1000, min_samples_leaf=2, random_state=42, n_jobs=1),
                "target_transform": "log1p",
            },
            {
                "name": "rf_waviness_noiseaware_structure",
                "num_cols": [
                    "fe_power", "fe_thickness", "anneal_time", "fe_deposition_index",
                    "anneal_time_x_thickness", "anneal_time_x_power", "deposition_time_index",
                    "thickness_sq", "time_sq", *structure_cols,
                ],
                "cat_cols": ["power_bin"],
                "estimator": RandomForestRegressor(n_estimators=1000, min_samples_leaf=2, random_state=42, n_jobs=1),
                "target_transform": "log1p",
            },
        ],
        "tortuosity": [
            {
                "name": "extratrees_tortuosity_noiseaware_structure",
                "num_cols": [
                    "fe_power", "fe_thickness", "anneal_time", "fe_deposition_index",
                    "anneal_time_x_thickness", "anneal_time_x_power", "deposition_time_index",
                    "thickness_sq", "time_sq", *structure_cols,
                ],
                "cat_cols": ["power_bin", "anneal_power_combo"],
                "estimator": ExtraTreesRegressor(n_estimators=1000, min_samples_leaf=2, random_state=42, n_jobs=1),
                "target_transform": "tortuosity_delta",
            },
            {
                "name": "rf_tortuosity_noiseaware_structure",
                "num_cols": [
                    "fe_power", "fe_thickness", "anneal_time", "fe_deposition_index",
                    "anneal_time_x_thickness", "anneal_time_x_power", "deposition_time_index",
                    "thickness_sq", "time_sq", *structure_cols,
                ],
                "cat_cols": ["power_bin"],
                "estimator": RandomForestRegressor(n_estimators=1000, min_samples_leaf=2, random_state=42, n_jobs=1),
                "target_transform": "tortuosity_delta",
            },
        ],
        "alignment": [
            {
                "name": "extratrees_alignment_noiseaware_structure",
                "num_cols": [
                    "fe_power", "fe_thickness", "anneal_time", "fe_deposition_index",
                    "anneal_time_x_thickness", "anneal_time_x_power", "deposition_time_index",
                    "thickness_sq", "power_sq", "time_sq", *structure_cols,
                ],
                "cat_cols": ["power_bin", "anneal_power_combo"],
                "estimator": ExtraTreesRegressor(n_estimators=1000, min_samples_leaf=2, random_state=42, n_jobs=1),
                "target_transform": None,
            },
            {
                "name": "rf_alignment_noiseaware_structure",
                "num_cols": [
                    "fe_power", "fe_thickness", "anneal_time", "fe_deposition_index",
                    "anneal_time_x_thickness", "anneal_time_x_power", "deposition_time_index",
                    "thickness_sq", "power_sq", "time_sq", *structure_cols,
                ],
                "cat_cols": ["power_bin"],
                "estimator": RandomForestRegressor(n_estimators=1000, min_samples_leaf=2, random_state=42, n_jobs=1),
                "target_transform": None,
            },
            {
                "name": "ridge_alignment_noiseaware_structure",
                "num_cols": [
                    "fe_thickness", "anneal_time", "fe_deposition_index",
                    "anneal_time_x_thickness", "deposition_time_index", "thickness_sq", *structure_cols,
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
    df[cfg["output_name"]] = pd.to_numeric(df[cfg["target_col"]], errors="coerce") * float(cfg["scale"])
    df["source_L"] = cfg["l_label"]
    return df


def build_noise_weights(df: pd.DataFrame, target: str) -> pd.DataFrame:
    work = df.copy()
    stats = (
        work.groupby(PROCESS_COLS)[target]
        .agg(["size", "var"])
        .reset_index()
        .rename(columns={"size": "combo_count", "var": "combo_var"})
    )
    nonzero_var = stats["combo_var"].replace(0, np.nan).dropna()
    global_var = float(nonzero_var.median()) if len(nonzero_var) else 1.0
    if not np.isfinite(global_var) or global_var <= 0:
        global_var = 1.0
    stats["combo_var"] = stats["combo_var"].fillna(global_var)
    stats["shrunk_var"] = (
        (stats["combo_count"] / (stats["combo_count"] + 2.0)) * stats["combo_var"]
        + (2.0 / (stats["combo_count"] + 2.0)) * global_var
    )
    eps = global_var * 0.05
    stats["raw_weight"] = 1.0 / (stats["shrunk_var"] + eps)
    lower = float(stats["raw_weight"].quantile(0.05))
    upper = float(stats["raw_weight"].quantile(0.95))
    stats["sample_weight"] = stats["raw_weight"].clip(lower=lower, upper=upper)
    stats["sample_weight"] = stats["sample_weight"] / float(stats["sample_weight"].mean())
    return work.merge(stats[PROCESS_COLS + ["sample_weight"]], on=PROCESS_COLS, how="left")


def cross_val_predict_with_weights(pipe: Pipeline, x: pd.DataFrame, y: np.ndarray, groups: pd.Series, sample_weight: np.ndarray) -> np.ndarray:
    preds = np.full(shape=len(y), fill_value=np.nan, dtype=float)
    n_splits = min(5, int(groups.nunique()))
    cv = GroupKFold(n_splits=n_splits)
    for train_idx, test_idx in cv.split(x, y, groups):
        pipe.fit(x.iloc[train_idx], y[train_idx], model__sample_weight=sample_weight[train_idx])
        preds[test_idx] = pipe.predict(x.iloc[test_idx])
    return preds


def evaluate() -> pd.DataFrame:
    rows: list[dict] = []
    target_specs = build_target_specs()
    for target, specs in target_specs.items():
        df = load_target_dataframe(target)
        df = df.dropna(subset=[target, "density", "junction_ratio", "branch_count"]).copy()
        df = build_noise_weights(df, target)
        y = df[target].to_numpy(dtype=float)
        groups = df["group_key"].astype(str)
        sample_weight = df["sample_weight"].to_numpy(dtype=float)
        for spec in specs:
            feature_cols = spec["num_cols"] + spec["cat_cols"]
            x = df[feature_cols].copy()
            pipe = build_pipeline(spec["num_cols"], spec["cat_cols"], spec["estimator"], spec["target_transform"])
            y_pred = cross_val_predict_with_weights(pipe, x, y, groups, sample_weight)
            rows.append(
                {
                    "target": target,
                    "source_L": TARGET_CONFIG[target]["l_label"],
                    "model": spec["name"],
                    "n": int(len(df)),
                    "group_count": int(groups.nunique()),
                    "feature_columns": ",".join(feature_cols),
                    "target_transform": spec["target_transform"] or "",
                    "mae": float(mean_absolute_error(y, y_pred)),
                    "rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
                    "r2": float(r2_score(y, y_pred)),
                }
            )
    return pd.DataFrame(rows)


def compare_to_previous(best: pd.DataFrame) -> pd.DataFrame:
    prev = pd.read_csv(PREV_DIR / "best_results.csv")
    compare = prev.rename(
        columns={"model": "prev_model", "mae": "prev_mae", "rmse": "prev_rmse", "r2": "prev_r2"}
    ).merge(
        best.rename(columns={"model": "new_model", "mae": "new_mae", "rmse": "new_rmse", "r2": "new_r2"}),
        on="target",
        how="outer",
        suffixes=("", "_dup"),
    )
    dup_cols = [col for col in compare.columns if col.endswith("_dup")]
    if dup_cols:
        compare = compare.drop(columns=dup_cols)
    compare["r2_gain_vs_prev"] = compare["new_r2"] - compare["prev_r2"]
    compare["mae_delta_vs_prev"] = compare["new_mae"] - compare["prev_mae"]
    compare["rmse_delta_vs_prev"] = compare["new_rmse"] - compare["prev_rmse"]
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
    vs_prev = compare_to_previous(best)
    vs_prev.to_csv(OUTPUT_DIR / "vs_noiseaware.csv", index=False, encoding="utf-8-sig")
    summary = {
        "input_dir": str(INPUT_DIR),
        "output_dir": str(OUTPUT_DIR),
        "best_results": best.to_dict(orient="records"),
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
