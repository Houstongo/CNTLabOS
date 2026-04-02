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
BASE_DIR = PROJECT_ROOT / "reports" / "zzy_50000_fe_time_model_20260402"
INPUT_CSV = BASE_DIR / "zzy_50000_fe_time_modeling_table.csv"
OUTPUT_DIR = BASE_DIR / "knowledge_residual_experiment"
TARGETS = ["curvature", "waviness_ratio", "tortuosity", "alignment"]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def add_base_features(df: pd.DataFrame) -> pd.DataFrame:
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
    out["fe_deposition_index"] = out["fe_power"] * out["fe_thickness"]
    out["anneal_time_x_thickness"] = out["anneal_time"] * out["fe_thickness"]
    out["anneal_time_x_power"] = out["anneal_time"] * out["fe_power"]
    out["thickness_sq"] = out["fe_thickness"] ** 2
    out["time_sq"] = out["anneal_time"] ** 2
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


def wrap_target_transform(estimator, mode: str | None):
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
    raise ValueError(f"Unknown target transform: {mode}")


def build_pipeline(num_cols: list[str], cat_cols: list[str], estimator, target_transform: str | None) -> Pipeline:
    return Pipeline(
        [
            ("pre", build_preprocessor(num_cols, cat_cols)),
            ("model", wrap_target_transform(estimator, target_transform)),
        ]
    )


def similarity_weight(df: pd.DataFrame, row: pd.Series) -> pd.Series:
    delta_power = (df["fe_power"] - row["fe_power"]).abs()
    delta_thickness = (df["fe_thickness"] - row["fe_thickness"]).abs()
    delta_time = (df["anneal_time"] - row["anneal_time"]).abs()

    # Focus similarity on the three varying process variables only.
    power_score = np.exp(-(delta_power / 10.0))
    thickness_score = np.exp(-(delta_thickness / 0.5))
    time_score = np.exp(-(delta_time / 0.25))

    # Favor exact power-bin / time matches but don't require them.
    power_bin_bonus = np.where(df["power_bin"] == row["power_bin"], 1.15, 1.0)
    anneal_bonus = np.where(df["anneal_label"] == row["anneal_label"], 1.10, 1.0)

    return power_score * thickness_score * time_score * power_bin_bonus * anneal_bonus


def compute_similarity_baseline(df: pd.DataFrame, target: str) -> pd.DataFrame:
    records = []
    for idx, row in df.iterrows():
        pool = df[df["image_id"] != row["image_id"]].copy()
        weights = similarity_weight(pool, row)
        pool = pool.assign(_weight=weights)
        pool = pool[pool["_weight"] > 0]

        if pool.empty:
            baseline = float(df[target].mean())
            support = 0
            nearest_gap = float("nan")
            weighted_std = float(df[target].std(ddof=0))
        else:
            weight_sum = float(pool["_weight"].sum())
            baseline = float((pool[target] * pool["_weight"]).sum() / max(weight_sum, 1e-12))
            support = int((pool["_weight"] > np.percentile(pool["_weight"], 75)).sum())
            nearest_gap = float(pool["_weight"].max())
            weighted_var = float((((pool[target] - baseline) ** 2) * pool["_weight"]).sum() / max(weight_sum, 1e-12))
            weighted_std = float(np.sqrt(max(weighted_var, 0.0)))

        records.append(
            {
                "image_id": row["image_id"],
                f"{target}_kb_baseline": baseline,
                f"{target}_kb_support_count": support,
                f"{target}_kb_weighted_std": weighted_std,
                f"{target}_kb_nearest_weight": nearest_gap,
            }
        )
    return pd.DataFrame(records)


def residual_specs(target: str) -> list[dict]:
    common_num = [
        "fe_power",
        "fe_thickness",
        "anneal_time",
        "fe_deposition_index",
        "anneal_time_x_thickness",
        "anneal_time_x_power",
        "thickness_sq",
        "time_sq",
    ]
    common_cat = ["power_bin", "anneal_power_combo"]

    if target == "curvature":
        return [
            {
                "name": "rf_residual_curvature",
                "num_cols": common_num,
                "cat_cols": ["power_bin"],
                "estimator": RandomForestRegressor(n_estimators=900, min_samples_leaf=2, random_state=42, n_jobs=1),
                "target_transform": None,
            },
            {
                "name": "extratrees_residual_curvature",
                "num_cols": common_num,
                "cat_cols": common_cat,
                "estimator": ExtraTreesRegressor(n_estimators=900, min_samples_leaf=2, random_state=42, n_jobs=1),
                "target_transform": None,
            },
            {
                "name": "elastic_residual_curvature",
                "num_cols": ["fe_thickness", "anneal_time", "fe_deposition_index", "anneal_time_x_thickness", "thickness_sq"],
                "cat_cols": common_cat,
                "estimator": ElasticNet(alpha=0.006, l1_ratio=0.2, max_iter=30000),
                "target_transform": None,
            },
        ]

    if target == "waviness_ratio":
        return [
            {
                "name": "extratrees_residual_waviness",
                "num_cols": common_num,
                "cat_cols": common_cat,
                "estimator": ExtraTreesRegressor(n_estimators=1000, min_samples_leaf=2, random_state=42, n_jobs=1),
                "target_transform": "log1p",
            },
            {
                "name": "rf_residual_waviness",
                "num_cols": common_num,
                "cat_cols": ["power_bin"],
                "estimator": RandomForestRegressor(n_estimators=1000, min_samples_leaf=2, random_state=42, n_jobs=1),
                "target_transform": "log1p",
            },
            {
                "name": "ridge_residual_waviness",
                "num_cols": ["fe_thickness", "anneal_time", "fe_deposition_index", "anneal_time_x_thickness", "time_sq"],
                "cat_cols": common_cat,
                "estimator": Ridge(alpha=1.0),
                "target_transform": "log1p",
            },
        ]

    if target == "tortuosity":
        return [
            {
                "name": "extratrees_residual_tortuosity",
                "num_cols": common_num,
                "cat_cols": common_cat,
                "estimator": ExtraTreesRegressor(n_estimators=1000, min_samples_leaf=2, random_state=42, n_jobs=1),
                "target_transform": None,
            },
            {
                "name": "rf_residual_tortuosity",
                "num_cols": common_num,
                "cat_cols": ["power_bin"],
                "estimator": RandomForestRegressor(n_estimators=1000, min_samples_leaf=2, random_state=42, n_jobs=1),
                "target_transform": None,
            },
            {
                "name": "ridge_residual_tortuosity",
                "num_cols": ["fe_thickness", "anneal_time", "fe_deposition_index", "anneal_time_x_thickness", "time_sq"],
                "cat_cols": common_cat,
                "estimator": Ridge(alpha=1.0),
                "target_transform": None,
            },
        ]

    if target == "alignment":
        return [
            {
                "name": "extratrees_residual_alignment",
                "num_cols": common_num,
                "cat_cols": common_cat,
                "estimator": ExtraTreesRegressor(n_estimators=1000, min_samples_leaf=2, random_state=42, n_jobs=1),
                "target_transform": None,
            },
            {
                "name": "rf_residual_alignment",
                "num_cols": common_num,
                "cat_cols": ["power_bin"],
                "estimator": RandomForestRegressor(n_estimators=1000, min_samples_leaf=2, random_state=42, n_jobs=1),
                "target_transform": None,
            },
            {
                "name": "ridge_residual_alignment",
                "num_cols": ["fe_thickness", "anneal_time", "fe_deposition_index", "anneal_time_x_thickness", "time_sq"],
                "cat_cols": common_cat,
                "estimator": Ridge(alpha=1.0),
                "target_transform": None,
            },
        ]

    raise ValueError(f"Unsupported target: {target}")


def fit_predict_residual_cv(df: pd.DataFrame, target: str, spec: dict) -> tuple[np.ndarray, np.ndarray]:
    groups = df["group_key"].astype(str)
    cv = GroupKFold(n_splits=min(5, int(groups.nunique())))
    feature_cols = spec["num_cols"] + spec["cat_cols"]
    final_pred = np.zeros(len(df), dtype=float)
    baseline_pred = np.zeros(len(df), dtype=float)

    for train_idx, test_idx in cv.split(df, df[target], groups):
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()

        baseline_train = compute_similarity_baseline(train_df, target)
        baseline_test = []
        for _, row in test_df.iterrows():
            weights = similarity_weight(train_df, row)
            pool = train_df.assign(_weight=weights)
            weight_sum = float(pool["_weight"].sum())
            baseline = float((pool[target] * pool["_weight"]).sum() / max(weight_sum, 1e-12))
            weighted_var = float((((pool[target] - baseline) ** 2) * pool["_weight"]).sum() / max(weight_sum, 1e-12))
            baseline_test.append(
                {
                    "image_id": row["image_id"],
                    f"{target}_kb_baseline": baseline,
                    f"{target}_kb_support_count": int((pool["_weight"] > np.percentile(pool["_weight"], 75)).sum()),
                    f"{target}_kb_weighted_std": float(np.sqrt(max(weighted_var, 0.0))),
                    f"{target}_kb_nearest_weight": float(pool["_weight"].max()),
                }
            )
        baseline_test = pd.DataFrame(baseline_test)

        train_aug = train_df.merge(baseline_train, on="image_id", how="left")
        test_aug = test_df.merge(baseline_test, on="image_id", how="left")

        residual_col = f"{target}_residual_target"
        train_aug[residual_col] = train_aug[target] - train_aug[f"{target}_kb_baseline"]

        pipe = build_pipeline(spec["num_cols"], spec["cat_cols"], spec["estimator"], spec["target_transform"])
        pipe.fit(train_aug[feature_cols], train_aug[residual_col].to_numpy(dtype=float))
        residual_pred = pipe.predict(test_aug[feature_cols])

        baseline_values = test_aug[f"{target}_kb_baseline"].to_numpy(dtype=float)
        baseline_pred[test_idx] = baseline_values
        final_pred[test_idx] = baseline_values + residual_pred

    return baseline_pred, final_pred


def evaluate(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    result_rows = []
    baseline_rows = []

    for target in TARGETS:
        target_df = df.dropna(subset=[target]).copy().reset_index(drop=True)
        y_true = target_df[target].to_numpy(dtype=float)

        # Baseline-only CV using leave-group-out style constrained neighbors.
        baseline_only = []
        groups = target_df["group_key"].astype(str)
        cv = GroupKFold(n_splits=min(5, int(groups.nunique())))
        for train_idx, test_idx in cv.split(target_df, y_true, groups):
            train_df = target_df.iloc[train_idx].copy()
            test_df = target_df.iloc[test_idx].copy()
            for _, row in test_df.iterrows():
                weights = similarity_weight(train_df, row)
                pool = train_df.assign(_weight=weights)
                baseline = float((pool[target] * pool["_weight"]).sum() / max(float(pool["_weight"].sum()), 1e-12))
                baseline_only.append((row["image_id"], baseline))
        baseline_map = {image_id: pred for image_id, pred in baseline_only}
        baseline_pred = target_df["image_id"].map(baseline_map).to_numpy(dtype=float)

        baseline_rows.append(
            {
                "target": target,
                "model": "knowledge_baseline_only",
                "n": int(len(target_df)),
                "group_count": int(groups.nunique()),
                "mae": float(mean_absolute_error(y_true, baseline_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_true, baseline_pred))),
                "r2": float(r2_score(y_true, baseline_pred)),
            }
        )

        for spec in residual_specs(target):
            _, final_pred = fit_predict_residual_cv(target_df, target, spec)
            result_rows.append(
                {
                    "target": target,
                    "model": spec["name"],
                    "n": int(len(target_df)),
                    "group_count": int(groups.nunique()),
                    "feature_columns": ",".join(spec["num_cols"] + spec["cat_cols"]),
                    "target_transform": spec["target_transform"] or "",
                    "mae": float(mean_absolute_error(y_true, final_pred)),
                    "rmse": float(np.sqrt(mean_squared_error(y_true, final_pred))),
                    "r2": float(r2_score(y_true, final_pred)),
                }
            )

    return pd.DataFrame(result_rows), pd.DataFrame(baseline_rows)


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    df = pd.read_csv(INPUT_CSV, dtype={"image_id": str})
    df = add_base_features(df)

    results, baseline_only = evaluate(df)
    results.to_csv(OUTPUT_DIR / "results.csv", index=False, encoding="utf-8-sig")
    baseline_only.to_csv(OUTPUT_DIR / "baseline_only_results.csv", index=False, encoding="utf-8-sig")

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
    compare.to_csv(OUTPUT_DIR / "vs_current.csv", index=False, encoding="utf-8-sig")

    specialized = pd.read_csv(BASE_DIR / "target_specialized_experiment" / "best_results.csv")
    specialized = specialized[specialized["target"].isin(TARGETS)].copy()
    compare_specialized = specialized.rename(
        columns={"model": "specialized_model", "mae": "specialized_mae", "rmse": "specialized_rmse", "r2": "specialized_r2"}
    ).merge(
        best.rename(columns={"model": "new_model", "mae": "new_mae", "rmse": "new_rmse", "r2": "new_r2"}),
        on="target",
        how="outer",
        suffixes=("", "_dup"),
    )
    dup_cols = [col for col in compare_specialized.columns if col.endswith("_dup")]
    if dup_cols:
        compare_specialized = compare_specialized.drop(columns=dup_cols)
    compare_specialized["r2_gain_vs_specialized"] = compare_specialized["new_r2"] - compare_specialized["specialized_r2"]
    compare_specialized["mae_delta_vs_specialized"] = compare_specialized["new_mae"] - compare_specialized["specialized_mae"]
    compare_specialized["rmse_delta_vs_specialized"] = compare_specialized["new_rmse"] - compare_specialized["specialized_rmse"]
    compare_specialized.to_csv(OUTPUT_DIR / "vs_target_specialized.csv", index=False, encoding="utf-8-sig")

    summary = {
        "input_csv": str(INPUT_CSV),
        "row_count": int(len(df)),
        "targets": TARGETS,
        "best_results": best.to_dict(orient="records"),
        "baseline_only_results": baseline_only.to_dict(orient="records"),
        "note": "Fixed process background is used as retrieval constraint by construction; residual models learn only from fe_power, fe_thickness, anneal_time, and their derived interactions.",
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
