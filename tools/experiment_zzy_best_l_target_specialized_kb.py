from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Iterable

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
sys.path.insert(0, str(PROJECT_ROOT))

from backend.core.knowledge_rag import RAGRetriever  # noqa: E402


INPUT_DIR = PROJECT_ROOT / "reports" / "zzy_50000_length_threshold_reextract_20260402_033320"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "zzy_50000_fe_time_model_20260402" / "best_l_target_specialized_kb_experiment"
CURRENT_BASE_DIR = PROJECT_ROOT / "reports" / "zzy_50000_fe_time_model_20260402"
BEST_L_SPECIALIZED_DIR = CURRENT_BASE_DIR / "best_l_target_specialized_experiment"
KB_DB_PATH = PROJECT_ROOT / "database" / "cnta_knowledge_base.sqlite"

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
    out["anneal_time_x_thickness"] = out["anneal_time"] * out["fe_thickness"]
    out["anneal_time_x_power"] = out["anneal_time"] * out["fe_power"]
    out["deposition_time_index"] = out["fe_deposition_index"] * out["anneal_time"]
    out["thickness_sq"] = out["fe_thickness"] ** 2
    out["power_sq"] = out["fe_power"] ** 2
    out["time_sq"] = out["anneal_time"] ** 2
    out["inv_thickness"] = 1.0 / out["fe_thickness"]
    out["power_x_thickness_sq"] = out["fe_power"] * out["thickness_sq"]
    return out


def normalize_series(series: pd.Series) -> pd.Series:
    min_v = float(series.min())
    max_v = float(series.max())
    if math.isclose(min_v, max_v):
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - min_v) / (max_v - min_v)


def effect_direction_sign(direction: str | None) -> float:
    text = str(direction or "").lower()
    positive_tokens = {"increase", "promote", "positive", "enhance", "促进", "提高", "增大", "增加"}
    negative_tokens = {"decrease", "negative", "reduce", "inhibit", "抑制", "降低", "减少"}
    nonlinear_tokens = {"nonlinear_or_tradeoff", "tradeoff", "复杂", "非线性"}
    if any(token in text for token in positive_tokens):
        return 1.0
    if any(token in text for token in negative_tokens):
        return -1.0
    if any(token in text for token in nonlinear_tokens):
        return 0.35
    return 0.0


def process_factor_value(row: pd.Series, process_factor: str, normalized: dict[str, float]) -> float:
    text = str(process_factor or "").lower()
    if any(token in text for token in ["铁催化剂厚度", "fe", "iron", "catalyst thickness"]):
        return normalized["fe_thickness"]
    if any(token in text for token in ["退火时间", "生长时间", "anneal", "time"]):
        return normalized["anneal_time"]
    if any(token in text for token in ["功率", "power"]):
        return normalized["fe_power"]
    return 0.25


def build_query_for_target(target: str) -> str:
    morphology_terms = {
        "curvature": "anneal time iron catalyst thickness power curvature waviness bending CNT morphology",
        "waviness_ratio": "anneal time iron catalyst thickness power waviness curvature CNT morphology",
        "tortuosity": "anneal time iron catalyst thickness power tortuosity waviness curvature CNT morphology",
        "alignment": "anneal time iron catalyst thickness power alignment oriented CNT morphology",
    }
    return morphology_terms[target]


def build_target_prior_features(df: pd.DataFrame, rag: RAGRetriever, target: str) -> pd.DataFrame:
    query = build_query_for_target(target)
    links = rag.knowledge_base.search_links(query, top_k=12)
    chain = rag.knowledge_base.get_relation_chain_summary(query, top_k=20)
    chain_counts = {f"kb_{target}_chain_{key}": len(value) for key, value in chain.items()}

    rows = []
    for _, row in df.iterrows():
        normalized = {
            "fe_power": float(row["fe_power_norm"]),
            "fe_thickness": float(row["fe_thickness_norm"]),
            "anneal_time": float(row["anneal_time_norm"]),
        }
        positive_sum = 0.0
        negative_sum = 0.0
        nonlinear_sum = 0.0
        signed_drive = 0.0
        match_score_sum = 0.0
        confidence_sum = 0.0
        matched = 0

        for item in links:
            confidence = float(item.get("confidence") or 0.0)
            match_score = float(item.get("_match_score") or 0.0)
            sign = effect_direction_sign(item.get("effect_direction"))
            factor_value = process_factor_value(row, str(item.get("process_factor") or ""), normalized)
            weighted = confidence * factor_value
            confidence_sum += confidence
            match_score_sum += match_score
            if sign > 0.5:
                positive_sum += weighted
                signed_drive += weighted
                matched += 1
            elif sign < -0.5:
                negative_sum += weighted
                signed_drive -= weighted
                matched += 1
            else:
                nonlinear_sum += confidence * max(0.2, factor_value)

        rows.append(
            {
                "image_id": row["image_id"],
                f"kb_{target}_link_count": len(links),
                f"kb_{target}_matched_link_count": matched,
                f"kb_{target}_positive_sum": positive_sum,
                f"kb_{target}_negative_sum": negative_sum,
                f"kb_{target}_nonlinear_sum": nonlinear_sum,
                f"kb_{target}_signed_drive": signed_drive,
                f"kb_{target}_match_score_sum": match_score_sum,
                f"kb_{target}_confidence_sum": confidence_sum,
                **chain_counts,
            }
        )
    return pd.DataFrame(rows)


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


def build_target_specs(kb_cols: Iterable[str]) -> dict[str, list[dict]]:
    kb_cols = list(kb_cols)
    return {
        "curvature": [
            {
                "name": "extratrees_curvature_bestl_kb",
                "num_cols": [
                    "fe_power", "fe_thickness", "anneal_time", "fe_deposition_index",
                    "anneal_time_x_thickness", "anneal_time_x_power", "deposition_time_index",
                    "thickness_sq", "inv_thickness", "power_x_thickness_sq", *kb_cols,
                ],
                "cat_cols": ["power_bin", "anneal_power_combo"],
                "estimator": ExtraTreesRegressor(n_estimators=1000, min_samples_leaf=2, random_state=42, n_jobs=1),
                "target_transform": None,
            },
            {
                "name": "rf_curvature_bestl_kb",
                "num_cols": [
                    "fe_power", "fe_thickness", "anneal_time", "fe_deposition_index",
                    "anneal_time_x_thickness", "anneal_time_x_power", "deposition_time_index",
                    "thickness_sq", "inv_thickness", "power_x_thickness_sq", *kb_cols,
                ],
                "cat_cols": ["power_bin"],
                "estimator": RandomForestRegressor(n_estimators=1000, min_samples_leaf=2, random_state=42, n_jobs=1),
                "target_transform": None,
            },
            {
                "name": "elastic_curvature_bestl_kb",
                "num_cols": [
                    "fe_thickness", "anneal_time", "fe_deposition_index",
                    "anneal_time_x_thickness", "thickness_sq", "inv_thickness", *kb_cols,
                ],
                "cat_cols": ["power_bin", "anneal_power_combo"],
                "estimator": ElasticNet(alpha=0.006, l1_ratio=0.2, max_iter=30000),
                "target_transform": "log1p",
            },
        ],
        "waviness_ratio": [
            {
                "name": "extratrees_waviness_bestl_kb",
                "num_cols": [
                    "fe_power", "fe_thickness", "anneal_time", "fe_deposition_index",
                    "anneal_time_x_thickness", "anneal_time_x_power", "deposition_time_index",
                    "thickness_sq", "time_sq", *kb_cols,
                ],
                "cat_cols": ["power_bin", "anneal_power_combo"],
                "estimator": ExtraTreesRegressor(n_estimators=1000, min_samples_leaf=2, random_state=42, n_jobs=1),
                "target_transform": "log1p",
            },
            {
                "name": "rf_waviness_bestl_kb",
                "num_cols": [
                    "fe_power", "fe_thickness", "anneal_time", "fe_deposition_index",
                    "anneal_time_x_thickness", "anneal_time_x_power", "deposition_time_index",
                    "thickness_sq", "time_sq", *kb_cols,
                ],
                "cat_cols": ["power_bin"],
                "estimator": RandomForestRegressor(n_estimators=1000, min_samples_leaf=2, random_state=42, n_jobs=1),
                "target_transform": "log1p",
            },
        ],
        "tortuosity": [
            {
                "name": "extratrees_tortuosity_bestl_kb",
                "num_cols": [
                    "fe_power", "fe_thickness", "anneal_time", "fe_deposition_index",
                    "anneal_time_x_thickness", "anneal_time_x_power", "deposition_time_index",
                    "thickness_sq", "time_sq", *kb_cols,
                ],
                "cat_cols": ["power_bin", "anneal_power_combo"],
                "estimator": ExtraTreesRegressor(n_estimators=1000, min_samples_leaf=2, random_state=42, n_jobs=1),
                "target_transform": "tortuosity_delta",
            },
            {
                "name": "rf_tortuosity_bestl_kb",
                "num_cols": [
                    "fe_power", "fe_thickness", "anneal_time", "fe_deposition_index",
                    "anneal_time_x_thickness", "anneal_time_x_power", "deposition_time_index",
                    "thickness_sq", "time_sq", *kb_cols,
                ],
                "cat_cols": ["power_bin"],
                "estimator": RandomForestRegressor(n_estimators=1000, min_samples_leaf=2, random_state=42, n_jobs=1),
                "target_transform": "tortuosity_delta",
            },
        ],
        "alignment": [
            {
                "name": "extratrees_alignment_bestl_kb",
                "num_cols": [
                    "fe_power", "fe_thickness", "anneal_time", "fe_deposition_index",
                    "anneal_time_x_thickness", "anneal_time_x_power", "deposition_time_index",
                    "thickness_sq", "power_sq", "time_sq", *kb_cols,
                ],
                "cat_cols": ["power_bin", "anneal_power_combo"],
                "estimator": ExtraTreesRegressor(n_estimators=1000, min_samples_leaf=2, random_state=42, n_jobs=1),
                "target_transform": None,
            },
            {
                "name": "rf_alignment_bestl_kb",
                "num_cols": [
                    "fe_power", "fe_thickness", "anneal_time", "fe_deposition_index",
                    "anneal_time_x_thickness", "anneal_time_x_power", "deposition_time_index",
                    "thickness_sq", "power_sq", "time_sq", *kb_cols,
                ],
                "cat_cols": ["power_bin"],
                "estimator": RandomForestRegressor(n_estimators=1000, min_samples_leaf=2, random_state=42, n_jobs=1),
                "target_transform": None,
            },
            {
                "name": "ridge_alignment_bestl_kb",
                "num_cols": [
                    "fe_thickness", "anneal_time", "fe_deposition_index",
                    "anneal_time_x_thickness", "deposition_time_index", "thickness_sq", *kb_cols,
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
    df = add_base_features(df)
    df["fe_power_norm"] = normalize_series(pd.to_numeric(df["fe_power"], errors="coerce"))
    df["fe_thickness_norm"] = normalize_series(pd.to_numeric(df["fe_thickness"], errors="coerce"))
    df["anneal_time_norm"] = normalize_series(pd.to_numeric(df["anneal_time"], errors="coerce"))
    df[cfg["output_name"]] = pd.to_numeric(df[cfg["target_col"]], errors="coerce") * float(cfg["scale"])
    df["source_L"] = cfg["l_label"]
    return df


def evaluate() -> pd.DataFrame:
    rows: list[dict] = []
    rag = RAGRetriever(db_path=str(KB_DB_PATH))

    for target in TARGET_CONFIG:
        df = load_target_dataframe(target)
        kb_features = build_target_prior_features(df, rag, target)
        df = df.merge(kb_features, on="image_id", how="left")
        kb_cols = [col for col in df.columns if col.startswith(f"kb_{target}_")]
        specs = build_target_specs(kb_cols)[target]

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


def compare_to_best_l_specialized(best: pd.DataFrame) -> pd.DataFrame:
    prev = pd.read_csv(BEST_L_SPECIALIZED_DIR / "best_results.csv")
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

    vs_current = compare_to_current(best)
    vs_current.to_csv(OUTPUT_DIR / "vs_current.csv", index=False, encoding="utf-8-sig")

    vs_prev = compare_to_best_l_specialized(best)
    vs_prev.to_csv(OUTPUT_DIR / "vs_best_l_specialized.csv", index=False, encoding="utf-8-sig")

    summary = {
        "input_dir": str(INPUT_DIR),
        "output_dir": str(OUTPUT_DIR),
        "target_config": TARGET_CONFIG,
        "best_results": best.to_dict(orient="records"),
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
