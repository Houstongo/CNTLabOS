from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(r"D:\CNTDATA\CNTA_ML_Project")
sys.path.insert(0, str(PROJECT_ROOT))

from backend.core.knowledge_rag import RAGRetriever  # noqa: E402


BASE_DIR = PROJECT_ROOT / "reports" / "zzy_50000_fe_time_model_20260402"
INPUT_CSV = BASE_DIR / "zzy_50000_fe_time_modeling_table.csv"
OUTPUT_DIR = BASE_DIR / "knowledge_prior_feature_experiment"
DB_PATH = PROJECT_ROOT / "database" / "cnta_experiments.sqlite"
KB_DB_PATH = PROJECT_ROOT / "database" / "cnta_knowledge_base.sqlite"
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
    out["anneal_time_x_thickness"] = out["anneal_time"] * out["fe_thickness"]
    out["anneal_time_x_power"] = out["anneal_time"] * out["fe_power"]
    out["deposition_time_index"] = out["fe_deposition_index"] * out["anneal_time"]
    out["thickness_sq"] = out["fe_thickness"] ** 2
    out["time_sq"] = out["anneal_time"] ** 2
    out["power_sq"] = out["fe_power"] ** 2
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
        "curvature": "curvature waviness bending CNT morphology",
        "waviness_ratio": "waviness curvature CNT morphology",
        "tortuosity": "tortuosity waviness curvature CNT morphology",
        "alignment": "alignment oriented CNT morphology",
    }
    return f"anneal time iron catalyst thickness power {morphology_terms[target]}"


def build_target_prior_features(df: pd.DataFrame, rag: RAGRetriever, target: str) -> pd.DataFrame:
    query = build_query_for_target(target)
    links = rag.knowledge_base.search_links(query, top_k=12)
    chain = rag.knowledge_base.get_relation_chain_summary(query, top_k=20)

    chain_counts = {f"kb_{target}_chain_{key}": len(value) for key, value in chain.items()}
    out_rows = []

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

        out_rows.append(
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

    return pd.DataFrame(out_rows)


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


def build_pipeline(num_cols: list[str], cat_cols: list[str], estimator) -> Pipeline:
    return Pipeline([("pre", build_preprocessor(num_cols, cat_cols)), ("model", estimator)])


def evaluate_target(df: pd.DataFrame, target: str, kb_cols: Iterable[str]) -> pd.DataFrame:
    specs = [
        {
            "name": "rf_with_kb_prior",
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
                *kb_cols,
            ],
            "cat_cols": ["power_bin", "anneal_power_combo"],
            "estimator": RandomForestRegressor(
                n_estimators=700,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=1,
            ),
        },
        {
            "name": "extratrees_with_kb_prior",
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
                *kb_cols,
            ],
            "cat_cols": ["power_bin", "anneal_power_combo"],
            "estimator": ExtraTreesRegressor(
                n_estimators=900,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=1,
            ),
        },
    ]

    rows = []
    target_df = df.dropna(subset=[target]).copy()
    y = target_df[target].to_numpy(dtype=float)
    groups = target_df["group_key"].astype(str)
    cv = GroupKFold(n_splits=min(5, int(groups.nunique())))

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
    rag = RAGRetriever(str(DB_PATH), knowledge_db_path=str(KB_DB_PATH))

    df = pd.read_csv(INPUT_CSV, dtype={"image_id": str})
    df = add_base_features(df)
    df["fe_power_norm"] = normalize_series(df["fe_power"])
    df["fe_thickness_norm"] = normalize_series(df["fe_thickness"])
    df["anneal_time_norm"] = normalize_series(df["anneal_time"])

    results_frames = []
    prior_meta: dict[str, object] = {}

    for target in TARGETS:
        priors = build_target_prior_features(df, rag, target)
        df_target = df.merge(priors, on="image_id", how="left")
        kb_cols = [col for col in priors.columns if col != "image_id"]
        target_results = evaluate_target(df_target, target, kb_cols)
        results_frames.append(target_results)
        prior_meta[target] = {
            "query": build_query_for_target(target),
            "kb_feature_columns": kb_cols,
        }

    results = pd.concat(results_frames, ignore_index=True)
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
        "prior_meta": prior_meta,
        "best_results": best.to_dict(orient="records"),
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
