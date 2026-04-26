"""
Knowledge-enhanced morphology predictor v2.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


XR_REPORT_PREFIX = "slice_standard_batch_"
ZZY_REPORT_PREFIX = "zzy_feature_engineering_gt10000_"

SUPPORTED_TARGETS = {
    "diameter": "diameter",
    "density": "density",
    "alignment": "alignment",
    "curvature": "curvature_proxy",
    "bend_score": "bend_score",
    "dk_bend_index": "dk_bend_index",
    "junction_ratio": "junction_ratio",
}

SOURCE_PROCESS_COLS = {
    "XR": ["actual_temp", "growth_time", "ar_flow", "catalyst_weight", "membrane_pos_cm", "magnification"],
    "ZZY": [
        "fe_thickness",
        "fe_power",
        "al2o3_thickness",
        "al2o3_power",
        "ar_flow",
        "h2_flow",
        "c2h4_flow",
        "growth_time",
        "magnification",
    ],
}

XR_FEATURE_KEYS = [
    "actual_temp",
    "growth_time_h",
    "ar_flow",
    "catalyst_weight",
    "membrane_pos_cm",
    "magnification",
    "temp_normalized",
    "position_normalized",
    "temp_x_catalyst",
    "flow_temp_interaction",
    "flow_per_catalyst",
]

ZZY_FEATURE_KEYS = [
    "growth_temp_effective",
    "growth_time_h",
    "magnification",
    "fe_thickness",
    "fe_power",
    "al2o3_thickness",
    "al2o3_power",
    "fe_thickness_sq",
    "catalyst_ratio",
    "gas_total",
    "carbon_supply",
    "reduction_ratio",
    "ar_to_c2h4_ratio",
    "fe_power_loading",
    "al2o3_power_loading",
]


@dataclass
class PredictionResult:
    predicted_value: float
    confidence: float
    knowledge_baseline: float
    ml_residual: float
    similar_experiments: List[Dict]
    rag_evidence: List[Dict]
    physical_constraints: List[str]


class KnowledgeDrivenPredictor:
    def __init__(self, db_path: str, rag_retriever):
        self.db_path = db_path
        self.rag = rag_retriever
        self.project_root = Path(db_path).resolve().parents[1]
        self.reports_root = self.project_root / "reports"
        self.source_frames: Dict[str, pd.DataFrame] = {}
        self.model_artifacts: Dict[str, Dict[str, Dict[str, Any]]] = {"XR": {}, "ZZY": {}}
        self.training_summaries: Dict[str, Dict[str, Dict[str, float]]] = {"XR": {}, "ZZY": {}}

    @staticmethod
    def _to_float(value: Any, default: Optional[float] = None) -> Optional[float]:
        if value is None:
            return default
        try:
            if pd.isna(value):
                return default
        except Exception:
            pass
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _resolve_target(self, target: str) -> str:
        key = str(target or "").strip().lower()
        if key not in SUPPORTED_TARGETS:
            raise ValueError(f"不支持的预测目标: {target}")
        return SUPPORTED_TARGETS[key]

    def check_physical_constraints(self, params: Dict) -> List[str]:
        constraints = []
        temp = params.get("growth_temp") or params.get("actual_temp")
        if temp is not None:
            if temp < 600:
                constraints.append("温度过低(<600℃)，碳源裂解不足，形核困难")
            elif temp > 900:
                constraints.append("温度过高(>900℃)，催化剂易团聚，密度下降")

        fe = params.get("fe_thickness")
        if fe is not None:
            if fe < 0.5:
                constraints.append("Fe厚度过薄(<0.5nm)，催化活性低，生长困难")
            elif fe > 3.0:
                constraints.append("Fe厚度过厚(>3.0nm)，颗粒团聚，密度降低")

        ar = params.get("ar_flow")
        c2h4 = params.get("c2h4_flow")
        if ar and c2h4:
            ratio = ar / c2h4 if c2h4 > 0 else float("inf")
            if ratio < 5:
                constraints.append(f"Ar/C2H4比过低({ratio:.1f}<5)，碳浓度过高易致无定形碳")
            elif ratio > 50:
                constraints.append(f"Ar/C2H4比过高({ratio:.1f}>50)，碳浓度不足密度降低")
        return constraints

    def build_knowledge_enhanced_features(self, params: Dict) -> Dict[str, float]:
        source = str(params.get("source") or "ZZY").upper()
        temp = self._to_float(params.get("growth_temp") or params.get("actual_temp"), default=750.0)
        growth_time = self._to_float(params.get("growth_time"), default=3.0)
        ar = self._to_float(params.get("ar_flow"), default=500.0)
        h2 = self._to_float(params.get("h2_flow"), default=100.0)
        c2h4 = self._to_float(params.get("c2h4_flow"), default=50.0)
        magnification = self._to_float(params.get("magnification"), default=50000.0)

        features: Dict[str, float] = {
            "growth_temp_effective": temp,
            "growth_time_h": growth_time,
            "magnification": magnification,
            "temp_normalized": temp / 750.0 if temp else 1.0,
            "gas_total": ar + h2 + c2h4,
            "carbon_supply": c2h4 / ar if ar > 0 else 0.0,
            "reduction_ratio": h2 / ar if ar > 0 else 0.0,
            "ar_to_c2h4_ratio": ar / c2h4 if c2h4 > 0 else 0.0,
        }

        if source == "XR":
            catalyst_weight = self._to_float(params.get("catalyst_weight"), default=1.0)
            membrane_pos = self._to_float(params.get("inlet_distance_cm"))
            if membrane_pos is None:
                membrane_pos = self._to_float(params.get("membrane_pos_cm"), default=18.0)
            features.update(
                {
                    "actual_temp": temp,
                    "ar_flow": ar,
                    "catalyst_weight": catalyst_weight,
                    "membrane_pos_cm": membrane_pos,
                    "position_normalized": membrane_pos / 36.0 if membrane_pos else 0.0,
                    "temp_x_catalyst": temp * catalyst_weight,
                    "flow_temp_interaction": ar * (temp / 800.0),
                    "flow_per_catalyst": ar / max(catalyst_weight, 0.1),
                }
            )
        else:
            fe = self._to_float(params.get("fe_thickness"), default=1.0)
            fe_power = self._to_float(params.get("fe_power"), default=5.0)
            al2o3 = self._to_float(params.get("al2o3_thickness"), default=10.0)
            al2o3_power = self._to_float(params.get("al2o3_power"), default=200.0)
            features.update(
                {
                    "fe_thickness": fe,
                    "fe_power": fe_power,
                    "al2o3_thickness": al2o3,
                    "al2o3_power": al2o3_power,
                    "ar_flow": ar,
                    "h2_flow": h2,
                    "c2h4_flow": c2h4,
                    "fe_thickness_sq": fe**2,
                    "catalyst_ratio": fe / al2o3 if al2o3 > 0 else 0.0,
                    "fe_power_loading": fe * fe_power,
                    "al2o3_power_loading": al2o3 * al2o3_power,
                }
            )
        return features

    def _latest_report_file(self, prefix: str, filename: str) -> Path:
        candidates = [
            directory / filename
            for directory in self.reports_root.iterdir()
            if directory.is_dir() and directory.name.startswith(prefix) and (directory / filename).exists()
        ]
        if not candidates:
            raise FileNotFoundError(f"未找到报表文件: prefix={prefix}, filename={filename}")
        return max(candidates, key=lambda path: path.stat().st_mtime)

    def _get_source_frame(self, source: str) -> pd.DataFrame:
        if source not in self.source_frames:
            self.source_frames[source] = self._load_source_frame(source)
        return self.source_frames[source]

    def _load_source_frame(self, source: str) -> pd.DataFrame:
        if source == "XR":
            return self._load_xr_frame()
        if source == "ZZY":
            return self._load_zzy_frame()
        raise ValueError(f"未知数据源: {source}")

    def _load_xr_frame(self) -> pd.DataFrame:
        summary_path = self._latest_report_file(XR_REPORT_PREFIX, "summary.csv")
        summary = pd.read_csv(summary_path)
        summary = summary[summary["status"].astype(str).str.lower().eq("success")].copy()

        conn = sqlite3.connect(self.db_path)
        meta = pd.read_sql_query(
            """
            SELECT
                id AS image_id,
                sample_id AS db_sample_id,
                growth_temp,
                actual_temp,
                growth_time,
                ar_flow,
                catalyst_weight,
                membrane_pos_cm,
                magnification
            FROM images
            WHERE source = 'XR' AND COALESCE(is_deleted, 0) = 0
            """,
            conn,
        )
        conn.close()

        df = summary.merge(meta, on="image_id", how="left")
        numeric_cols = [
            "actual_temp",
            "growth_time",
            "ar_flow",
            "catalyst_weight",
            "membrane_pos_cm",
            "magnification",
            "density",
            "alignment",
            "diameter_mean_nm",
            "l2_curvature_trimmed_mean_sqrt_length_nm",
            "l2_waviness_ratio_v2",
            "l2_tortuosity_v2",
            "junction_ratio",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        frame = pd.DataFrame(
            {
                "source": "XR",
                "image_id": df["image_id"],
                "sample_id": df["sample_id"],
                "file_name": df["file_name"],
                "file_path": df["file_path"],
                "actual_temp": df["actual_temp"],
                "growth_time": df["growth_time"],
                "ar_flow": df["ar_flow"],
                "catalyst_weight": df["catalyst_weight"],
                "membrane_pos_cm": df["membrane_pos_cm"],
                "magnification": df["magnification"],
                "diameter": df["diameter_mean_nm"],
                "density": df["density"],
                "alignment": df["alignment"],
                "curvature_proxy": df["l2_curvature_trimmed_mean_sqrt_length_nm"],
                "waviness_proxy": df["l2_waviness_ratio_v2"],
                "tortuosity_proxy": df["l2_tortuosity_v2"],
                "junction_ratio": pd.to_numeric(df.get("junction_ratio"), errors="coerce"),
            }
        )
        frame["dk_bend_index"] = frame["diameter"] * frame["curvature_proxy"]
        frame["bend_score"] = self._compute_bend_score(frame)
        return frame

    def _load_zzy_frame(self) -> pd.DataFrame:
        dataset_path = self._latest_report_file(ZZY_REPORT_PREFIX, "engineered_dataset_active.csv")
        eng = pd.read_csv(dataset_path)

        conn = sqlite3.connect(self.db_path)
        proc = pd.read_sql_query(
            """
            SELECT
                id AS image_id,
                fe_thickness,
                fe_power,
                al2o3_thickness,
                al2o3_power,
                c2h4_flow,
                ar_flow,
                h2_flow,
                actual_temp,
                growth_time,
                anneal_temp,
                anneal_time,
                magnification
            FROM images
            WHERE source='ZZY' AND COALESCE(is_deleted,0)=0
            """,
            conn,
        )
        conn.close()

        df = eng.merge(proc, on="image_id", how="left", suffixes=("", "_db"))
        numeric_cols = [
            "fe_thickness",
            "fe_power",
            "al2o3_thickness",
            "al2o3_power",
            "c2h4_flow",
            "ar_flow",
            "h2_flow",
            "actual_temp",
            "growth_time",
            "anneal_temp",
            "anneal_time",
            "magnification",
            "diameter",
            "density",
            "alignment",
            "curvature_nm_v3_trimmed_mean_sqrt_length",
            "waviness_ratio_v2",
            "tortuosity_v2",
            "junction_ratio",
            "dk_bend_index",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        frame = pd.DataFrame(
            {
                "source": "ZZY",
                "image_id": df["image_id"],
                "sample_id": df["sample_id"],
                "file_name": df["file_name"],
                "file_path": df["file_path"],
                "fe_thickness": df["fe_thickness"],
                "fe_power": df["fe_power"],
                "al2o3_thickness": df["al2o3_thickness"],
                "al2o3_power": df["al2o3_power"],
                "c2h4_flow": df.get("c2h4_flow_db", df.get("c2h4_flow")),
                "ar_flow": df.get("ar_flow_db", df.get("ar_flow")),
                "h2_flow": df.get("h2_flow_db", df.get("h2_flow")),
                "actual_temp": df.get("actual_temp_db", df.get("actual_temp")),
                "growth_time": df.get("growth_time_db", df.get("growth_time")),
                "anneal_temp": df.get("anneal_temp_db", df.get("anneal_temp")),
                "anneal_time": df.get("anneal_time_db", df.get("anneal_time")),
                "magnification": df.get("magnification_db", df.get("magnification")),
                "diameter": df["diameter"],
                "density": df["density"],
                "alignment": df["alignment"],
                "curvature_proxy": df["curvature_nm_v3_trimmed_mean_sqrt_length"],
                "waviness_proxy": df["waviness_ratio_v2"],
                "tortuosity_proxy": df["tortuosity_v2"],
                "junction_ratio": df["junction_ratio"],
                "dk_bend_index": df["dk_bend_index"],
            }
        )
        frame["bend_score"] = self._compute_bend_score(frame)
        return frame

    def _compute_bend_score(self, frame: pd.DataFrame) -> pd.Series:
        required = {
            "dk_bend_index": True,
            "curvature_proxy": True,
            "waviness_proxy": True,
            "tortuosity_proxy": True,
            "junction_ratio": True,
            "alignment": False,
        }
        scores = []
        for col, ascending in required.items():
            series = pd.to_numeric(frame[col], errors="coerce")
            rank = series.rank(method="average", pct=True, ascending=ascending)
            if col == "alignment":
                rank = 1.0 - rank
            scores.append(rank)
        return pd.concat(scores, axis=1).mean(axis=1, skipna=True)

    def _retrieve_similar_experiments(
        self,
        frame: pd.DataFrame,
        source: str,
        params: Dict,
        target_key: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        work = frame[pd.notna(frame[target_key])].copy()
        current_id = params.get("current_id")
        if current_id is not None and "image_id" in work.columns:
            work = work[work["image_id"] != current_id]
        if work.empty:
            return []

        process_cols = SOURCE_PROCESS_COLS[source]
        scales = {
            col: max(float(work[col].std(ddof=0) or 0.0), 1e-6)
            for col in process_cols
            if col in work.columns
        }

        def distance(row: pd.Series) -> float:
            total = 0.0
            used = 0
            for col in process_cols:
                param_value = self._to_float(params.get(col))
                if col == "membrane_pos_cm" and param_value is None:
                    param_value = self._to_float(params.get("inlet_distance_cm"))
                row_value = self._to_float(row.get(col))
                if param_value is None or row_value is None:
                    continue
                total += abs(param_value - row_value) / scales.get(col, 1.0)
                used += 1
            if used == 0:
                return float("inf")
            return total / used

        work["similarity_distance"] = work.apply(distance, axis=1)
        work = work[np.isfinite(work["similarity_distance"])].copy()
        work = work.sort_values("similarity_distance").head(top_k)

        result_cols = [col for col in ["image_id", "sample_id", "file_name", "file_path", target_key] if col in work.columns]
        result = []
        for row in work[result_cols + ["similarity_distance"]].to_dict(orient="records"):
            row["target_value"] = row.pop(target_key)
            result.append(row)
        return result

    def _compute_knowledge_baseline(
        self,
        similar_exps: List[Dict[str, Any]],
        source: str,
        target_key: str,
    ) -> float:
        if not similar_exps:
            return self._get_default_baseline(source, target_key)

        weights = []
        values = []
        for exp in similar_exps:
            value = self._to_float(exp.get("target_value"))
            if value is None:
                continue
            distance = max(float(exp.get("similarity_distance") or 0.0), 1e-6)
            weights.append(1.0 / distance)
            values.append(value)

        if not values:
            return self._get_default_baseline(source, target_key)

        weights_arr = np.asarray(weights, dtype=float)
        weights_arr = weights_arr / weights_arr.sum()
        return float(np.dot(weights_arr, np.asarray(values, dtype=float)))

    def _ensure_model_artifact(self, source: str, target_key: str) -> Optional[Dict[str, Any]]:
        if target_key not in self.model_artifacts[source]:
            artifact = self._train_target_model(source, target_key)
            if artifact is not None:
                self.model_artifacts[source][target_key] = artifact
                self.training_summaries[source][target_key] = {
                    "n_samples": float(artifact["n_samples"]),
                    "cv_r2": float(artifact["cv_r2"]),
                    "cv_mae": float(artifact["cv_mae"]),
                }
        return self.model_artifacts[source].get(target_key)

    def _train_target_model(self, source: str, target_key: str) -> Optional[Dict[str, Any]]:
        frame = self._get_source_frame(source).copy()
        feature_keys = XR_FEATURE_KEYS if source == "XR" else ZZY_FEATURE_KEYS
        work = frame[pd.notna(frame[target_key])].copy()
        if len(work) < 24:
            return None

        feature_rows = []
        baseline_values = []
        for idx, row in work.iterrows():
            params = {col: row.get(col) for col in SOURCE_PROCESS_COLS[source]}
            params["source"] = source
            feature_rows.append(self.build_knowledge_enhanced_features(params))

            other = work.drop(index=idx)
            similar = self._retrieve_similar_experiments(other, source, params, target_key, top_k=5)
            baseline_values.append(self._compute_knowledge_baseline(similar, source, target_key))

        feature_df = pd.DataFrame(feature_rows)
        for key in feature_keys:
            if key not in feature_df.columns:
                feature_df[key] = np.nan
        feature_df = feature_df[feature_keys]

        residual = pd.to_numeric(work[target_key], errors="coerce") - np.asarray(baseline_values, dtype=float)
        valid_mask = residual.notna()
        feature_df = feature_df.loc[valid_mask].reset_index(drop=True)
        residual = residual.loc[valid_mask].reset_index(drop=True)
        if len(feature_df) < 24:
            return None

        candidates = {
            "ridge": Pipeline(
                [
                    ("imp", SimpleImputer(strategy="median")),
                    ("sc", StandardScaler()),
                    ("model", Ridge(alpha=1.0)),
                ]
            ),
            "rf": Pipeline(
                [
                    ("imp", SimpleImputer(strategy="median")),
                    ("model", RandomForestRegressor(n_estimators=160, random_state=42)),
                ]
            ),
            "gb": Pipeline(
                [
                    ("imp", SimpleImputer(strategy="median")),
                    ("model", GradientBoostingRegressor(random_state=42)),
                ]
            ),
        }

        n_splits = min(5, max(3, len(feature_df) // 20))
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        best_name = None
        best_scores = None
        best_estimator = None

        for name, estimator in candidates.items():
            scores = cross_validate(
                estimator,
                feature_df,
                residual,
                cv=cv,
                scoring={"r2": "r2", "mae": "neg_mean_absolute_error"},
                error_score="raise",
            )
            mean_r2 = float(np.mean(scores["test_r2"]))
            mean_mae = float(-np.mean(scores["test_mae"]))
            if best_scores is None or mean_r2 > best_scores["r2"] or (
                np.isclose(mean_r2, best_scores["r2"]) and mean_mae < best_scores["mae"]
            ):
                best_name = name
                best_scores = {"r2": mean_r2, "mae": mean_mae}
                best_estimator = clone(estimator)

        if best_estimator is None or best_scores is None:
            return None

        best_estimator.fit(feature_df, residual)
        return {
            "model_name": best_name,
            "model": best_estimator,
            "feature_keys": feature_keys,
            "n_samples": int(len(feature_df)),
            "cv_r2": best_scores["r2"],
            "cv_mae": best_scores["mae"],
        }

    def _compute_confidence(
        self,
        similar_exps: List[Dict[str, Any]],
        rag_evidence: List[Dict[str, Any]],
        artifact: Optional[Dict[str, Any]],
        constraints: List[str],
    ) -> float:
        confidence = 0.35
        confidence += min(len(similar_exps) * 0.07, 0.28)
        confidence += min(len(rag_evidence) * 0.04, 0.12)
        if artifact is not None:
            confidence += max(min(float(artifact["cv_r2"]), 0.20), -0.05)
        if constraints:
            confidence = min(confidence, 0.35)
        return float(max(0.1, min(confidence, 0.98)))

    def _get_default_baseline(self, source: str, target_key: str) -> float:
        frame = self._get_source_frame(source)
        series = pd.to_numeric(frame[target_key], errors="coerce").dropna()
        if not series.empty:
            return float(series.median())
        return 0.0

    def _get_conservative_estimate(
        self,
        source: str,
        target_key: str,
        constraints: List[str],
    ) -> PredictionResult:
        baseline = self._get_default_baseline(source, target_key)
        return PredictionResult(
            predicted_value=float(baseline),
            confidence=0.3,
            knowledge_baseline=float(baseline),
            ml_residual=0.0,
            similar_experiments=[],
            rag_evidence=[],
            physical_constraints=constraints,
        )

    def _build_prediction_query(self, source: str, target_key: str) -> str:
        if source == "XR":
            return (
                f"CNT {source} morphology prediction {target_key} "
                f"temperature catalyst flow alignment density curvature"
            )
        return (
            f"CNT {source} morphology prediction {target_key} "
            f"Fe thickness catalyst flow waviness tortuosity curvature"
        )

    def predict(
        self,
        params: Dict,
        target: str = "diameter",
        query: Optional[str] = None,
        use_knowledge: bool = True,
    ) -> PredictionResult:
        source = str(params.get("source") or "ZZY").upper()
        if source not in {"XR", "ZZY"}:
            raise ValueError(f"不支持的数据源: {source}")

        target_key = self._resolve_target(target)
        constraints = self.check_physical_constraints(params)
        if constraints:
            return self._get_conservative_estimate(source, target_key, constraints)

        frame = self._get_source_frame(source)
        similar_exps = self._retrieve_similar_experiments(frame, source, params, target_key, top_k=5)
        knowledge_baseline = self._compute_knowledge_baseline(similar_exps, source, target_key)

        artifact = self._ensure_model_artifact(source, target_key)
        ml_residual = 0.0
        if use_knowledge and artifact is not None:
            features = self.build_knowledge_enhanced_features({"source": source, **params})
            feature_vector = pd.DataFrame([{key: features.get(key) for key in artifact["feature_keys"]}])
            ml_residual = float(artifact["model"].predict(feature_vector)[0])

        predicted_value = knowledge_baseline + ml_residual
        query_text = query or self._build_prediction_query(source, target_key)
        rag_evidence = self.rag.retrieve_from_pdf(
            query_text,
            top_k=3,
            task_name="prediction_explanation",
        )
        confidence = self._compute_confidence(similar_exps, rag_evidence, artifact, constraints)

        return PredictionResult(
            predicted_value=float(predicted_value),
            confidence=float(confidence),
            knowledge_baseline=float(knowledge_baseline),
            ml_residual=float(ml_residual),
            similar_experiments=similar_exps,
            rag_evidence=rag_evidence,
            physical_constraints=[],
        )

    def train_models(self, source: Optional[str] = None):
        sources = [source.upper()] if source else ["XR", "ZZY"]
        for src in sources:
            if src not in {"XR", "ZZY"}:
                continue
            for requested_target in SUPPORTED_TARGETS:
                target_key = self._resolve_target(requested_target)
                artifact = self._train_target_model(src, target_key)
                if artifact is None:
                    continue
                self.model_artifacts[src][target_key] = artifact
                self.training_summaries[src][target_key] = {
                    "n_samples": float(artifact["n_samples"]),
                    "cv_r2": float(artifact["cv_r2"]),
                    "cv_mae": float(artifact["cv_mae"]),
                }
                print(
                    f"{src}:{target_key} "
                    f"n={artifact['n_samples']} cv_r2={artifact['cv_r2']:.3f} cv_mae={artifact['cv_mae']:.4f}"
                )

    def batch_predict(
        self,
        params_list: List[Dict],
        target: str = "diameter",
        query: Optional[str] = None,
    ) -> List[PredictionResult]:
        return [self.predict(params, target, query) for params in params_list]
