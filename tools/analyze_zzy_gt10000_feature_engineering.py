from __future__ import annotations

import csv
import json
import math
import os
import sqlite3
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_ROOT = PROJECT_ROOT / "reports"
DB_PATH = PROJECT_ROOT / "database" / "cnta_experiments.sqlite"
SUMMARY_PATH = (
    REPORTS_ROOT / "zzy_feature_panels_cldice_20260331_gt10000_with_junction" / "summary.json"
)

FEATURE_QUALITY_KEYS = [
    "density",
    "alignment",
    "diameter",
    "curvature_nm_v3",
    "curvature_nm_v3_mean_sqrt_length",
    "curvature_nm_v3_trimmed_mean_sqrt_length",
    "dk_bend_index",
    "surface_strain_proxy",
    "tortuosity_v2",
    "waviness_ratio_v2",
    "n_branches",
    "junction_count",
    "junction_ratio",
    "skeleton_length_um",
    "junctions_per_100um",
    "branches_per_100um",
    "junction_to_branch_ratio",
    "removed_short_component_count",
    "removed_spur_count",
    "spur_per_branch",
    "short_component_per_branch",
]

CORE_FEATURES = [
    "density",
    "alignment",
    "diameter",
    "curvature_nm_v3",
    "curvature_nm_v3_trimmed_mean_sqrt_length",
    "dk_bend_index",
    "tortuosity_v2",
    "waviness_ratio_v2",
    "n_branches",
    "junction_ratio",
    "junctions_per_100um",
    "branches_per_100um",
    "junction_to_branch_ratio",
]

GROUP_MEAN_FEATURES = [
    "density",
    "alignment",
    "diameter",
    "curvature_nm_v3",
    "dk_bend_index",
    "tortuosity_v2",
    "waviness_ratio_v2",
    "n_branches",
    "junction_ratio",
    "junctions_per_100um",
]

DK_COMPARISON_FEATURES = [
    "alignment",
    "tortuosity_v2",
    "waviness_ratio_v2",
    "junction_ratio",
    "density",
    "n_branches",
]


@dataclass
class JoinedRow:
    image_id: int
    file_path: str
    file_name: str
    sample_id: str | None
    sample_no: str | None
    magnification: int | None
    magnification_bucket: str | None
    position: str | None
    c2h4_flow: float | None
    ar_flow: float | None
    h2_flow: float | None
    gas_condition: str
    gas_total_flow: float | None
    gas_level: str | None
    features: Dict[str, Any]


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and value is not None and not isinstance(value, bool)


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text or text.lower() in {"none", "nan", "null", "na", "n/a"}:
        return None
    return float(text)


def mean(values: Iterable[float]) -> float | None:
    values = list(values)
    if not values:
        return None
    return statistics.fmean(values)


def stdev(values: Iterable[float]) -> float | None:
    values = list(values)
    if len(values) < 2:
        return None
    return statistics.pstdev(values)


def pearson(xs: List[float], ys: List[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    mx = statistics.fmean(xs)
    my = statistics.fmean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    if den_x == 0.0 or den_y == 0.0:
        return None
    return num / den_x / den_y


def rankdata(values: List[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = rank
        i = j + 1
    return ranks


def spearman(xs: List[float], ys: List[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    return pearson(rankdata(xs), rankdata(ys))


def safe_ratio(numerator: Any, denominator: Any, scale: float = 1.0) -> float | None:
    num = to_float(numerator)
    den = to_float(denominator)
    if num is None or den is None or den == 0.0:
        return None
    return num / den * scale


def safe_product(left: Any, right: Any, scale: float = 1.0) -> float | None:
    left_value = to_float(left)
    right_value = to_float(right)
    if left_value is None or right_value is None:
        return None
    return left_value * right_value * scale


def format_flow(value: float | None) -> str:
    if value is None:
        return "NA"
    if float(value).is_integer():
        return str(int(value))
    return f"{value:g}"


def load_summary_rows(summary_path: Path) -> List[dict]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    return [row for row in payload.get("rows", []) if str(row.get("status") or "").lower() == "ok"]


def fetch_candidate_rows(db_path: Path) -> List[sqlite3.Row]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    rows = cur.execute(
        """
        SELECT
            id,
            file_path,
            sample_id,
            magnification,
            position_label,
            c2h4_flow,
            ar_flow,
            h2_flow,
            COALESCE(is_deleted, 0) AS is_deleted
        FROM images
        WHERE source = 'ZZY'
          AND magnification > 10000
        ORDER BY id
        """
    ).fetchall()
    conn.close()
    return rows


def build_gas_level_map(rows: List[sqlite3.Row]) -> Dict[float, str]:
    totals = sorted(
        {
            float((row["c2h4_flow"] or 0) + (row["ar_flow"] or 0) + (row["h2_flow"] or 0))
            for row in rows
            if int(row["is_deleted"] or 0) == 0
        }
    )
    if len(totals) == 3:
        return {totals[0]: "low", totals[1]: "mid", totals[2]: "high"}
    return {value: f"level_{idx + 1}" for idx, value in enumerate(totals)}


def align_rows(candidate_rows: List[sqlite3.Row], summary_rows: List[dict]) -> List[Tuple[sqlite3.Row, dict]]:
    pairs: List[Tuple[sqlite3.Row, dict]] = []
    db_index = 0
    summary_index = 0

    while db_index < len(candidate_rows) and summary_index < len(summary_rows):
        db_row = candidate_rows[db_index]
        summary_row = summary_rows[summary_index]
        db_name = Path(str(db_row["file_path"])).name.lower()
        summary_name = str(summary_row["file_name"]).lower()

        if db_name == summary_name:
            pairs.append((db_row, summary_row))
            db_index += 1
            summary_index += 1
            continue

        if int(db_row["is_deleted"] or 0) == 1:
            db_index += 1
            continue

        raise RuntimeError(
            "Summary alignment failed at "
            f"db id={db_row['id']} name='{Path(str(db_row['file_path'])).name}' "
            f"summary='{summary_row['file_name']}'"
        )

    while db_index < len(candidate_rows):
        if int(candidate_rows[db_index]["is_deleted"] or 0) == 0:
            raise RuntimeError(
                f"Unmatched active database row remains: id={candidate_rows[db_index]['id']}"
            )
        db_index += 1

    if summary_index != len(summary_rows):
        raise RuntimeError(f"Unmatched summary rows remain: {len(summary_rows) - summary_index}")

    return pairs


def build_joined_rows(candidate_rows: List[sqlite3.Row], summary_rows: List[dict]) -> List[JoinedRow]:
    gas_level_map = build_gas_level_map(candidate_rows)
    aligned = align_rows(candidate_rows, summary_rows)
    joined: List[JoinedRow] = []

    for db_row, summary_row in aligned:
        if int(db_row["is_deleted"] or 0) != 0:
            continue

        c2h4 = to_float(db_row["c2h4_flow"])
        ar = to_float(db_row["ar_flow"])
        h2 = to_float(db_row["h2_flow"])
        gas_total = float((c2h4 or 0) + (ar or 0) + (h2 or 0))
        features = dict(summary_row)
        features["branches_per_100um"] = safe_ratio(
            features.get("n_branches"),
            features.get("skeleton_length_um"),
            scale=100.0,
        )
        features["junction_to_branch_ratio"] = safe_ratio(
            features.get("junction_count"),
            features.get("n_branches"),
        )
        features["dk_bend_index"] = safe_product(
            features.get("diameter"),
            features.get("curvature_nm_v3_trimmed_mean_sqrt_length"),
        )
        features["surface_strain_proxy"] = safe_product(
            features.get("diameter"),
            features.get("curvature_nm_v3_trimmed_mean_sqrt_length"),
            scale=0.5,
        )
        features["spur_per_branch"] = safe_ratio(
            features.get("removed_spur_count"),
            features.get("n_branches"),
        )
        features["short_component_per_branch"] = safe_ratio(
            features.get("removed_short_component_count"),
            features.get("n_branches"),
        )

        magnification = int(db_row["magnification"]) if db_row["magnification"] is not None else None
        joined.append(
            JoinedRow(
                image_id=int(db_row["id"]),
                file_path=str(db_row["file_path"]),
                file_name=Path(str(db_row["file_path"])).name,
                sample_id=str(db_row["sample_id"]) if db_row["sample_id"] is not None else None,
                sample_no=(str(db_row["sample_id"]).split("-")[0] if db_row["sample_id"] else None),
                magnification=magnification,
                magnification_bucket=("50k" if magnification is not None and magnification < 100000 else "100k"),
                position=str(db_row["position_label"]) if db_row["position_label"] is not None else None,
                c2h4_flow=c2h4,
                ar_flow=ar,
                h2_flow=h2,
                gas_condition="/".join([format_flow(c2h4), format_flow(ar), format_flow(h2)]),
                gas_total_flow=gas_total,
                gas_level=gas_level_map.get(gas_total),
                features=features,
            )
        )

    return joined


def flatten_row(row: JoinedRow) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "image_id": row.image_id,
        "file_path": row.file_path,
        "file_name": row.file_name,
        "sample_id": row.sample_id,
        "sample_no": row.sample_no,
        "magnification": row.magnification,
        "magnification_bucket": row.magnification_bucket,
        "position": row.position,
        "c2h4_flow": row.c2h4_flow,
        "ar_flow": row.ar_flow,
        "h2_flow": row.h2_flow,
        "gas_condition": row.gas_condition,
        "gas_total_flow": row.gas_total_flow,
        "gas_level": row.gas_level,
    }
    for key, value in row.features.items():
        payload[key] = value
    return payload


def feature_quality(rows: List[JoinedRow], keys: List[str]) -> List[dict]:
    total = len(rows)
    output: List[dict] = []
    for key in keys:
        values = [to_float(row.features.get(key)) for row in rows if to_float(row.features.get(key)) is not None]
        output.append(
            {
                "feature": key,
                "non_null_count": len(values),
                "missing_pct": round((1.0 - len(values) / max(total, 1)) * 100.0, 2),
                "mean": round(mean(values), 6) if values else None,
                "std": round(stdev(values), 6) if len(values) > 1 else None,
                "min": round(min(values), 6) if values else None,
                "max": round(max(values), 6) if values else None,
            }
        )
    return output


def grouped_means(rows: List[JoinedRow], attr: str, features: List[str]) -> List[dict]:
    grouped: Dict[str, List[JoinedRow]] = defaultdict(list)
    for row in rows:
        value = getattr(row, attr)
        grouped[str(value)].append(row)

    output: List[dict] = []
    for group_name in sorted(grouped):
        subset = grouped[group_name]
        result: Dict[str, Any] = {"group": group_name, "count": len(subset)}
        for feature in features:
            values = [to_float(row.features.get(feature)) for row in subset if to_float(row.features.get(feature)) is not None]
            result[f"{feature}_mean"] = round(mean(values), 6) if values else None
        output.append(result)
    return output


def meta_correlations(rows: List[JoinedRow], label: str, extractor, features: List[str]) -> List[dict]:
    output: List[dict] = []
    for feature in features:
        pairs = []
        for row in rows:
            meta_value = extractor(row)
            feature_value = to_float(row.features.get(feature))
            if meta_value is None or feature_value is None:
                continue
            pairs.append((float(meta_value), float(feature_value)))
        xs = [item[0] for item in pairs]
        ys = [item[1] for item in pairs]
        output.append(
            {
                "meta": label,
                "feature": feature,
                "n": len(pairs),
                "pearson": round(pearson(xs, ys), 6) if pearson(xs, ys) is not None else None,
                "spearman": round(spearman(xs, ys), 6) if spearman(xs, ys) is not None else None,
            }
        )
    return output


def top_correlation_pairs(rows: List[JoinedRow], features: List[str], limit: int = 20) -> List[dict]:
    output: List[dict] = []
    for idx, left in enumerate(features):
        for right in features[idx + 1 :]:
            pairs = []
            for row in rows:
                left_value = to_float(row.features.get(left))
                right_value = to_float(row.features.get(right))
                if left_value is None or right_value is None:
                    continue
                pairs.append((left_value, right_value))
            xs = [item[0] for item in pairs]
            ys = [item[1] for item in pairs]
            corr = pearson(xs, ys)
            if corr is None:
                continue
            output.append(
                {
                    "left": left,
                    "right": right,
                    "n": len(pairs),
                    "corr": round(corr, 6),
                    "abs_corr": round(abs(corr), 6),
                }
            )
    output.sort(key=lambda item: item["abs_corr"], reverse=True)
    return output[:limit]


def feature_selection_recommendations() -> List[dict]:
    rows: List[dict] = []

    def add(feature: str, role: str, recommendation: str, preprocess: str, reason: str) -> None:
        rows.append(
            {
                "feature": feature,
                "role": role,
                "recommendation": recommendation,
                "preprocess": preprocess,
                "reason": reason,
            }
        )

    add("density", "core_feature", "keep", "robust_scale_optional", "Coverage dimension is relatively independent and stable.")
    add("diameter", "core_feature", "keep", "winsorize_or_robust_scale", "Size dimension is informative but right-skewed.")
    add("alignment", "core_feature", "keep", "none_or_robust_scale", "Orientation signal is strong and interpretable.")
    add("n_branches", "core_feature", "keep", "log1p_then_scale", "Adds topology complexity with moderate overlap to the core morphology set.")
    add("dk_bend_index", "derived_feature", "keep", "robust_scale", "Dimensionless bend proxy with good physical meaning and much weaker magnification coupling than diameter or curvature alone.")
    add("magnification_bucket", "control_variable", "keep_as_control", "one_hot", "Strong confounder; control for it rather than treating it as morphology.")
    add("gas_level", "control_variable", "keep_as_control", "one_hot", "Use the three total-flow tiers instead of separate gas columns.")
    add("sample_no", "group_split_key", "keep_for_split_only", "group_kfold_key", "Avoid leakage between the same family during validation.")
    add("curvature_nm_v3_trimmed_mean_sqrt_length", "redundant_shape_feature", "drop_from_core", "optional_extended_only", "Useful on its own, but overlaps strongly with d*k and orientation/shape features.")
    add("curvature_nm_v3", "redundant_shape_feature", "drop", "none", "Less robust than the trimmed-length-weighted curvature summary.")
    add("tortuosity_v2", "redundant_shape_feature", "drop_from_core", "optional_extended_only", "Highly collinear with waviness and correlated with alignment and junction topology.")
    add("waviness_ratio_v2", "redundant_shape_feature", "drop", "none", "Nearly duplicates tortuosity_v2.")
    add("junction_ratio", "redundant_topology_feature", "drop_from_core", "optional_extended_only", "Useful for topology interpretation but overlaps with d*k, tortuosity, and alignment.")
    add("junctions_per_100um", "confounded_topology_feature", "drop", "none", "Strongly driven by magnification and highly correlated with curvature.")
    add("branches_per_100um", "confounded_topology_feature", "drop", "none", "Strongly driven by magnification and redundant with branch count.")
    add("junction_to_branch_ratio", "redundant_topology_feature", "drop", "none", "Built from already available topology counts and correlated with junction_ratio.")
    add("c2h4_flow", "meta", "drop", "none", "Do not split gases into separate variables for this dataset.")
    add("ar_flow", "meta", "drop", "none", "Do not split gases into separate variables for this dataset.")
    add("h2_flow", "meta", "drop", "none", "Do not split gases into separate variables for this dataset.")
    add("gas_total_flow", "meta", "drop", "none", "Gas level already captures the intended tiering.")
    add("image_id", "identifier", "drop", "none", "Identifier only.")
    add("file_path", "identifier", "drop", "none", "Identifier only.")
    add("file_name", "identifier", "drop", "none", "Identifier only.")
    add("sample_id", "identifier", "drop", "none", "Identifier only.")
    add("status", "pipeline_state", "drop", "none", "Processing state is not a modeling feature.")
    add("hof_method", "pipeline_state", "drop", "none", "Pipeline metadata only.")
    add("diameter_method", "pipeline_state", "drop", "none", "Pipeline metadata only.")
    add("alignment_raw", "pipeline_diagnostic", "drop", "none", "Raw intermediate statistic.")
    add("mean_phi_raw_deg", "pipeline_diagnostic", "drop", "none", "Raw intermediate statistic.")
    add("rotation_correction_deg", "pipeline_diagnostic", "drop", "none", "Preprocessing metadata.")
    add("removed_short_component_count", "diagnostic", "drop", "none", "Cleanup diagnostic rather than morphology target.")
    add("removed_spur_count", "diagnostic", "drop", "none", "Cleanup diagnostic rather than morphology target.")
    add("spur_per_branch", "diagnostic", "drop", "none", "Derived from cleanup counters and branch count.")
    add("short_component_per_branch", "diagnostic", "drop", "none", "Derived from cleanup counters and branch count.")
    return rows


def dk_correlations(rows: List[JoinedRow]) -> List[dict]:
    output: List[dict] = []
    gas_level_order = {"low": 1, "mid": 2, "high": 3}
    for feature in DK_COMPARISON_FEATURES:
        pairs = []
        for row in rows:
            dk = to_float(row.features.get("dk_bend_index"))
            value = to_float(row.features.get(feature))
            if dk is None or value is None:
                continue
            pairs.append((dk, value))
        xs = [item[0] for item in pairs]
        ys = [item[1] for item in pairs]
        pearson_value = pearson(xs, ys)
        spearman_value = spearman(xs, ys)
        output.append(
            {
                "feature": feature,
                "n": len(pairs),
                "pearson": round(pearson_value, 6) if pearson_value is not None else None,
                "spearman": round(spearman_value, 6) if spearman_value is not None else None,
            }
        )

    for meta_name, extractor in [
        ("magnification_bucket", lambda row: 1 if row.magnification_bucket == "50k" else 2),
        ("gas_level", lambda row: gas_level_order.get(row.gas_level)),
    ]:
        pairs = []
        for row in rows:
            dk = to_float(row.features.get("dk_bend_index"))
            meta = extractor(row)
            if dk is None or meta is None:
                continue
            pairs.append((dk, float(meta)))
        xs = [item[0] for item in pairs]
        ys = [item[1] for item in pairs]
        pearson_value = pearson(xs, ys)
        spearman_value = spearman(xs, ys)
        output.append(
            {
                "feature": meta_name,
                "n": len(pairs),
                "pearson": round(pearson_value, 6) if pearson_value is not None else None,
                "spearman": round(spearman_value, 6) if spearman_value is not None else None,
            }
        )

    return output


def family_gas_coverage(rows: List[JoinedRow]) -> List[dict]:
    levels = sorted({row.gas_level for row in rows if row.gas_level is not None})
    counter: Dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        if row.sample_no and row.gas_level:
            counter[row.sample_no][row.gas_level] += 1
    output: List[dict] = []
    for sample_no in sorted(counter):
        result: Dict[str, Any] = {"sample_no": sample_no}
        for level in levels:
            result[level] = counter[sample_no][level]
        output.append(result)
    return output


def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def render_markdown(
    rows: List[JoinedRow],
    quality_rows: List[dict],
    gas_group_rows: List[dict],
    mag_group_rows: List[dict],
    gas_corr_rows: List[dict],
    mag_corr_rows: List[dict],
    top_pairs: List[dict],
    family_rows: List[dict],
    dk_corr_rows: List[dict],
    selection_rows: List[dict],
) -> str:
    sample_counts = Counter(row.sample_no for row in rows if row.sample_no)
    gas_counts = Counter(row.gas_condition for row in rows)
    mag_counts = Counter(row.magnification for row in rows if row.magnification is not None)
    position_counts = Counter(row.position or "unknown" for row in rows)

    lines: List[str] = []
    lines.append("# ZZY >10000X 特征工程与相关性分析")
    lines.append("")
    lines.append(f"- 生成时间: `{datetime.now().isoformat(timespec='seconds')}`")
    lines.append("- 数据口径: `source='ZZY' AND is_deleted=0 AND magnification>10000`")
    lines.append(f"- 样本总数: `{len(rows)}`")
    lines.append("- 气体条件分组原则: 不拆分三种气体，而是按同一比例下的总流量档位分成 `low / mid / high`。")
    lines.append("")
    lines.append("## 数据情况")
    lines.append("")
    lines.append(f"- 样本组分布: `{dict(sorted(sample_counts.items()))}`")
    lines.append(f"- 倍率分布: `{dict(sorted(mag_counts.items()))}`")
    lines.append(f"- 位置分布: `{dict(position_counts)}`")
    lines.append(f"- 气体条件分布: `{dict(sorted(gas_counts.items()))}`")
    lines.append("")
    lines.append("## 特征工程")
    lines.append("")
    lines.append("- 基础特征: `density`, `alignment`, `diameter`, `curvature_nm_v3`, `tortuosity_v2`, `waviness_ratio_v2`, `n_branches`。")
    lines.append("- 拓扑特征: `junction_count`, `junction_ratio`, `skeleton_length_um`, `junctions_per_100um`。")
    lines.append("- 派生归一化特征: `branches_per_100um`, `junction_to_branch_ratio`, `spur_per_branch`, `short_component_per_branch`。")
    lines.append("- 新增弯曲代理特征: `dk_bend_index = diameter * curvature_nm_v3_trimmed_mean_sqrt_length`。")
    lines.append("- 力学解释辅助量: `surface_strain_proxy = 0.5 * d*k`，更接近表面弯曲应变代理。")
    lines.append("- 分层变量: `gas_level`, `magnification_bucket`, `sample_no`。")
    lines.append("")
    lines.append("## 特征覆盖")
    lines.append("")
    lines.append("| feature | non_null | missing_pct | mean | std | min | max |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for item in quality_rows:
        lines.append(
            f"| {item['feature']} | {item['non_null_count']} | {item['missing_pct']} | "
            f"{item['mean']} | {item['std']} | {item['min']} | {item['max']} |"
        )
    lines.append("")
    lines.append("## 三档均值")
    lines.append("")
    lines.append("| gas_level | count | density | alignment | diameter | curvature_nm_v3 | dk_bend_index | tortuosity_v2 | waviness_ratio_v2 | n_branches | junction_ratio | junctions_per_100um |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for item in gas_group_rows:
        lines.append(
            f"| {item['group']} | {item['count']} | {item['density_mean']} | {item['alignment_mean']} | "
            f"{item['diameter_mean']} | {item['curvature_nm_v3_mean']} | {item['dk_bend_index_mean']} | {item['tortuosity_v2_mean']} | "
            f"{item['waviness_ratio_v2_mean']} | {item['n_branches_mean']} | {item['junction_ratio_mean']} | "
            f"{item['junctions_per_100um_mean']} |"
        )
    lines.append("")
    lines.append("## 倍率均值")
    lines.append("")
    lines.append("| magnification_bucket | count | density | alignment | diameter | curvature_nm_v3 | dk_bend_index | tortuosity_v2 | waviness_ratio_v2 | n_branches | junction_ratio | junctions_per_100um |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for item in mag_group_rows:
        lines.append(
            f"| {item['group']} | {item['count']} | {item['density_mean']} | {item['alignment_mean']} | "
            f"{item['diameter_mean']} | {item['curvature_nm_v3_mean']} | {item['dk_bend_index_mean']} | {item['tortuosity_v2_mean']} | "
            f"{item['waviness_ratio_v2_mean']} | {item['n_branches_mean']} | {item['junction_ratio_mean']} | "
            f"{item['junctions_per_100um_mean']} |"
        )
    lines.append("")
    lines.append("## d*k 相关性")
    lines.append("")
    lines.append("| feature | n | pearson | spearman |")
    lines.append("| --- | ---: | ---: | ---: |")
    for item in dk_corr_rows:
        lines.append(f"| {item['feature']} | {item['n']} | {item['pearson']} | {item['spearman']} |")
    lines.append("")
    lines.append("## 元数据相关性")
    lines.append("")
    lines.append("| meta | feature | n | pearson | spearman |")
    lines.append("| --- | --- | ---: | ---: | ---: |")
    for item in gas_corr_rows + mag_corr_rows:
        lines.append(
            f"| {item['meta']} | {item['feature']} | {item['n']} | {item['pearson']} | {item['spearman']} |"
        )
    lines.append("")
    lines.append("## 冗余特征")
    lines.append("")
    lines.append("| left | right | corr |")
    lines.append("| --- | --- | ---: |")
    for item in top_pairs[:15]:
        lines.append(f"| {item['left']} | {item['right']} | {item['corr']} |")
    lines.append("")
    lines.append("## 家族覆盖")
    lines.append("")
    levels = [key for key in family_rows[0].keys() if key != "sample_no"] if family_rows else []
    header = "| sample_no | " + " | ".join(levels) + " |"
    align = "| --- | " + " | ".join("---:" for _ in levels) + " |"
    lines.append(header)
    lines.append(align)
    for item in family_rows:
        values = " | ".join(str(item[level]) for level in levels)
        lines.append(f"| {item['sample_no']} | {values} |")
    lines.append("")
    lines.append("## 结论建议")
    lines.append("")
    lines.append("- 这批数据最强的混杂因素不是气体三档，而是 `magnification`。")
    lines.append("- `tortuosity_v2` 与 `waviness_ratio_v2` 高度冗余，建模时二选一即可。")
    lines.append("- `curvature_nm_v3` 与 `junctions_per_100um`、`alignment` 与弯曲/波动特征之间相关性很强，说明拓扑复杂度和形貌弯曲在这批样本里联动明显。")
    lines.append("- `dk_bend_index` 与 `alignment` 负相关、与 `tortuosity_v2 / waviness_ratio_v2 / junction_ratio` 正相关，能作为有物理意义的弯曲强度代理。")
    lines.append("- `dk_bend_index` 相比单独的 `diameter` 或 `curvature`，对倍率的耦合明显更弱，适合进入首版建模特征集。")
    lines.append("- 后续做建模或相关性解释，建议优先分层控制 `magnification_bucket`，再讨论 `gas_level` 的影响。")
    lines.append("")
    lines.append("## 正式建模前筛特征")
    lines.append("")
    lines.append("| feature | role | recommendation | preprocess | reason |")
    lines.append("| --- | --- | --- | --- | --- |")
    for item in selection_rows:
        lines.append(
            f"| {item['feature']} | {item['role']} | {item['recommendation']} | "
            f"{item['preprocess']} | {item['reason']} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    candidate_rows = fetch_candidate_rows(DB_PATH)
    summary_rows = load_summary_rows(SUMMARY_PATH)
    rows = build_joined_rows(candidate_rows, summary_rows)

    quality_rows = feature_quality(rows, FEATURE_QUALITY_KEYS)
    gas_group_rows = grouped_means(rows, "gas_level", GROUP_MEAN_FEATURES)
    mag_group_rows = grouped_means(rows, "magnification_bucket", GROUP_MEAN_FEATURES)
    gas_level_order = {"low": 1, "mid": 2, "high": 3}
    gas_corr_rows = meta_correlations(
        rows,
        "gas_level",
        extractor=lambda row: gas_level_order.get(row.gas_level),
        features=CORE_FEATURES,
    )
    mag_corr_rows = meta_correlations(
        rows,
        "magnification_bucket",
        extractor=lambda row: 1 if row.magnification_bucket == "50k" else 2,
        features=CORE_FEATURES,
    )
    top_pairs = top_correlation_pairs(rows, CORE_FEATURES)
    family_rows = family_gas_coverage(rows)
    dk_corr_rows = dk_correlations(rows)
    selection_rows = feature_selection_recommendations()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = REPORTS_ROOT / f"zzy_feature_engineering_gt10000_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    flattened_rows = [flatten_row(row) for row in rows]
    write_csv(out_dir / "engineered_dataset_active.csv", flattened_rows)
    write_csv(out_dir / "feature_quality.csv", quality_rows)
    write_csv(out_dir / "group_means_by_gas_level.csv", gas_group_rows)
    write_csv(out_dir / "group_means_by_magnification.csv", mag_group_rows)
    write_csv(out_dir / "meta_correlations.csv", gas_corr_rows + mag_corr_rows)
    write_csv(out_dir / "top_correlation_pairs.csv", top_pairs)
    write_csv(out_dir / "family_gas_coverage.csv", family_rows)
    write_csv(out_dir / "dk_correlations.csv", dk_corr_rows)
    write_csv(out_dir / "feature_selection_recommendations.csv", selection_rows)
    core_columns = [
        "image_id",
        "sample_no",
        "gas_level",
        "magnification_bucket",
        "density",
        "diameter",
        "alignment",
        "n_branches",
        "dk_bend_index",
    ]
    write_csv(
        out_dir / "model_dataset_core.csv",
        [{key: row.get(key) for key in core_columns} for row in flattened_rows],
    )

    report_path = out_dir / "report.md"
    report_path.write_text(
        render_markdown(
            rows=rows,
            quality_rows=quality_rows,
            gas_group_rows=gas_group_rows,
            mag_group_rows=mag_group_rows,
            gas_corr_rows=gas_corr_rows,
            mag_corr_rows=mag_corr_rows,
            top_pairs=top_pairs,
            family_rows=family_rows,
            dk_corr_rows=dk_corr_rows,
            selection_rows=selection_rows,
        ),
        encoding="utf-8",
    )

    print(out_dir)
    print(report_path)


if __name__ == "__main__":
    main()
