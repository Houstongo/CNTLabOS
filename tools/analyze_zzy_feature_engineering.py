from __future__ import annotations

import csv
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_ROOT = PROJECT_ROOT / "reports"
SUMMARY_PATHS = [
    REPORTS_ROOT / "zzy_feature_panels_cldice_20260330_202128" / "summary.json",
    REPORTS_ROOT / "zzy_feature_panels_cldice_20260331_003404" / "summary.json",
]

GAS_RATIO_RE = re.compile(r"\b\d+w\s+([0-9]+(?:\.[0-9]+)?)nm\b")
MAG_RE = re.compile(r"\b(10000|50000|100000)\b")
NO_RE = re.compile(r"^(No\d+)")
POSITION_RE = re.compile(r"\b(top|mid|bottom)\b", re.IGNORECASE)

NUMERIC_FEATURES = [
    "density",
    "alignment",
    "diameter",
    "curvature_nm_v3",
    "curvature_nm_v3_length",
    "curvature_nm_v3_mean_length",
    "curvature_nm_v3_trimmed_mean_length",
    "tortuosity_v2",
    "waviness_ratio_v2",
    "n_branches",
    "curvature_v3_branch_count",
    "removed_short_component_count",
    "removed_short_pixel_count",
    "removed_spur_count",
    "removed_spur_pixel_count",
]

CORE_MODEL_FEATURES = [
    "density",
    "alignment",
    "diameter",
    "curvature_nm_v3",
    "tortuosity_v2",
    "waviness_ratio_v2",
    "n_branches",
]


@dataclass
class EngineeredRow:
    file_name: str
    sample_no: str | None
    condition_value_proxy: float | None
    condition_tier_proxy: str | None
    magnification: int | None
    position: str | None
    status: str | None
    source_batch: str
    row: Dict[str, Any]


def choose_better_row(current: Dict[str, Any] | None, candidate: Dict[str, Any]) -> Dict[str, Any]:
    if current is None:
        return candidate
    current_rank = 0 if current.get("status") == "ok" else 1
    candidate_rank = 0 if candidate.get("status") == "ok" else 1
    if candidate_rank < current_rank:
        return candidate
    return current


def load_summary_rows() -> list[EngineeredRow]:
    by_name: dict[str, Dict[str, Any]] = {}
    source_batch_by_name: dict[str, str] = {}

    for path in SUMMARY_PATHS:
        payload = json.loads(path.read_text(encoding="utf-8"))
        deduped_in_batch: dict[str, Dict[str, Any]] = {}
        for row in payload["rows"]:
            name = str(row["file_name"])
            deduped_in_batch[name] = choose_better_row(deduped_in_batch.get(name), row)
        for name, row in deduped_in_batch.items():
            by_name[name] = row
            source_batch_by_name[name] = path.parent.name

    engineered: list[EngineeredRow] = []
    for name, row in sorted(by_name.items()):
        condition_value = parse_condition_value(name)
        engineered.append(
            EngineeredRow(
                file_name=name,
                sample_no=parse_sample_no(name),
                condition_value_proxy=condition_value,
                condition_tier_proxy=bin_condition_value(condition_value),
                magnification=parse_magnification(name),
                position=parse_position(name),
                status=row.get("status"),
                source_batch=source_batch_by_name[name],
                row=row,
            )
        )
    return engineered


def parse_condition_value(file_name: str) -> float | None:
    matches = GAS_RATIO_RE.findall(file_name)
    return float(matches[-1]) if matches else None


def parse_sample_no(file_name: str) -> str | None:
    match = NO_RE.match(file_name)
    return match.group(1) if match else None


def parse_magnification(file_name: str) -> int | None:
    matches = MAG_RE.findall(file_name)
    return int(matches[-1]) if matches else None


def parse_position(file_name: str) -> str | None:
    match = POSITION_RE.search(file_name)
    return match.group(1).lower() if match else None


def bin_condition_value(value: float | None) -> str | None:
    if value is None:
        return None
    if value <= 1.0:
        return "low"
    if value < 2.0:
        return "mid"
    return "high"


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and value is not None and not isinstance(value, bool)


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


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    mx = statistics.fmean(xs)
    my = statistics.fmean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    if den_x == 0.0 or den_y == 0.0:
        return None
    return num / (den_x * den_y)


def rankdata(values: list[float]) -> list[float]:
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


def spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    return pearson(rankdata(xs), rankdata(ys))


def feature_quality(rows: list[EngineeredRow]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    total = len(rows)
    for key in NUMERIC_FEATURES:
        values = [float(r.row[key]) for r in rows if is_number(r.row.get(key))]
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


def tier_summary(rows: list[EngineeredRow], metrics: list[str]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for tier in ("low", "mid", "high"):
        subset = [r for r in rows if r.condition_tier_proxy == tier]
        row: dict[str, Any] = {"condition_tier_proxy": tier, "count": len(subset)}
        for metric in metrics:
            values = [float(r.row[metric]) for r in subset if is_number(r.row.get(metric))]
            row[f"{metric}_mean"] = round(mean(values), 6) if values else None
        output.append(row)
    return output


def feature_correlations(rows: list[EngineeredRow], metrics: list[str]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for metric in metrics:
        pairs = [
            (float(r.condition_value_proxy), float(r.row[metric]))
            for r in rows
            if r.condition_value_proxy is not None and is_number(r.row.get(metric))
        ]
        xs = [a for a, _ in pairs]
        ys = [b for _, b in pairs]
        output.append(
            {
                "feature": metric,
                "n": len(pairs),
                "pearson": round(pearson(xs, ys), 6) if pearson(xs, ys) is not None else None,
                "spearman": round(spearman(xs, ys), 6) if spearman(xs, ys) is not None else None,
            }
        )
    return output


def redundancy_pairs(rows: list[EngineeredRow], limit: int = 20) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for idx, left in enumerate(NUMERIC_FEATURES):
        for right in NUMERIC_FEATURES[idx + 1 :]:
            pairs = [
                (float(r.row[left]), float(r.row[right]))
                for r in rows
                if is_number(r.row.get(left)) and is_number(r.row.get(right))
            ]
            if len(pairs) < 8:
                continue
            xs = [a for a, _ in pairs]
            ys = [b for _, b in pairs]
            corr = pearson(xs, ys)
            if corr is None:
                continue
            results.append(
                {
                    "left": left,
                    "right": right,
                    "abs_corr": abs(corr),
                    "corr": corr,
                }
            )
    results.sort(key=lambda item: item["abs_corr"], reverse=True)
    return results[:limit]


def family_tier_counts(rows: list[EngineeredRow]) -> list[dict[str, Any]]:
    counter: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        if row.sample_no and row.condition_tier_proxy:
            counter[row.sample_no][row.condition_tier_proxy] += 1
    output: list[dict[str, Any]] = []
    for sample_no in sorted(counter):
        output.append(
            {
                "sample_no": sample_no,
                "low": counter[sample_no]["low"],
                "mid": counter[sample_no]["mid"],
                "high": counter[sample_no]["high"],
            }
        )
    return output


def write_engineered_csv(rows: list[EngineeredRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "file_name",
        "sample_no",
        "condition_value_proxy",
        "condition_tier_proxy",
        "magnification",
        "position",
        "status",
        "source_batch",
    ]
    numeric_keys = sorted({key for row in rows for key, value in row.row.items() if is_number(value)})
    fieldnames.extend(numeric_keys)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload = {
                "file_name": row.file_name,
                "sample_no": row.sample_no,
                "condition_value_proxy": row.condition_value_proxy,
                "condition_tier_proxy": row.condition_tier_proxy,
                "magnification": row.magnification,
                "position": row.position,
                "status": row.status,
                "source_batch": row.source_batch,
            }
            for key in numeric_keys:
                payload[key] = row.row.get(key)
            writer.writerow(payload)


def render_markdown(
    rows: list[EngineeredRow],
    quality: list[dict[str, Any]],
    tier_stats: list[dict[str, Any]],
    correlations: list[dict[str, Any]],
    redundancy: list[dict[str, Any]],
    family_counts: list[dict[str, Any]],
) -> str:
    total = len(rows)
    sample_families = sorted({row.sample_no for row in rows if row.sample_no})
    mag_counts = Counter(row.magnification for row in rows if row.magnification is not None)
    position_counts = Counter(row.position or "unknown" for row in rows)
    missing_junction = all("junction_count" not in row.row for row in rows)

    lines: list[str] = []
    lines.append("# ZZY Feature Engineering Report")
    lines.append("")
    lines.append(f"- Generated at: `{datetime.now().isoformat(timespec='seconds')}`")
    lines.append(f"- Rows after cross-batch dedup: `{total}`")
    lines.append(f"- Sample families: `{', '.join(sample_families)}`")
    lines.append(
        "- Source summaries: "
        + ", ".join(f"`{path.parent.name}`" for path in SUMMARY_PATHS)
    )
    lines.append("")
    lines.append("## Data Status")
    lines.append("")
    lines.append(f"- Magnification distribution: `{dict(sorted(mag_counts.items()))}`")
    lines.append(f"- Position distribution: `{dict(position_counts)}`")
    lines.append(
        "- Condition-tier proxy counts: "
        + f"`{ {item['condition_tier_proxy']: item['count'] for item in tier_stats} }`"
    )
    lines.append(
        "- Condition-tier proxy meaning: extracted from the trailing filename setpoint (`... 5w Xnm ...`), "
        + "used only as a grouped process-condition proxy, not as the true ethylene/argon/hydrogen volume ratio."
    )
    lines.append(
        "- Junction metrics in historical ZZY summaries: "
        + ("`missing`" if missing_junction else "`present`")
    )
    lines.append("")
    lines.append("## Feature Quality")
    lines.append("")
    lines.append("| feature | non_null | missing_pct | mean | std | min | max |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for item in quality:
        lines.append(
            f"| {item['feature']} | {item['non_null_count']} | {item['missing_pct']} | "
            f"{item['mean']} | {item['std']} | {item['min']} | {item['max']} |"
        )
    lines.append("")
    lines.append("## Tier Means")
    lines.append("")
    lines.append("| tier | count | density | alignment | diameter | curvature_nm_v3 | tortuosity_v2 | waviness_ratio_v2 | n_branches |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for item in tier_stats:
        lines.append(
            f"| {item['condition_tier_proxy']} | {item['count']} | {item['density_mean']} | "
            f"{item['alignment_mean']} | {item['diameter_mean']} | {item['curvature_nm_v3_mean']} | "
            f"{item['tortuosity_v2_mean']} | {item['waviness_ratio_v2_mean']} | {item['n_branches_mean']} |"
        )
    lines.append("")
    lines.append("## Condition Proxy Correlations")
    lines.append("")
    lines.append("| feature | n | pearson | spearman |")
    lines.append("| --- | ---: | ---: | ---: |")
    for item in correlations:
        lines.append(
            f"| {item['feature']} | {item['n']} | {item['pearson']} | {item['spearman']} |"
        )
    lines.append("")
    lines.append("## Redundant Features")
    lines.append("")
    lines.append("| left | right | corr |")
    lines.append("| --- | --- | ---: |")
    for item in redundancy:
        lines.append(f"| {item['left']} | {item['right']} | {round(item['corr'], 6)} |")
    lines.append("")
    lines.append("## Family x Tier Coverage")
    lines.append("")
    lines.append("| sample_no | low | mid | high |")
    lines.append("| --- | ---: | ---: | ---: |")
    for item in family_counts:
        lines.append(f"| {item['sample_no']} | {item['low']} | {item['mid']} | {item['high']} |")
    lines.append("")
    lines.append("## Recommendations")
    lines.append("")
    lines.append(
        "- Directly usable core features: "
        + ", ".join(f"`{name}`" for name in CORE_MODEL_FEATURES)
    )
    lines.append(
        "- Collapse duplicate representations before modeling: "
        + "`tortuosity` vs `tortuosity_v2`, `waviness_*` vs `waviness_*_v2`, "
        + "`curvature_nm_v3` vs `curvature_nm_v3_sqrt_length`, "
        + "`curvature_nm_v3_length` vs `curvature_nm_v3_p75_length`."
    )
    lines.append(
        "- Treat `magnification`, `sample_no`, and `condition_tier_proxy` as stratification variables, not ordinary continuous predictors."
    )
    lines.append(
        "- The current `condition_value_proxy` / `condition_tier_proxy` columns are placeholders for grouped process-condition analysis. "
        + "If you later provide the true gas-composition mapping, these should be replaced rather than interpreted literally."
    )
    lines.append(
        "- Cleanup-load features such as `removed_spur_count` and `removed_short_component_count` are useful as process diagnostics, "
        + "but they are strongly pipeline-dependent and should be used cautiously in any scientific interpretation."
    )
    lines.append(
        "- Historical ZZY batches still lack `junction_*` outputs. Re-running the new ZZY pipeline is the cleanest path before any graph-topology modeling."
    )
    lines.append(
        "- Current family coverage is imbalanced across tiers, so family-conditioned analysis is safer than a single pooled regression."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    rows = load_summary_rows()
    quality = feature_quality(rows)
    tier_stats = tier_summary(rows, ["density", "alignment", "diameter", "curvature_nm_v3", "tortuosity_v2", "waviness_ratio_v2", "n_branches"])
    correlations = feature_correlations(rows, ["density", "alignment", "diameter", "curvature_nm_v3", "tortuosity_v2", "waviness_ratio_v2", "n_branches"])
    redundancy = redundancy_pairs(rows)
    family_counts = family_tier_counts(rows)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = REPORTS_ROOT / f"zzy_feature_engineering_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "engineered_dataset.csv"
    md_path = out_dir / "report.md"
    write_engineered_csv(rows, csv_path)
    md_path.write_text(
        render_markdown(rows, quality, tier_stats, correlations, redundancy, family_counts),
        encoding="utf-8",
    )

    print(out_dir)
    print(csv_path)
    print(md_path)


if __name__ == "__main__":
    main()
