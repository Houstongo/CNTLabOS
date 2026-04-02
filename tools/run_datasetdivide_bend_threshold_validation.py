from __future__ import annotations

from datetime import datetime
import hashlib
from pathlib import Path
import re
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_ROOT = PROJECT_ROOT / "reports"
DATASETDIVIDE_DIR = PROJECT_ROOT / "datasetdivide"
ENGINEERING_PREFIX = "zzy_feature_engineering_gt10000_"
OUTPUT_PREFIX = "datasetdivide_bend_threshold_validation_"

CLASS_ORDER = ["straight", "wavy", "coiled"]
CLASS_LABELS_ZH = ["平直型", "波曲型", "卷曲型"]
HIGH_FEATURES = [
    "dk_bend_index",
    "curvature_nm_v3_trimmed_mean_sqrt_length",
    "waviness_ratio_v2",
    "tortuosity_v2",
    "junction_ratio",
]
LOW_FEATURES = ["alignment"]
SCORE_VARIANTS = {
    "bend_score": ("composite", None),
    "tortuosity_v2": ("high", "tortuosity_v2"),
    "alignment_inverse": ("low", "alignment"),
    "dk_bend_index": ("high", "dk_bend_index"),
    "curvature_only": ("high", "curvature_nm_v3_trimmed_mean_sqrt_length"),
}
CMAP = LinearSegmentedColormap.from_list(
    "datasetdivide_confusion",
    ["#F8FBFF", "#BFD7EA", "#4A90D9", "#08306B"],
)
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def find_latest_engineering_dataset() -> Path:
    candidates = [
        path / "engineered_dataset_active.csv"
        for path in REPORTS_ROOT.iterdir()
        if path.is_dir()
        and path.name.startswith(ENGINEERING_PREFIX)
        and (path / "engineered_dataset_active.csv").exists()
    ]
    if not candidates:
        raise FileNotFoundError("No engineered_dataset_active.csv found for ZZY reports.")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def normalize_selected_name(path_str: str) -> str:
    name = Path(path_str).name
    name = re.sub(r"^\d+_(50000|100000)_(straight|wavy|coiled)_", "", name, flags=re.I)
    return name.lower().strip()


def file_sha1(path_str: str) -> str:
    path = Path(path_str)
    hasher = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def selected_image_path(copied_path: str) -> Path:
    preferred = DATASETDIVIDE_DIR / Path(copied_path).name
    if preferred.exists():
        return preferred
    fallback = Path(copied_path)
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Selected image file not found: {copied_path}")


def percentile_from_train(train_values: pd.Series, values: pd.Series) -> pd.Series:
    clean_train = pd.to_numeric(train_values, errors="coerce").dropna().to_numpy(dtype=float)
    if clean_train.size == 0:
        return pd.Series(np.nan, index=values.index, dtype=float)
    clean_train.sort()
    clean_values = pd.to_numeric(values, errors="coerce")
    percentiles = clean_values.map(
        lambda x: np.searchsorted(clean_train, float(x), side="right") / clean_train.size
        if pd.notna(x)
        else np.nan
    )
    return percentiles.astype(float)


def compute_bend_score(train_df: pd.DataFrame, eval_df: pd.DataFrame) -> pd.Series:
    components: List[pd.Series] = []
    for col in HIGH_FEATURES:
        components.append(percentile_from_train(train_df[col], eval_df[col]))
    for col in LOW_FEATURES:
        components.append(1.0 - percentile_from_train(train_df[col], eval_df[col]))
    return pd.concat(components, axis=1).mean(axis=1, skipna=True)


def compute_variant_score(
    variant_name: str,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
) -> pd.Series:
    mode, column = SCORE_VARIANTS[variant_name]
    if mode == "composite":
        return compute_bend_score(train_df, eval_df)
    if column is None:
        raise ValueError(f"Variant {variant_name} is missing a source column.")
    score = percentile_from_train(train_df[column], eval_df[column])
    if mode == "low":
        score = 1.0 - score
    return score


def threshold_candidates(values: Iterable[float]) -> List[float]:
    arr = np.sort(np.unique(np.asarray(list(values), dtype=float)))
    if arr.size == 1:
        return [float(arr[0])]
    mids = ((arr[:-1] + arr[1:]) / 2.0).tolist()
    return [float(arr[0] - 1e-6)] + [float(x) for x in mids] + [float(arr[-1] + 1e-6)]


def classify_with_thresholds(scores: pd.Series, t1: float, t2: float) -> pd.Series:
    labels = pd.Series(index=scores.index, dtype=object)
    labels.loc[scores <= t1] = "straight"
    labels.loc[(scores > t1) & (scores <= t2)] = "wavy"
    labels.loc[scores > t2] = "coiled"
    return labels


def metric_bundle(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "weighted_kappa": float(
            cohen_kappa_score(y_true, y_pred, labels=CLASS_ORDER, weights="quadratic")
        ),
    }


def find_best_thresholds(train_df: pd.DataFrame) -> Tuple[float, float, Dict[str, float]]:
    candidates = threshold_candidates(train_df["bend_score"].dropna().tolist())
    best_pair: Tuple[float, float] | None = None
    best_metrics: Dict[str, float] | None = None

    for i, t1 in enumerate(candidates[:-1]):
        for t2 in candidates[i + 1 :]:
            pred = classify_with_thresholds(train_df["bend_score"], t1, t2)
            metrics = metric_bundle(train_df["waviness_class"], pred)
            if (
                best_metrics is None
                or metrics["macro_f1"] > best_metrics["macro_f1"]
                or (
                    np.isclose(metrics["macro_f1"], best_metrics["macro_f1"])
                    and metrics["balanced_accuracy"] > best_metrics["balanced_accuracy"]
                )
                or (
                    np.isclose(metrics["macro_f1"], best_metrics["macro_f1"])
                    and np.isclose(metrics["balanced_accuracy"], best_metrics["balanced_accuracy"])
                    and metrics["accuracy"] > best_metrics["accuracy"]
                )
            ):
                best_pair = (float(t1), float(t2))
                best_metrics = metrics

    if best_pair is None or best_metrics is None:
        raise RuntimeError("Failed to determine thresholds.")
    return best_pair[0], best_pair[1], best_metrics


def save_confusion_matrix(cm: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASS_ORDER)))
    ax.set_yticks(range(len(CLASS_ORDER)))
    ax.set_xticklabels(CLASS_LABELS_ZH)
    ax.set_yticklabels(CLASS_LABELS_ZH)
    ax.set_xlabel("预测标签")
    ax.set_ylabel("真实标签")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", color="#0F172A", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix_percent(
    cm: np.ndarray,
    out_path: Path,
    labels: List[str],
) -> np.ndarray:
    row_sums = cm.sum(axis=1, keepdims=True)
    percent = np.divide(
        cm.astype(float),
        row_sums,
        out=np.zeros_like(cm, dtype=float),
        where=row_sums != 0,
    ) * 100.0

    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    ax.imshow(percent, cmap=CMAP, vmin=0, vmax=100)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("预测标签", fontsize=15, fontweight="bold")
    ax.set_ylabel("真实标签", fontsize=15, fontweight="bold")

    for i in range(percent.shape[0]):
        for j in range(percent.shape[1]):
            value = percent[i, j]
            count = int(cm[i, j])
            text_color = "white" if value >= 55 else "#2F2F2F"
            ax.text(
                j,
                i,
                f"{count}\n{value:.1f}%",
                ha="center",
                va="center",
                fontsize=10.5,
                color=text_color,
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return percent


def save_score_boxplot(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    data = [df.loc[df["waviness_class"].eq(label), "bend_score"].dropna().to_numpy() for label in CLASS_ORDER]
    box = ax.boxplot(data, tick_labels=CLASS_ORDER, patch_artist=True)
    colors = ["#3B82F6", "#F97316", "#8B5CF6"]
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
    ax.set_ylabel("bend_score")
    ax.set_title("bend_score Distribution by Manual Class", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_rank_plot(df: pd.DataFrame, t1: float, t2: float, out_path: Path) -> None:
    rank_df = df.sort_values("bend_score").reset_index(drop=True).copy()
    color_map = {"straight": "#2563EB", "wavy": "#F97316", "coiled": "#7C3AED"}
    colors = rank_df["waviness_class"].map(color_map)
    fig, ax = plt.subplots(figsize=(9.4, 5.6))
    ax.scatter(np.arange(len(rank_df)), rank_df["bend_score"], c=colors, s=34, alpha=0.85)
    ax.axhline(t1, color="#0F172A", linestyle="--", linewidth=1.5, label=f"t1={t1:.4f}")
    ax.axhline(t2, color="#7C2D12", linestyle="--", linewidth=1.5, label=f"t2={t2:.4f}")
    ax.set_xlabel("Sample rank by bend_score")
    ax.set_ylabel("bend_score")
    ax.set_title("bend_score Ranking with Thresholds", fontweight="bold")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_report(
    out_dir: Path,
    dataset_path: Path,
    engineering_path: Path,
    train_metrics: Dict[str, float],
    test_metrics: Dict[str, float],
    t1: float,
    t2: float,
) -> None:
    lines = [
        "# Datasetdivide 4.3 Threshold Validation",
        "",
        f"- Dataset: `{dataset_path}`",
        f"- Engineering features: `{engineering_path}`",
        "- Split: stratified 50/50 hold-out",
        "- Score: bend_score from 4.3 feature fusion",
        "- Boundary selection: threshold search on training split only",
        "",
        "## Thresholds",
        "",
        f"- `t1 = {t1:.6f}`",
        f"- `t2 = {t2:.6f}`",
        "",
        "## Train Metrics",
        "",
        f"- Accuracy: `{train_metrics['accuracy']:.4f}`",
        f"- Macro-F1: `{train_metrics['macro_f1']:.4f}`",
        f"- Balanced Accuracy: `{train_metrics['balanced_accuracy']:.4f}`",
        f"- Weighted Kappa: `{train_metrics['weighted_kappa']:.4f}`",
        "",
        "## Test Metrics",
        "",
        f"- Accuracy: `{test_metrics['accuracy']:.4f}`",
        f"- Macro-F1: `{test_metrics['macro_f1']:.4f}`",
        f"- Balanced Accuracy: `{test_metrics['balanced_accuracy']:.4f}`",
        f"- Weighted Kappa: `{test_metrics['weighted_kappa']:.4f}`",
        "",
        "## Figures",
        "",
        "![rank](02_bend_score_rank_with_thresholds.png)",
        "",
        "![box](01_bend_score_boxplot.png)",
        "",
        "![cm](03_confusion_matrix_test.png)",
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def write_variant_report(
    out_dir: Path,
    variant_rows: List[Dict[str, float | str]],
) -> None:
    ordered = sorted(
        variant_rows,
        key=lambda row: (
            float(row["test_macro_f1"]),
            float(row["test_balanced_accuracy"]),
            float(row["test_accuracy"]),
        ),
        reverse=True,
    )
    lines = [
        "# 全部评分项测试集混淆矩阵",
        "",
        "以下图件均为测试集按行归一化后的百分比混淆矩阵。",
        "",
        "| variant | test accuracy | test macro-F1 | test balanced accuracy | test weighted kappa |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in ordered:
        lines.append(
            f"| {row['variant']} | {float(row['test_accuracy']):.4f} | "
            f"{float(row['test_macro_f1']):.4f} | {float(row['test_balanced_accuracy']):.4f} | "
            f"{float(row['test_weighted_kappa']):.4f} |"
        )
    lines += ["", "## 图件", ""]
    for row in ordered:
        variant = str(row["variant"])
        lines.append(f"### {variant}")
        lines.append("")
        lines.append(
            f"![{variant}](variant_confusion_matrices/{variant}_confusion_matrix_percent.png)"
        )
        lines.append("")
    (out_dir / "variant_confusion_matrices" / "index.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def resolve_selected_features(selected: pd.DataFrame, engineering: pd.DataFrame) -> pd.DataFrame:
    engineering = engineering.copy()
    counts = engineering["norm_name"].value_counts()
    duplicate_names = set(counts[counts > 1].index)
    hash_cache: Dict[str, str] = {}
    engineering_hashes: Dict[str, str] = {}

    records: List[Dict[str, object]] = []
    grouped = {name: group.copy() for name, group in engineering.groupby("norm_name", sort=False)}

    for row in selected.itertuples(index=False):
        candidates = grouped.get(row.norm_name)
        if candidates is None or candidates.empty:
            raise ValueError(f"No engineered feature row found for {row.copied_path}")

        chosen = None
        if len(candidates) == 1:
            chosen = candidates.iloc[0]
        else:
            selected_file = str(selected_image_path(row.copied_path))
            selected_hash = hash_cache.get(selected_file)
            if selected_hash is None:
                selected_hash = file_sha1(selected_file)
                hash_cache[selected_file] = selected_hash

            candidate_matches = []
            for candidate in candidates.itertuples(index=False):
                candidate_hash = engineering_hashes.get(candidate.file_path)
                if candidate_hash is None:
                    candidate_hash = file_sha1(candidate.file_path)
                    engineering_hashes[candidate.file_path] = candidate_hash
                if candidate_hash == selected_hash:
                    candidate_matches.append(candidate)

            if len(candidate_matches) == 1:
                chosen = pd.Series(candidate_matches[0]._asdict())
            elif len(candidate_matches) > 1:
                chosen = pd.Series(candidate_matches[0]._asdict())
            else:
                raise ValueError(
                    f"Duplicate norm_name could not be disambiguated by hash for {row.copied_path}"
                )

        merged_row = dict(row._asdict())
        if duplicate_names and row.norm_name in duplicate_names:
            merged_row["match_mode"] = "hash"
        else:
            merged_row["match_mode"] = "name"
        for key in [
            "file_name",
            "file_path",
            "sample_no",
            "magnification",
            "dk_bend_index",
            "curvature_nm_v3_trimmed_mean_sqrt_length",
            "waviness_ratio_v2",
            "tortuosity_v2",
            "junction_ratio",
            "alignment",
        ]:
            merged_row[key] = chosen[key]
        records.append(merged_row)

    return pd.DataFrame(records)


def main() -> None:
    dataset_path = DATASETDIVIDE_DIR / "summary.csv"
    engineering_path = find_latest_engineering_dataset()

    selected = pd.read_csv(dataset_path)
    selected["norm_name"] = selected["copied_path"].map(normalize_selected_name)

    engineering = pd.read_csv(engineering_path)
    engineering["norm_name"] = engineering["file_name"].str.lower().str.strip()

    merged = resolve_selected_features(
        selected,
        engineering[
            [
                "norm_name",
                "file_name",
                "file_path",
                "sample_no",
                "magnification",
                "dk_bend_index",
                "curvature_nm_v3_trimmed_mean_sqrt_length",
                "waviness_ratio_v2",
                "tortuosity_v2",
                "junction_ratio",
                "alignment",
            ]
        ],
    )

    required_cols = HIGH_FEATURES + LOW_FEATURES
    missing_mask = merged[required_cols].isna().any(axis=1)
    if missing_mask.any():
        missing_names = merged.loc[missing_mask, "copied_path"].tolist()
        raise ValueError(f"Missing required features for {len(missing_names)} rows: {missing_names[:5]}")

    train_idx, test_idx = train_test_split(
        merged.index,
        test_size=0.5,
        random_state=42,
        stratify=merged["waviness_class"],
    )
    train_df = merged.loc[train_idx].copy().reset_index(drop=True)
    test_df = merged.loc[test_idx].copy().reset_index(drop=True)

    train_df["bend_score"] = compute_bend_score(train_df, train_df)
    test_df["bend_score"] = compute_bend_score(train_df, test_df)

    t1, t2, train_metrics = find_best_thresholds(train_df)
    train_df["predicted_class"] = classify_with_thresholds(train_df["bend_score"], t1, t2)
    test_df["predicted_class"] = classify_with_thresholds(test_df["bend_score"], t1, t2)
    test_metrics = metric_bundle(test_df["waviness_class"], test_df["predicted_class"])

    variant_rows: List[Dict[str, float | str]] = []
    variant_predictions: Dict[str, pd.DataFrame] = {}
    for variant_name in SCORE_VARIANTS:
        train_variant = train_df.copy()
        test_variant = test_df.copy()
        train_variant["variant_score"] = compute_variant_score(variant_name, train_df, train_variant)
        test_variant["variant_score"] = compute_variant_score(variant_name, train_df, test_variant)

        search_df = train_variant[["waviness_class", "variant_score"]].rename(
            columns={"variant_score": "bend_score"}
        )
        vt1, vt2, train_variant_metrics = find_best_thresholds(search_df)
        train_pred = classify_with_thresholds(train_variant["variant_score"], vt1, vt2)
        test_pred = classify_with_thresholds(test_variant["variant_score"], vt1, vt2)
        test_variant_metrics = metric_bundle(test_variant["waviness_class"], test_pred)
        test_variant["predicted_class"] = test_pred
        test_variant["variant"] = variant_name
        variant_predictions[variant_name] = test_variant.copy()
        variant_rows.append(
            {
                "variant": variant_name,
                "t1": float(vt1),
                "t2": float(vt2),
                "train_accuracy": float(train_variant_metrics["accuracy"]),
                "train_macro_f1": float(train_variant_metrics["macro_f1"]),
                "train_balanced_accuracy": float(train_variant_metrics["balanced_accuracy"]),
                "train_weighted_kappa": float(train_variant_metrics["weighted_kappa"]),
                "test_accuracy": float(test_variant_metrics["accuracy"]),
                "test_macro_f1": float(test_variant_metrics["macro_f1"]),
                "test_balanced_accuracy": float(test_variant_metrics["balanced_accuracy"]),
                "test_weighted_kappa": float(test_variant_metrics["weighted_kappa"]),
            }
        )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = REPORTS_ROOT / f"{OUTPUT_PREFIX}{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    variant_dir = out_dir / "variant_confusion_matrices"
    variant_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(out_dir / "train_predictions.csv", index=False, encoding="utf-8-sig")
    test_df.to_csv(out_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame(
        [
            {"split": "train", **train_metrics},
            {"split": "test", **test_metrics},
        ]
    ).to_csv(out_dir / "metrics.csv", index=False, encoding="utf-8-sig")

    thresholds = pd.DataFrame([{"t1": t1, "t2": t2}])
    thresholds.to_csv(out_dir / "thresholds.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(variant_rows).sort_values(
        ["test_macro_f1", "test_balanced_accuracy", "test_accuracy"],
        ascending=False,
    ).to_csv(out_dir / "score_variant_comparison.csv", index=False, encoding="utf-8-sig")
    write_variant_report(out_dir, variant_rows)

    cm = confusion_matrix(test_df["waviness_class"], test_df["predicted_class"], labels=CLASS_ORDER)
    pd.DataFrame(cm, index=CLASS_ORDER, columns=CLASS_ORDER).to_csv(
        out_dir / "confusion_matrix_test.csv", encoding="utf-8-sig"
    )
    percent_main = save_confusion_matrix_percent(
        cm,
        out_dir / "03_confusion_matrix_test_percent.png",
        CLASS_LABELS_ZH,
    )
    pd.DataFrame(percent_main, index=CLASS_ORDER, columns=CLASS_ORDER).to_csv(
        out_dir / "confusion_matrix_test_percent.csv",
        encoding="utf-8-sig",
    )

    for row in variant_rows:
        variant = str(row["variant"])
        variant_test = variant_predictions[variant]
        variant_cm = confusion_matrix(
            variant_test["waviness_class"],
            variant_test["predicted_class"],
            labels=CLASS_ORDER,
        )
        pd.DataFrame(variant_cm, index=CLASS_ORDER, columns=CLASS_ORDER).to_csv(
            variant_dir / f"{variant}_confusion_matrix_counts.csv",
            encoding="utf-8-sig",
        )
        variant_percent = save_confusion_matrix_percent(
            variant_cm,
            variant_dir / f"{variant}_confusion_matrix_percent.png",
            CLASS_LABELS_ZH,
        )
        pd.DataFrame(variant_percent, index=CLASS_ORDER, columns=CLASS_ORDER).to_csv(
            variant_dir / f"{variant}_confusion_matrix_percent.csv",
            encoding="utf-8-sig",
        )

    full_df = pd.concat([train_df.assign(split="train"), test_df.assign(split="test")], ignore_index=True)
    save_score_boxplot(full_df, out_dir / "01_bend_score_boxplot.png")
    save_rank_plot(full_df, t1, t2, out_dir / "02_bend_score_rank_with_thresholds.png")
    save_confusion_matrix(cm, out_dir / "03_confusion_matrix_test.png")
    write_report(out_dir, dataset_path, engineering_path, train_metrics, test_metrics, t1, t2)

    print(f"output_dir={out_dir}")
    print(f"t1={t1:.6f}")
    print(f"t2={t2:.6f}")
    print(
        "test_metrics "
        f"accuracy={test_metrics['accuracy']:.4f} "
        f"macro_f1={test_metrics['macro_f1']:.4f} "
        f"balanced_accuracy={test_metrics['balanced_accuracy']:.4f} "
        f"weighted_kappa={test_metrics['weighted_kappa']:.4f}"
    )


if __name__ == "__main__":
    main()
