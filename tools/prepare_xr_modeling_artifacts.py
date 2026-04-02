from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_SUMMARY = Path(
    r"D:\CNTDATA\CNTA_ML_Project\reports\slice_standard_batch_20260331_005741\summary.csv"
)
DEFAULT_OUTPUT_DIR = Path(
    r"D:\CNTDATA\CNTA_ML_Project\reports\slice_standard_batch_20260331_005741\data_cleaning_review\modeling_prep"
)

FEATURES = [
    "density",
    "alignment",
    "diameter_mean_nm",
    "l2_curvature_trimmed_mean_sqrt_length_nm",
    "l2_waviness_ratio_v2",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare XR modeling artifacts from the standard batch summary."
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_SUMMARY,
        help="Path to the XR batch summary.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where modeling artifacts will be written",
    )
    return parser.parse_args()


def ensure_numeric(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    out = df.copy()
    for feature in features:
        out[feature] = pd.to_numeric(out[feature], errors="coerce")
    return out


def save_distribution_plot(df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(len(FEATURES), 2, figsize=(12, 16), constrained_layout=True)

    for row_idx, feature in enumerate(FEATURES):
        series = df[feature].dropna()

        ax_hist = axes[row_idx, 0]
        ax_box = axes[row_idx, 1]

        ax_hist.hist(series, bins=24, color="#4C78A8", edgecolor="white", alpha=0.9)
        ax_hist.axvline(series.median(), color="#F58518", linestyle="--", linewidth=1.2)
        ax_hist.axvline(series.mean(), color="#54A24B", linestyle="-.", linewidth=1.2)
        ax_hist.set_title(f"{feature} distribution")
        ax_hist.set_xlabel(feature)
        ax_hist.set_ylabel("count")

        ax_box.boxplot(
            series,
            vert=False,
            patch_artist=True,
            boxprops={"facecolor": "#E45756", "alpha": 0.55},
            medianprops={"color": "black", "linewidth": 1.2},
        )
        ax_box.set_title(f"{feature} boxplot")
        ax_box.set_xlabel(feature)
        ax_box.set_yticks([])

    fig.suptitle("XR Modeling Features: Raw Distributions", fontsize=16)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_standardized_table(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    stats: dict[str, dict[str, float]] = {}
    out = df[
        [
            "image_id",
            "sample_id",
            "file_name",
            "status",
            *FEATURES,
        ]
    ].copy()

    for feature in FEATURES:
        series = out[feature]
        mean = float(series.mean())
        std = float(series.std(ddof=0))
        stats[feature] = {"mean": mean, "std": std}
        if std <= 0 or np.isnan(std):
            out[f"{feature}_zscore"] = np.nan
        else:
            out[f"{feature}_zscore"] = (series - mean) / std

    return out, stats


def build_anomaly_table(df: pd.DataFrame, zscore_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    thresholds: dict[str, dict[str, float]] = {}
    flags = pd.DataFrame(index=df.index)

    for feature in FEATURES:
        series = df[feature]
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        thresholds[feature] = {
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "lower": lower,
            "upper": upper,
        }
        flags[feature] = series.lt(lower) | series.gt(upper)

    anomaly_rows = []
    zscore_cols = {feature: f"{feature}_zscore" for feature in FEATURES}

    for idx, row in df.iterrows():
        reasons: list[str] = []
        for feature in FEATURES:
            if pd.isna(row[feature]):
                reasons.append(f"{feature}:missing")
                continue
            if bool(flags.at[idx, feature]):
                value = float(row[feature])
                lower = thresholds[feature]["lower"]
                upper = thresholds[feature]["upper"]
                direction = "low" if value < lower else "high"
                reasons.append(f"{feature}:{direction}_iqr")
        anomaly_rows.append(reasons)

    out = zscore_df.copy()
    out["anomaly_reasons"] = [",".join(reasons) for reasons in anomaly_rows]
    out["anomaly_flag_count"] = [len(reasons) for reasons in anomaly_rows]
    out["is_anomaly"] = out["anomaly_flag_count"] > 0

    ordered_cols = [
        "image_id",
        "sample_id",
        "file_name",
        "status",
        *FEATURES,
        *zscore_cols.values(),
        "anomaly_flag_count",
        "anomaly_reasons",
        "is_anomaly",
    ]
    out = out[ordered_cols].sort_values(
        ["anomaly_flag_count", "sample_id", "image_id"], ascending=[False, True, True]
    )
    return out, thresholds


def build_summary(
    df: pd.DataFrame,
    standardized_df: pd.DataFrame,
    anomaly_df: pd.DataFrame,
    stats: dict[str, dict[str, float]],
    thresholds: dict[str, dict[str, float]],
) -> dict[str, object]:
    feature_missing = {feature: int(df[feature].isna().sum()) for feature in FEATURES}
    anomaly_counts = {
        "total_rows": int(len(anomaly_df)),
        "flagged_rows": int(anomaly_df["is_anomaly"].sum()),
        "clean_rows": int((~anomaly_df["is_anomaly"]).sum()),
    }
    return {
        "source_summary": str(DEFAULT_SUMMARY),
        "features": FEATURES,
        "row_count": int(len(df)),
        "feature_missing_counts": feature_missing,
        "zscore_stats": stats,
        "iqr_thresholds": thresholds,
        "anomaly_counts": anomaly_counts,
        "top_anomalies": anomaly_df.loc[anomaly_df["is_anomaly"]]
        .head(20)[["image_id", "sample_id", "anomaly_flag_count", "anomaly_reasons"]]
        .to_dict(orient="records"),
    }


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.summary)
    df = df.loc[df["status"].eq("success")].copy()
    df = ensure_numeric(df, FEATURES)

    distribution_path = output_dir / "xr_modeling_raw_distributions.png"
    save_distribution_plot(df, distribution_path)

    standardized_df, stats = build_standardized_table(df)
    standardized_path = output_dir / "xr_modeling_standardized_table.csv"
    standardized_df.to_csv(standardized_path, index=False, encoding="utf-8-sig")

    anomaly_df, thresholds = build_anomaly_table(df, standardized_df)
    anomaly_path = output_dir / "xr_modeling_anomaly_flags.csv"
    anomaly_df.to_csv(anomaly_path, index=False, encoding="utf-8-sig")

    summary = build_summary(df, standardized_df, anomaly_df, stats, thresholds)
    summary_path = output_dir / "xr_modeling_prep_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"distribution_plot={distribution_path}")
    print(f"standardized_table={standardized_path}")
    print(f"anomaly_flags={anomaly_path}")
    print(f"summary_json={summary_path}")
    print(
        "flagged_rows="
        f"{int(anomaly_df['is_anomaly'].sum())}/{int(len(anomaly_df))}"
    )


if __name__ == "__main__":
    main()
