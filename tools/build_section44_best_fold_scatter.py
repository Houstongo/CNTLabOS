from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold


matplotlib.use("Agg")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(r"D:\CNTDATA\CNTA_ML_Project")
TOOLS_DIR = PROJECT_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import experiment_zzy_best_l_target_specialized_noiseaware as noise_mod


BUNDLE_DIR = PROJECT_ROOT / "reports" / "paper_section_4_4_data_bundle_20260402"
NOISEAWARE_RESULTS = PROJECT_ROOT / "reports" / "zzy_50000_fe_time_model_20260402" / "best_l_target_specialized_noiseaware_experiment" / "best_results.csv"

TARGET_ORDER = ["alignment", "curvature", "waviness_ratio", "tortuosity"]
TARGET_LABELS = {
    "alignment": "取向度",
    "curvature": "有效平均曲率 / μm^-1",
    "waviness_ratio": "波曲度",
    "tortuosity": "迂曲度",
}
BEST_FOLD_MAP = {
    "alignment": 4,
    "curvature": 4,
    "waviness_ratio": 4,
    "tortuosity": 3,
}
COLORS = {
    "alignment": "#0072B2",
    "curvature": "#D55E00",
    "waviness_ratio": "#009E73",
    "tortuosity": "#CC79A7",
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_best_fold_records() -> pd.DataFrame:
    best_df = pd.read_csv(NOISEAWARE_RESULTS)
    target_specs = noise_mod.build_target_specs()
    rows = []

    for target in TARGET_ORDER:
        target_df = noise_mod.load_target_dataframe(target)
        target_df = target_df.dropna(subset=[target]).copy()
        target_df, _ = noise_mod.build_noise_weights(target_df, target)
        y = target_df[target].to_numpy(dtype=float)
        groups = target_df["group_key"].astype(str)
        sample_weight = target_df["sample_weight"].to_numpy(dtype=float)

        best_row = best_df[best_df["target"] == target].iloc[0]
        model_name = best_row["model"]
        spec = next(item for item in target_specs[target] if item["name"] == model_name)
        feature_cols = spec["num_cols"] + spec["cat_cols"]
        x = target_df[feature_cols].copy()

        pipe = noise_mod.build_pipeline(
            spec["num_cols"],
            spec["cat_cols"],
            clone(spec["estimator"]),
            spec["target_transform"] or None,
        )

        cv = GroupKFold(n_splits=min(5, int(groups.nunique())))
        wanted_fold = BEST_FOLD_MAP[target]

        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(x, y, groups), start=1):
            if fold_idx != wanted_fold:
                continue
            x_train = x.iloc[train_idx]
            x_test = x.iloc[test_idx]
            y_train = y[train_idx]
            y_test = y[test_idx]
            w_train = sample_weight[train_idx]
            pipe.fit(x_train, y_train, model__sample_weight=w_train)
            y_pred = pipe.predict(x_test)

            fold_r2 = float(r2_score(y_test, y_pred))
            fold_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            for local_idx, row_idx in enumerate(test_idx):
                rows.append(
                    {
                        "target": target,
                        "fold": fold_idx,
                        "image_id": str(target_df.iloc[row_idx]["image_id"]),
                        "group_key": str(target_df.iloc[row_idx]["group_key"]),
                        "y_true": float(y_test[local_idx]),
                        "y_pred": float(y_pred[local_idx]),
                        "residual": float(y_test[local_idx] - y_pred[local_idx]),
                        "fold_r2": fold_r2,
                        "fold_rmse": fold_rmse,
                    }
                )
            break

    return pd.DataFrame(rows)


def save_scatter_panel(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.6), constrained_layout=True)
    axes = axes.flatten()

    for ax, target in zip(axes, TARGET_ORDER):
        sub = df[df["target"] == target].copy()
        lo = min(sub["y_true"].min(), sub["y_pred"].min())
        hi = max(sub["y_true"].max(), sub["y_pred"].max())
        pad = (hi - lo) * 0.07 if hi > lo else 0.1
        lo -= pad
        hi += pad
        ax.scatter(sub["y_true"], sub["y_pred"], s=30, alpha=0.75, color=COLORS[target], edgecolors="none")
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.1, color="#444444")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("真实值")
        ax.set_ylabel("预测值")
        ax.set_title(TARGET_LABELS[target], fontsize=11)
        ax.grid(alpha=0.18, linestyle="--")
        fold_no = int(sub["fold"].iloc[0])
        r2_val = float(sub["fold_r2"].iloc[0])
        rmse_val = float(sub["fold_rmse"].iloc[0])
        ax.text(
            0.04,
            0.95,
            f"第 {fold_no} 折\nR² = {r2_val:.3f}\nRMSE = {rmse_val:.3f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.86, "edgecolor": "#cccccc", "boxstyle": "round,pad=0.24"},
        )
    fig.suptitle("四个关键形貌指标最佳折预测散点图", fontsize=14)
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ensure_dir(BUNDLE_DIR)
    records = build_best_fold_records()
    records.to_csv(BUNDLE_DIR / "best_fold_scatter_data.csv", index=False, encoding="utf-8-sig")
    save_scatter_panel(records, BUNDLE_DIR / "best_fold_prediction_scatter.png")

    summary = (
        records.groupby(["target", "fold"], as_index=False)
        .agg({"fold_r2": "first", "fold_rmse": "first", "image_id": "count"})
        .rename(columns={"image_id": "test_n"})
    )
    summary.to_csv(BUNDLE_DIR / "best_fold_prediction_summary.csv", index=False, encoding="utf-8-sig")
    (BUNDLE_DIR / "best_fold_prediction_summary.json").write_text(
        json.dumps(summary.to_dict(orient="records"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
