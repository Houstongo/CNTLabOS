from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict


matplotlib.use("Agg")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(r"D:\CNTDATA\CNTA_ML_Project")
TOOLS_DIR = PROJECT_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import experiment_zzy_best_l_target_specialized_noiseaware as noise_mod
import experiment_zzy_best_l_target_specialized as spec_mod


BUNDLE_DIR = PROJECT_ROOT / "reports" / "paper_section_4_4_data_bundle_20260402"
CONTROLLED_CSV = BUNDLE_DIR / "controlled_subset_raw.csv"
MODEL_BASE_DIR = PROJECT_ROOT / "reports" / "zzy_50000_fe_time_model_20260402"
BASELINE_RESULTS = MODEL_BASE_DIR / "best_results_by_target.csv"
SPECIALIZED_RESULTS = MODEL_BASE_DIR / "best_l_target_specialized_experiment" / "best_results.csv"
NOISEAWARE_RESULTS = MODEL_BASE_DIR / "best_l_target_specialized_noiseaware_experiment" / "best_results.csv"
ALIGNMENT_OOF = MODEL_BASE_DIR / "alignment_noiseaware_cv_detail" / "alignment_oof_predictions.csv"
OTHER_OOF_DIR = MODEL_BASE_DIR / "noiseaware_cv_detail_all_targets"


TARGET_ORDER = ["alignment", "curvature", "waviness_ratio", "tortuosity"]
TARGET_LABELS = {
    "alignment": "取向度",
    "curvature": "有效平均曲率",
    "waviness_ratio": "波曲度",
    "tortuosity": "迂曲度",
}
TARGET_LABELS_WITH_UNIT = {
    "alignment": "取向度",
    "curvature": "有效平均曲率 / μm^-1",
    "waviness_ratio": "波曲度",
    "tortuosity": "迂曲度",
}
FEATURE_LABELS = {
    "fe_power": "铁功率",
    "fe_thickness": "铁厚度",
    "anneal_time": "退火时间",
    "fe_deposition_index": "铁沉积指数",
    "anneal_time_x_thickness": "退火时间×铁厚度",
    "anneal_time_x_power": "退火时间×铁功率",
    "deposition_time_index": "沉积指数×退火时间",
    "thickness_sq": "铁厚度平方",
    "power_sq": "铁功率平方",
    "time_sq": "退火时间平方",
    "inv_thickness": "铁厚度倒数",
    "power_x_thickness_sq": "铁功率×厚度平方",
    "power_bin": "铁功率分档",
    "anneal_power_combo": "退火-功率组合",
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_controlled_data() -> pd.DataFrame:
    return pd.read_csv(CONTROLLED_CSV)


def save_better_correlation_heatmap(df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    cols = [
        "fe_power",
        "fe_thickness",
        "anneal_time",
        "fe_deposition_index",
        "alignment",
        "curvature",
        "waviness_ratio",
        "tortuosity",
    ]
    corr = df[cols].corr(method="spearman")
    labels = [
        "铁功率",
        "铁厚度",
        "退火时间",
        "铁沉积指数",
        "取向度",
        "有效平均曲率",
        "波曲度",
        "迂曲度",
    ]

    fig, ax = plt.subplots(figsize=(8.8, 7.2), constrained_layout=True)
    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    im = ax.imshow(corr.values, cmap="coolwarm", norm=norm, aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=10)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_title("工艺参数与关键形貌特征相关性热图", fontsize=14)

    for i in range(len(labels)):
        for j in range(len(labels)):
            value = corr.values[i, j]
            color = "white" if abs(value) > 0.45 else "#202020"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8.6, color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.92)
    cbar.set_label("秩相关系数", fontsize=10)
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return corr


def save_fe_response_panel(df: pd.DataFrame, out_path: Path) -> None:
    metrics = ["curvature", "waviness_ratio", "tortuosity", "alignment"]
    cmaps = {
        "curvature": matplotlib.colormaps["magma"].copy(),
        "waviness_ratio": matplotlib.colormaps["viridis"].copy(),
        "tortuosity": matplotlib.colormaps["PuBu"].copy(),
        "alignment": matplotlib.colormaps["YlGn"].copy(),
    }
    for cmap in cmaps.values():
        cmap.set_bad("#f1f1f1")
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 8.4), constrained_layout=True)
    axes = axes.flatten()

    for ax, metric in zip(axes, metrics):
        pivot = df.pivot_table(index="fe_thickness", columns="fe_power", values=metric, aggfunc="mean")
        pivot = pivot.sort_index().sort_index(axis=1)
        masked = np.ma.masked_invalid(pivot.values.astype(float))
        im = ax.imshow(masked, cmap=cmaps[metric], aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{int(v)}" if float(v).is_integer() else f"{v}" for v in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{v:g}" for v in pivot.index])
        ax.set_xlabel("铁功率 / W")
        ax.set_ylabel("铁厚度 / nm")
        ax.set_title(TARGET_LABELS_WITH_UNIT[metric], fontsize=11)
        norm = im.norm
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if np.isfinite(val):
                    text = f"{val:.2f}" if metric != "alignment" else f"{val:.3f}"
                    rgba = im.cmap(norm(val))
                    luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                    text_color = "white" if luminance < 0.45 else "#202020"
                    ax.text(j, i, text, ha="center", va="center", fontsize=7.6, color=text_color)
        cbar = fig.colorbar(im, ax=ax, shrink=0.88)
        cbar.ax.tick_params(labelsize=8)

    fig.suptitle("铁参数对关键形貌特征的响应分布", fontsize=14)
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def load_noiseaware_oof_predictions() -> pd.DataFrame:
    frames = []
    alignment_df = pd.read_csv(ALIGNMENT_OOF, dtype={"image_id": str})
    alignment_df["target"] = "alignment"
    alignment_df["stage"] = "噪声感知增强"
    frames.append(alignment_df)
    for target in ["curvature", "waviness_ratio", "tortuosity"]:
        df = pd.read_csv(OTHER_OOF_DIR / f"{target}_oof_predictions.csv", dtype={"image_id": str})
        df["target"] = target
        df["stage"] = "噪声感知增强"
        frames.append(df)
    return pd.concat(frames, ignore_index=True)

def build_specialized_oof_predictions() -> pd.DataFrame:
    best_df = pd.read_csv(SPECIALIZED_RESULTS)
    target_specs = spec_mod.build_target_specs()
    frames = []

    for target in TARGET_ORDER:
        target_df = spec_mod.load_target_dataframe(target)
        target_df = target_df.dropna(subset=[target]).copy()
        y = target_df[target].to_numpy(dtype=float)
        groups = target_df["group_key"].astype(str)
        n_splits = min(5, int(groups.nunique()))
        cv = GroupKFold(n_splits=n_splits)

        best_row = best_df[best_df["target"] == target].iloc[0]
        model_name = best_row["model"]
        spec = next(item for item in target_specs[target] if item["name"] == model_name)
        feature_cols = spec["num_cols"] + spec["cat_cols"]
        x = target_df[feature_cols].copy()
        pipe = spec_mod.build_pipeline(spec["num_cols"], spec["cat_cols"], clone(spec["estimator"]), spec["target_transform"] or None)
        y_pred = cross_val_predict(pipe, x, y, cv=cv, groups=groups)
        frames.append(
            pd.DataFrame(
                {
                    "image_id": target_df["image_id"].astype(str).to_numpy(),
                    "group_key": groups.to_numpy(),
                    "y_true": y,
                    "y_pred": y_pred,
                    "residual": y - y_pred,
                    "target": target,
                    "stage": "目标专门化",
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def build_prediction_metrics(all_preds: pd.DataFrame) -> pd.DataFrame:
    metric_rows = []
    for target in TARGET_ORDER:
        for stage in ["目标专门化", "噪声感知增强"]:
            g = all_preds[(all_preds["target"] == target) & (all_preds["stage"] == stage)]
            metric_rows.append(
                {
                    "target": target,
                    "stage": stage,
                    "r2": float(r2_score(g["y_true"], g["y_pred"])),
                    "rmse": float(np.sqrt(mean_squared_error(g["y_true"], g["y_pred"]))),
                }
            )
    return pd.DataFrame(metric_rows)


def save_prediction_scatter(all_preds: pd.DataFrame, metrics: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.6), constrained_layout=True)
    axes = axes.flatten()
    stage_colors = {
        "目标专门化": "#6c757d",
        "噪声感知增强": "#D55E00",
    }

    for ax, target in zip(axes, TARGET_ORDER):
        df = all_preds[all_preds["target"] == target].copy()
        lo = min(df["y_true"].min(), df["y_pred"].min())
        hi = max(df["y_true"].max(), df["y_pred"].max())
        pad = (hi - lo) * 0.06 if hi > lo else 0.1
        lo -= pad
        hi += pad
        for stage in ["目标专门化", "噪声感知增强"]:
            sub = df[df["stage"] == stage]
            ax.scatter(
                sub["y_true"],
                sub["y_pred"],
                s=24,
                alpha=0.58,
                color=stage_colors[stage],
                edgecolors="none",
                label=stage,
            )
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.2, color="#404040")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(TARGET_LABELS_WITH_UNIT[target], fontsize=11)
        ax.set_xlabel("真实值")
        ax.set_ylabel("预测值")
        ax.grid(alpha=0.18, linestyle="--")
        spec_row = metrics[(metrics["target"] == target) & (metrics["stage"] == "目标专门化")].iloc[0]
        noise_row = metrics[(metrics["target"] == target) & (metrics["stage"] == "噪声感知增强")].iloc[0]
        ax.text(
            0.04,
            0.95,
            (
                f"专门化 R² = {spec_row['r2']:.3f}\n"
                f"噪声增强 R² = {noise_row['r2']:.3f}"
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc", "boxstyle": "round,pad=0.25"},
        )
    axes[0].legend(frameon=False, loc="lower right")
    fig.suptitle("目标专门化与噪声感知增强模型的预测结果对比", fontsize=14)
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def build_stage_comparison() -> pd.DataFrame:
    baseline = pd.read_csv(BASELINE_RESULTS)
    specialized = pd.read_csv(SPECIALIZED_RESULTS)
    noiseaware = pd.read_csv(NOISEAWARE_RESULTS)

    base = baseline[baseline["target"].isin(TARGET_ORDER)][["target", "r2"]].rename(columns={"r2": "基础模型"})
    spec = specialized[["target", "r2"]].rename(columns={"r2": "目标专门化"})
    noise = noiseaware[["target", "r2"]].rename(columns={"r2": "噪声感知增强"})
    merged = base.merge(spec, on="target").merge(noise, on="target")
    merged["指标"] = merged["target"].map(TARGET_LABELS)
    return merged[["target", "指标", "基础模型", "目标专门化", "噪声感知增强"]]


def save_noiseaware_improvement(stage_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.6, 5.6), constrained_layout=True)
    x = np.arange(len(stage_df))
    width = 0.24
    colors = ["#999999", "#0072B2", "#009E73"]
    cols = ["基础模型", "目标专门化", "噪声感知增强"]

    for idx, col in enumerate(cols):
        ax.bar(x + (idx - 1) * width, stage_df[col], width=width, label=col, color=colors[idx], alpha=0.92)

    ax.set_xticks(x)
    ax.set_xticklabels(stage_df["指标"])
    ax.set_ylabel("R²")
    ax.set_ylim(0, max(stage_df[cols].max()) + 0.12)
    ax.set_title("噪声感知增强前后模型性能对比", fontsize=14)
    ax.grid(axis="y", alpha=0.2, linestyle="--")
    ax.legend(frameon=False, ncol=3, loc="upper left")
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def compute_feature_importance() -> pd.DataFrame:
    best_df = pd.read_csv(NOISEAWARE_RESULTS)
    target_specs = noise_mod.build_target_specs()
    rows = []

    for target in TARGET_ORDER:
        target_df = noise_mod.load_target_dataframe(target)
        target_df = target_df.dropna(subset=[target]).copy()
        target_df, _ = noise_mod.build_noise_weights(target_df, target)

        best_row = best_df[best_df["target"] == target].iloc[0]
        model_name = best_row["model"]
        spec = next(item for item in target_specs[target] if item["name"] == model_name)
        feature_cols = spec["num_cols"] + spec["cat_cols"]
        x = target_df[feature_cols].copy()
        y = target_df[target].to_numpy(dtype=float)
        sample_weight = target_df["sample_weight"].to_numpy(dtype=float)

        estimator = noise_mod.build_pipeline(
            spec["num_cols"],
            spec["cat_cols"],
            clone(spec["estimator"]),
            spec["target_transform"] or None,
        )
        estimator.fit(x, y, model__sample_weight=sample_weight)
        perm = permutation_importance(
            estimator,
            x,
            y,
            n_repeats=20,
            random_state=42,
            scoring="r2",
            n_jobs=1,
        )

        for feature, mean, std in zip(feature_cols, perm.importances_mean, perm.importances_std):
            rows.append(
                {
                    "target": target,
                    "feature": feature,
                    "feature_cn": FEATURE_LABELS.get(feature, feature),
                    "importance_mean": float(mean),
                    "importance_std": float(std),
                }
            )
    return pd.DataFrame(rows)


def save_feature_importance_panel(importance_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.6), constrained_layout=True)
    axes = axes.flatten()
    colors = {
        "alignment": "#4c78a8",
        "curvature": "#e15759",
        "waviness_ratio": "#59a14f",
        "tortuosity": "#f28e2b",
    }

    for ax, target in zip(axes, TARGET_ORDER):
        sub = importance_df[importance_df["target"] == target].copy()
        sub = sub.sort_values("importance_mean", ascending=False).head(8).sort_values("importance_mean", ascending=True)
        ax.barh(sub["feature_cn"], sub["importance_mean"], color=colors[target], alpha=0.88)
        ax.set_title(TARGET_LABELS[target], fontsize=11)
        ax.set_xlabel("置换重要性")
        ax.grid(axis="x", alpha=0.18, linestyle="--")
    fig.suptitle("关键形貌指标的变量重要性分析", fontsize=14)
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ensure_dir(BUNDLE_DIR)
    controlled = load_controlled_data()

    corr = save_better_correlation_heatmap(controlled, BUNDLE_DIR / "spearman_correlation_heatmap.png")
    corr.to_csv(BUNDLE_DIR / "spearman_correlation_matrix.csv", encoding="utf-8-sig")

    save_fe_response_panel(controlled, BUNDLE_DIR / "fe_parameter_response_panel.png")

    fe_waviness = controlled.pivot_table(index="fe_thickness", columns="fe_power", values="waviness_ratio", aggfunc="mean")
    fe_waviness.to_csv(BUNDLE_DIR / "fe_power_thickness_waviness_mean.csv", encoding="utf-8-sig")
    fe_tortuosity = controlled.pivot_table(index="fe_thickness", columns="fe_power", values="tortuosity", aggfunc="mean")
    fe_tortuosity.to_csv(BUNDLE_DIR / "fe_power_thickness_tortuosity_mean.csv", encoding="utf-8-sig")

    all_preds = pd.concat([build_specialized_oof_predictions(), load_noiseaware_oof_predictions()], ignore_index=True)
    pred_metrics = build_prediction_metrics(all_preds)
    save_prediction_scatter(all_preds, pred_metrics, BUNDLE_DIR / "prediction_scatter_best_models.png")
    pred_metrics.to_csv(BUNDLE_DIR / "prediction_scatter_metrics.csv", index=False, encoding="utf-8-sig")

    stage_df = build_stage_comparison()
    stage_df.to_csv(BUNDLE_DIR / "noiseaware_stage_comparison.csv", index=False, encoding="utf-8-sig")
    save_noiseaware_improvement(stage_df, BUNDLE_DIR / "noiseaware_r2_improvement.png")

    importance_df = compute_feature_importance()
    importance_df.to_csv(BUNDLE_DIR / "feature_importance.csv", index=False, encoding="utf-8-sig")
    save_feature_importance_panel(importance_df, BUNDLE_DIR / "feature_importance_panel.png")

    summary = {
        "output_dir": str(BUNDLE_DIR),
        "generated_files": [
            "spearman_correlation_heatmap.png",
            "spearman_correlation_matrix.csv",
            "fe_parameter_response_panel.png",
            "fe_power_thickness_waviness_mean.csv",
            "fe_power_thickness_tortuosity_mean.csv",
            "prediction_scatter_best_models.png",
            "prediction_scatter_metrics.csv",
            "noiseaware_stage_comparison.csv",
            "noiseaware_r2_improvement.png",
            "feature_importance.csv",
            "feature_importance_panel.png",
        ],
    }
    (BUNDLE_DIR / "section44_model_figures_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
