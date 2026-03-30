"""ZZY 50000x 工艺参数-图像特征相关性分析（clDice 管线数据）"""
import re, json, warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr

warnings.filterwarnings("ignore")
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# ── 路径 ──
JSON_DIR = Path(r"D:\CNTDATA\CNTA_ML_Project\reports\zzy_feature_panels_cldice_20260330_202128")
DB_PATH = Path(r"D:\CNTDATA\CNTA_ML_Project\database\cnta_experiments.sqlite")
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = Path(r"D:\CNTDATA\CNTA_ML_Project\output") / f"zzy_50k_correlation_{stamp}"

# ── 列定义 ──
PARAM_COLS = [
    "al2o3_power", "al2o3_thickness", "fe_power", "fe_thickness",
    "ar_flow", "h2_flow", "c2h4_flow",
]
CORE_FEATURES = [
    "density", "diameter", "alignment", "curvature_nm_v3",
    "tortuosity_v2", "waviness_ratio_v2",
]
ALL_FEATURES = CORE_FEATURES + [
    "mean_phi_deg", "curvature_nm_v3_p50_length",
    "curvature_nm_v3_p75_length", "curvature_nm_v3_mean_length",
    "curvature_nm_v3_trimmed_mean_length",
    "waviness_height_nm_v2", "waviness_wavelength_nm_v2",
    "n_branches",
]

PARAM_CN = {
    "al2o3_power": "Al₂O₃功率", "al2o3_thickness": "Al₂O₃厚度",
    "fe_power": "Fe功率", "fe_thickness": "Fe厚度",
    "ar_flow": "Ar流量", "h2_flow": "H₂流量", "c2h4_flow": "C₂H₄流量",
}
FEAT_CN = {
    "density": "密度(%)", "diameter": "直径(nm)", "alignment": "对齐度(HOF)",
    "curvature_nm_v3": "曲率V3", "tortuosity_v2": "扭率V2",
    "waviness_ratio_v2": "波纹比V2", "mean_phi_deg": "取向角(°)",
    "curvature_nm_v3_p50_length": "曲率P50", "curvature_nm_v3_p75_length": "曲率P75",
    "curvature_nm_v3_mean_length": "曲率Mean", "curvature_nm_v3_trimmed_mean_length": "曲率TrimMean",
    "waviness_height_nm_v2": "波高(nm)", "waviness_wavelength_nm_v2": "波长(nm)",
    "n_branches": "分支数",
}


# ── Step 1: 加载数据 ──
def load_data():
    import sqlite3
    rows = []
    for fp in sorted(JSON_DIR.glob("*50000*__features.json")):
        feat = json.loads(fp.read_text("utf-8"))
        stem = fp.name.replace("__features.json", "")
        rows.append({"stem": stem, **feat})

    df = pd.DataFrame(rows)
    print(f"加载 {len(df)} 张 50000x 特征数据")

    # 从数据库取工艺参数
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(
        "SELECT file_path, sample_id, al2o3_power, al2o3_thickness, fe_power, fe_thickness, "
        "ar_flow, h2_flow, c2h4_flow, anneal_temp, growth_temp, anneal_time, growth_time "
        "FROM images WHERE source='ZZY' AND magnification=50000 AND COALESCE(is_deleted,0)=0"
    )
    db_map = {}
    for r in cur.fetchall():
        s = str(Path(r["file_path"]).stem).replace(" ", "_")
        db_map[s] = {k: r[k] for k in PARAM_COLS}
    conn.close()

    # 匹配
    params_list = []
    missed = 0
    for _, row in df.iterrows():
        p = db_map.get(row["stem"])
        if p:
            params_list.append(p)
        else:
            params_list.append({k: np.nan for k in PARAM_COLS + ["sample_no"]})
            missed += 1
    df_params = pd.DataFrame(params_list)
    df = pd.concat([df.reset_index(drop=True), df_params], axis=1)

    # 从文件名提取纯样品号 (No26, No39 等)
    df["sample_no"] = df["stem"].apply(lambda s: re.match(r"(No\d+)", s).group(1) if re.match(r"(No\d+)", s) else "Unknown")

    if missed:
        print(f"  {missed} 张未匹配到数据库记录")
    print(f"  样品分布: {df['sample_no'].value_counts().sort_index().to_dict()}")
    return df


# ── Step 2: 按工艺条件取均值 ──
def compute_sample_means(df):
    # 分组键：样品号 + 7个工艺参数（数值型去重后取均值）
    group_key = ["sample_no"] + PARAM_COLS
    agg_dict = {f: ["mean", "std", "count"] for f in ALL_FEATURES}
    df_mean = df.groupby(group_key).agg(agg_dict).reset_index()
    df_mean.columns = ["_".join(c) if c[1] else c[0] for c in df_mean.columns]
    print(f"按工艺条件分组: {len(df_mean)} 组（每组合并同条件重复）")
    return df_mean


# ── Step 3: Spearman / Pearson 相关 + FDR ──
def compute_correlations(df, features, label=""):
    from statsmodels.stats.multitest import multipletests
    results = []
    for param in PARAM_COLS:
        for feat in features:
            col_mean = f"{feat}_mean" if f"{feat}_mean" in df.columns else feat
            valid = df[[param, col_mean]].dropna()
            if len(valid) < 5:
                continue
            rho, p_sp = spearmanr(valid[param], valid[col_mean])
            r, p_pe = pearsonr(valid[param], valid[col_mean])
            results.append({
                "parameter": param, "feature": feat,
                "spearman_rho": round(rho, 4), "spearman_p": round(p_sp, 6),
                "pearson_r": round(r, 4), "pearson_p": round(p_pe, 6),
                "n": len(valid),
            })
    if not results:
        return pd.DataFrame()
    res = pd.DataFrame(results)
    _, res["fdr_p"], _, _ = multipletests(res["spearman_p"], method="fdr_bh")
    res["fdr_p"] = res["fdr_p"].round(6)
    res["sig_fdr05"] = res["fdr_p"] < 0.05
    res["sig_fdr10"] = res["fdr_p"] < 0.10
    print(f"\n{'[' + label + '] ' if label else ''}相关性结果: {len(res)} 对, "
          f"FDR<0.05: {res['sig_fdr05'].sum()}, |rho|>0.3: {(res['spearman_rho'].abs()>0.3).sum()}")
    return res


# ── Step 4: 热力图 ──
def plot_heatmap(df, features, results_df, suffix, title_suffix=""):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    feat_col = [f"{f}_mean" if f"{f}_mean" in df.columns else f for f in features]
    corr = df[PARAM_COLS + feat_col].corr(method="spearman").loc[PARAM_COLS, feat_col]

    # 矩阵热力图
    fig, ax = plt.subplots(figsize=(max(10, len(feat_col) * 1.1), max(6, len(PARAM_COLS) * 0.9)))
    xlabels = [FEAT_CN.get(f.replace("_mean", ""), f) for f in feat_col]
    ylabels = [PARAM_CN.get(p, p) for p in PARAM_COLS]

    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, xticklabels=xlabels, yticklabels=ylabels,
                ax=ax, annot_kws={"size": 9}, linewidths=0.5)

    # 显著性标记
    if results_df is not None and len(results_df) > 0:
        for idx, row in results_df.iterrows():
            pi = list(PARAM_COLS).index(row["parameter"]) if row["parameter"] in PARAM_COLS else -1
            fi = -1
            for j, fc in enumerate(feat_col):
                if fc.replace("_mean", "") == row["feature"]:
                    fi = j
                    break
            if pi >= 0 and fi >= 0:
                stars = ""
                if row["spearman_p"] < 0.001:
                    stars = "***"
                elif row["spearman_p"] < 0.01:
                    stars = "**"
                elif row["spearman_p"] < 0.05:
                    stars = "*"
                if stars:
                    ax.text(fi + 0.5, pi + 0.75, stars, ha="center", va="center",
                            fontsize=7, color="black", fontweight="bold")

    ax.set_title(f"50000x Spearman 相关性矩阵{title_suffix}", fontsize=13)
    plt.xticks(rotation=40, ha="right")
    plt.tight_layout()
    fig.savefig(OUT_DIR / f"heatmap_spearman_{suffix}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 共线性热力图（工艺参数间）
    param_corr = df[PARAM_COLS].corr(method="spearman")
    fig2, ax2 = plt.subplots(figsize=(7, 6))
    sns.heatmap(param_corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, xticklabels=ylabels, yticklabels=ylabels,
                ax=ax2, annot_kws={"size": 10}, linewidths=0.5)
    ax2.set_title("工艺参数间共线性（Spearman）", fontsize=13)
    plt.tight_layout()
    fig2.savefig(OUT_DIR / f"param_collinearity_{suffix}.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)


# ── Step 5: 散点图 ──
def plot_scatter_top(df_raw, results_df, top_n=12, suffix=""):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    top = results_df.reindex(results_df["spearman_rho"].abs().sort_values(ascending=False).index).head(top_n)
    if len(top) == 0:
        return

    ncols = min(4, len(top))
    nrows = (len(top) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    if nrows == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    samples = df_raw["sample_no"].dropna().unique()
    colors = plt.cm.tab20(np.linspace(0, 1, max(len(samples), 1)))

    for i, (_, row) in enumerate(top.iterrows()):
        ax = axes[i]
        param = row["parameter"]
        feat = row["feature"]
        valid = df_raw[[param, feat, "sample_no"]].dropna()

        for j, s in enumerate(samples):
            mask = valid["sample_no"] == s
            if mask.sum() == 0:
                continue
            ax.scatter(valid.loc[mask, param], valid.loc[mask, feat],
                       color=colors[j % len(colors)], label=str(s), alpha=0.7, s=30, edgecolors="w", linewidths=0.3)

        rho = row["spearman_rho"]
        p = row["spearman_p"]
        stars = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
        ax.set_xlabel(PARAM_CN.get(param, param), fontsize=9)
        ax.set_ylabel(FEAT_CN.get(feat, feat), fontsize=9)
        ax.set_title(f"ρ={rho:.3f}{stars}", fontsize=10)
        ax.tick_params(labelsize=8)
        if i == 0:
            ax.legend(fontsize=6, title="样品", loc="best", ncol=2)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("50000x 最强相关配对（点按样品着色）", fontsize=13, y=1.01)
    plt.tight_layout()
    fig.savefig(OUT_DIR / f"scatter_top_pairs_{suffix}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Step 6: 箱线图 ──
def plot_boxplots(df_raw):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    feats = CORE_FEATURES[:6]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    df_plot = df_raw.dropna(subset=["sample_no"])

    for i, feat in enumerate(feats):
        ax = axes[i]
        order = sorted(df_plot["sample_no"].dropna().unique())
        data_list = [df_plot.loc[df_plot["sample_no"] == s, feat].dropna().values for s in order]
        bp = ax.boxplot(data_list, labels=order, patch_artist=True, widths=0.6)
        colors = plt.cm.Set3(np.linspace(0, 1, len(order)))
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
        ax.set_title(FEAT_CN.get(feat, feat), fontsize=10)
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("50000x 各样品特征分布（clDice 管线）", fontsize=13)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "boxplot_features_by_sample.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Step 7: 导出 ──
def export_results(df_raw, df_mean, results_core, results_all):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # 合并数据
    export_cols = ["sample_no", "stem"] + PARAM_COLS + ALL_FEATURES
    available = [c for c in export_cols if c in df_raw.columns]
    df_raw[available].to_csv(OUT_DIR / "merged_data_raw.csv", index=False, encoding="utf-8-sig")

    mean_cols = ["sample_no"] + PARAM_COLS + [f"{f}_mean" for f in ALL_FEATURES if f"{f}_mean" in df_mean.columns]
    mean_cols = [c for c in mean_cols if c in df_mean.columns]
    df_mean[mean_cols].to_csv(OUT_DIR / "merged_data_grouped.csv", index=False, encoding="utf-8-sig")

    if len(results_all) > 0:
        results_all.to_csv(OUT_DIR / "correlation_results_all.csv", index=False, encoding="utf-8-sig")
    if len(results_core) > 0:
        results_core.to_csv(OUT_DIR / "correlation_results_core.csv", index=False, encoding="utf-8-sig")

    # 样品汇总
    summary = df_raw.groupby("sample_no")[ALL_FEATURES].agg(["mean", "std", "count"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary.to_csv(OUT_DIR / "summary_by_sample.csv", encoding="utf-8-sig")

    print(f"\n输出目录: {OUT_DIR}")


# ── main ──
def main():
    df = load_data()

    # 去除特征为 None 的行
    for f in ALL_FEATURES:
        if f in df.columns:
            df[f] = pd.to_numeric(df[f], errors="coerce")

    df_mean = compute_sample_means(df)

    # 相关性
    res_core = compute_correlations(df_mean, CORE_FEATURES, "核心特征")
    res_all = compute_correlations(df_mean, ALL_FEATURES, "全部特征")

    # 热力图
    plot_heatmap(df_mean, CORE_FEATURES, res_core, "core", "（核心特征）")
    plot_heatmap(df_mean, ALL_FEATURES, res_all, "all", "（全部特征）")

    # 散点图
    plot_scatter_top(df, res_all, top_n=12, suffix="all")

    # 箱线图
    plot_boxplots(df)

    # 导出
    export_results(df, df_mean, res_core, res_all)

    # 打印关键发现
    if len(res_all) > 0:
        print("\n=== |rho|>0.3 的配对 ===")
        strong = res_all[res_all["spearman_rho"].abs() > 0.3].sort_values("spearman_rho", key=abs, ascending=False)
        for _, r in strong.iterrows():
            sig = "FDR<0.05" if r["sig_fdr05"] else ("FDR<0.10" if r["sig_fdr10"] else "")
            print(f"  {PARAM_CN.get(r['parameter'],r['parameter']):>12s} <-> {FEAT_CN.get(r['feature'],r['feature']):>14s}: ρ={r['spearman_rho']:+.3f}  p={r['spearman_p']:.4f}  {sig}")


if __name__ == "__main__":
    main()
