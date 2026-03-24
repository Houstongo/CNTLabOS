# 批量处理结果对比汇总

import json
from pathlib import Path

import pandas as pd

FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR if (FILE_DIR / "backend").exists() else FILE_DIR.parents[1]
BATCH_OUTPUT_ROOT = PROJECT_ROOT / "output" / "batch_runs"
THRESHOLD_SUMMARY = BATCH_OUTPUT_ROOT / "batch_output_threshold_full" / "batch_summary.json"
CNTSEGNET_SUMMARY = BATCH_OUTPUT_ROOT / "batch_output_cntsegnet" / "batch_summary.json"
COMPARISON_CSV = PROJECT_ROOT / "output" / "batch_comparison.csv"

# 读取两种方法的结果
with THRESHOLD_SUMMARY.open("r", encoding="utf-8") as f:
    threshold_data = json.load(f)

with CNTSEGNET_SUMMARY.open("r", encoding="utf-8") as f:
    cntsegnet_data = json.load(f)

# 构建对比表格
results = []
for t, c in zip(threshold_data, cntsegnet_data):
    t_features = t.get('threshold', {}).get('features', {})
    c_features = c.get('cntsegnet', {}).get('features', {})

    row = {
        'image_name': t['image_name'][:60],
        'mag': t['magnification'],
        # Threshold 特征
        'density_t': t_features.get('density'),
        'curvature_t': t_features.get('curvature'),
        'tortuosity_t': t_features.get('tortuosity'),
        'alignment_t': t_features.get('alignment'),
        'diameter_nm_t': t_features.get('diameter_nm'),
        # CNTSegNet 特征
        'density_c': c_features.get('density'),
    }
    results.append(row)

df = pd.DataFrame(results)

# 保存为CSV
df.to_csv(COMPARISON_CSV, index=False, encoding="utf-8-sig")

# 打印统计汇总
print("=" * 100)
print("批量处理结果对比汇总")
print("=" * 100)
print(f"\n总图片数: {len(df)}")
print(f"\n倍率分布:")
print(df['mag'].value_counts().sort_index())

print(f"\n--- Threshold 方法 ---")
print(f"密度: mean={df['density_t'].mean():.4f}, std={df['density_t'].std():.4f}")
print(f"曲率: mean={df['curvature_t'].mean():.4f}, std={df['curvature_t'].std():.4f}")
print(f"波曲度: mean={df['tortuosity_t'].mean():.4f}, std={df['tortuosity_t'].std():.4f}")
print(f"取向度: mean={df['alignment_t'].mean():.4f}, std={df['alignment_t'].std():.4f}")
print(f"直径(nm): mean={df['diameter_nm_t'].mean():.1f}, std={df['diameter_nm_t'].std():.1f}")

print(f"\n--- CNTSegNet 方法 ---")
print(f"密度: mean={df['density_c'].mean():.4f}, std={df['density_c'].std():.4f}")

print(f"\n--- 密度差异 ---")
df['density_diff'] = df['density_c'] - df['density_t']
print(f"密度差异 (CNTSegNet - Threshold): mean={df['density_diff'].mean():.4f}")
print(f"  CNTSegNet 更高的图片: {(df['density_diff'] > 0).sum()}")
print(f"  Threshold 更高的图片: {(df['density_diff'] < 0).sum()}")

print(f"\n结果已保存至: {COMPARISON_CSV}")
