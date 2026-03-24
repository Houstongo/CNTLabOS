import json
from pathlib import Path

FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR if (FILE_DIR / "backend").exists() else FILE_DIR.parents[1]
BATCH_OUTPUT_ROOT = PROJECT_ROOT / "output" / "batch_runs"

# 读取两个结果文件
with (BATCH_OUTPUT_ROOT / "batch_output_threshold" / "batch_summary.json").open("r", encoding="utf-8") as f:
    threshold_data = json.load(f)

with (BATCH_OUTPUT_ROOT / "batch_output_cntsegnet" / "batch_summary.json").open("r", encoding="utf-8") as f:
    cntsegnet_data = json.load(f)

print("=" * 90)
print(f"{'Image':<50} {'Threshold':>12} {'CNTSegNet':>12} {'Diff':>10}")
print("=" * 90)

t_densities = []
c_densities = []

for t, c in zip(threshold_data, cntsegnet_data):
    name = t['image_name'][:48]
    t_d = t.get('threshold', {}).get('features', {}).get('density', 0)
    c_d = c.get('cntsegnet', {}).get('features', {}).get('density', 0)
    diff = c_d - t_d
    t_densities.append(t_d)
    c_densities.append(c_d)
    print(f"{name:<50} {t_d:>12.4f} {c_d:>12.4f} {diff:>+10.4f}")

print("=" * 90)
t_avg = sum(t_densities) / len(t_densities)
c_avg = sum(c_densities) / len(c_densities)
print(f"{'Average':<50} {t_avg:>12.4f} {c_avg:>12.4f} {c_avg - t_avg:>+10.4f}")
print(f"\nTotal images: {len(threshold_data)}")
