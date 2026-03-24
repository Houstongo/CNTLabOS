import json
from pathlib import Path

import pandas as pd

FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR if (FILE_DIR / "backend").exists() else FILE_DIR.parents[1]
BATCH_OUTPUT_ROOT = PROJECT_ROOT / "output" / "batch_runs"

# Read results
with (BATCH_OUTPUT_ROOT / "batch_output_threshold_full" / "batch_summary.json").open("r", encoding="utf-8") as f:
    threshold_data = json.load(f)

with (BATCH_OUTPUT_ROOT / "batch_output_cntsegnet" / "batch_summary.json").open("r", encoding="utf-8") as f:
    cntsegnet_data = json.load(f)

# Build comparison table
results = []
for t, c in zip(threshold_data, cntsegnet_data):
    t_f = t.get('threshold', {}).get('features', {})
    c_f = c.get('cntsegnet', {}).get('features', {})

    row = {
        'image': t['image_name'][:50],
        'mag': t['magnification'],
        'density_t': t_f.get('density'),
        'density_c': c_f.get('density'),
        'density_diff': (c_f.get('density', 0) or 0) - (t_f.get('density', 0) or 0),
        'alignment': t_f.get('alignment'),
        'diameter_nm': t_f.get('diameter_nm'),
        'curvature': t_f.get('curvature'),
        'tortuosity': t_f.get('tortuosity'),
    }
    results.append(row)

df = pd.DataFrame(results)

# Print table
print("=" * 130)
print(f"{'Image':<50} {'Mag':>6} {'Density_T':>10} {'Density_C':>10} {'Diff':>8} {'Align':>8} {'Diam':>8} {'Curv':>8} {'Tort':>8}")
print("=" * 130)
for _, row in df.iterrows():
    print(f"{row['image']:<50} {row['mag']:>6} {row['density_t']:>10.4f} {row['density_c']:>10.4f} {row['density_diff']:>+8.4f} {row['alignment']:>8.3f} {row['diameter_nm']:>8.1f} {row['curvature']:>8.4f} {row['tortuosity']:>8.3f}")

print("=" * 130)

# Summary
print("\n=== SUMMARY ===")
print(f"Total images: {len(df)}")
print(f"\nMagnification distribution:")
print(df['mag'].value_counts().sort_index())
print(f"\n--- Density Comparison ---")
print(f"Threshold mean: {df['density_t'].mean():.4f}")
print(f"CNTSegNet mean: {df['density_c'].mean():.4f}")
print(f"Difference: {df['density_diff'].mean():+.4f}")
print(f"  CNTSegNet higher: {(df['density_diff'] > 0).sum()} images")
print(f"  Threshold higher: {(df['density_diff'] < 0).sum()} images")
print(f"\n--- Alignment (Threshold) ---")
print(f"Mean: {df['alignment'].mean():.3f}, Std: {df['alignment'].std():.3f}")
print(f"\n--- Diameter (Threshold) ---")
print(f"Mean: {df['diameter_nm'].mean():.1f} nm, Std: {df['diameter_nm'].std():.1f} nm")
print(f"\n--- Curvature (Threshold) ---")
print(f"Mean: {df['curvature'].mean():.4f} nm^-1")
print(f"\n--- Tortuosity (Threshold) ---")
print(f"Mean: {df['tortuosity'].mean():.3f}")
