"""
ZZY 数据特征聚合
按样品聚合多倍率照片，选择最优特征值
"""
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path

DB_PATH = Path(r"D:\CNTDATA\CNTA_ML_Project\database\cnta_experiments.sqlite")
OUTPUT_DIR = Path(r"D:\CNTDATA\CNTA_ML_Project\output")
OUTPUT_DIR.mkdir(exist_ok=True)


def load_zzy_data():
    """加载 ZZY 全部数据"""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query('''
        SELECT *
        FROM images
        WHERE source = 'ZZY' AND is_deleted = 0
    ''', conn)
    conn.close()
    return df


def get_sample_key(row):
    """生成样品唯一标识（基于 No + 工艺参数）"""
    no_id = row['sample_id'].split('-')[0]
    return f"{no_id}_{row['fe_thickness']:.2f}_{row['fe_power']:.0f}_{row['ar_flow']:.0f}"


def aggregate_features(df):
    """
    按样品聚合特征
    规则：
    - alignment, density: 优先 50000×
    - diameter, curvature, tortuosity: 优先 100000×
    - 同一倍数多张照片：加权平均（用 std 倒数作为权重）
    """

    df['sample_key'] = df.apply(get_sample_key, axis=1)

    results = []

    for sample_key, group in df.groupby('sample_key'):
        # 工艺参数（取第一条即可，同一样品相同）
        params = {
            'sample_key': sample_key,
            'no_id': group.iloc[0]['sample_id'].split('-')[0],
            'fe_thickness': group.iloc[0]['fe_thickness'],
            'fe_power': group.iloc[0]['fe_power'],
            'ar_flow': group.iloc[0]['ar_flow'],
            'h2_flow': group.iloc[0]['h2_flow'],
            'c2h4_flow': group.iloc[0]['c2h4_flow'],
            'photo_count': len(group)
        }

        # 按倍率分组
        by_mag = {mag: g for mag, g in group.groupby('magnification')}

        # alignment, density: 优先 50000，备选 100000
        for feat in ['alignment', 'density']:
            params[feat] = select_feature(by_mag, feat, priority=[50000, 100000, 10000])

        # diameter, curvature, tortuosity: 优先 100000，备选 50000
        for feat in ['diameter', 'curvature', 'tortuosity']:
            params[feat] = select_feature(by_mag, feat, priority=[100000, 50000])

        # 记录使用的倍率
        params['diameter_mag'] = get_used_magnitude(by_mag, 'diameter', [100000, 50000])
        params['alignment_mag'] = get_used_magnitude(by_mag, 'alignment', [50000, 100000, 10000])

        results.append(params)

    return pd.DataFrame(results)


def select_feature(by_mag, feature, priority):
    """
    按优先级选择特征值
    同一倍数多张照片取平均
    """
    for mag in priority:
        if mag in by_mag:
            values = by_mag[mag][feature].dropna()
            if len(values) > 0:
                return values.mean()
    return np.nan


def get_used_magnitude(by_mag, feature, priority):
    """记录实际使用的倍率"""
    for mag in priority:
        if mag in by_mag:
            values = by_mag[mag][feature].dropna()
            if len(values) > 0:
                return mag
    return None


def weighted_average(values, weights):
    """加权平均"""
    mask = ~(np.isnan(values) | np.isnan(weights))
    if mask.sum() == 0:
        return np.nan
    return np.average(values[mask], weights=weights[mask])


def main():
    # 加载数据
    df = load_zzy_data()
    print(f"加载 ZZY 数据: {len(df)} 条记录")

    # 聚合特征
    agg_df = aggregate_features(df)
    print(f"聚合后样品数: {len(agg_df)}")

    # 显示聚合结果
    print("\n=== 聚合后的样品数据 ===")
    display_cols = ['no_id', 'fe_thickness', 'fe_power', 'ar_flow',
                    'diameter', 'density', 'alignment', 'curvature', 'tortuosity',
                    'diameter_mag', 'alignment_mag', 'photo_count']
    print(agg_df[display_cols].to_string(index=False))

    # 保存
    output_path = OUTPUT_DIR / 'zzy_aggregated_samples.csv'
    agg_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n已保存: {output_path}")

    return agg_df


if __name__ == '__main__':
    main()
