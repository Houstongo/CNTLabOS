"""
XR 数据相关性矩阵分析 - 简化版
工艺参数: 温度、催化剂浓度、氩气流速
图像表征: 直径、密度、对齐度、曲率、曲折度
"""
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

DB_PATH = Path(r"D:\CNTDATA\CNTA_ML_Project\database\cnta_experiments.sqlite")
OUTPUT_DIR = Path(r"D:\CNTDATA\CNTA_ML_Project\output")
OUTPUT_DIR.mkdir(exist_ok=True)

def load_xr_data():
    """加载 XR 数据"""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT
            actual_temp, ar_flow, catalyst_weight,
            diameter, density, alignment, curvature, tortuosity
        FROM images
        WHERE source = 'XR' AND is_deleted = 0
    """, conn)
    conn.close()
    return df

def main():
    # 加载数据
    df = load_xr_data()
    print(f"加载 XR 数据: {len(df)} 条")
    print(f"\n各列非空数量:")
    print(df.notna().sum())

    # 列名映射
    col_names_cn = {
        'actual_temp': '实际温度',
        'ar_flow': '氩气流量',
        'catalyst_weight': '催化剂浓度',
        'diameter': '直径',
        'density': '密度',
        'alignment': '对齐度',
        'curvature': '曲率',
        'tortuosity': '曲折度'
    }

    # 计算相关性矩阵
    corr_matrix = df.corr()

    # 重命名行列标签
    labels = [col_names_cn.get(col, col) for col in corr_matrix.columns]

    # 绘制热力图
    fig, ax = plt.subplots(figsize=(10, 8))

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt='.2f',
        cmap='RdBu_r',
        center=0,
        vmin=-1, vmax=1,
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        annot_kws={'size': 10}
    )

    ax.set_title('XR 数据相关性矩阵', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # 保存图片
    output_path = OUTPUT_DIR / 'xr_correlation_matrix.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n相关性矩阵已保存: {output_path}")

    # 输出相关性矩阵
    print("\n=== 相关性矩阵 ===")
    corr_matrix_cn = corr_matrix.copy()
    corr_matrix_cn.index = labels
    corr_matrix_cn.columns = labels
    print(corr_matrix_cn.round(3).to_string())

    # 输出显著相关性
    print("\n=== 显著相关性 (|r| > 0.3) ===")
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
            r = corr_matrix.iloc[i, j]
            if abs(r) > 0.3:
                print(f"{col_names_cn.get(col1, col1)} <-> {col_names_cn.get(col2, col2)}: r = {r:.3f}")

    # plt.show()
    return corr_matrix

if __name__ == '__main__':
    main()
