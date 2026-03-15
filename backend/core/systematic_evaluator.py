"""
CNTA特征提取系统性评估脚本  v1.0
=================================
功能：
1. 批量运行特征提取算法
2. 记录详细的失败原因和成功指标
3. 分析失败样品的特征分布（倍率、来源、密度等）
4. 生成评估报告（文本 + 可视化）

运行方式：
    cd d:\\CNTDATA\\CNTA_ML_Project
    python backend/core/systematic_evaluator.py

可选参数：
    --reprocess       重新处理已处理过的图像
    --limit N         只处理前 N 张（用于测试）
    --source ZZY      只处理指定来源
    --output DIR      输出目录（默认：reports/eval_YYYYMMDD_HHMMSS）
"""

import os
import sys
import sqlite3
import argparse
import json
import traceback
from datetime import datetime
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Any, Tuple

import cv2
import numpy as np
import pandas as pd

# 确保可以 import src.analysis.feature_extractor
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.analysis.feature_extractor import FeatureExtractor

DB_PATH = r'd:\CNTDATA\CNTA_ML_Project\database\cnta_experiments.sqlite'


class SystematicEvaluator:
    """系统性评估器"""

    def __init__(self, output_dir: str = None):
        self.output_dir = Path(output_dir) if output_dir else Path(
            'reports') / f'eval_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 评估结果存储
        self.results = {
            'summary': {
                'total': 0,
                'success': 0,
                'skipped': 0,
                'error': 0,
                'warning': 0
            },
            'errors': [],      # 完全失败
            'warnings': [],    # 部分问题
            'success_stats': defaultdict(list),  # 成功样品的统计信息
            'by_magnification': defaultdict(lambda: {'total': 0, 'success': 0, 'error': 0, 'warning': 0}),
            'by_source': defaultdict(lambda: {'total': 0, 'success': 0, 'error': 0, 'warning': 0}),
        }

    def evaluate_single(self, row: sqlite3.Row) -> Dict[str, Any]:
        """评估单张图像"""

        img_id = row['id']
        path = row['file_path']
        source = row['source']
        mag = row['magnification']
        sample_id = row['sample_id'] if 'sample_id' in row.keys() else 'Unknown'

        result = {
            'id': img_id,
            'path': path,
            'source': source,
            'magnification': mag,
            'sample_id': sample_id,
            'status': 'unknown',
            'error': None,
            'features': {},
            'warnings': [],
            'details': {}
        }

        # 检查文件是否存在
        if not os.path.exists(path):
            result['status'] = 'skipped'
            result['error'] = f'文件不存在: {path}'
            return result

        # 检查图像读取
        try:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                result['status'] = 'skipped'
                result['error'] = 'cv2.imread 返回 None'
                return result
        except Exception as e:
            result['status'] = 'error'
            result['error'] = f'图像读取异常: {str(e)}'
            return result

        # 提取特征
        try:
            extractor = FeatureExtractor(magnification=int(mag) if mag else None)
            features = extractor.extract_all(img)
            result['features'] = features

            # 检查提取结果的质量
            quality_issues = self._check_feature_quality(features)

            if quality_issues:
                result['status'] = 'warning'
                result['warnings'] = quality_issues
                result['details'] = quality_issues
            else:
                result['status'] = 'success'

        except Exception as e:
            result['status'] = 'error'
            result['error'] = str(e)
            result['traceback'] = traceback.format_exc()

        return result

    def _check_feature_quality(self, features: Dict[str, Any]) -> List[str]:
        """检查特征提取质量，返回问题列表"""

        issues = []

        # 检查 diameter
        diam = features.get('diameter')
        if diam is None:
            issues.append('diameter = N/A (低倍率或提取失败)')
        elif diam < 0:
            issues.append('diameter < 0 (提取失败)')
        elif diam < 5:
            issues.append(f'diameter 过小: {diam:.1f}nm (可能是噪声)')
        elif diam > 50:
            issues.append(f'diameter 过大: {diam:.1f}nm (可能是束径或团簇)')

        # 检查 density
        density = features.get('density', 0)
        if density == 0:
            issues.append('density = 0% (可能是二值化失败)')
        elif density > 95:
            issues.append(f'density 过高: {density:.1f}% (可能是背景误判)')

        # 检查 alignment
        alignment = features.get('alignment', 0)
        if alignment == 0:
            issues.append('alignment = 0 (可能是各向同性或提取失败)')
        elif alignment < -0.5:
            issues.append(f'alignment 异常: {alignment:.3f} (超出理论范围)')
        elif alignment > 1.0:
            issues.append(f'alignment 异常: {alignment:.3f} (超出理论范围)')

        # 检查 curvature
        curvature = features.get('curvature', 'Unknown')
        if curvature == 'Unknown':
            issues.append('curvature = Unknown (骨架提取失败)')
        elif curvature == 'N/A':
            issues.append('curvature = N/A (低倍率)')

        # 检查 hof_method
        hof_method = features.get('hof_method', '')
        if hof_method == 'structure_tensor':
            issues.append('alignment 使用结构张量法（低倍率，系统偏置）')

        # 检查辅助指标
        n_branches = features.get('n_branches')
        if n_branches is not None and n_branches == 0:
            issues.append('骨架分支数 = 0 (骨架化失败)')

        coherence = features.get('coherence')
        if coherence is not None and coherence < 0.1:
            issues.append(f'相干性过低: {coherence:.4f} (结构张量信号弱)')

        tortuosity = features.get('tortuosity', 0)
        if tortuosity > 5:
            issues.append(f'tortuosity 过高: {tortuosity:.3f} (可能骨架追踪错误)')

        return issues

    def evaluate_all(self, reprocess: bool = False, limit: int = None, source: str = None):
        """批量评估所有图像"""

        print(f"开始系统性评估...")
        print(f"输出目录: {self.output_dir}")
        print()

        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # 构建查询
        where_clauses = []
        params = []

        if not reprocess:
            where_clauses.append("processed = 0")
        if source:
            where_clauses.append("source = ?")
            params.append(source)

        where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
        limit_sql = f"LIMIT {limit}" if limit else ""

        cursor.execute(
            f"SELECT id, file_path, source, magnification, sample_id FROM images {where_sql} {limit_sql}",
            params
        )
        rows = cursor.fetchall()
        total = len(rows)

        print(f"待评估图像: {total} 张")
        print("-" * 80)

        # 逐个评估
        for i, row in enumerate(rows, 1):
            result = self.evaluate_single(row)

            # 更新统计
            self.results['summary']['total'] += 1
            mag = result['magnification'] or 0
            src = result['source']

            # 按倍率和来源统计
            self.results['by_magnification'][mag]['total'] += 1
            self.results['by_source'][src]['total'] += 1

            # 分类记录
            if result['status'] == 'success':
                self.results['summary']['success'] += 1
                self.results['by_magnification'][mag]['success'] += 1
                self.results['by_source'][src]['success'] += 1

                # 记录成功样品的特征统计
                self._record_success_stats(result)

                print(f"[{i:4d}/{total}] [OK] {os.path.basename(result['path'])}")

            elif result['status'] == 'warning':
                self.results['summary']['warning'] += 1
                self.results['by_magnification'][mag]['warning'] += 1
                self.results['by_source'][src]['warning'] += 1
                self.results['warnings'].append(result)

                print(f"[{i:4d}/{total}] [!] {os.path.basename(result['path'])}")
                for w in result['warnings']:
                    print(f"      - {w}")

            elif result['status'] == 'error':
                self.results['summary']['error'] += 1
                self.results['by_magnification'][mag]['error'] += 1
                self.results['by_source'][src]['error'] += 1
                self.results['errors'].append(result)

                print(f"[{i:4d}/{total}] [X] {os.path.basename(result['path'])}")
                print(f"      Error: {result['error']}")

            else:  # skipped
                self.results['summary']['skipped'] += 1
                print(f"[{i:4d}/{total}] [S] {os.path.basename(result['path'])}")
                print(f"      Skip: {result['error']}")

            # 更新数据库（如果成功或警告）
            if result['status'] in ['success', 'warning']:
                self._update_database(conn, result)

        conn.close()

        print("-" * 80)
        print(f"评估完成！")

    def _record_success_stats(self, result: Dict[str, Any]):
        """记录成功样品的特征统计"""
        features = result['features']

        for key in ['density', 'alignment', 'diameter', 'tortuosity', 'mean_phi_deg']:
            val = features.get(key)
            if val is not None:
                self.results['success_stats'][key].append(val)

        # 按倍率分组
        mag = result['magnification'] or 0
        if mag not in self.results['success_stats']:
            self.results['success_stats'][f'mag_{mag}'] = {}

        for key in ['density', 'alignment', 'diameter']:
            val = features.get(key)
            if val is not None:
                mag_key = f'mag_{mag}_{key}'
                if mag_key not in self.results['success_stats']:
                    self.results['success_stats'][mag_key] = []
                self.results['success_stats'][mag_key].append(val)

    def _update_database(self, conn: sqlite3.Connection, result: Dict[str, Any]):
        """更新数据库"""
        cursor = conn.cursor()
        features = result['features']

        cursor.execute("""
            UPDATE images
            SET diameter=?, density=?, alignment=?, curvature=?, processed=1
            WHERE id=?
        """, (
            features['diameter'],
            features['density'],
            features['alignment'],
            features['curvature'],
            result['id']
        ))
        conn.commit()

    def generate_report(self):
        """生成评估报告"""
        print("\n正在生成评估报告...")

        # 1. 生成文本报告
        self._generate_text_report()

        # 2. 生成JSON数据
        self._generate_json_report()

        # 3. 生成统计表格
        self._generate_statistics()

        print(f"\n报告已生成到: {self.output_dir}")

    def _generate_text_report(self):
        """生成文本报告"""
        report_path = self.output_dir / 'evaluation_report.txt'

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("CNTA 特征提取系统性评估报告\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"评估版本: FeatureExtractor v2.1\n\n")

            # 总体统计
            f.write("一、总体统计\n")
            f.write("-" * 80 + "\n")
            summary = self.results['summary']
            f.write(f"总图像数: {summary['total']}\n")
            f.write(f"成功:     {summary['success']:4d} ({summary['success']/summary['total']*100:.1f}%)\n")
            f.write(f"警告:     {summary['warning']:4d} ({summary['warning']/summary['total']*100:.1f}%)\n")
            f.write(f"错误:     {summary['error']:4d} ({summary['error']/summary['total']*100:.1f}%)\n")
            f.write(f"跳过:     {summary['skipped']:4d} ({summary['skipped']/summary['total']*100:.1f}%)\n")
            f.write("\n")

            # 按来源统计
            f.write("二、按来源统计\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'来源':<10} {'总计':>6} {'成功':>6} {'警告':>6} {'错误':>6} {'成功率':>8}\n")
            f.write("-" * 80 + "\n")
            for source, stats in sorted(self.results['by_source'].items()):
                success_rate = stats['success'] / stats['total'] * 100 if stats['total'] > 0 else 0
                f.write(f"{source:<10} {stats['total']:>6} {stats['success']:>6} "
                       f"{stats['warning']:>6} {stats['error']:>6} {success_rate:>7.1f}%\n")
            f.write("\n")

            # 按倍率统计
            f.write("三、按倍率统计\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'倍率':>10} {'总计':>6} {'成功':>6} {'警告':>6} {'错误':>6} {'成功率':>8}\n")
            f.write("-" * 80 + "\n")
            for mag, stats in sorted(self.results['by_magnification'].items()):
                mag_str = f"{mag}x" if mag > 0 else "Unknown"
                success_rate = stats['success'] / stats['total'] * 100 if stats['total'] > 0 else 0
                f.write(f"{mag_str:>10} {stats['total']:>6} {stats['success']:>6} "
                       f"{stats['warning']:>6} {stats['error']:>6} {success_rate:>7.1f}%\n")
            f.write("\n")

            # 特征统计
            f.write("四、成功样品特征统计\n")
            f.write("-" * 80 + "\n")
            for key in ['density', 'alignment', 'diameter', 'tortuosity', 'mean_phi_deg']:
                values = self.results['success_stats'].get(key, [])
                if len(values) > 0:
                    values = [v for v in values if v is not None]
                    f.write(f"{key:<20} "
                          f"均值: {np.mean(values):8.3f}  "
                          f"中位数: {np.median(values):8.3f}  "
                          f"最小: {np.min(values):8.3f}  "
                          f"最大: {np.max(values):8.3f}\n")
            f.write("\n")

            # 错误分析
            if self.results['errors']:
                f.write("五、错误详情\n")
                f.write("-" * 80 + "\n")
                f.write(f"共 {len(self.results['errors'])} 个错误\n\n")
                for err in self.results['errors'][:20]:  # 只显示前20个
                    f.write(f"ID: {err['id']}  |  {os.path.basename(err['path'])}\n")
                    f.write(f"  Error: {err['error']}\n\n")
                if len(self.results['errors']) > 20:
                    f.write(f"... 还有 {len(self.results['errors']) - 20} 个错误\n")
                f.write("\n")

            # 警告分析
            if self.results['warnings']:
                f.write("六、警告详情\n")
                f.write("-" * 80 + "\n")
                f.write(f"共 {len(self.results['warnings'])} 个警告\n\n")

                # 警告类型统计
                warning_types = Counter()
                for w in self.results['warnings']:
                    for warn in w['warnings']:
                        warning_types[warn] += 1

                f.write("警告类型分布:\n")
                for warn_type, count in warning_types.most_common(20):
                    f.write(f"  {count:4d}: {warn_type}\n")
                f.write("\n")

                # 具体警告详情（只显示前20个）
                f.write("具体警告列表 (前20个):\n\n")
                for warn in self.results['warnings'][:20]:
                    f.write(f"ID: {warn['id']}  |  {os.path.basename(warn['path'])}\n")
                    for w in warn['warnings']:
                        f.write(f"  - {w}\n")
                    f.write("\n")
                if len(self.results['warnings']) > 20:
                    f.write(f"... 还有 {len(self.results['warnings']) - 20} 个警告\n")

    def _generate_json_report(self):
        """生成JSON数据"""
        json_path = self.output_dir / 'evaluation_data.json'

        # 准备可序列化的数据
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'summary': self.results['summary'],
            'by_source': dict(self.results['by_source']),
            'by_magnification': {str(k): v for k, v in self.results['by_magnification'].items()},
            'errors': self.results['errors'],
            'warnings': self.results['warnings'],
            'success_stats': {}
        }

        # 转换numpy类型为Python原生类型
        for key, values in self.results['success_stats'].items():
            if values:
                export_data['success_stats'][key] = {
                    'mean': float(np.mean(values)),
                    'median': float(np.median(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'count': len(values)
                }

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

    def _generate_statistics(self):
        """生成统计表格（CSV格式）"""
        csv_path = self.output_dir / 'statistics.csv'

        rows = []
        for result in self.results['warnings'] + self.results['errors']:
            row = {
                'id': result['id'],
                'path': result['path'],
                'source': result['source'],
                'magnification': result['magnification'],
                'sample_id': result['sample_id'],
                'status': result['status'],
                'error': result.get('error', ''),
                'warnings': '; '.join(result.get('warnings', [])),
                'density': result.get('features', {}).get('density', ''),
                'alignment': result.get('features', {}).get('alignment', ''),
                'diameter': result.get('features', {}).get('diameter', ''),
                'curvature': result.get('features', {}).get('curvature', ''),
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')

        # 生成成功样品的统计
        success_csv_path = self.output_dir / 'success_statistics.csv'
        success_rows = []

        # 从结果中重建成功样品数据
        # 这里简化处理，实际应该从数据库读取
        print(f"成功样品统计将保存在数据库中")


def main():
    parser = argparse.ArgumentParser(description="CNTA特征提取系统性评估")
    parser.add_argument("--reprocess", action="store_true", help="重新处理已处理过的图像")
    parser.add_argument("--limit", type=int, default=None, help="只处理前 N 张")
    parser.add_argument("--source", type=str, default=None, help="只处理指定来源 (ZZY/XR)")
    parser.add_argument("--output", type=str, default=None, help="输出目录")

    args = parser.parse_args()

    evaluator = SystematicEvaluator(output_dir=args.output)
    evaluator.evaluate_all(reprocess=args.reprocess, limit=args.limit, source=args.source)
    evaluator.generate_report()


if __name__ == "__main__":
    main()
