"""
快速评估脚本 - 用于小规模测试
=================================
只评估前50张图像，快速检查算法表现。

运行方式：
    cd d:\\CNTDATA\\CNTA_ML_Project
    python backend/core/quick_eval.py
"""

import sys
import os

# 添加项目根目录到路径
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from systematic_evaluator import SystematicEvaluator

if __name__ == "__main__":
    print("快速评估模式 - 只处理ZZY数据集前50张图像（PNG格式）")
    print("=" * 60)

    evaluator = SystematicEvaluator()

    # 只处理ZZY数据的前50张，重新处理已处理的
    evaluator.evaluate_all(reprocess=True, limit=50, source='ZZY')
    evaluator.generate_report()

    print("\n快速评估完成！")
    print(f"查看报告: {evaluator.output_dir}")
