"""
第三章实验：运行所有实验
Chapter 3 Experiments: Run All Experiments
"""

import subprocess
import sys
from pathlib import Path


def run_experiment(exp_name: str, script_name: str):
    """运行单个实验"""
    print(f"\n{'='*80}")
    print(f"运行实验: {exp_name}")
    print(f"{'='*80}")

    script_path = Path(__file__).parent / script_name
    if not script_path.exists():
        print(f"错误：实验脚本不存在: {script_path}")
        return False

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(script_path.parent),
            capture_output=False,
            text=True,
            timeout=300  # 5 分钟超时
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"错误：实验超时")
        return False
    except Exception as e:
        print(f"错误：{e}")
        return False


def main():
    """主函数"""
    base_dir = Path(__file__).parent

    print("=" * 80)
    print("第三章实验：知识建模与检索增强方法")
    print("=" * 80)

    # 实验列表
    experiments = [
        ("实验 1: 知识库数据统计", "exp_01_data_statistics.py"),
        ("实验 2: 深度扩展功能验证", "exp_02_depth_extension.py"),
        ("实验 3: 任务类型对比实验", "exp_03_task_comparison.py"),
        ("实验 4: 消融实验", "exp_04_ablation_study.py"),
    ]

    results = {}

    # 运行每个实验
    for exp_name, script_name in experiments:
        success = run_experiment(exp_name, script_name)
        results[exp_name] = {
            "success": success,
            "script": script_name
        }

    # 输出总结
    print("\n" + "=" * 80)
    print("实验运行总结")
    print("=" * 80)

    for exp_name, result in results.items():
        status = "✅ 成功" if result["success"] else "❌ 失败"
        print(f"{exp_name}: {status}")

    # 检查结果文件
    results_dir = base_dir / "results"
    if results_dir.exists():
        result_files = list(results_dir.glob("exp_*.json"))
        print(f"\n结果文件: {len(result_files)} 个")
        for f in result_files:
            print(f"  - {f.name}")

    print("\n" + "=" * 80)
    print("所有实验完成！")
    print("=" * 80)

    # 提示查看结果
    print(f"\n结果保存在: {results_dir}")
    print("可以使用以下命令查看结果:")
    print(f"  cat {results_dir} / exp_*.json")


if __name__ == "__main__":
    main()
