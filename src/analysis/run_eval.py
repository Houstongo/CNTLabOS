"""
运行评估脚本的包装器（处理编码问题）
"""
import sys
import os
import subprocess

# 设置标准输出编码为 UTF-8
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# 切换目录
os.chdir(r"d:\CNTDATA\CNTA_ML_Project\src\analysis")

# 运行评估脚本
subprocess.run([r"D:\ProgramData\miniconda3\envs\lab_agent\python.exe", "evaluate_improvements.py"],
               encoding='utf-8', errors='replace')
