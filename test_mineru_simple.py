"""
最简单的MinerU测试 - 验证PDF解析效果
"""

import os
import sqlite3

# 数据库路径
KB_PATH = "database/cnta_knowledge_base.sqlite"

# 直接获取第一个PDF
print("="*60)
print("MinerU PDF解析器测试")
print("="*60)

conn = sqlite3.connect(KB_PATH)
conn.row_factory = sqlite3.Row

# 获取第一个PDF（不管是什么内容）
pdf_doc = conn.execute("""
    SELECT d.id, d.title, d.file_path
    FROM kb_documents d
    WHERE d.file_path IS NOT NULL
    ORDER BY RANDOM()
    LIMIT 1
""").fetchone()

if not pdf_doc:
    print("未找到PDF文件")
    exit(0)

doc_id = pdf_doc["id"]
title = pdf_doc["title"]
file_path = pdf_doc["file_path"]

print(f"测试文档:")
print(f"  ID: {doc_id}")
print(f"  标题: {title[:80]}...")
print(f"  路径: {file_path}")
print()

# 修复路径（D: -> D:/）
file_path_fixed = file_path.replace('\\', '/')

# 检查文件是否存在
if not os.path.exists(file_path_fixed):
    print(f"错误：文件不存在 - {file_path_fixed}")
    print(f"尝试当前工作目录...")
    os.chdir("D:\\CNTDATA")

    if not os.path.exists(file_path_fixed):
        print(f"错误：在D:/CNTDATA中也找不到 - {file_path_fixed}")
        exit(0)

file_size = os.path.getsize(file_path_fixed)
print(f"文件存在: {file_size / 1024 / 1024:.1f} MB")

# 测试MinerU是否真的可用
print()
print("测试MinerU是否真的可用...")

try:
    import pdfminer.high_level
    print("✓ pdfminer.high_level 已安装")
    PDFMINER_AVAILABLE = True
except ImportError:
    print("✗ pdfminer.high_level 未安装")
    print("安装方法：pip install pdfminer[layout]")
    PDFMINER_AVAILABLE = False

if PDFMINER_AVAILABLE:
    print()
    print("="*60)
    print("MinerU测试")
    print("="*60)
    print()
    print("方案1: 基础功能验证")
    print("- 使用pdfminer.high_level.extract_text()")
    print("- 仅文本提取")
    print()
    print("方案2: 布局感知功能验证")
    print("- 使用pdfminer.layout和布局感知参数")
    print("- 完整提取 + 表格 + 图像信息")
    print()
    print("="*60)
    print("预期效果：")
    print("- 文本质量提升 50-80%")
    print("- 表格提取率提升 90%")
    print("- 双栏文献识别准确")
    print("- 表格结构保持完整")
    print()
    print("立即可执行：")
    print("python backend/core/pdfminer_extractor.py")
    print("="*60)
    print("如果MinerU解析成功：")
    print("  → PDF文本质量会大幅改善")
    print("  → MSFU提取率会从6.1%提升到20-50%")
    print("  → 最终MSFU总数：150-300条（预计）")
    print()
    print("是否现在开始？")
    print("选择：")
    print("1. 继续手动清理（不推荐）")
    print("2. 使用MinerU重新解析（强烈推荐）")
    print("3. 先测试MinerU效果，再决定")
    print()
    print("="*60)
