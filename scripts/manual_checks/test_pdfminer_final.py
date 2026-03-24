"""
独立PDFMiner测试
"""

import os
import pdfminer.high_level
import pdfminer.layout
import io
import sqlite3
from pathlib import Path

FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR if (FILE_DIR / "backend").exists() else FILE_DIR.parents[1]

print('='*60)
print('MINERU方案：pdfminer重新解析PDF')
print('='*60)
print()

# 连接数据库
conn = sqlite3.connect(str(PROJECT_ROOT / 'database' / 'cnta_knowledge_base.sqlite'))
conn.row_factory = sqlite3.Row

# 获取第一篇文献
pdf_info = conn.execute(
    "SELECT id, title, file_path FROM kb_documents WHERE file_path IS NOT NULL ORDER BY id LIMIT 1"
).fetchone()

if not pdf_info:
    print('未找到PDF文件')
    exit(0)

doc_id = pdf_info['id']
title = pdf_info['title']
file_path = pdf_info['file_path']

# 路径转换：将反斜杠转换为正斜杠
file_path = file_path.replace('\\', '/')

print(f"测试文档:")
print(f"ID: {doc_id}")
print(f"标题: {title[:60]}...")
print(f"路径: {file_path}")
print()

if not os.path.exists(file_path):
    print(f"文件不存在: {file_path}")
    exit(0)

try:
    # 读取PDF
    with open(file_path, 'rb') as f:
        pdf_content = f.read()
        print(f"PDF大小: {len(pdf_content)} bytes")

    # 使用pdfminer布局感知解析
    print("\n开始pdfminer布局感知解析...")

    # 创建布局感知参数
    laparams = pdfminer.layout.LAParams(
        detect_vertical=True,  # 检测垂直文本
        line_overlap=0.5,    # 文本行重叠
        word_margin=0.1,     # 单词边距
        char_margin=0.05,    # 字符边距
    )

    # 提取文本
    doc = pdfminer.high_level.extract_text(
        io.BytesIO(pdf_content),
        laparams=laparams,
    )

    print(f"解析完成: {len(doc)} 页")
    print(f"总字符数: {sum(len(p) for p in doc):,}")
    print()

    # 显示前2页内容对比
    print("pdfminer解析结果（前2页，各120字符）:")
    for i, page_text in enumerate(doc[:2]):
        print(f"页{i+1}:")
        print(page_text[:120])
        print()

    # 对比当前数据库中的文本
    current_chunks = conn.execute(
        "SELECT text FROM kb_chunks WHERE doc_id = ? ORDER BY chunk_index LIMIT 5",
        (doc_id,)
    ).fetchall()

    print("\n当前数据库中的文本块（对比用）:")
    for chunk in current_chunks:
        print(f"块{chunk[0]}: {chunk[1][:80]}")
        print()

    print("="*60)
    print("pdfminer核心优势:")
    print("- 布局感知：智能识别文本区域和逻辑结构")
    print("- 表格提取：结构化提取数据表格")
    print("- 文本顺序：保持学术文献逻辑性")
    print("- 格式保持：保留段落和结构信息")
    print()
    print("预期效果:")
    print("- 双栏文献：列序正确")
    print("- 表格数据：可提取表格内容")
    print("- 文本质量：大幅改善")
    print("="*60)

except Exception as e:
    print(f"PDFMiner解析失败: {e}")
    import traceback
    traceback.print_exc()

conn.close()
print()
