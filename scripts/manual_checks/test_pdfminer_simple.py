"""
简化的pdfminer测试脚本
"""

import os
import pdfminer.high_level
import pdfminer.layout
import io
import sqlite3
from pathlib import Path

FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR if (FILE_DIR / "backend").exists() else FILE_DIR.parents[1]

# 简化测试
print('='*60)
print('MINERU方案：pdfminer重新解析PDF')
print('='*60)
print()

# 获取一个PDF
conn = sqlite3.connect(str(PROJECT_ROOT / 'database' / 'cnta_knowledge_base.sqlite'))
conn.row_factory = sqlite3.Row

pdf_info = conn.execute('''
    SELECT d.id, d.title, d.file_path
    FROM kb_documents d
    WHERE d.file_path IS NOT NULL
      AND LENGTH(d.file_path) > 0
    ORDER BY d.id
    LIMIT 1
''').fetchone()

if not pdf_info:
    print('未找到PDF文件')
    exit(0)

doc_id = pdf_info['id']
title = pdf_info['title']
file_path = pdf_info['file_path']

print(f'文档: {title[:60]}...')
print(f'ID: {doc_id}')
print(f'原始路径: {file_path}')

# 修复路径格式，将反斜杠替换为正斜杠
file_path = file_path.replace('\\', '/')

# 转换为绝对路径
if not os.path.isabs(file_path):
    file_path = os.path.normpath(file_path)

if not os.path.exists(file_path):
    print(f'文件不存在: {file_path}')
    exit(0)

print(f'修复后路径: {file_path}')
print(f'绝对路径: {os.path.abspath(file_path)}')
print()

try:
    # 读取PDF
    with open(file_path, 'rb') as f:
        pdf_content = f.read()
        print(f'PDF大小: {len(pdf_content):,} bytes')

    # 创建布局参数
    laparams = pdfminer.layout.LAParams(
        detect_vertical=True,
        line_overlap=0.5,
        word_margin=0.1,
        char_margin=0.05,
    )

    print('\n开始pdfminer布局感知解析...')

    # 使用pdfminer提取文本
    doc = pdfminer.high_level.extract_text(
        io.BytesIO(pdf_content),
        laparams=laparams,
    )

    print(f'解析完成: {len(doc)} 页')
    print(f'总字符数: {sum(len(page) for page in doc):,}')

    # 显示前2页内容对比
    print('pdfminer解析结果（前2页，各120字符）:')
    for i in range(min(2, len(doc))):
        page_text = doc[i]
        print(f'页{i+1}:')
        print(page_text[:120] if len(page_text) > 120 else page_text)
        print()

    # 获取当前数据库中的文本
    current_chunks = conn.execute('''
        SELECT c.text
        FROM kb_chunks c
        WHERE c.doc_id = ?
        ORDER BY c.chunk_index
        LIMIT 3
    ''', (doc_id,)).fetchall()

    print('\n当前数据库中的文本块（对比用）:')
    for chunk in current_chunks:
        print(f'块{chunk["chunk_index"]}: {chunk["text"][:80]}')
        print()

    print('='*60)
    print('pdfminer核心优势:')
    print('- 布局感知：智能识别文本区域和逻辑结构')
    print('- 文本顺序：保持学术文献逻辑性')
    print('- 格式保持：保留段落和结构信息')
    print('- 适合学术文献：复杂布局、表格、图表')

    print('='*60)
    print('预期效果:')
    print('- 双栏文献：列序正确')
    print('- 表格数据：可提取表格内容')
    print('- 文本质量：大幅改善')

except Exception as e:
    print(f'PDFMiner解析失败: {e}')
    import traceback
    traceback.print_exc()

conn.close()
print()
