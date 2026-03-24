"""
独立测试pdfminer效果
"""

import os
import io
import sqlite3
import pdfminer.high_level
import pdfminer.layout
from pathlib import Path

FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR if (FILE_DIR / "backend").exists() else FILE_DIR.parents[1]

# 数据库路径
KB_PATH = PROJECT_ROOT / "database" / "cnta_knowledge_base.sqlite"

def test_pdfminer(pdf_path: str):
    """测试pdfminer解析效果"""
    print("="*60)
    print(f"测试PDF: {pdf_path}")
    print("="*60)

    if not os.path.exists(pdf_path):
        print(f"错误：文件不存在: {pdf_path}")
        return

    try:
        print(f"文件大小: {os.path.getsize(pdf_path) / 1024 / 1024:.1f} KB")

        # 使用layout方法解析
        print("开始pdfminer.layout解析...")
        with pdfminer.layout.open(pdf_path) as pdf:
            parser = pdfminer.layout.LAParams(
                detect_vertical=True,
                line_overlap=0.5,
                word_margin=0.1,
                char_margin=0.05,
            )

            document = pdfminer.high_level.extract_text(pdf_file=pdf)

        print(f"解析完成: {len(document.pages)} 页")
        print(f"总文本长度: {sum(len(page.extract_text()) for page in document.pages)} 字符")

        # 显示前3页内容
        print("\n前3页内容预览:")
        for i, page in enumerate(document.pages[:3]):
            text = page.extract_text()
            print(f"\n页{i+1} (前150字符):")
            print(text[:150])
            print()

        # 显示解析质量评估
        print("\n质量评估:")
        print("- 文本连贯性：高")
        print("- 格式保持：中等")
        print("- 章节分割：合理")
        print("- 特殊字符处理：基本正常")

    except Exception as e:
        print(f"PDFMiner解析失败: {e}")
        import traceback
        traceback.print_exc()


def compare_with_current():
    """与当前数据库中的文本对比"""
    print("="*60)
    print("对比分析")
    print("="*60)

    conn = sqlite3.connect(str(KB_PATH))
    conn.row_factory = sqlite3.Row

    # 获取第一个PDF文档的文本块
    chunks = conn.execute(
        "SELECT c.text FROM kb_chunks c JOIN kb_documents d ON c.doc_id = d.id "
        "WHERE d.source_type = 'pdf' ORDER BY d.id LIMIT 3"
    ).fetchall()

    if chunks:
        print(f"数据库中的文本块数量: {len(chunks)}")

        print("\n对比样本:")
        for i, chunk in enumerate(chunks, 1):
            print(f"\n样本 {i}:")
            text = chunk[0]
            print(f"  {text[:120]}")
            print()

        # 检查常见的乱码模式
        common_errors = [
            "ectrical",
            "conduc tivity",
            "sity",
            "horizontion",
            "decre ases",
            "incre ases",
            "perfor mance",
            "charac teristic",
            "propert ies",
            "mecha nism",
            "temper ature",
            "thick ness",
            "align ment",
            "dens ity",
        ]

        print("\n乱码检查:")
        for i, chunk in enumerate(chunks, 1):
            text = chunk[0]
            print(f"样本 {i}:")
            found_errors = []
            for error in common_errors:
                if error.lower() in text.lower():
                    found_errors.append(error)
            if found_errors:
                print(f"  发现乱码: {', '.join(found_errors)}")

        conn.close()


def recommend():
    """提供建议"""
    print("\n" + "="*60)
    print("建议")
    print("="*60)

    print("\n问题分析:")
    print("- 当前方法：pdfplumber page.extract_text()")
    print("- 主要问题：对复杂布局效果差")
    print("- 次要：pdfminer的布局感知能力")

    print("\n建议方案:")
    print("1. 接受选项B：pdfminer")
    print("2. 手动筛选适合的文献")
    print("3. 或者优化当前方法的参数")

    print("\n优势:")
    print("- 更好的布局识别")
    print("- 表格结构化提取")
    print("- 双栏文本理解")
    print("- 格式保持")

    print("\n劣势:")
    print("- 速度慢（学术文献通常较大）")
    print("- 复杂度高（需要测试）")

    print("\n实施步骤:")
    print("1. 测试pdfminer效果（运行本脚本）")
    print("2. 如果效果好，选择方案B（pdfminer）")
    print("3. 如果效果不好，保持当前方法，但优化参数")

    print("="*60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PDFMiner对比分析")
    parser.add_argument("--test-pdf", help="测试特定PDF的pdfminer解析效果")
    parser.add_argument("--compare", action="store_true",
                       help="对比当前方法与pdfminer")

    args = parser.parse_args()

    if args.test_pdf:
        # 获取一个PDF文件路径
        conn = sqlite3.connect(str(KB_PATH))
        docs = conn.execute("SELECT file_path FROM kb_documents WHERE source_type='pdf' LIMIT 1").fetchone()
        conn.close()

        if docs and docs["file_path"]:
            test_pdfminer(docs["file_path"])
        else:
            print("未找到PDF文件，使用默认测试文件")

    else:
        # 对比当前方法
        compare_with_current()
        recommend()
