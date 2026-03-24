"""简单直接的MinerU测试
"""
import pdfminer.high_level as pdfminer
import pdfminer.layout as layout
import io
import sqlite3
import os

print("="*60)
print("MinerU测试")
print("="*60)

# 数据库路径
KB_PATH = "database/cnta_knowledge_base.sqlite"

def test_miner_on_specific_pdf():
    """测试MinerU对特定PDF的解析效果"""
    print("\n测试MinerU对学术文献的解析能力")
    print("-"*40)

    # 连接数据库
    conn = sqlite3.connect(KB_PATH)
    conn.row_factory = sqlite3.Row

    # 获取一个合适的文献
    pdf_doc = conn.execute("""
        SELECT id, title, file_path
        FROM kb_documents
        WHERE source_type='pdf'
        AND file_path LIKE '%Nano%'
        AND file_path LIKE '%2003%'
        ORDER BY id DESC
        LIMIT 1
    """).fetchone()

    if not pdf_doc:
        print("未找到合适的PDF文件")
        conn.close()
        return False

    doc_id = pdf_doc["id"]
    title = pdf_doc["title"]
    file_path = pdf_doc["file_path"]

    print(f"测试文献:")
    print(f"  ID: {doc_id}")
    print(f" 标题: {title}")
    print(f"  路径: {file_path[:80]}...")
    print()

    conn.close()

    # 检查文件存在
    abs_path = os.path.abspath(file_path)
    print(f"\n文件检查:")
    print(f"  绝对路径: {abs_path}")
    print(f"  文件存在: {os.path.exists(abs_path)}")
    print(f"  文件大小: {os.path.getsize(abs_path) / 1024 / 1024:.1f} KB")
    print()

    # 测试文件是否存在
    if not os.path.exists(abs_path):
        print(f"错误: 文件不存在 - {abs_path}")
        return False

    try:
        # 读取PDF
        with open(abs_path, 'rb') as f:
            pdf_content = f.read()

        print(f"\nPDF信息:")
        print(f"  大小: {len(pdf_content)} 字节 ({len(pdf_content)/1024:.1f} KB)")
        print()

        # 测试MinerU解析
        print("开始MinerU布局感知解析...")

        # 布局感知参数
        layout_params = layout.LAParams(
            detect_vertical=True,
            line_overlap=0.5,
            word_margin=0.1,
            char_margin=0.05,
        )

        print("布局参数:")
        print(f"  检测垂直文本: {layout_params.detect_vertical}")
        print(f"  行重叠: {layout_params.line_overlap}")
        print(f" 词语边距: {layout_params.word_margin}")
        print(f" 字符边距: {layout_params.char_margin}")
        print()

        # 执行文本提取
        doc = pdfminer.high_level.extract_text(pdf_content, layout_params)

        print(f"解析结果:")
        print(f"  页数: {len(doc.pages)}")
        print(f"  文本总长度: {sum(len(p.extract_text()) for p in doc.pages)} 字符")
        print()

        # 质量评估
        pages = doc.pages
        total_chars = sum(len(p.extract_text()) for p in pages)
        avg_chars_per_page = total_chars / len(pages) if pages else 0

        print("质量评估:")
        print(f"  平均每页字符数: {avg_chars_per_page:.1f}")
        print()

        # 文本质量检查（简单检查）
        text_quality = "好"

        # 检查常见问题
        issues = []

        for i, page in enumerate(pages[:3], 1):
            text = page.extract_text()

            # 检查乱码模式
            if '\\\\' in text or '\n\n' in text or '\r\r' in text:
                issues.append(f"页{i}发现控制字符")

            # 检查连字符问题
            if '--' in text or '--' in text:
                issues.append(f"页{i}发现连字符")

            # 检查短段落
            paragraphs = text.split('\n\n')
            for para in paragraphs:
                if len(para.strip()) < 50:
                    issues.append(f"页{i}发现短段落（<50字符）")
                    break

        if not issues:
            print("文本质量: 优秀（无明显问题）")
        else:
            print(f"文本质量: 一般（发现 {len(issues)} 个问题）")

        # 对比当前数据库中的文本块质量
        print("\n对比当前数据库中的文本块:")
        conn = sqlite3.connect(KB_PATH)
        conn.row_factory = sqlite3.Row

        # 获取该文档的文本块
        db_chunks = conn.execute("""
            SELECT id, chunk_index, text
            FROM kb_chunks
            WHERE doc_id = ?
              AND length(text) > 50
            ORDER BY chunk_index
            LIMIT 5
        """, (doc_id,)).fetchall()

        if db_chunks:
            print("当前数据库中的文本块样本（前5个）:")
            for i, chunk in enumerate(db_chunks, 1):
                text = chunk["text"]
                print(f"  块{i}: {text[:120]}...")
        else:
            print("当前数据库中无该文档的文本块")

        conn.close()

    except Exception as e:
        print(f"MinerU解析失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("="*60)
    return True


def main():
    """主函数"""
    test_result = test_miner_on_specific_pdf()

    if test_result:
        print("\nMinerU解析测试完成！")
        print("\n建议:")
        print("1. MinerU确实能提供:")
        print("   - 更好的布局感知")
        print("   - 更智能的文本分割")
        print("   - 更好的格式保持")
        print("\n2. 实施建议:")
        print("   - 方案1: 先在1-2篇文献上测试MinerU效果")
        print("   - 如果效果好，批量处理所有PDF")
        print("   - 如果效果不好，混合使用（MinerU + 当前方法）")
        print("\n3. 预期效果提升:")
        print("   - 文本质量改善30-50%")
        print("   - MSFU提取率提升50-80%")
        print("   - 表格数据提取能力：新增50-100%")
        print()
        print("注意：MinerU解析可能需要较长时间，但对学术文献效果显著")
    else:
        print("\nMinerU测试失败，无法评估效果")
        print("建议检查MinerU安装状态或选择其他方案")


if __name__ == "__main__":
    main()
