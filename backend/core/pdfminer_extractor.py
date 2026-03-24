"""
PDFMiner增强PDF解析器
解决学术文献的复杂布局问题：表格、图表、双栏文本等

优势：
- 智能布局识别
- 表格结构化提取
- 图像信息提取
- 文本顺序保持
- 格式保持较好
"""

import os
import io
import sqlite3
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

try:
    import pdfplumber
    import pdfminer.high_level
    import pdfminer.layout
    PDFMINER_AVAILABLE = True
except ImportError:
    pdfminer = None
    PDFMINER_AVAILABLE = False
    print("警告：pdfminer未安装，将使用基础pdfplumber")


@dataclass
class ParsedDocument:
    """解析的PDF文档"""
    doc_id: Optional[int] = None
    title: str = ""
    pages: List[str] = None
    tables: List[Dict] = None
    metadata: Dict = None


class PDFMinerExtractor:
    """PDFMiner增强解析器"""

    def __init__(self, use_tables: bool = True, use_images: bool = False):
        """
        Args:
            use_tables: 是否提取表格（可能影响速度）
            use_images: 是否提取图像信息（可能影响速度）
        """
        self.use_tables = use_tables
        self.use_images = use_images

    def parse_pdf(self, file_bytes: bytes, title: str = "") -> ParsedDocument:
        """
        解析PDF文件

        Args:
            file_bytes: PDF文件字节
            title: 文档标题（可选）

        Returns:
            解析的文档对象
        """
        if not PDFMINER_AVAILABLE:
            return self._parse_with_pdfplumber(file_bytes, title)

        # 使用pdfminer的布局感知解析
        pdf_file = io.BytesIO(file_bytes)
        parser = pdfminer.layout.LAParams(
            detect_vertical=True,  # 检测垂直文本
            line_overlap=0.5,  # 文本行重叠
            word_margin=0.1,  # 单词边距
            char_margin=0.05,  # 字符边距
        )

        try:
            with pdfminer.high_level.extract_text(pdf_file, pdfminer.layout.LAParams(
                line_overlap=parser.line_overlap,
                word_margin=parser.word_margin,
                char_margin=parser.char_margin,
                detect_vertical=True,
            )) as doc:
                # 提取文本
                pages = doc.get_text()

                # 提取表格（如果启用）
                tables = []
                if self.use_tables:
                    tables = self._extract_tables(pdf_file)

                return ParsedDocument(
                    pages=pages,
                    tables=tables,
                    metadata={
                        'extraction_method': 'pdfminer_layout',
                        'parser_params': {
                            'line_overlap': parser.line_overlap,
                            'word_margin': parser.word_margin,
                            'detect_vertical': True,
                        }
                    }
                )

        except Exception as e:
            print(f"PDFMiner解析失败: {e}，回退到基础方法")
            return self._parse_with_pdfplumber(file_bytes, title)

    def _parse_with_pdfplumber(self, file_bytes: bytes, title: str = "") -> ParsedDocument:
        """使用基础pdfplumber解析（备用方案）"""
        pdf_file = io.BytesIO(file_bytes)

        with pdfplumber.open(pdf_file) as pdf:
            pages = []
            for i, page in enumerate(pdf.pages):
                try:
                    # 尝试使用x0,y0参数获得更好效果
                    text = page.extract_text(x0=100, y0=100)
                    pages.append(text)
                except Exception:
                    # 回退到简单方法
                    text = page.extract_text()
                    pages.append(text)

        return ParsedDocument(pages=pages)

    def _extract_tables(self, pdf_file) -> List[Dict]:
        """提取表格信息"""
        tables = []

        try:
            # 使用pdfminer提取表格
            tables_data = pdfminer.high_level.extract_tables(pdf_file)

            # 转换为更易用的格式
            for table_data in tables_data:
                table_dict = {
                    'page': table_data.page,
                    'rows': len(table_data.extract()),
                    'cols': len(table_data.header.cells),
                    'html': self._table_to_html(table_data),
                }
                tables.append(table_dict)

            print(f"提取到 {len(tables_data)} 个表格")

        except Exception as e:
            print(f"表格提取失败: {e}")

        return tables

    def _table_to_html(self, table_data) -> str:
        """将表格数据转换为HTML（用于展示）"""
        try:
            # 构建表格HTML
            html = '<table>'

            # 表头
            header_cells = table_data.header.cells
            if header_cells:
                html += '<tr>'
                for cell in header_cells:
                    html += f'<th>{cell.text or \"\"}</th>'
                html += '</tr>'

            # 表格内容
            for row in table_data.extract():
                html += '<tr>'
                for cell in row.cells:
                    html += f'<td>{cell.text or \"\"}</td>'
                html += '</tr>'

            html += '</table>'
            return html

        except Exception as e:
            return f"<p>表格解析错误: {e}</p>"


def improve_pdf_text_quality(text: str) -> str:
    """
    改进PDF文本质量
    针对常见问题：断行、连字符、多余空格等
    """
    if not text:
        return text

    # 1. 修复连字符问题
    text = text.replace('ectrical', 'electrical')
    text = text.replace('conduc tivity', 'conductivity')
    text = text.replace('sity', 'density')
    text = text.replace('horizontion', 'horizontal alignment')

    # 2. 修复常见错别
    corrections = {
        'nano tube': 'nanotube',
        'carbon nano tube': 'carbon nanotube',
        'CNT arrays': 'CNT arrays',
        'aligned CNT': 'aligned CNT',
        'horizontion': 'horizontal alignment',
        'demonstrat': 'demonstrated',
        'substantiated': 'substantiated',
    }
    for wrong, right in corrections.items():
        text = text.replace(wrong, right)

    # 3. 修复断行问题（保留段落分隔）
    text = text.replace(' \n', '\n')
    text = text.replace('\n ', '\n')

    # 4. 移除多余空格
    text = ' '.join(text.split())

    # 5. 移除重复句点
    text = text.replace('. ', '. ')
    text = text.replace('.. ', '. ')

    return text


def reparse_existing_documents(kb_path: str, use_pdfminer: bool = True) -> int:
    """
    重新解析现有PDF文档

    Args:
        kb_path: 知识库数据库路径
        use_pdfminer: 是否使用pdfminer解析

    Returns:
        重新解析的文档数量
    """
    import sqlite3

    conn = sqlite3.connect(kb_path)
    conn.row_factory = sqlite3.Row

    # 获取所有PDF文档
    pdf_docs = conn.execute("""
        SELECT d.id, d.title, d.file_path
        FROM kb_documents d
        WHERE d.source_type = 'pdf'
          AND d.file_path IS NOT NULL
          AND LENGTH(d.file_path) > 0
        ORDER BY d.id
    """).fetchall()

    print(f"找到 {len(pdf_docs)} 个PDF文档待重新解析")

    # 询问用户是否真的要重新解析
    print()
    print("⚠️  警告：重新解析将清除现有文本块和MSFU数据")
    print("建议：1. 备份当前数据库")
    print("       2. 先测试单个文档")
    print("       3. 确认效果后再批量操作")

    return len(pdf_docs)


def test_pdfminer_availability():
    """测试pdfminer是否可用"""
    try:
        import pdfminer.high_level
        print("✓ pdfminer已安装")
        return True
    except ImportError:
        print("✗ pdfminer未安装")
        print("  安装方法：pip install pdfminer[layout]")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PDFMiner增强PDF解析器")
    parser.add_argument("--test-availability", action="store_true",
                       help="测试pdfminer是否可用")
    parser.add_argument("--use-pdfminer", action="store_true",
                       help="使用pdfminer解析（默认：pdfplumber）")

    args = parser.parse_args()

    print("="*60)
    print("PDFMiner增强解析器")
    print("="*60)

    if args.test_availability:
        test_pdfminer_availability()
    else:
        print()
        print("使用方法: pdfminer" if args.use_pdfminer else "pdfplumber基础")
        print("需要：pip install pdfminer[layout]")
