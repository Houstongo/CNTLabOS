"""
PDF文本清理工具
解决PDF解析导致的乱码和换行问题
"""

import sqlite3
import re
from pathlib import Path


def clean_pdf_text(text: str) -> str:
    """
    清理PDF解析的常见问题

    Args:
        text: 原始文本

    Returns:
        清理后的文本
    """
    # 1. 修复连字符换行（如 "electri- \ncal" → "electrical"）
    text = re.sub(r'(?<!\w)-\n(?=\w)', '', text)

    # 2. 修复单词换行（如 "electri- \ncal" → "electrical"）
    text = re.sub(r'(?<!\w)\n(?=\w)', ' ', text)

    # 3. 修复句号换行（如 "sentence. \nNext" → "sentence. Next"）
    text = re.sub(r'\.\s*\n(?=[A-Z])', '. ', text)

    # 4. 修复多余空格
    text = re.sub(r'\s+', ' ', text)

    # 5. 修复常见错拼和分割
    corrections = {
        'ectrical': 'electrical',
        'conduc tivity': 'conductivity',
        'nano tube': 'nanotube',
        'carbon nano tube': 'carbon nanotube',
        'sity': 'density',
        'horizontion': 'horizontal alignment',
        'ofthe': 'of the',
        'tothe': 'to the',
        'andthe': 'and the',
        'withthe': 'with the',
        'fromthe': 'from the',
        'forthe': 'for the',
        'onthe': 'on the',
        'atthe': 'at the',
        'inthe': 'in the',
        'decre ases': 'decreases',
        'incre ases': 'increases',
        'perfor mance': 'performance',
        'charac teristic': 'characteristic',
        'propert ies': 'properties',
        'mecha nism': 'mechanism',
        'temper ature': 'temperature',
        'thick ness': 'thickness',
        'align ment': 'alignment',
        'dens ity': 'density',
    }

    for wrong, right in corrections.items():
        text = text.replace(wrong, right)

    # 6. 移除重复单词
    text = re.sub(r'\b(\w+)\s+\1\b', r'\1', text)

    # 7. 修复空格和标点
    text = re.sub(r'\s+([.,;!?:])', r'\1', text)  # 标点前移除空格
    text = re.sub(r'([.,;!?:])\s+', r'\1 ', text)  # 标点后添加一个空格

    # 8. 修复连在一起的英文单词
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)

    # 9. 移除控制字符
    text = ''.join(char for char in text if char.isprintable() or char.isspace())

    return text.strip()


def clean_all_chunks(kb_path: str, output_path: str = None):
    """
    清理所有文本块

    Args:
        kb_path: 知识库数据库路径
        output_path: 输出数据库路径（可选，默认原库）
    """
    if output_path is None:
        output_path = kb_path

    print(f"连接数据库: {kb_path}")
    conn = sqlite3.connect(kb_path)
    conn.row_factory = sqlite3.Row

    # 获取所有文本块
    chunks = conn.execute("SELECT id, text FROM kb_chunks ORDER BY id").fetchall()
    print(f"找到 {len(chunks)} 个文本块")

    # 清理每个块
    updated_count = 0
    for chunk in chunks:
        chunk_id = chunk["id"]
        original_text = chunk["text"]

        cleaned_text = clean_pdf_text(original_text)

        if cleaned_text != original_text:
            conn.execute(
                "UPDATE kb_chunks SET text = ? WHERE id = ?",
                (cleaned_text, chunk_id)
            )
            updated_count += 1

            if updated_count % 100 == 0:
                print(f"  已清理 {updated_count}/{len(chunks)} 个块")
                conn.commit()

    conn.commit()
    conn.close()

    print(f"\n清理完成!")
    print(f"总计更新: {updated_count} 个块")
    print(f"覆盖率: {updated_count}/{len(chunks)} ({updated_count/len(chunks)*100:.1f}%)")


def preview_cleaning(kb_path: str, num_samples: int = 5):
    """
    预览清理效果

    Args:
        kb_path: 知识库数据库路径
        num_samples: 样本数量
    """
    conn = sqlite3.connect(kb_path)
    conn.row_factory = sqlite3.Row

    chunks = conn.execute("SELECT text FROM kb_chunks ORDER BY id LIMIT ?", (num_samples,)).fetchall()

    print("清理效果预览:")
    print("="*60)

    for i, chunk in enumerate(chunks, 1):
        original = chunk["text"][:150]
        cleaned = clean_pdf_text(original)[:150]

        print(f"\n样本 {i}:")
        print(f"原始: {original}")
        print(f"清理: {cleaned}")

    conn.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="PDF文本清理工具")
    parser.add_argument("--action", choices=["clean", "preview", "stats"], default="clean",
                       help="操作类型")
    parser.add_argument("--kb-path", default="database/cnta_knowledge_base.sqlite",
                       help="知识库数据库路径")
    parser.add_argument("--samples", type=int, default=5,
                       help="预览样本数量")
    parser.add_argument("--output", help="输出数据库路径（可选）")

    args = parser.parse_args()

    print("="*60)
    print("PDF文本清理工具")
    print("="*60)

    if args.action == "preview":
        print(f"预览清理效果 (样本数: {args.samples})")
        preview_cleaning(args.kb_path, args.samples)
    elif args.action == "stats":
        print("统计信息")
        conn = sqlite3.connect(args.kb_path)
        total = conn.execute("SELECT COUNT(*) FROM kb_chunks").fetchone()[0]
        print(f"总文本块数: {total}")
        conn.close()
    else:  # clean
        print("开始清理所有文本块...")
        clean_all_chunks(args.kb_path, args.output)

    print("="*60)


if __name__ == "__main__":
    main()
