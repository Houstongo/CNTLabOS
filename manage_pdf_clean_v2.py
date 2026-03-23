"""
强力PDF文本清理工具 - 解决乱码和格式问题（修复版）
"""

import sqlite3
import re


def aggressive_clean_pdf_text(text: str) -> str:
    """
    强力清理PDF解析问题

    Args:
        text: 原始文本

    Returns:
        清理后的文本
    """
    # 1. 移除控制字符
    text = ''.join(char for char in text if char.isprintable() or char.isspace())

    # 2. 修复连字符换行
    text = re.sub(r'(?<!\w)-\n(?=\w)', '', text)

    # 3. 修复单词换行（包括各种连字符）
    text = re.sub(r'(?<!\w)[\-\s]\n(?=\w)', ' ', text)

    # 4. 修复句号换行
    text = re.sub(r'\.\s*\n(?=[A-Z])', '. ', text)
    text = re.sub(r'\.\s*\n\s*(?=[A-Z][a-z])', '. ', text)

    # 5. 强力清理常见错拼和分割
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

        # 领域特定修正
        'CNT arrays': 'CNT arrays',
        'carbon nanotube arrays': 'carbon nanotube arrays',
        'aligned CNT': 'aligned CNT',
        'horizontally aligned CNT': 'horizontally aligned CNT',

        # 常见学术写作问题
        'resultsdemonstratethat': 'results demonstrated that',
        'further substantiated through': 'further substantiated through',
        'Theresultsclearlydemonstratethat': 'The results clearly demonstrated that',
        'couldbeattributed eter': 'could be attributed to other',
        'ForHRTEManalysis': 'For HRTEM analysis',
        'somesingle-walled': 'some single-walled',
        'tosubstrate': 'to substrate',
        'anddisrupt': 'and disrupt',
    }

    for wrong, right in corrections.items():
        text = text.replace(wrong, right)

    # 7. 移除重复单词（激进）
    text = re.sub(r'\b(\w+)\s+\1\b', r'\1', text)

    # 8. 移除连续空格
    text = re.sub(r'\s{2,}', ' ', text)

    # 9. 修复标点周围空格
    text = re.sub(r'\s+([.,;:!?)', r' \1', text)
    text = re.sub(r'([.,;:!?)\s+', r'\1 ', text)

    # 10. 移除孤立的标点
    text = re.sub(r'\s+([.,;:!?)\s+', ' ', text)

    # 11. 修复括号周围空格
    text = re.sub(r'\s*\(\s*', ' (', text)
    text = re.sub(r'\)\s*', ') ', text)

    # 12. 移除行首/行尾空格
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        cleaned_lines.append(line.strip())
    text = '\n'.join(cleaned_lines)

    # 13. 移除空行
    text = re.sub(r'\n\s*\n', '\n', text)
    text = re.sub(r'\n\s*\n', '\n', text)

    return text.strip()


def smart_clean_all_chunks(kb_path: str):
    """
    智能清理文本块（只清理有问题的块）

    Args:
        kb_path: 知识库数据库路径
    """
    print(f"连接数据库: {kb_path}")
    conn = sqlite3.connect(kb_path)
    conn.row_factory = sqlite3.Row

    # 获取所有文本块
    chunks = conn.execute("SELECT id, text, length(text) as length FROM kb_chunks ORDER BY id").fetchall()
    print(f"找到 {len(chunks)} 个文本块")

    # 分析文本质量
    problem_chunks = []
    good_chunks = []

    for chunk in chunks:
        chunk_id = chunk["id"]
        text = chunk["text"]
        text_len = chunk["length"]

        # 检测问题指标
        has_problems = False
        reasons = []

        # 1. 检查连在一起的单词
        if re.search(r'[a-z][A-Z]', text):
            has_problems = True
            reasons.append("大小写边界问题")

        # 2. 检查常见错拼
        problem_words = ['ectrical', 'conduc tivity', 'nano tube', 'sity', 'horizontion']
        if any(word in text.lower() for word in problem_words):
            has_problems = True
            reasons.append("常见错拼")

        # 3. 检查长度异常
        if text_len < 50 or text_len > 500:
            has_problems = True
            reasons.append(f"长度异常({text_len})")

        # 4. 检查重复模式
        duplicate_patterns = [r'(\w+)\s+\1', r'\.{2,}', r';{2,}']
        for pattern in duplicate_patterns:
            if re.search(pattern, text):
                has_problems = True
                reasons.append("重复模式")
                break

        if has_problems:
            problem_chunks.append((chunk_id, text, reasons))
        else:
            good_chunks.append((chunk_id, text))

    print(f"\n文本质量分析:")
    print(f"  有问题的文本块: {len(problem_chunks)}")
    print(f"  正常的文本块: {len(good_chunks)}")

    # 清理有问题的块
    updated_count = 0
    for chunk_id, text in problem_chunks:
        cleaned_text = aggressive_clean_pdf_text(text)

        if cleaned_text != text:
            conn.execute(
                "UPDATE kb_chunks SET text = ? WHERE id = ?",
                (cleaned_text, chunk_id)
            )
            updated_count += 1

            if updated_count % 100 == 0:
                print(f"  已清理 {updated_count}/{len(problem_chunks)} 个块")
                conn.commit()

    conn.commit()
    conn.close()

    print(f"\n清理完成!")
    print(f"总计更新: {updated_count} 个有问题的块")
    print(f"预计改善率: {updated_count}/{len(chunks)} ({updated_count/len(chunks)*100:.1f}%)")


def preview_cleaning(kb_path: str, num_samples: int = 3):
    """预览清理效果"""
    print("强力清理效果预览:")
    print("="*60)

    conn = sqlite3.connect(kb_path)
    conn.row_factory = sqlite3.Row

    # 获取有问题的文本块
    problem_chunks = conn.execute("""
        SELECT c.id, c.text
        FROM kb_chunks c
        LEFT JOIN kb_msfu m ON m.chunk_id = c.id
        WHERE m.id IS NULL
          AND (c.text LIKE '%ectrical%'
               OR c.text LIKE '%conduc tivity%'
               OR c.text LIKE '%nano tube%'
               OR c.text LIKE '%sity%'
               OR c.text LIKE '%horizontion%')
        LIMIT ?
    """, (num_samples,)).fetchall()

    for i, chunk in enumerate(problem_chunks, 1):
        original = chunk["text"][:150]
        cleaned = aggressive_clean_pdf_text(original)[:150]

        print(f"\n样本 {i}:")
        print(f"原始: {original}")
        print(f"清理: {cleaned}")

    conn.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="强力PDF文本清理工具")
    parser.add_argument("--action", choices=["clean", "preview", "stats"],
                       default="clean", help="操作类型")
    parser.add_argument("--kb-path", default="database/cnta_knowledge_base.sqlite",
                       help="知识库数据库路径")
    parser.add_argument("--samples", type=int, default=3,
                       help="预览样本数量")

    args = parser.parse_args()

    print("="*60)
    print("强力PDF文本清理工具")
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
    elif args.action == "clean":
        print("开始智能清理有问题的文本块...")
        smart_clean_all_chunks(args.kb_path)

    print("="*60)


if __name__ == "__main__":
    main()