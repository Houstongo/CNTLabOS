"""
RAG 检索模块 - 三路知识检索
1. 实验数据库（SQLite）：检索特征/参数相似的历史实验
2. 领域文献（PDF）：BM25关键词检索
3. 预设专家知识：通过 AIInterpreter 的 System Prompt 注入
"""
import sqlite3
import re
import math
from typing import Optional


class RAGRetriever:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_schema()

    # ------------------------------------------------------------------ #
    #  Schema 初始化
    # ------------------------------------------------------------------ #
    def init_schema(self):
        conn = sqlite3.connect(self.db_path)
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS rag_documents (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                filename   TEXT    NOT NULL,
                created_at TEXT    DEFAULT (datetime('now','localtime'))
            );

            CREATE TABLE IF NOT EXISTS rag_chunks (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id      INTEGER NOT NULL REFERENCES rag_documents(id) ON DELETE CASCADE,
                chunk_index INTEGER NOT NULL,
                text        TEXT    NOT NULL,
                keywords    TEXT
            );
        """)
        conn.commit()
        conn.close()

    # ------------------------------------------------------------------ #
    #  PDF 文献管理
    # ------------------------------------------------------------------ #
    def add_pdf(self, file_bytes: bytes, filename: str) -> dict:
        """提取PDF文本并分块存入数据库，返回 {doc_id, chunk_count}"""
        import pdfplumber
        import io

        text_pages = []
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text_pages.append(t)

        full_text = "\n".join(text_pages)
        chunks = self._split_text(full_text, chunk_size=600, overlap=80)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO rag_documents (filename) VALUES (?)", (filename,))
        doc_id = cursor.lastrowid

        for idx, chunk in enumerate(chunks):
            keywords = self._extract_keywords(chunk)
            cursor.execute(
                "INSERT INTO rag_chunks (doc_id, chunk_index, text, keywords) VALUES (?,?,?,?)",
                (doc_id, idx, chunk, " ".join(keywords))
            )
        conn.commit()
        conn.close()
        return {"doc_id": doc_id, "chunk_count": len(chunks)}

    def list_documents(self) -> list:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT d.id, d.filename, d.created_at, COUNT(c.id) AS chunk_count "
            "FROM rag_documents d LEFT JOIN rag_chunks c ON c.doc_id = d.id "
            "GROUP BY d.id ORDER BY d.id DESC"
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def delete_document(self, doc_id: int):
        conn = sqlite3.connect(self.db_path)
        conn.execute("DELETE FROM rag_documents WHERE id = ?", (doc_id,))
        conn.commit()
        conn.close()

    # ------------------------------------------------------------------ #
    #  检索：实验数据库相似实验
    # ------------------------------------------------------------------ #
    def retrieve_from_db(self, features: dict, params: dict, top_k: int = 5) -> list:
        """
        按工艺参数相似度检索，返回 top_k 条相似实验摘要。
        相似度 = 温度差 + 气流差（归一化加权）
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # 只检索已处理（有特征值）的实验
        cursor.execute("""
            SELECT id, sample_id, source, growth_temp, growth_time,
                   ar_flow, h2_flow, c2h4_flow,
                   fe_thickness, al2o3_thickness,
                   diameter, density, alignment, curvature,
                   position_label
            FROM images
            WHERE processed = 1
              AND id != ?
            ORDER BY id DESC
            LIMIT 200
        """, (params.get('current_id', -1),))
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return []

        # 计算简单相似度分数（越小越相似）
        def score(row):
            s = 0.0
            gt = params.get('growth_temp')
            if gt and row['growth_temp']:
                s += abs(gt - row['growth_temp']) / max(gt, 1) * 2.0
            fe = params.get('fe_thickness')
            if fe and row['fe_thickness']:
                s += abs(fe - row['fe_thickness']) / max(fe, 1) * 1.5
            ar = params.get('ar_flow')
            if ar and row['ar_flow']:
                s += abs(ar - row['ar_flow']) / max(ar, 1)
            return s

        scored = sorted(rows, key=score)[:top_k]

        results = []
        for r in scored:
            results.append({
                "id": r["id"],
                "sample_id": r["sample_id"],
                "source": r["source"],
                "growth_temp": r["growth_temp"],
                "growth_time": r["growth_time"],
                "ar_flow": r["ar_flow"],
                "h2_flow": r["h2_flow"],
                "c2h4_flow": r["c2h4_flow"],
                "fe_thickness": r["fe_thickness"],
                "al2o3_thickness": r["al2o3_thickness"],
                "diameter": r["diameter"],
                "density": r["density"],
                "alignment": r["alignment"],
                "curvature": r["curvature"],
                "position_label": r["position_label"],
            })
        return results

    # ------------------------------------------------------------------ #
    #  检索：PDF文献 BM25
    # ------------------------------------------------------------------ #
    def retrieve_from_pdf(self, query: str, top_k: int = 3) -> list:
        """BM25 检索文献片段，返回 top_k 条文本片段"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT c.id, c.text, c.keywords, d.filename "
            "FROM rag_chunks c JOIN rag_documents d ON d.id = c.doc_id"
        ).fetchall()
        conn.close()

        if not rows:
            return []

        query_terms = set(self._tokenize(query))
        if not query_terms:
            return []

        corpus = [(r["text"], set(r["keywords"].split()) if r["keywords"] else set()) for r in rows]
        scores = self._bm25_score(query_terms, corpus)

        top_indices = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append({
                    "filename": rows[idx]["filename"],
                    "text": rows[idx]["text"][:400],  # 截断避免token过多
                    "score": round(scores[idx], 3),
                })
        return results

    # ------------------------------------------------------------------ #
    #  合并三路检索结果
    # ------------------------------------------------------------------ #
    def retrieve_all(self, features: dict, params: dict, query: str) -> dict:
        similar_exps = self.retrieve_from_db(features, params)
        pdf_docs = self.retrieve_from_pdf(query) if query else []
        return {
            "similar_experiments": similar_exps,
            "pdf_passages": pdf_docs,
        }

    # ------------------------------------------------------------------ #
    #  工具方法
    # ------------------------------------------------------------------ #
    @staticmethod
    def _split_text(text: str, chunk_size: int = 600, overlap: int = 80) -> list:
        """按固定字符数分块，带重叠"""
        text = re.sub(r'\s+', ' ', text).strip()
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(text[start:end])
            if end == len(text):
                break
            start += chunk_size - overlap
        return chunks

    @staticmethod
    def _tokenize(text: str) -> list:
        """简单分词：英文单词 + 中文字符"""
        words = re.findall(r'[a-zA-Z]+|[\u4e00-\u9fff]+', text.lower())
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'on', 'of',
            'for', 'to', 'and', 'or', 'with', 'at', 'by', 'from', 'as',
            '的', '了', '在', '是', '和', '与', '对', '中', '为', '以',
        }
        return [w for w in words if w not in stopwords and len(w) > 1]

    def _extract_keywords(self, text: str, top_n: int = 20) -> list:
        """提取文本中频率最高的词作为关键词"""
        tokens = self._tokenize(text)
        freq: dict = {}
        for t in tokens:
            freq[t] = freq.get(t, 0) + 1
        sorted_tokens = sorted(freq, key=lambda x: -freq[x])
        return sorted_tokens[:top_n]

    @staticmethod
    def _bm25_score(query_terms: set, corpus: list, k1: float = 1.5, b: float = 0.75) -> list:
        """BM25 打分"""
        N = len(corpus)
        avgdl = sum(len(text.split()) for text, _ in corpus) / max(N, 1)

        scores = []
        for text, kw_set in corpus:
            tokens = text.lower().split()
            dl = len(tokens)
            score = 0.0
            for term in query_terms:
                tf = tokens.count(term)
                df = sum(1 for _, kws in corpus if term in kws or term in text.lower())
                if df == 0:
                    continue
                idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
                norm_tf = tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))
                score += idf * norm_tf
            scores.append(score)
        return scores
