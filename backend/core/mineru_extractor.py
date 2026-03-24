"""
MinerU 云端 API PDF解析器（上海人工智能实验室）
通过 REST API 调用云端解析服务，无需本地模型。

API 文档: https://mineru.net/apiManage/docs
"""

import os
import io
import json
import re
import time
import zipfile
import tempfile
import requests

# Markdown 本地备份目录
MINERU_OUTPUT_DIR = r"D:\CNTDATA\CNTA_ML_Project\database\mineru_output"
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

# MinerU 精准解析 API (v4)
MINERU_API_BASE = "https://mineru.net/api/v4"
MINERU_BATCH_UPLOAD_URL = f"{MINERU_API_BASE}/file-urls/batch"
MINERU_BATCH_RESULT_URL = f"{MINERU_API_BASE}/extract-results/batch/{{batch_id}}"

# 配置文件路径
DEFAULT_CONFIG_PATH = os.path.join(os.path.expanduser("~"), "magic-pdf.json")


def _load_api_key(config_path: str = DEFAULT_CONFIG_PATH) -> str:
    """从配置文件加载 API Key"""
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        return config.get("api-key", "")
    return os.environ.get("MINERU_API_KEY", "")


@dataclass
class MinerUParsedDocument:
    """MinerU解析的PDF文档"""
    doc_id: int = 0
    title: str = ""
    pages: List[str] = field(default_factory=list)
    tables: List[Dict] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    markdown: str = ""

    # 性能指标
    parse_time: float = 0.0
    text_quality_score: float = 0.0
    layout_quality_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "page_count": len(self.pages) if self.pages else 0,
            "table_count": len(self.tables) if self.tables else 0,
            "parse_time": self.parse_time,
            "text_quality_score": self.text_quality_score,
            "layout_quality_score": self.layout_quality_score,
            "metadata": self.metadata or {}
        }


class MinerUExtractor:
    """基于 MinerU 云端 API (v4) 的 PDF 解析器"""

    def __init__(self, api_key: str = "", model_version: str = "vlm"):
        """
        Args:
            api_key: MinerU API Key，不传则自动从配置文件读取
            model_version: 模型版本 - "vlm"(推荐), "pipeline", "MinerU-HTML"
        """
        self.api_key = api_key or _load_api_key()
        self.model_version = model_version

    def test_installation(self) -> bool:
        """测试 API Key 是否可用"""
        if not self.api_key:
            print("API Key 未配置")
            print("   请设置 ~/magic-pdf.json 中的 api-key 字段")
            return False
        print(f"MinerU API Key 已配置 (长度={len(self.api_key)})")
        return True

    def parse_pdf_bytes(self, pdf_bytes: bytes, title: str = "",
                        doc_id: int = 0) -> MinerUParsedDocument:
        """
        通过云端 API 解析 PDF 字节数据

        流程: 申请上传URL → PUT上传 → 轮询结果 → 下载zip解压 → 提取markdown
        """
        start_time = time.time()

        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }
            file_name = title or "document.pdf"
            if not file_name.endswith(".pdf"):
                file_name += ".pdf"

            # 1. 申请上传链接
            data = {
                "files": [{"name": file_name}],
                "model_version": self.model_version,
                "enable_formula": True,
                "enable_table": True,
            }
            resp = requests.post(MINERU_BATCH_UPLOAD_URL, headers=headers, json=data, timeout=30)
            resp.raise_for_status()
            result = resp.json()

            if result.get("code") != 0:
                raise RuntimeError(f"申请上传链接失败: {result.get('msg')}")

            batch_id = result["data"]["batch_id"]
            upload_url = result["data"]["file_urls"][0]
            print(f"  已获取上传链接, batch_id={batch_id}")

            # 2. PUT 上传文件到 OSS（不需要 Content-Type）
            put_resp = requests.put(upload_url, data=pdf_bytes, timeout=120)
            if put_resp.status_code not in (200, 201):
                raise RuntimeError(f"文件上传失败: HTTP {put_resp.status_code}")
            print(f"  文件上传成功 ({len(pdf_bytes)/1024:.0f}KB)")

            # 3. 轮询等待解析完成
            zip_url = self._poll_batch_result(batch_id)
            if not zip_url:
                raise RuntimeError("解析超时或失败")

            # 4. 下载 zip 并提取 markdown，同时本地备份
            markdown = self._download_markdown(zip_url, file_name=file_name)
            if not markdown:
                raise RuntimeError("下载解析结果失败")

            # 5. 提取表格和按页文本
            pages, tables = self._extract_from_markdown(markdown)
            parse_time = time.time() - start_time

            return MinerUParsedDocument(
                doc_id=doc_id,
                title=title or "untitled.pdf",
                pages=pages,
                tables=tables,
                markdown=markdown,
                metadata={
                    "model_version": self.model_version,
                    "file_size": len(pdf_bytes) / 1024 / 1024,
                    "page_count": len(pages),
                    "batch_id": batch_id,
                },
                parse_time=parse_time,
                text_quality_score=self._assess_text_quality(pages),
                layout_quality_score=self._assess_layout_quality(pages, tables),
            )

        except Exception as e:
            print(f"MinerU API 解析失败: {e}")
            import traceback
            traceback.print_exc()
            return MinerUParsedDocument(
                doc_id=doc_id,
                title=title,
                parse_time=time.time() - start_time,
            )

    def parse_pdf_document(self, doc_id: int, pdf_path: str) -> Optional[MinerUParsedDocument]:
        """解析单个PDF文件"""
        if not os.path.exists(pdf_path):
            print(f"文件不存在: {pdf_path}")
            return None

        with open(pdf_path, 'rb') as f:
            pdf_bytes = f.read()

        title = os.path.basename(pdf_path)
        result = self.parse_pdf_bytes(pdf_bytes, title=title, doc_id=doc_id)

        if not result.markdown:
            print(f"MinerU 解析无有效内容: {title}")
            return None

        return result

    def _poll_batch_result(self, batch_id: str, poll_interval: int = 5,
                           max_wait: int = 300) -> Optional[str]:
        """
        轮询批量任务结果

        Returns:
            完成后返回 full_zip_url，失败返回 None
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        url = MINERU_BATCH_RESULT_URL.format(batch_id=batch_id)
        elapsed = 0

        while elapsed < max_wait:
            time.sleep(poll_interval)
            elapsed += poll_interval

            try:
                resp = requests.get(url, headers=headers, timeout=30)
                resp.raise_for_status()
                result = resp.json()

                if result.get("code") != 0:
                    print(f"  查询失败: {result.get('msg')}")
                    continue

                extract_results = result.get("data", {}).get("extract_result", [])
                if not extract_results:
                    continue

                item = extract_results[0]
                state = item.get("state", "")

                if state == "done":
                    zip_url = item.get("full_zip_url")
                    print(f"  解析完成 ({elapsed}s)")
                    return zip_url

                elif state == "failed":
                    print(f"  解析失败: {item.get('err_msg', '未知错误')}")
                    return None

                else:
                    progress = item.get("extract_progress", {})
                    if progress:
                        print(f"  {state}... ({progress.get('extracted_pages', '?')}/{progress.get('total_pages', '?')}页, {elapsed}s)")
                    else:
                        print(f"  {state}... ({elapsed}s)")

            except requests.RequestException as e:
                print(f"  轮询异常: {e}, 继续等待...")

        print(f"  解析超时 ({max_wait}s)")
        return None

    @staticmethod
    def _download_markdown(zip_url: str, file_name: str = "",
                            max_retries: int = 3) -> Optional[str]:
        """下载 zip 并提取 full.md（带重试），同时保存本地备份"""
        for attempt in range(1, max_retries + 1):
            try:
                resp = requests.get(zip_url, timeout=120)
                resp.raise_for_status()

                with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                    md_names = [n for n in zf.namelist() if n.endswith("full.md")]
                    if not md_names:
                        md_names = [n for n in zf.namelist() if n.endswith(".md")]

                    if not md_names:
                        print(f"  zip 中未找到 .md 文件: {zf.namelist()}")
                        return None

                    markdown = zf.read(md_names[0]).decode("utf-8")

                    # 本地备份 markdown
                    if file_name and markdown:
                        MinerUExtractor._save_markdown(file_name, markdown, zf)

                    return markdown

            except Exception as e:
                print(f"  下载尝试 {attempt}/{max_retries} 失败: {e}")
                if attempt < max_retries:
                    time.sleep(3)

        print("  下载/解压最终失败")
        return None

    @staticmethod
    def _save_markdown(file_name: str, markdown: str, zip_file: zipfile.ZipFile):
        """将 markdown 和 zip 保存到本地备份目录"""
        os.makedirs(MINERU_OUTPUT_DIR, exist_ok=True)

        # 用文件名生成安全目录名
        safe_name = os.path.splitext(file_name)[0].replace(" ", "_")[:80]

        # 每个文档一个子目录，避免重名覆盖
        doc_dir = os.path.join(MINERU_OUTPUT_DIR, safe_name)
        os.makedirs(doc_dir, exist_ok=True)

        # 保存 full.md
        md_path = os.path.join(doc_dir, "full.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(markdown)

        # 保存原始 zip（包含 json 等额外输出）
        zip_path = os.path.join(doc_dir, f"{safe_name}.zip")
        if not os.path.exists(zip_path):
            try:
                zip_bytes = zip_file.fp.seek(0) or zip_file.fp.read()
                with open(zip_path, "wb") as f:
                    f.write(zip_bytes)
            except Exception:
                pass

        print(f"  已备份到: {doc_dir}")

    @staticmethod
    def _extract_from_markdown(markdown: str) -> tuple:
        """从 Markdown 中提取按页文本和表格"""
        tables = []
        table_pattern = re.compile(
            r'(\|.+\|\n\|[-| :]+\|\n(?:\|.+\|\n)*)', re.MULTILINE
        )

        table_count = 0
        for match in table_pattern.finditer(markdown):
            table_count += 1
            md_table = match.group(1)
            tables.append({
                "page": 0,
                "index": table_count,
                "markdown": md_table.strip(),
                "html": MinerUExtractor._table_md_to_html(md_table),
            })

        # 按标题切分页面
        sections = re.split(r'\n(?=#{1,3} )', markdown)
        page_texts = []
        for section in sections:
            text = section.strip()
            if text:
                text = table_pattern.sub('', text).strip()
                if text:
                    page_texts.append(text)

        return page_texts if page_texts else [markdown], tables

    @staticmethod
    def _table_md_to_html(md_text: str) -> str:
        """将 Markdown 表格转为 HTML"""
        if not md_text:
            return ""
        lines = [l.strip() for l in md_text.strip().split("\n") if l.strip()]
        if not lines:
            return ""
        html = "<table>"
        for i, line in enumerate(lines):
            if set(line.replace("|", "").replace("-", "").replace(":", "").strip()) == set():
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            tag = "th" if i == 0 else "td"
            html += "<tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in cells) + "</tr>"
        html += "</table>"
        return html

    @staticmethod
    def _assess_text_quality(pages: List[str]) -> float:
        if not pages:
            return 0.0
        total_chars = sum(len(p) for p in pages)
        avg_chars = total_chars / len(pages)
        score = 0.5
        if avg_chars >= 1000:
            score += 0.3
        elif avg_chars >= 500:
            score += 0.2
        if avg_chars < 200:
            score -= 0.2
        return max(0.0, min(1.0, score))

    @staticmethod
    def _assess_layout_quality(pages: List[str], tables: List[Dict]) -> float:
        if not pages:
            return 0.0
        score = 0.5
        if tables:
            score += 0.3
        if len(pages) >= 3:
            score += 0.2
        return min(1.0, score)


# ==================== 便捷函数 ====================

def parse_pdf_to_markdown(pdf_path: str) -> str:
    """将 PDF 文件解析为 Markdown 文本"""
    extractor = MinerUExtractor()
    doc = extractor.parse_pdf_document(0, pdf_path)
    return doc.markdown if doc else ""


def parse_pdf_to_text(pdf_bytes: bytes, title: str = "") -> str:
    """将 PDF 字节数据解析为文本（兼容旧接口）"""
    extractor = MinerUExtractor()
    doc = extractor.parse_pdf_bytes(pdf_bytes, title=title)
    return doc.markdown if doc else ""
