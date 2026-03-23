"""
One-click bulk ingestion for CNT literature candidates.

Features:
1) Deduplication (candidate-internal, already-ingested, historical success ledger)
2) Retry on download/ingestion failures
3) JSONL run log + persistent success ledger

Usage examples:
  python scripts/bulk_ingest_candidates.py --dry-run --max-docs 20
  python scripts/bulk_ingest_candidates.py --download-only --max-docs 20
  python scripts/bulk_ingest_candidates.py --max-docs 30 --rebuild-links
  python scripts/bulk_ingest_candidates.py --status-allow direct_pdf_candidate,oa_pdf
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import re
import sqlite3
import sys
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import quote, urlparse

import requests


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KB_DB = ROOT / "database" / "cnta_knowledge_base.sqlite"
DEFAULT_DOWNLOAD_DIR = ROOT / "reports" / "rag_pdf_cache"
DEFAULT_LOG_DIR = ROOT / "reports" / "rag_ingest_logs"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STATUS_ALLOW_DEFAULT = {"direct_pdf_verified", "direct_pdf_candidate", "oa_pdf"}
PRIORITY_RANK = {"P0": 3, "P1": 2, "P2": 1}
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0.0.0 Safari/537.36"
)


@dataclass
class Candidate:
    priority: str
    bucket: str
    title: str
    year: str
    journal: str
    doi: str
    relation_targets: str
    notes: str
    source_url: str
    download_status: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bulk-ingest CNT candidate papers into KB.")
    parser.add_argument("--candidate-csv", default="", help="Path to candidate CSV. Default uses project config.")
    parser.add_argument("--kb-db", default="", help="Path to knowledge DB. Empty means auto-detect.")
    parser.add_argument("--download-dir", default=str(DEFAULT_DOWNLOAD_DIR), help="Directory to store downloaded PDFs.")
    parser.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR), help="Directory for run logs and success ledger.")
    parser.add_argument("--status-allow", default="direct_pdf_verified,direct_pdf_candidate,oa_pdf")
    parser.add_argument("--max-docs", type=int, default=0, help="0 means no limit.")
    parser.add_argument("--timeout", type=int, default=45)
    parser.add_argument("--retry", type=int, default=3)
    parser.add_argument("--retry-backoff", type=float, default=1.2)
    parser.add_argument("--sleep", type=float, default=0.1, help="Sleep between documents (seconds).")
    parser.add_argument("--source-type", default="paper")
    parser.add_argument("--theme", default="growth_mechanism")
    parser.add_argument("--is-core", action="store_true")
    parser.add_argument("--download-only", action="store_true", help="Only download PDFs, do not ingest into KB.")
    parser.add_argument("--rebuild-links", action="store_true", help="Rebuild kb_links after ingestion.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def candidate_csv_path(cli_value: str) -> Path:
    if cli_value:
        return Path(cli_value).expanduser()
    from backend.core.knowledge_base import KnowledgeBaseService

    return Path(KnowledgeBaseService._candidate_csv_path())  # noqa: SLF001 (intentional)


def has_kb_documents_table(path: Path) -> bool:
    if not path.exists():
        return False
    conn = sqlite3.connect(str(path))
    try:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='kb_documents'"
        ).fetchone()
        return bool(row)
    finally:
        conn.close()


def resolve_kb_db_path(cli_value: str) -> Path:
    if cli_value:
        return Path(cli_value).expanduser()
    candidates = [
        ROOT / "database" / "cnta_knowledge_base.sqlite",
        ROOT / "database" / "cnta_knowledge.sqlite",
        ROOT / "database" / "cnta_knowledge_fresh.sqlite",
    ]
    for path in candidates:
        if has_kb_documents_table(path):
            return path
    return DEFAULT_KB_DB


def read_candidates(path: Path) -> List[Candidate]:
    if not path.exists():
        raise FileNotFoundError(f"Candidate CSV not found: {path}")
    rows: List[Dict[str, str]] = []
    last_exc: Optional[Exception] = None
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            with path.open("r", encoding=encoding, newline="") as handle:
                rows = list(csv.DictReader(handle))
            break
        except Exception as exc:  # pragma: no cover - fallback path
            last_exc = exc
            rows = []
    if not rows and last_exc:
        raise RuntimeError(f"Failed to read candidate CSV: {last_exc}") from last_exc

    out: List[Candidate] = []
    for row in rows:
        out.append(
            Candidate(
                priority=str((row or {}).get("priority", "")).strip(),
                bucket=str((row or {}).get("bucket", "")).strip(),
                title=str((row or {}).get("title", "")).strip(),
                year=str((row or {}).get("year", "")).strip(),
                journal=str((row or {}).get("journal", "")).strip(),
                doi=normalize_doi(str((row or {}).get("doi", "")).strip()),
                relation_targets=str((row or {}).get("relation_targets", "")).strip(),
                notes=str((row or {}).get("notes", "")).strip(),
                source_url=str((row or {}).get("source_url", "")).strip(),
                download_status=str((row or {}).get("download_status", "")).strip(),
            )
        )
    return out


def normalize_doi(value: str) -> str:
    doi = (value or "").strip().lower()
    doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi)
    return doi


def normalize_url(value: str) -> str:
    raw = (value or "").strip()
    if not raw:
        return ""
    parsed = urlparse(raw)
    scheme = (parsed.scheme or "https").lower()
    host = (parsed.netloc or "").lower()
    path = parsed.path or ""
    query = parsed.query or ""
    return f"{scheme}://{host}{path}" + (f"?{query}" if query else "")


def normalize_title_key(value: str) -> str:
    text = (value or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\u4e00-\u9fff]+", "", text)
    return text


def status_allow_set(raw: str) -> Set[str]:
    values = {item.strip() for item in (raw or "").split(",") if item.strip()}
    return values or set(STATUS_ALLOW_DEFAULT)


def sort_candidates(rows: List[Candidate]) -> List[Candidate]:
    def sort_key(row: Candidate) -> Tuple[int, int, str]:
        pr = PRIORITY_RANK.get(row.priority, 0)
        year = int(row.year) if row.year.isdigit() else 0
        return (-pr, -year, row.title.lower())

    return sorted(rows, key=sort_key)


def build_signatures(row: Candidate) -> Set[str]:
    signatures: Set[str] = set()
    if row.doi:
        signatures.add(f"doi:{row.doi}")
    if row.source_url:
        signatures.add(f"url:{normalize_url(row.source_url)}")
    title_key = normalize_title_key(row.title)
    if title_key:
        signatures.add(f"title:{title_key}")
    return signatures


def slugify(value: str, max_len: int = 80) -> str:
    text = (value or "").strip().lower()
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    if not text:
        text = "paper"
    return text[:max_len].strip("-") or "paper"


def build_pdf_filename(row: Candidate) -> str:
    title_slug = slugify(row.title, max_len=70)
    year = row.year if row.year.isdigit() else "na"
    fingerprint = row.doi or normalize_url(row.source_url) or row.title
    digest = hashlib.sha1(fingerprint.encode("utf-8")).hexdigest()[:10]
    return f"{year}_{title_slug}_{digest}.pdf"


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def append_jsonl(path: Path, payload: Dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def load_success_signatures(ledger_path: Path) -> Set[str]:
    if not ledger_path.exists():
        return set()
    seen: Set[str] = set()
    with ledger_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            signatures = payload.get("signatures") or []
            if isinstance(signatures, list):
                for sig in signatures:
                    if isinstance(sig, str) and sig:
                        seen.add(sig)
    return seen


def load_existing_doc_keys(db_path: Path) -> Tuple[Set[str], Set[str]]:
    if not db_path.exists():
        return set(), set()
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute("SELECT title, file_path FROM kb_documents").fetchall()
    finally:
        conn.close()
    title_keys: Set[str] = set()
    file_keys: Set[str] = set()
    for title, file_path in rows:
        title_key = normalize_title_key(str(title or ""))
        if title_key:
            title_keys.add(title_key)
        name = Path(str(file_path or "")).stem.strip().lower()
        if name:
            file_keys.add(name)
    return title_keys, file_keys


def looks_like_pdf_url(url: str) -> bool:
    lowered = (url or "").lower()
    return lowered.endswith(".pdf") or ".pdf?" in lowered or "/pdf" in lowered


def make_http_session() -> requests.Session:
    session = requests.Session()
    session.trust_env = False
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/pdf,*/*"})
    return session


def request_json(session: requests.Session, url: str, timeout: int = 20) -> Dict:
    resp = session.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def openalex_pdf_candidates(session: requests.Session, doi: str, timeout: int = 20) -> List[str]:
    if not doi:
        return []
    encoded = quote(f"https://doi.org/{doi}", safe="")
    url = f"https://api.openalex.org/works/{encoded}"
    try:
        payload = request_json(session, url, timeout=timeout)
    except Exception:
        return []
    urls: List[str] = []
    oa = payload.get("open_access") or {}
    for item in (
        oa.get("oa_url"),
        ((payload.get("best_oa_location") or {}).get("pdf_url")),
        ((payload.get("best_oa_location") or {}).get("landing_page_url")),
        ((payload.get("primary_location") or {}).get("pdf_url")),
    ):
        if isinstance(item, str) and item.strip():
            urls.append(item.strip())
    for loc in payload.get("locations") or []:
        if not isinstance(loc, dict):
            continue
        for item in (loc.get("pdf_url"), loc.get("landing_page_url")):
            if isinstance(item, str) and item.strip():
                urls.append(item.strip())
    return urls


def crossref_pdf_candidates(session: requests.Session, doi: str, timeout: int = 20) -> List[str]:
    if not doi:
        return []
    url = f"https://api.crossref.org/works/{quote(doi, safe='')}"
    try:
        payload = request_json(session, url, timeout=timeout)
    except Exception:
        return []
    message = payload.get("message") or {}
    out: List[str] = []
    for link in message.get("link") or []:
        if not isinstance(link, dict):
            continue
        content_type = str(link.get("content-type") or "").lower()
        candidate_url = str(link.get("URL") or "").strip()
        if not candidate_url:
            continue
        if "pdf" in content_type or looks_like_pdf_url(candidate_url):
            out.append(candidate_url)
    return out


def build_download_url_candidates(row: Candidate, session: requests.Session, timeout: int) -> List[str]:
    candidates: List[str] = []
    if row.source_url:
        candidates.append(row.source_url)
    if row.doi:
        candidates.extend(openalex_pdf_candidates(session=session, doi=row.doi, timeout=timeout))
        candidates.extend(crossref_pdf_candidates(session=session, doi=row.doi, timeout=timeout))
        candidates.append(f"https://doi.org/{row.doi}")

    normalized: List[str] = []
    seen: Set[str] = set()
    for item in candidates:
        key = normalize_url(item)
        if not key or key in seen:
            continue
        seen.add(key)
        normalized.append(key)
    return normalized


def download_pdf(
    session: requests.Session,
    url: str,
    out_path: Path,
    timeout: int,
) -> Tuple[int, str, str]:
    normalized = normalize_url(url)
    if not normalized:
        raise ValueError("Missing source_url.")

    with session.get(normalized, timeout=timeout, stream=True, allow_redirects=True) as response:
        response.raise_for_status()
        content_type = str(response.headers.get("Content-Type") or "").lower()
        final_path = out_path.with_name(
            f"{out_path.stem}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}{out_path.suffix}"
        )
        first_bytes = b""
        total_bytes = 0
        with final_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=64 * 1024):
                if not chunk:
                    continue
                if len(first_bytes) < 8:
                    needed = 8 - len(first_bytes)
                    first_bytes += chunk[:needed]
                handle.write(chunk)
                total_bytes += len(chunk)

        is_pdf_header = first_bytes.startswith(b"%PDF-")
        if ("pdf" not in content_type) and (not is_pdf_header) and (not looks_like_pdf_url(normalized)):
            final_path.unlink(missing_ok=True)
            raise ValueError(f"Downloaded content is not PDF (content-type={content_type or 'unknown'}).")
        return total_bytes, content_type, str(final_path)


def download_pdf_with_fallback(
    row: Candidate,
    out_path: Path,
    timeout: int,
    retry: int,
    backoff: float,
    session: requests.Session,
) -> Tuple[Dict[str, object], int]:
    url_candidates = build_download_url_candidates(row=row, session=session, timeout=timeout)
    if not url_candidates:
        raise RuntimeError("No download URL candidates available.")

    errors: List[str] = []
    for url_idx, candidate_url in enumerate(url_candidates, start=1):
        try:
            (result, attempt) = retry_call(
                max_retry=retry,
                backoff=backoff,
                func=download_pdf,
                session=session,
                url=candidate_url,
                out_path=out_path,
                timeout=timeout,
            )
            payload = {
                "bytes": int(result[0]),
                "content_type": str(result[1] or ""),
                "pdf_path": str(result[2]),
                "source_url_used": candidate_url,
                "url_candidate_count": len(url_candidates),
                "url_candidate_index": url_idx,
            }
            return payload, attempt
        except Exception as exc:
            errors.append(f"[{url_idx}] {candidate_url} -> {exc}")
            continue

    raise RuntimeError("All fallback URLs failed: " + " | ".join(errors[:4]))


def retry_call(max_retry: int, backoff: float, func, *args, **kwargs):
    last_exc: Optional[Exception] = None
    for attempt in range(1, max_retry + 1):
        try:
            return func(*args, **kwargs), attempt
        except Exception as exc:  # pragma: no cover - runtime path
            last_exc = exc
            if attempt >= max_retry:
                break
            time.sleep(max(0.1, backoff * attempt))
    raise RuntimeError(f"Retries exhausted: {last_exc}") from last_exc


def should_skip_by_existing(
    row: Candidate,
    existing_title_keys: Set[str],
    existing_file_keys: Set[str],
    historical_signatures: Set[str],
    run_signatures: Set[str],
) -> Optional[str]:
    signatures = build_signatures(row)
    if not signatures:
        return "no_signature"

    if any(sig in historical_signatures for sig in signatures):
        return "already_ingested_by_ledger"
    if any(sig in run_signatures for sig in signatures):
        return "duplicate_in_current_run"

    title_key = normalize_title_key(row.title)
    if title_key and title_key in existing_title_keys:
        return "title_exists_in_kb"

    file_stem = Path(build_pdf_filename(row)).stem.lower()
    if file_stem in existing_file_keys:
        return "file_exists_in_kb"

    return None


def ingest_pdf_with_service(
    kb_db: Path,
    pdf_path: Path,
    source_type: str,
    theme: str,
    is_core: bool,
) -> Dict[str, int]:
    from backend.core.knowledge_base import KnowledgeBaseService

    svc = KnowledgeBaseService(str(kb_db))
    return svc.ingest_file(
        file_path=str(pdf_path),
        source_type=source_type,
        theme=theme,
        is_core=is_core,
    )


def rebuild_links(kb_db: Path) -> Dict[str, int]:
    from backend.core.knowledge_base import KnowledgeBaseService

    svc = KnowledgeBaseService(str(kb_db))
    return svc.rebuild_links(clear_existing=True)


def main() -> int:
    args = parse_args()
    candidate_csv = candidate_csv_path(args.candidate_csv)
    kb_db = resolve_kb_db_path(args.kb_db)
    download_dir = Path(args.download_dir).expanduser()
    log_dir = Path(args.log_dir).expanduser()
    ensure_dirs(download_dir, log_dir)

    run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_log = log_dir / f"bulk_ingest_{run_id}.jsonl"
    success_ledger = log_dir / "bulk_ingest_success_ledger.jsonl"

    rows = sort_candidates(read_candidates(candidate_csv))
    allow_status = status_allow_set(args.status_allow)
    rows = [row for row in rows if row.download_status in allow_status and row.source_url]
    if args.max_docs and args.max_docs > 0:
        rows = rows[: args.max_docs]

    existing_title_keys, existing_file_keys = load_existing_doc_keys(kb_db)
    historical_signatures = load_success_signatures(success_ledger)
    run_signatures: Set[str] = set()
    stats = Counter()
    http_session = make_http_session()

    print(f"[bulk-ingest] candidate_csv={candidate_csv}")
    print(f"[bulk-ingest] kb_db={kb_db}")
    print(f"[bulk-ingest] selected={len(rows)} status_allow={sorted(allow_status)}")
    print(f"[bulk-ingest] dry_run={args.dry_run} retry={args.retry}")
    print(f"[bulk-ingest] download_only={args.download_only}")

    for idx, row in enumerate(rows, start=1):
        signatures = build_signatures(row)
        base_event = {
            "ts": dt.datetime.now().isoformat(timespec="seconds"),
            "run_id": run_id,
            "index": idx,
            "title": row.title,
            "year": row.year,
            "journal": row.journal,
            "doi": row.doi,
            "source_url": row.source_url,
            "download_status": row.download_status,
            "bucket": row.bucket,
            "priority": row.priority,
            "signatures": sorted(signatures),
        }

        skip_reason = should_skip_by_existing(
            row=row,
            existing_title_keys=existing_title_keys,
            existing_file_keys=existing_file_keys,
            historical_signatures=historical_signatures,
            run_signatures=run_signatures,
        )
        if skip_reason:
            stats[f"skip_{skip_reason}"] += 1
            append_jsonl(run_log, {**base_event, "result": "skipped", "reason": skip_reason})
            continue

        pdf_name = build_pdf_filename(row)
        pdf_path = download_dir / pdf_name
        if args.dry_run:
            run_signatures.update(signatures)
            stats["dryrun_ready"] += 1
            append_jsonl(
                run_log,
                {
                    **base_event,
                    "result": "dryrun_ready",
                    "pdf_path": str(pdf_path),
                    "reason": "ready_for_download_and_ingest",
                },
            )
            continue

        try:
            downloaded_pdf_path = pdf_path
            if pdf_path.exists() and pdf_path.stat().st_size > 1000:
                download_attempt = 0
                download_meta = {
                    "bytes": pdf_path.stat().st_size,
                    "content_type": "cached",
                    "pdf_path": str(pdf_path),
                }
            else:
                (download_meta_raw, download_attempt) = download_pdf_with_fallback(
                    row=row,
                    out_path=pdf_path,
                    timeout=args.timeout,
                    retry=args.retry,
                    backoff=args.retry_backoff,
                    session=http_session,
                )
                download_meta = dict(download_meta_raw)
                downloaded_pdf_path = Path(str(download_meta.get("pdf_path") or pdf_path))

            if args.download_only:
                run_signatures.update(signatures)
                historical_signatures.update(signatures)
                title_key = normalize_title_key(row.title)
                if title_key:
                    existing_title_keys.add(title_key)
                existing_file_keys.add(downloaded_pdf_path.stem.lower())
                stats["downloaded_only"] += 1
                append_jsonl(
                    run_log,
                    {
                        **base_event,
                        "result": "downloaded_only",
                        "pdf_path": str(downloaded_pdf_path),
                        "download_attempt": download_attempt,
                        "download": download_meta,
                    },
                )
                append_jsonl(
                    success_ledger,
                    {
                        "ts": dt.datetime.now().isoformat(timespec="seconds"),
                        "run_id": run_id,
                        "title": row.title,
                        "doi": row.doi,
                        "source_url": row.source_url,
                        "signatures": sorted(signatures),
                        "doc_id": 0,
                        "pdf_path": str(downloaded_pdf_path),
                        "mode": "download_only",
                    },
                )
                if args.sleep > 0:
                    time.sleep(args.sleep)
                continue

            (ingest_result_raw, ingest_attempt) = retry_call(
                max_retry=args.retry,
                backoff=args.retry_backoff,
                func=ingest_pdf_with_service,
                kb_db=kb_db,
                pdf_path=downloaded_pdf_path,
                source_type=args.source_type,
                theme=args.theme,
                is_core=bool(args.is_core),
            )
            ingest_result = {
                "doc_id": int(ingest_result_raw.get("doc_id", 0)),
                "chunk_count": int(ingest_result_raw.get("chunk_count", 0)),
            }

            run_signatures.update(signatures)
            historical_signatures.update(signatures)
            title_key = normalize_title_key(row.title)
            if title_key:
                existing_title_keys.add(title_key)
            existing_file_keys.add(downloaded_pdf_path.stem.lower())

            stats["ingested"] += 1
            append_jsonl(
                run_log,
                {
                    **base_event,
                    "result": "ingested",
                    "pdf_path": str(downloaded_pdf_path),
                    "download_attempt": download_attempt,
                    "ingest_attempt": ingest_attempt,
                    "download": download_meta,
                    "ingest": ingest_result,
                },
            )
            append_jsonl(
                success_ledger,
                {
                    "ts": dt.datetime.now().isoformat(timespec="seconds"),
                    "run_id": run_id,
                    "title": row.title,
                    "doi": row.doi,
                    "source_url": row.source_url,
                    "signatures": sorted(signatures),
                    "doc_id": ingest_result["doc_id"],
                    "pdf_path": str(downloaded_pdf_path),
                },
            )
        except Exception as exc:  # pragma: no cover - runtime path
            stats["failed"] += 1
            append_jsonl(
                run_log,
                {
                    **base_event,
                    "result": "failed",
                    "pdf_path": str(pdf_path),
                    "error": str(exc),
                },
            )
            pdf_tmp = pdf_path.with_suffix(".tmp")
            try:
                if pdf_tmp.exists():
                    pdf_tmp.unlink(missing_ok=True)
            except Exception:
                pass
            if args.sleep > 0:
                time.sleep(args.sleep)
            continue

        if args.sleep > 0:
            time.sleep(args.sleep)

    http_session.close()

    if args.rebuild_links and not args.dry_run:
        try:
            rebuild = rebuild_links(kb_db)
            append_jsonl(
                run_log,
                {
                    "ts": dt.datetime.now().isoformat(timespec="seconds"),
                    "run_id": run_id,
                    "result": "rebuild_links",
                    "payload": rebuild,
                },
            )
            stats["rebuild_links_done"] += 1
        except Exception as exc:  # pragma: no cover - runtime path
            append_jsonl(
                run_log,
                {
                    "ts": dt.datetime.now().isoformat(timespec="seconds"),
                    "run_id": run_id,
                    "result": "rebuild_links_failed",
                    "error": str(exc),
                },
            )
            stats["rebuild_links_failed"] += 1

    summary_path = log_dir / f"bulk_ingest_{run_id}_summary.json"
    summary_payload = {
        "ts": dt.datetime.now().isoformat(timespec="seconds"),
        "run_id": run_id,
        "candidate_csv": str(candidate_csv),
        "kb_db": str(kb_db),
        "download_dir": str(download_dir),
        "run_log": str(run_log),
        "success_ledger": str(success_ledger),
        "status_allow": sorted(allow_status),
        "dry_run": bool(args.dry_run),
        "stats": dict(stats),
    }
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[bulk-ingest] done. run_log={run_log}")
    print(f"[bulk-ingest] summary={summary_path}")
    print(f"[bulk-ingest] stats={dict(stats)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
