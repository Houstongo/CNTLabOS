"""
Batch-fetch CNT literature metadata and export candidate CSV rows.

This script is designed for the CNTA RAG workflow and writes rows with fields:
priority,bucket,title,year,journal,doi,relation_targets,notes,source_url,download_status

Usage examples:
  python scripts/fetch_cnt_literature_candidates.py
  python scripts/fetch_cnt_literature_candidates.py --rows-per-query 30 --max-per-bucket 15
  python scripts/fetch_cnt_literature_candidates.py --direct-pdf-only --verify-pdf-url
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import quote

import requests


CSV_FIELDS = [
    "priority",
    "bucket",
    "title",
    "year",
    "journal",
    "doi",
    "relation_targets",
    "notes",
    "source_url",
    "download_status",
]

RELATION_ORDER = [
    "process_to_morphology",
    "morphology_to_performance",
    "process_to_performance",
    "process_to_mechanism",
    "mechanism_to_morphology",
    "mechanism_evidence",
]

DOWNLOAD_STATUS_RANK = {
    "direct_pdf_verified": 6,
    "direct_pdf_candidate": 5,
    "oa_pdf": 4,
    "publisher_page_with_abstract": 3,
    "publisher_page_with_preview": 2,
    "publisher_page_only": 1,
    "metadata_index_only": 0,
}

PRIORITY_RANK = {"P0": 3, "P1": 2, "P2": 1}


@dataclass(frozen=True)
class SearchPreset:
    priority: str
    bucket: str
    query: str
    relation_targets: str
    notes: str


PRESETS: List[SearchPreset] = [
    SearchPreset(
        priority="P0",
        bucket="01_工艺-机理-形貌",
        query="carbon nanotube array growth mechanism boundary layer catalyst aggregation",
        relation_targets="process_to_mechanism;mechanism_to_morphology;process_to_morphology",
        notes="优先补工艺-机理-形貌主链证据",
    ),
    SearchPreset(
        priority="P0",
        bucket="02_形貌-性能",
        query="aligned carbon nanotube array conductivity tensile strength modulus morphology",
        relation_targets="morphology_to_performance;process_to_morphology",
        notes="优先补取向/密度/波曲与性能关系",
    ),
    SearchPreset(
        priority="P1",
        bucket="03_工艺-性能",
        query="carbon nanotube array process parameter conductivity mechanical property",
        relation_targets="process_to_performance;process_to_morphology",
        notes="补充工艺参数对导电/力学性能的直接关联",
    ),
    SearchPreset(
        priority="P1",
        bucket="04_机理证据",
        query="carbon nanotube forest review growth kinetics diffusion limitation deactivation",
        relation_targets="process_to_mechanism;mechanism_evidence",
        notes="补机理术语和证据锚点，增强解释链路",
    ),
]


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_output_csv() -> Path:
    root = project_root().parent
    return root / "RagDocument" / "CORE" / "13. 工艺-形貌-性能补充" / "文献候选清单.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch CNT literature candidate metadata.")
    parser.add_argument("--output-csv", default=str(default_output_csv()))
    parser.add_argument("--rows-per-query", type=int, default=25)
    parser.add_argument("--max-per-bucket", type=int, default=20)
    parser.add_argument("--min-year", type=int, default=2000)
    parser.add_argument("--max-year", type=int, default=dt.date.today().year)
    parser.add_argument("--mailto", default="")
    parser.add_argument("--sleep", type=float, default=0.25)
    parser.add_argument("--append-existing", action="store_true", default=True)
    parser.add_argument("--no-append-existing", action="store_true")
    parser.add_argument("--openalex-enrich", action="store_true", default=True)
    parser.add_argument("--no-openalex-enrich", action="store_true")
    parser.add_argument("--verify-pdf-url", action="store_true")
    parser.add_argument("--direct-pdf-only", action="store_true")
    parser.add_argument("--backup-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def http_json(url: str, params: Optional[Dict[str, str]] = None, timeout: int = 30) -> Dict:
    last_error = None
    for attempt in range(3):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # pragma: no cover - network dependent
            last_error = exc
            if attempt < 2:
                time.sleep(0.6 * (attempt + 1))
    raise RuntimeError(f"HTTP request failed: {url} -> {last_error}")  # pragma: no cover


def normalize_doi(value: str) -> str:
    if not value:
        return ""
    doi = value.strip()
    doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi, flags=re.IGNORECASE)
    return doi.lower()


def extract_year(item: Dict) -> str:
    for field in ("issued", "published-print", "published-online", "created"):
        date_info = item.get(field) or {}
        date_parts = date_info.get("date-parts") or []
        if date_parts and isinstance(date_parts[0], list) and date_parts[0]:
            year = date_parts[0][0]
            if isinstance(year, int):
                return str(year)
    return ""


def pick_title(item: Dict) -> str:
    titles = item.get("title") or []
    if not titles:
        return ""
    return str(titles[0]).strip()


def pick_journal(item: Dict) -> str:
    journals = item.get("container-title") or []
    if not journals:
        return ""
    return str(journals[0]).strip()


def pick_direct_pdf_url(item: Dict) -> str:
    links = item.get("link") or []
    for link in links:
        if not isinstance(link, dict):
            continue
        content_type = str(link.get("content-type") or "").lower()
        url = str(link.get("URL") or "").strip()
        if not url:
            continue
        if "pdf" in content_type or looks_like_pdf_url(url):
            return url
    return ""


def looks_like_pdf_url(url: str) -> bool:
    u = url.lower()
    return u.endswith(".pdf") or ".pdf?" in u or "/pdf" in u


def verify_pdf_url(url: str, timeout: int = 15) -> bool:
    try:
        head = requests.head(url, timeout=timeout, allow_redirects=True)
        content_type = str(head.headers.get("Content-Type") or "").lower()
        if "pdf" in content_type:
            return True
        final_url = str(head.url or "").lower()
        return looks_like_pdf_url(final_url)
    except Exception:  # pragma: no cover - network dependent
        return False


def openalex_oa_url(doi: str, mailto: str = "") -> str:
    if not doi:
        return ""
    encoded = quote(f"https://doi.org/{doi}", safe="")
    url = f"https://api.openalex.org/works/{encoded}"
    params: Dict[str, str] = {}
    if mailto:
        params["mailto"] = mailto
    try:
        payload = http_json(url, params=params, timeout=25)
    except Exception:  # pragma: no cover - network dependent
        return ""
    oa = payload.get("open_access") or {}
    best = str(oa.get("oa_url") or "").strip()
    if best:
        return best
    best_location = payload.get("best_oa_location") or {}
    return str(best_location.get("pdf_url") or best_location.get("landing_page_url") or "").strip()


def classify_source(
    crossref_item: Dict,
    doi: str,
    openalex_url: str,
    do_verify: bool,
) -> Tuple[str, str]:
    pdf_url = pick_direct_pdf_url(crossref_item)
    if pdf_url:
        if do_verify:
            if verify_pdf_url(pdf_url):
                return pdf_url, "direct_pdf_verified"
            return pdf_url, "direct_pdf_candidate"
        return pdf_url, "direct_pdf_candidate"

    if openalex_url:
        if looks_like_pdf_url(openalex_url):
            return openalex_url, "oa_pdf"
        return openalex_url, "publisher_page_with_abstract"

    if crossref_item.get("URL"):
        return str(crossref_item["URL"]).strip(), "publisher_page_only"

    if doi:
        return f"https://doi.org/{doi}", "publisher_page_only"

    return "", "metadata_index_only"


def relation_union(left: str, right: str) -> str:
    all_values = set()
    for raw in (left, right):
        for token in (raw or "").split(";"):
            token = token.strip()
            if token:
                all_values.add(token)
    ordered = [x for x in RELATION_ORDER if x in all_values]
    ordered.extend(sorted(x for x in all_values if x not in RELATION_ORDER))
    return ";".join(ordered)


def merge_priority(old: str, new: str) -> str:
    return old if PRIORITY_RANK.get(old, 0) >= PRIORITY_RANK.get(new, 0) else new


def merge_download_status(old: str, new: str) -> str:
    return old if DOWNLOAD_STATUS_RANK.get(old, -1) >= DOWNLOAD_STATUS_RANK.get(new, -1) else new


def best_source_url(old_row: Dict[str, str], new_row: Dict[str, str]) -> str:
    winner = merge_download_status(old_row.get("download_status", ""), new_row.get("download_status", ""))
    if winner == old_row.get("download_status", ""):
        return old_row.get("source_url", "")
    return new_row.get("source_url", "")


def merge_notes(left: str, right: str) -> str:
    left_clean = (left or "").strip()
    right_clean = (right or "").strip()
    if not left_clean:
        return right_clean
    if not right_clean:
        return left_clean
    if right_clean in left_clean:
        return left_clean
    if left_clean in right_clean:
        return right_clean
    return f"{left_clean} | {right_clean}"


def row_key(row: Dict[str, str]) -> str:
    doi = normalize_doi(row.get("doi", ""))
    if doi:
        return f"doi:{doi}"
    title = re.sub(r"\s+", " ", (row.get("title", "") or "").strip().lower())
    return f"title:{title}"


def upsert_rows(rows: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    merged: Dict[str, Dict[str, str]] = {}
    for row in rows:
        clean = {k: str(row.get(k, "") or "").strip() for k in CSV_FIELDS}
        key = row_key(clean)
        if key not in merged:
            merged[key] = clean
            continue
        old = merged[key]
        merged[key] = {
            "priority": merge_priority(old.get("priority", ""), clean.get("priority", "")),
            "bucket": old.get("bucket", "") or clean.get("bucket", ""),
            "title": old.get("title", "") if len(old.get("title", "")) >= len(clean.get("title", "")) else clean.get("title", ""),
            "year": old.get("year", "") or clean.get("year", ""),
            "journal": old.get("journal", "") or clean.get("journal", ""),
            "doi": normalize_doi(old.get("doi", "") or clean.get("doi", "")),
            "relation_targets": relation_union(old.get("relation_targets", ""), clean.get("relation_targets", "")),
            "notes": merge_notes(old.get("notes", ""), clean.get("notes", "")),
            "source_url": best_source_url(old, clean),
            "download_status": merge_download_status(old.get("download_status", ""), clean.get("download_status", "")),
        }
    return list(merged.values())


def read_existing_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            with path.open("r", encoding=encoding, newline="") as fh:
                reader = csv.DictReader(fh)
                rows = [{k: row.get(k, "") for k in CSV_FIELDS} for row in reader]
            return rows
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("csv", b"", 0, 1, f"Unable to decode CSV: {path}")


def write_rows(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})


def fetch_bucket_rows(
    preset: SearchPreset,
    rows_per_query: int,
    max_per_bucket: int,
    min_year: int,
    max_year: int,
    mailto: str,
    sleep_s: float,
    enrich_openalex: bool,
    verify_pdf: bool,
) -> List[Dict[str, str]]:
    params: Dict[str, str] = {
        "query.bibliographic": preset.query,
        "rows": str(rows_per_query),
        "sort": "relevance",
        "order": "desc",
        "filter": f"type:journal-article,from-pub-date:{min_year}-01-01,until-pub-date:{max_year}-12-31",
    }
    if mailto:
        params["mailto"] = mailto
    payload = http_json("https://api.crossref.org/works", params=params, timeout=45)
    items = (payload.get("message") or {}).get("items") or []

    output: List[Dict[str, str]] = []
    for item in items:
        if len(output) >= max_per_bucket:
            break
        doi = normalize_doi(str(item.get("DOI") or ""))
        title = pick_title(item)
        if not title:
            continue
        year = extract_year(item)
        if year.isdigit():
            yv = int(year)
            if yv < min_year or yv > max_year:
                continue

        oa_url = ""
        if enrich_openalex and doi:
            oa_url = openalex_oa_url(doi=doi, mailto=mailto)
            if sleep_s > 0:
                time.sleep(min(sleep_s, 0.25))

        source_url, download_status = classify_source(
            crossref_item=item,
            doi=doi,
            openalex_url=oa_url,
            do_verify=verify_pdf,
        )
        output.append(
            {
                "priority": preset.priority,
                "bucket": preset.bucket,
                "title": title,
                "year": year,
                "journal": pick_journal(item),
                "doi": doi,
                "relation_targets": preset.relation_targets,
                "notes": preset.notes,
                "source_url": source_url,
                "download_status": download_status,
            }
        )

        if sleep_s > 0:
            time.sleep(sleep_s)
    return output


def keep_direct_pdf(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    allow = {"direct_pdf_verified", "direct_pdf_candidate", "oa_pdf"}
    return [r for r in rows if r.get("download_status") in allow]


def sort_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    def sort_key(row: Dict[str, str]) -> Tuple[int, int, str]:
        pr = PRIORITY_RANK.get(row.get("priority", ""), 0)
        year = int(row["year"]) if row.get("year", "").isdigit() else 0
        title = row.get("title", "")
        return (-pr, -year, title.lower())

    return sorted(rows, key=sort_key)


def backup_file(path: Path) -> Optional[Path]:
    if not path.exists():
        return None
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = path.with_name(f"{path.stem}.bak_{timestamp}{path.suffix}")
    backup.write_bytes(path.read_bytes())
    return backup


def main() -> None:
    args = parse_args()
    output_csv = Path(args.output_csv)
    append_existing = bool(args.append_existing and not args.no_append_existing)
    enrich_openalex = bool(args.openalex_enrich and not args.no_openalex_enrich)

    all_new_rows: List[Dict[str, str]] = []
    for preset in PRESETS:
        bucket_rows = fetch_bucket_rows(
            preset=preset,
            rows_per_query=args.rows_per_query,
            max_per_bucket=args.max_per_bucket,
            min_year=args.min_year,
            max_year=args.max_year,
            mailto=args.mailto.strip(),
            sleep_s=max(0.0, args.sleep),
            enrich_openalex=enrich_openalex,
            verify_pdf=bool(args.verify_pdf_url),
        )
        all_new_rows.extend(bucket_rows)

    existing_rows = read_existing_rows(output_csv) if append_existing else []
    combined = upsert_rows([*existing_rows, *all_new_rows])
    if args.direct_pdf_only:
        combined = keep_direct_pdf(combined)
    combined = sort_rows(combined)

    if args.dry_run:
        print(f"[DRY RUN] output={output_csv}")
        print(f"existing_rows={len(existing_rows)} new_rows={len(all_new_rows)} combined_rows={len(combined)}")
        status_counts: Dict[str, int] = {}
        for row in combined:
            status = row.get("download_status", "")
            status_counts[status] = status_counts.get(status, 0) + 1
        print(f"status_counts={status_counts}")
        return

    backup_path = backup_file(output_csv) if args.backup_existing else None
    write_rows(output_csv, combined)

    status_counts: Dict[str, int] = {}
    for row in combined:
        status = row.get("download_status", "")
        status_counts[status] = status_counts.get(status, 0) + 1

    print(f"Saved: {output_csv}")
    if backup_path:
        print(f"Backup: {backup_path}")
    print(f"rows={len(combined)} (existing={len(existing_rows)}, fetched={len(all_new_rows)})")
    print(f"download_status={status_counts}")


if __name__ == "__main__":
    main()
