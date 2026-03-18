import os
from typing import Dict, List


THEME_GROWTH_MECHANISM = "growth_mechanism"
THEME_PROCESS_MORPHOLOGY = "process_morphology"
THEME_ML_GROWTH = "ml_growth"
THEME_CHARACTERIZATION = "characterization"
THEME_APPLICATIONS = "applications"
THEME_REFERENCE_REVIEW = "reference_review"


def infer_theme_from_path(path: str) -> str:
    normalized = path.replace("/", "\\").lower()

    if "12. cnt生长" in normalized or "12. cnt" in normalized:
        return THEME_GROWTH_MECHANISM
    if "ai辅助碳管生长" in normalized:
        return THEME_ML_GROWTH
    if "重点参考" in normalized:
        return THEME_REFERENCE_REVIEW
    if "感受器" in normalized or "器件" in normalized or "仿生" in normalized:
        return THEME_APPLICATIONS
    if "表征" in normalized or "形貌" in normalized:
        return THEME_CHARACTERIZATION
    if "碳管" in normalized:
        return THEME_PROCESS_MORPHOLOGY
    return THEME_REFERENCE_REVIEW


DEFAULT_KB_SEED_SOURCES = [
    {
        "path": r"D:\CNTDATA\RagDocument\CORE\12. CNT生长",
        "source_type": "paper",
        "theme": infer_theme_from_path(r"D:\CNTDATA\RagDocument\CORE\12. CNT生长"),
        "is_core": True,
    },
    {
        "path": r"D:\CNTDATA\RagDocument\CORE\碳纳米管相关0312\2.AI辅助碳管生长",
        "source_type": "paper",
        "theme": infer_theme_from_path(r"D:\CNTDATA\RagDocument\CORE\碳纳米管相关0312\2.AI辅助碳管生长"),
        "is_core": True,
    },
    {
        "path": r"D:\CNTDATA\RagDocument\CORE\碳纳米管相关0312\1.碳管\重点参考",
        "source_type": "paper",
        "theme": infer_theme_from_path(r"D:\CNTDATA\RagDocument\CORE\碳纳米管相关0312\1.碳管\重点参考"),
        "is_core": False,
    },
]


def import_seed_sources(service, sources: List[Dict[str, object]]) -> Dict[str, int]:
    imported_documents = 0
    imported_sources = 0

    for source in sources:
        source_path = source["path"]
        if not os.path.exists(source_path):
            continue

        result = service.ingest_directory(
            source_dir=source_path,
            source_type=source.get("source_type", "paper"),
            theme=source.get("theme") or infer_theme_from_path(source_path),
            is_core=bool(source.get("is_core", False)),
            allowed_extensions=source.get("allowed_extensions"),
        )
        imported_sources += 1
        imported_documents += result["document_count"]

    return {
        "source_count": imported_sources,
        "document_count": imported_documents,
    }


def relabel_document_themes(service) -> Dict[str, int]:
    updated_count = 0
    for document in service.list_documents():
        file_path = document.get("file_path")
        if not file_path:
            continue
        inferred = infer_theme_from_path(file_path)
        if inferred != document.get("theme"):
            service.update_document_theme(document["id"], inferred)
            updated_count += 1
    return {"updated_count": updated_count}
