from __future__ import annotations


CANONICAL_WCNTSEGNET = "wcntsegnet"
LEGACY_THRESHOLD = "threshold"
CNTSEGNET = "cntsegnet"
BOTH = "both"

TRADITIONAL_ALIASES = {CANONICAL_WCNTSEGNET, LEGACY_THRESHOLD}
VALID_SEGMENTATION_BACKENDS = {CANONICAL_WCNTSEGNET, LEGACY_THRESHOLD, CNTSEGNET, BOTH}


def normalize_segmentation_backend(backend: str | None, *, allow_both: bool = True) -> str:
    value = (backend or CANONICAL_WCNTSEGNET).strip().lower()
    if value in TRADITIONAL_ALIASES:
        return CANONICAL_WCNTSEGNET
    if value == CNTSEGNET:
        return CNTSEGNET
    if allow_both and value == BOTH:
        return BOTH
    raise ValueError(f"Unsupported segmentation backend: {backend}")
