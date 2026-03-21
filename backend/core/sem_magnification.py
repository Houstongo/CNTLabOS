from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

from PIL import Image


STANDARD_SEM_MAGNIFICATIONS = (500, 1000, 5000, 10000, 50000, 100000)
_PHENOM_REFERENCE_FIELD_WIDTH_UM = 518000.0

_ZZY_MAG_SEGMENT_RE = re.compile(
    r"(?P<prefix>\b(?:top|mid|middle|bottom|all|bpttom|position-\w+)\s+)"
    r"(?P<mag>\d+)"
    r"(?P<suffix>(?:\s+\d+-|-)\d+(?:-.*)?$)",
    re.IGNORECASE,
)


def _load_fei_xml(image_path: Path) -> Optional[ET.Element]:
    with Image.open(image_path) as image:
        info = dict(image.info)
    xml_text = info.get("34683")
    if not xml_text:
        return None
    return ET.fromstring(xml_text)


def extract_magnification_from_png_metadata(image_path: Path) -> Optional[int]:
    root = _load_fei_xml(Path(image_path))
    if root is None:
        return None

    pixel_width_text = root.findtext(".//pixelWidth")
    if pixel_width_text is None:
        return None

    pixel_width_nm = float(pixel_width_text)
    with Image.open(image_path) as image:
        width_px = image.width

    crop_left = root.findtext(".//cropHint/left")
    crop_right = root.findtext(".//cropHint/right")
    if crop_left is not None and crop_right is not None:
        width_px = int(float(crop_right)) - int(float(crop_left))

    field_width_um = pixel_width_nm * width_px / 1000.0
    if field_width_um <= 0:
        return None

    raw_magnification = _PHENOM_REFERENCE_FIELD_WIDTH_UM / field_width_um
    return min(STANDARD_SEM_MAGNIFICATIONS, key=lambda mag: abs(mag - raw_magnification))


def build_zzy_filename_with_magnification(image_path: Path, magnification: int) -> Path:
    path = Path(image_path)
    stem = path.stem
    match = _ZZY_MAG_SEGMENT_RE.search(stem)
    if not match:
        raise ValueError(f"Cannot locate ZZY magnification segment in {path.name}")

    renamed_stem = (
        stem[: match.start()]
        + match.group("prefix")
        + str(int(magnification))
        + match.group("suffix")
    )
    return path.with_name(renamed_stem + path.suffix)
