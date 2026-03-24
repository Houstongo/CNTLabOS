"""Config helpers for the paper-reproduction pipeline."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    return _resolve_paths(config, base_dir=config_path.parent)


def save_config_snapshot(config: Dict[str, Any], path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh, allow_unicode=True, sort_keys=False)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_paths(value: Any, base_dir: Path) -> Any:
    if isinstance(value, dict):
        return {key: _resolve_paths(item, base_dir) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_paths(item, base_dir) for item in value]
    if isinstance(value, str):
        looks_like_path = ("\\" in value) or ("/" in value) or value.endswith((".yaml", ".yml", ".csv", ".pth", ".json"))
        if looks_like_path:
            path = Path(value)
            if not path.is_absolute():
                return str((base_dir / path).resolve())
    return value
