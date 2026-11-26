"""Training configuration files (versioned YAML).

A config may start from another one with ``base: <relative path>``; the two
are merged recursively (mappings merge key by key, anything else is
replaced; `null` replaces a mapping, since `{}` would merge into it). Ablations
are a base config plus the one setting they change.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

MAX_DEPTH = 8


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """``override`` on top of ``base`` (neither is modified)."""
    out = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | Path, _depth: int = 0) -> dict[str, Any]:
    """Loads a YAML config, resolving its ``base`` chain."""
    if _depth > MAX_DEPTH:
        raise ValueError(f"config base chain deeper than {MAX_DEPTH} (a cycle?)")
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: a config must be a mapping")
    base = data.pop("base", None)
    if base is None:
        return data
    return deep_merge(load_config(path.parent / base, _depth + 1), data)
