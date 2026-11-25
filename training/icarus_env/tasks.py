"""Task specifications: what the environments draw their episodes from.

A task is a plain dict understood by the Rust core (``icarus.Env`` /
``icarus.VecEnv``)::

    {
        "config": "easy",            # a vehicle configuration (icarus.configs())
        "scenario": "nominal",       # an evaluation scenario (icarus.scenarios())
        "overrides": {"start.altitude": [300, 500], ...},   # icarus.override_keys()
        "randomization": {"wind": [0.5, 1.5], "sensor_delay": [0, 6], ...},
    }

For readability in YAML, an override key may end in ``_deg`` (for example
``start.tilt_deg``): its values are given in degrees and converted to the
radians the core expects.
"""

from __future__ import annotations

import copy
import math
import numbers
from collections.abc import Mapping, Sequence
from typing import Any

import icarus

DEG_SUFFIX = "_deg"
TASK_KEYS = frozenset({"config", "scenario", "overrides", "randomization"})


def _number(value: Any, key: str) -> float:
    if isinstance(value, bool | str) or not isinstance(value, numbers.Real):
        raise ValueError(f"override {key!r}: expected a number, got {value!r}")
    return float(value)


def _as_list(value: Any, key: str) -> list[float]:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return [_number(v, key) for v in value]
    return [_number(value, key)]


def resolve_task(spec: Mapping[str, Any] | None) -> dict[str, Any]:
    """Returns the core's task dict for a (YAML-friendly) task spec.

    Converts ``*_deg`` override keys to radians and rejects unknown top-level
    keys. The core validates everything else when the task is used (see
    :func:`validate_task`).
    """
    if spec is None:
        return {}
    unknown = set(spec) - TASK_KEYS
    if unknown:
        raise ValueError(f"unknown task keys {sorted(unknown)}; expected {sorted(TASK_KEYS)}")
    task: dict[str, Any] = {k: copy.deepcopy(v) for k, v in spec.items() if k != "overrides"}
    raw = spec.get("overrides") or {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"task overrides must be a mapping, got {type(raw).__name__}")
    overrides: dict[str, list[float]] = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            raise ValueError(f"override keys must be strings, got {key!r}")
        values = _as_list(value, key)
        if key.endswith(DEG_SUFFIX):
            key = key.removesuffix(DEG_SUFFIX)
            values = [math.radians(v) for v in values]
        if key in overrides:
            raise ValueError(f"override {key!r} given twice")
        overrides[key] = values
    if overrides:
        task["overrides"] = overrides
    return task


def validate_task(task: Mapping[str, Any], env_config: Mapping[str, Any] | None = None) -> None:
    """Raises ``ValueError`` if the core rejects ``task`` (or the pair)."""
    icarus.Env(dict(task), dict(env_config) if env_config is not None else None)
