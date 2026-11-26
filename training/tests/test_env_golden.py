"""The Python extension reproduces the RL-environment golden hashes.

The same file is checked by the native Rust core (``tests/env_golden.rs``);
running this on every CI operating system pins the environment bit for bit.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

PARITY = Path(__file__).resolve().parents[2] / "parity"


def _generator():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location(
        "generate_env_golden", PARITY / "generate_env_golden.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_environment_matches_the_golden_hashes() -> None:
    golden = json.loads((PARITY / "env_golden.json").read_text(encoding="utf-8"))
    fresh = _generator().generate()
    assert fresh["physics_version"] == golden["physics_version"]
    for got, want in zip(fresh["cases"], golden["cases"], strict=True):
        assert got == want, got["name"]
