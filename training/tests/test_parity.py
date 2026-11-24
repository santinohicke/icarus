"""The native Python module reproduces the golden trajectories bit for bit."""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path

import icarus
import pytest

GOLDEN_DIR = Path(__file__).resolve().parents[2] / "parity" / "golden"
MANIFEST = json.loads((GOLDEN_DIR / "manifest.json").read_text(encoding="utf-8"))
GOLDENS = sorted(GOLDEN_DIR / f"{name}.json" for name in MANIFEST["cases"])
FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
MASK64 = (1 << 64) - 1


def f64_bits(value: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", value))[0]


def from_hex(text: str) -> float:
    return struct.unpack("<d", struct.pack("<Q", int(text, 16)))[0]


def fnv1a(h: int, values: list[float]) -> int:
    for value in values:
        word = f64_bits(value)
        for i in range(8):
            h ^= (word >> (8 * i)) & 0xFF
            h = (h * FNV_PRIME) & MASK64
    return h


def apply_event(sim: icarus.Sim | icarus.Recorder, event: dict[str, object]) -> None:
    a, b = from_hex(str(event["a"])), from_hex(str(event["b"]))
    kind = event["kind"]
    if kind == "gimbal_stuck":
        sim.inject_gimbal_stuck(a)
    elif kind == "thrust_loss":
        sim.inject_thrust_loss(a)
    elif kind == "gust":
        sim.inject_gust(a, b)
    else:
        raise AssertionError(f"unknown event kind {kind!r}")


def test_golden_files_match_the_manifest() -> None:
    on_disk = sorted(p.stem for p in GOLDEN_DIR.glob("*.json") if p.name != "manifest.json")
    assert MANIFEST["cases"], "empty manifest"
    assert on_disk == sorted(MANIFEST["cases"])


@pytest.mark.parametrize("path", GOLDENS, ids=[p.stem for p in GOLDENS])
def test_native_module_matches_golden(path: Path) -> None:
    golden = json.loads(path.read_text(encoding="utf-8"))
    assert golden["format"] == 1
    assert golden["name"] == path.stem
    assert golden["samples"][-1]["tick"] == golden["ticks"]
    assert golden["physics_version"] == icarus.PHYSICS_VERSION, (
        "golden recorded with another physics version: regenerate with parity/generate_golden.py"
    )
    assert golden["telemetry_fields"] == icarus.TELEMETRY_FIELDS

    sim = icarus.Sim(golden["config"], int(golden["seed"]))
    # Goldens flown by a controller: it must make the recorded decision every tick.
    controller = None
    if "controller" in golden:
        controller = icarus.Controller(golden["controller"], golden["config"])
    samples = {s["tick"]: s["telemetry"] for s in golden["samples"]}
    events = golden["events"]

    def check_sample(tick: int, telemetry: list[float]) -> None:
        if tick in samples:
            actual = [f"{f64_bits(v):016x}" for v in telemetry]
            for name, got, want in zip(
                golden["telemetry_fields"], actual, samples[tick], strict=True
            ):
                assert got == want, f"tick {tick}: {name} = {got}, expected {want}"

    telemetry = sim.telemetry()
    h = fnv1a(FNV_OFFSET, telemetry)
    check_sample(0, telemetry)
    tick = 0
    for count, q0, q1, q2 in golden["inputs"]:
        for _ in range(count):
            for event in events:
                if event["tick"] == tick:
                    apply_event(sim, event)
            if controller is not None:
                assert controller.act(sim) == (q0, q1, q2), f"controller differs at tick {tick}"
            sim.step_quantized(q0, q1, q2)
            tick += 1
            telemetry = sim.telemetry()
            assert all(math.isfinite(v) for v in telemetry), f"non-finite telemetry at tick {tick}"
            h = fnv1a(h, telemetry)
            check_sample(tick, telemetry)

    assert tick == golden["ticks"]
    assert sim.outcome == golden["outcome"]
    assert sim.score == golden["score"]
    assert f"{h:016x}" == golden["hash"]


@pytest.mark.parametrize("path", GOLDENS, ids=[p.stem for p in GOLDENS])
def test_replay_string_round_trips(path: Path) -> None:
    golden = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(golden["replay"], str), "every golden carries its replay string"
    # Re-recording the same inputs and events encodes the identical string...
    recorder = icarus.Recorder(golden["config"], int(golden["seed"]))
    tick = 0
    for count, q0, q1, q2 in golden["inputs"]:
        for _ in range(count):
            for event in golden["events"]:
                if event["tick"] == tick:
                    apply_event(recorder, event)
            recorder.step_quantized(q0, q1, q2)
            tick += 1
    assert recorder.replay() == golden["replay"]
    # ...and the string replays to the final state.
    replayed = icarus.run_replay(golden["replay"])
    final = golden["samples"][-1]["telemetry"]
    assert [f"{f64_bits(v):016x}" for v in replayed.telemetry()] == final
