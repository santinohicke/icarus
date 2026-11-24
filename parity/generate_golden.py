"""Generate the golden parity trajectories from the native module.

Every case runs the simulator with a fixed seed and a fixed per-tick input
sequence and records, as hex strings of the IEEE-754 bit patterns, the full
telemetry every SAMPLE_EVERY ticks plus at the end, and a 64-bit FNV-1a hash
over the telemetry after every tick. The same files are then checked against:

- the Python extension (training/tests/test_parity.py),
- the WebAssembly build under Node (parity/verify_wasm.mjs),
- the native Rust core (crates/icarus-core/tests/golden.rs).

The input sequences are stored in the goldens, so the checkers replay them
and never re-run these policies. `--check` does re-run them: the controller
case uses Python's `math` (the platform's libm), so on another OS `--check` can
differ in the inputs even when the physics is identical; the cross-platform
proof is the checkers, not `--check`.

Goldens are tied to a physics version. Regenerate them only when the physics
version changes on purpose:

    python parity/generate_golden.py            # refuses to overwrite same-version goldens
    python parity/generate_golden.py --force    # overwrite anyway
    python parity/generate_golden.py --check    # compare without writing (exit 1 on mismatch)
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import icarus

GOLDEN_DIR = Path(__file__).resolve().parent / "golden"
GOLDEN_FORMAT = 1
SAMPLE_EVERY = 30
FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
MASK64 = (1 << 64) - 1

Quantized = tuple[int, int, int]
Policy = Callable[[int, icarus.Sim], Quantized]


def f64_bits(value: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", value))[0]


def hex_bits(value: float) -> str:
    return f"{f64_bits(value):016x}"


def fnv1a(h: int, values: list[float]) -> int:
    """Folds the little-endian bytes of each value's bit pattern into the hash."""
    for value in values:
        word = f64_bits(value)
        for i in range(8):
            h ^= (word >> (8 * i)) & 0xFF
            h = (h * FNV_PRIME) & MASK64
    return h


@dataclass(frozen=True)
class Event:
    tick: int
    kind: str  # "gimbal_stuck" | "thrust_loss" | "gust"
    a: float
    b: float = 0.0

    def apply(self, target: icarus.Sim | icarus.Recorder) -> bool:
        if self.kind == "gimbal_stuck":
            return target.inject_gimbal_stuck(self.a)
        if self.kind == "thrust_loss":
            return target.inject_thrust_loss(self.a)
        if self.kind == "gust":
            return target.inject_gust(self.a, self.b)
        raise ValueError(f"unknown event kind {self.kind!r}")

    def to_json(self) -> dict[str, object]:
        return {"tick": self.tick, "kind": self.kind, "a": hex_bits(self.a), "b": hex_bits(self.b)}


@dataclass
class Case:
    name: str
    config: str
    seed: int
    policy: Policy
    events: list[Event] = field(default_factory=list)
    max_ticks: int = 7_200


# --- Input policies ---------------------------------------------------------


def scripted_easy(tick: int, _sim: icarus.Sim) -> Quantized:
    return (
        110 + (tick * 7) % 120,
        (0, 127, 254)[(tick // 40) % 3],
        (127, 200, 60)[(tick // 25) % 3],
    )


def scripted_realistic(tick: int, _sim: icarus.Sim) -> Quantized:
    throttle = 0 if tick < 420 else 180 + (tick % 60)
    return throttle, (127, 254, 127, 0)[(tick // 30) % 4], 127


def splitmix64(state: int) -> tuple[int, int]:
    """One SplitMix64 step: (new_state, output). Integer-only, so identical everywhere."""
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, z ^ (z >> 31)


def make_fuzz(seed: int) -> Policy:
    state = seed
    current: list[int] = [127, 127, 127]

    def policy(tick: int, _sim: icarus.Sim) -> Quantized:
        nonlocal state
        if tick % 10 == 0:
            for i in range(3):
                state, value = splitmix64(state)
                current[i] = value % 255
        return current[0], current[1], current[2]

    return policy


def descent_controller(_tick: int, sim: icarus.Sim) -> Quantized:
    """A simple continuous controller for the easy vehicle (exercises touchdown)."""
    t = sim.telemetry_dict()
    alt, vy, vx, x = t["clearance"], t["vy"], t["vx"], t["x"]
    theta = math.atan2(math.sin(t["theta"]), math.cos(t["theta"]))
    vref = -0.8 if alt < 4 else -min(max(0.15 * alt, 1.2), 40.0)
    accel = 1.5 * (vref - vy)
    frac = t["mass"] * (9.81 + accel) / 500_000.0 / math.cos(theta)
    frac = min(max(frac, 0.4), 1.0)
    u = -0.9 + (frac - 0.4) / 0.6 * 1.9
    ax = -0.05 * x - 0.5 * vx if alt > 40 else -0.8 * vx
    thrust_accel = max(t["thrust"] / t["mass"], 1.0)
    theta_ref = min(max(-ax / thrust_accel, -0.12), 0.12) if alt > 15 else 0.0
    gimbal = min(max(3.0 * (theta_ref - theta) - 4.0 * t["omega"], -1.0), 1.0)

    def q(v: float) -> int:
        return min(254, max(0, round(v * 127 + 127)))

    return q(u), q(gimbal), q(gimbal * 0.5)


def scripted_relight(tick: int, _sim: icarus.Sim) -> Quantized:
    """Easy vehicle: cut the engine three times and relight (spends all ignitions),
    then a fourth cut-off whose relight must be refused."""
    off = any(start <= tick < start + 20 for start in (100, 250, 400, 550))
    return (0 if off else 150), (0, 127, 254)[(tick // 50) % 3], 127


def hard_landing(tick: int, sim: icarus.Sim) -> Quantized:
    """Brakes correctly, then stops braking 25 m up: touches down on the pad too fast."""
    q = descent_controller(tick, sim)
    if sim.telemetry_dict()["clearance"] < 25:
        return 14, 127, 127  # minimum throttle
    return q


CASES = [
    Case("easy_scripted", "easy", 1, scripted_easy, max_ticks=1_500),
    Case("realistic_scripted", "realistic", 2, scripted_realistic),
    Case(
        "realistic_events",
        "realistic",
        3,
        scripted_realistic,
        events=[
            Event(60, "gust", 9.5, 4.0),
            Event(240, "thrust_loss", 0.8),
            Event(450, "gimbal_stuck", 0.012),
            Event(450, "gust", -7.25, 2.5),
        ],
    ),
    Case("easy_controller", "easy", 4, descent_controller),
    Case("realistic_fuzz", "realistic", 5, make_fuzz(1234), max_ticks=3_000),
    # Seeds above 2^53 catch BigInt → Number truncation and exercise the high
    # bits of the seeding.
    Case("easy_relight", "easy", 2**64 - 1, scripted_relight, max_ticks=1_200),
    Case("easy_hard_landing", "easy", 2**53 + 1, hard_landing),
]

MANIFEST = GOLDEN_DIR / "manifest.json"


# --- Running ----------------------------------------------------------------


def run(case: Case) -> dict[str, object]:
    sim = icarus.Sim(case.config, case.seed)
    recorder = icarus.Recorder(case.config, case.seed)
    events = sorted(case.events, key=lambda e: e.tick)
    inputs: list[Quantized] = []
    telemetry = sim.telemetry()
    h = fnv1a(FNV_OFFSET, telemetry)
    samples = [{"tick": 0, "telemetry": [hex_bits(v) for v in telemetry]}]
    tick = 0
    while not sim.done and tick < case.max_ticks:
        for e in events:
            if e.tick == tick:
                applied = e.apply(sim)
                if e.apply(recorder) != applied:
                    raise AssertionError("recorder and sim disagree on an injection")
        q = case.policy(tick, sim)
        inputs.append(q)
        sim.step_quantized(*q)
        recorder.step_quantized(*q)
        tick += 1
        telemetry = sim.telemetry()
        if not all(math.isfinite(v) for v in telemetry):
            raise AssertionError(f"{case.name}: non-finite telemetry at tick {tick}")
        h = fnv1a(h, telemetry)
        if tick % SAMPLE_EVERY == 0 or sim.done:
            samples.append({"tick": tick, "telemetry": [hex_bits(v) for v in telemetry]})
    if [f64_bits(v) for v in recorder.telemetry()] != [f64_bits(v) for v in telemetry]:
        raise AssertionError(f"{case.name}: recorder diverged from the simulation")

    runs: list[list[int]] = []
    for q in inputs:
        if runs and tuple(runs[-1][1:]) == q:
            runs[-1][0] += 1
        else:
            runs.append([1, *q])

    return {
        "format": GOLDEN_FORMAT,
        "physics_version": icarus.PHYSICS_VERSION,
        "name": case.name,
        "config": case.config,
        "seed": str(case.seed),
        "telemetry_fields": icarus.TELEMETRY_FIELDS,
        "inputs": runs,
        "events": [e.to_json() for e in events],
        "ticks": tick,
        "outcome": sim.outcome,
        "score": sim.score,
        "sample_every": SAMPLE_EVERY,
        "samples": samples,
        "hash": f"{h:016x}",
        # Always present: a replay of an unfinished run is valid and replays the
        # same ticks.
        "replay": recorder.replay(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--force", action="store_true", help="overwrite same-version goldens")
    group.add_argument("--check", action="store_true", help="compare without writing")
    args = parser.parse_args()

    # Compute everything first, so a refusal never leaves a partial update.
    results = {case.name: run(case) for case in CASES}
    manifest = {"format": GOLDEN_FORMAT, "cases": [case.name for case in CASES]}

    def existing(path: Path) -> object:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    if args.check:
        mismatches = 0
        for name, result in results.items():
            ok = existing(GOLDEN_DIR / f"{name}.json") == result
            mismatches += not ok
            print(f"{'ok      ' if ok else 'MISMATCH'} {name}")
        if existing(MANIFEST) != manifest:
            mismatches += 1
            print("MISMATCH manifest.json")
        return 1 if mismatches else 0

    if not args.force:
        for name, result in results.items():
            old = existing(GOLDEN_DIR / f"{name}.json")
            if (
                isinstance(old, dict)
                and old.get("physics_version") == icarus.PHYSICS_VERSION
                and old != result
            ):
                print(
                    f"refusing to overwrite {name}.json: same physics version "
                    f"({icarus.PHYSICS_VERSION}) but different trajectory. Bump PHYSICS_VERSION "
                    "if the physics changed on purpose, or pass --force. Nothing was written.",
                    file=sys.stderr,
                )
                return 1

    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    for stale in GOLDEN_DIR.glob("*.json"):
        if stale.stem not in results and stale != MANIFEST:
            stale.unlink()
            print(f"removed stale {stale.name}")
    for name, result in results.items():
        path = GOLDEN_DIR / f"{name}.json"
        path.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8", newline="\n")
        print(
            f"wrote {path.name}: {result['ticks']} ticks, "
            f"{result['outcome']}, score {result['score']}"
        )
    MANIFEST.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
