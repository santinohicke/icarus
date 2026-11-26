"""Generate the RL-environment golden hashes (``parity/env_golden.json``).

Each case runs ``icarus.Env`` on a task with a fixed seed and a fixed action
script, and hashes (64-bit FNV-1a) every observation (``f32`` bits), every
reward (``f64`` bits) and the termination flags. The same hashes are checked
by the native Rust core (``crates/icarus-core/tests/env_golden.rs``) and by
the Python extension on every CI operating system
(``training/tests/test_env_golden.py``), so the environment (observation,
reward, termination, domain randomization) is pinned bit for bit, not only
the physics.

The action script uses only small-integer arithmetic and one division, which
IEEE 754 rounds identically everywhere:

    action[k] = [((37k + 11) % 201) / 100 - 1,
                 ((53k + 7) % 201) / 100 - 1,
                 ((71k + 3) % 201) / 100 - 1]

Regenerate only when the environment changes on purpose:

    python parity/generate_env_golden.py            # refuses to overwrite
    python parity/generate_env_golden.py --force
    python parity/generate_env_golden.py --check    # exit 1 on mismatch
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path
from typing import Any

import icarus

GOLDEN = Path(__file__).resolve().parent / "env_golden.json"
FORMAT = 1
MAX_STEPS = 400
FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
MASK64 = (1 << 64) - 1

RANDOMIZED = {
    "dry_mass": [0.9, 1.1],
    "propellant": [0.85, 1.15],
    "wind": [0.0, 1.5],
    "thrust_time_constant": [0.8, 1.25],
    "gimbal_time_constant": [0.8, 1.25],
    "sensor_noise": [0.5, 1.5],
    "sensor_delay": [0, 9],
}

CASES: list[dict[str, Any]] = [
    {"name": "easy", "task": {"config": "easy"}, "seed": 1},
    {"name": "realistic", "task": {"config": "realistic"}, "seed": 2},
    {"name": "storm", "task": {"config": "storm"}, "seed": 3},
    {
        "name": "engine_out_randomized",
        "task": {"config": "engine_out", "randomization": RANDOMIZED},
        "seed": 4,
    },
    {
        "name": "easy_tilted_low",
        "task": {
            "config": "easy",
            "scenario": "initial_tilt",
            "overrides": {"start.altitude": [200.0, 300.0], "faults.probability": 0.5},
        },
        "seed": 5,
    },
    {
        "name": "training_repeat1",
        "task": {"config": "training"},
        "env_config": {"action_repeat": 1, "quantize_actions": False},
        "seed": 6,
    },
]


def action(k: int) -> list[float]:
    return [
        ((37 * k + 11) % 201) / 100 - 1,
        ((53 * k + 7) % 201) / 100 - 1,
        ((71 * k + 3) % 201) / 100 - 1,
    ]


def _feed(h: int, data: bytes) -> int:
    for byte in data:
        h ^= byte
        h = (h * FNV_PRIME) & MASK64
    return h


def hash_obs(h: int, obs: Any) -> int:
    return _feed(h, b"".join(struct.pack("<f", float(v)) for v in obs))


def run(case: dict[str, Any]) -> dict[str, Any]:
    env = icarus.Env(case["task"], case.get("env_config"))
    obs = env.reset(case["seed"])
    h = hash_obs(FNV_OFFSET, obs)
    steps, outcome = 0, None
    for k in range(MAX_STEPS):
        obs, reward, terminated, truncated, info = env.step(action(k))
        h = hash_obs(h, obs)
        h = _feed(h, struct.pack("<d", reward))
        h = _feed(h, bytes([int(terminated) | (int(truncated) << 1)]))
        steps = k + 1
        if terminated or truncated:
            outcome = info["outcome"]
            break
    return {
        "name": case["name"],
        "task": case["task"],
        "env_config": case.get("env_config"),
        "seed": case["seed"],
        "steps": steps,
        "outcome": outcome,
        "hash": f"{h:016x}",
    }


def generate() -> dict[str, Any]:
    return {
        "format": FORMAT,
        "physics_version": icarus.PHYSICS_VERSION,
        "max_steps": MAX_STEPS,
        "cases": [run(c) for c in CASES],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--force", action="store_true", help="overwrite the existing file")
    group.add_argument("--check", action="store_true", help="compare without writing")
    args = parser.parse_args()
    data = generate()
    if args.check:
        old = json.loads(GOLDEN.read_text(encoding="utf-8"))
        if old != data:
            print("env golden mismatch", file=sys.stderr)
            return 1
        print("env golden matches")
        return 0
    if GOLDEN.exists() and not args.force:
        print(f"{GOLDEN} exists; use --force to overwrite", file=sys.stderr)
        return 1
    GOLDEN.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")
    for c in data["cases"]:
        print(f"{c['name']}: {c['steps']} steps, {c['outcome']}, {c['hash']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
