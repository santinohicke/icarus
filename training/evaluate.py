"""Evaluation battery: every scenario, for a controller, over fixed seeds.

Runs the scenarios of ``icarus.scenarios()`` on the ``easy`` and ``realistic``
vehicles (the research configurations) and the nominal case of every
campaign level, then writes a Markdown report and the raw results as JSON.

Success rates come with Wilson score 95% confidence intervals (classical
controllers have no training seeds, so the uncertainty is per episode).

Evaluation seeds start at 0. Controllers are tuned on seeds >= 1_000_000, so
the reported numbers are never measured on the seeds the gains were fit to.

    python training/evaluate.py --controller pid --seeds 500
    python training/evaluate.py --controller pid --seeds 50 --out /tmp/eval
    python training/evaluate.py --agent models/ppo/agent.json --name ppo --seeds 500

An exported agent (``--agent``) runs through the same Rust environment it was
trained in (``icarus.VecEnv``, no domain randomization beyond the scenario),
with the NumPy reference inference of ``icarus_env.policy``. It was trained
on seeds with the top bit set and its checkpoint was selected on validation
seeds >= 2,000,000, so no evaluation seed was seen.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import icarus
from icarus_env.policy import MlpPolicy, run_agent

RESEARCH_CONFIGS = ("easy", "realistic")
CAMPAIGN = ("training", "easy", "crosswind", "realistic", "storm", "engine_out")
Z_95 = 1.959963984540054


def wilson_interval(successes: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (well-behaved at 0 and n)."""
    if n <= 0:
        return (0.0, 1.0)
    if not 0 <= successes <= n:
        raise ValueError(f"successes={successes} outside [0, {n}]")
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return statistics.fmean(values) if values else None


@dataclass(frozen=True)
class Summary:
    controller: str
    config: str
    scenario: str
    episodes: int
    landed: int
    success_rate: float
    ci_low: float
    ci_high: float
    outcomes: dict[str, int]
    # Means over landed episodes (None when nothing landed).
    contact_vy: float | None
    contact_vx: float | None
    position_error: float | None
    propellant_used: float | None
    act_us: float


def summarize(controller: str, config: str, scenario: str, rows: Sequence[dict]) -> Summary:
    if not rows:
        raise ValueError("no episodes to summarize")
    landed = [r for r in rows if r["landed"]]
    outcomes: dict[str, int] = {}
    for r in rows:
        outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1
    low, high = wilson_interval(len(landed), len(rows))
    return Summary(
        controller=controller,
        config=config,
        scenario=scenario,
        episodes=len(rows),
        landed=len(landed),
        success_rate=len(landed) / len(rows),
        ci_low=low,
        ci_high=high,
        outcomes=dict(sorted(outcomes.items(), key=lambda kv: (-kv[1], kv[0]))),
        contact_vy=_mean(abs(r["contact_vy"]) for r in landed),
        contact_vx=_mean(abs(r["contact_vx"]) for r in landed),
        position_error=_mean(r["position_error"] for r in landed),
        propellant_used=_mean(r["propellant_used"] for r in landed),
        act_us=statistics.fmean(r["act_ns"] for r in rows) / 1000.0,
    )


def run(
    controller: str, config: str, scenario: str, seeds: int, agent: MlpPolicy | None = None
) -> tuple[Summary, list[dict]]:
    if agent is None:
        rows = icarus.evaluate(controller, config, scenario, list(range(seeds)))
    else:
        rows = run_agent(agent, {"config": config, "scenario": scenario}, list(range(seeds)))
    return summarize(controller, config, scenario, rows), rows


def _fmt(v: float | None, digits: int = 2) -> str:
    return "—" if v is None else f"{v:.{digits}f}"


def _failures(s: Summary, top: int = 2) -> str:
    fails = [(k, v) for k, v in s.outcomes.items() if k != "landed"][:top]
    return ", ".join(f"{k.removeprefix('crashed_')} {v}" for k, v in fails) or "—"


def table(summaries: Sequence[Summary], key: str) -> str:
    header = (
        f"| {key.capitalize()} | Success (95% CI) | Contact v_y | Contact v_x "
        "| Pad error | Propellant | Main failures |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
    )
    lines = []
    for s in summaries:
        label = getattr(s, key)
        lines.append(
            f"| `{label}` | {100 * s.success_rate:.1f}% "
            f"({100 * s.ci_low:.1f} to {100 * s.ci_high:.1f}) "
            f"| {_fmt(s.contact_vy)} m/s | {_fmt(s.contact_vx)} m/s "
            f"| {_fmt(s.position_error, 1)} m "
            f"| {_fmt(s.propellant_used, 0)} kg | {_failures(s)} |"
        )
    return header + "\n".join(lines) + "\n"


def report(
    controller: str,
    seeds: int,
    battery: dict[str, list[Summary]],
    campaign: list[Summary],
    agent: MlpPolicy | None = None,
) -> str:
    act = statistics.fmean(s.act_us for group in battery.values() for s in group)
    parts = [
        f"# Evaluation: `{controller}`\n",
        f"{seeds} episodes per row (seeds 0 to {seeds - 1}), physics version "
        f"{icarus.PHYSICS_VERSION}. Success rates with Wilson 95% confidence intervals; "
        "contact speeds, pad error and propellant are means over landed episodes. "
        f"Mean controller time per decision: {act:.1f} µs "
        + (
            "(NumPy reference inference, batched, amortized per decision; the agent decides "
            "at 20 Hz, every third physics tick).\n"
            if agent is not None
            else "(native, single thread; timed per call, so it includes the timer's own "
            "overhead: an upper bound).\n"
        ),
        "Generated by `training/evaluate.py`. "
        + (
            _agent_line(agent)
            if agent is not None
            else "The controller was tuned on seeds ≥ 1,000,000.\n"
        ),
    ]
    for config, summaries in battery.items():
        parts.append(f"\n## Battery on `{config}`\n\n{table(summaries, 'scenario')}")
    parts.append(f"\n## Campaign levels (nominal)\n\n{table(campaign, 'config')}")
    return "".join(parts)


def _agent_line(agent: MlpPolicy) -> str:
    t = agent.manifest.get("training", {})
    return (
        f"Agent: config `{t.get('config')}`, training seed {t.get('seed')}, "
        f"{t.get('timesteps', 0):,} decisions, weights sha256 "
        f"`{agent.manifest['weights']['sha256'][:12]}`. Trained on seeds with the top bit set; "
        "checkpoint selected on validation seeds ≥ 2,000,000.\n"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--controller", default="pid", choices=icarus.controllers())
    parser.add_argument("--agent", type=Path, help="exported agent manifest (agent.json)")
    parser.add_argument("--name", default="ppo", help="report name for --agent")
    parser.add_argument("--seeds", type=int, default=500, help="episodes per row")
    parser.add_argument(
        "--out", type=Path, default=Path(__file__).resolve().parent.parent / "results"
    )
    args = parser.parse_args(argv)
    if args.seeds < 1:
        parser.error("--seeds must be at least 1")
    agent = MlpPolicy.load(args.agent) if args.agent else None
    name = args.name if agent is not None else args.controller

    started = time.perf_counter()
    battery: dict[str, list[Summary]] = {}
    raw: list[dict] = []
    for config in RESEARCH_CONFIGS:
        battery[config] = []
        for scenario in icarus.scenarios():
            summary, rows = run(name, config, scenario, args.seeds, agent)
            battery[config].append(summary)
            raw.extend({"config": config, "scenario": scenario, **r} for r in rows)
            print(f"{config:10} {scenario:20} {100 * summary.success_rate:5.1f}%")
    campaign = []
    for config in CAMPAIGN:
        summary, _ = run(name, config, "nominal", args.seeds, agent)
        campaign.append(summary)
        print(f"{config:10} {'nominal (campaign)':20} {100 * summary.success_rate:5.1f}%")

    args.out.mkdir(parents=True, exist_ok=True)
    md = args.out / f"{name}.md"
    md.write_text(report(name, args.seeds, battery, campaign, agent), encoding="utf-8")
    payload = {
        "controller": name,
        "seeds": args.seeds,
        "physics_version": icarus.PHYSICS_VERSION,
        "battery": {c: [asdict(s) for s in group] for c, group in battery.items()},
        "campaign": [asdict(s) for s in campaign],
    }
    if agent is not None:
        payload["agent"] = {
            "weights_sha256": agent.manifest["weights"]["sha256"],
            "training": agent.manifest.get("training"),
            "env_config": agent.env_config,
        }
    (args.out / f"{name}.json").write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {md} in {time.perf_counter() - started:.1f} s ({len(raw)} battery episodes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
