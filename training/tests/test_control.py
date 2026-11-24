"""The classical controllers and the evaluation battery, through Python."""

from __future__ import annotations

import json
import math

import evaluate
import icarus
import pytest


def fly(config: str, seed: int, scenario: str = "nominal") -> icarus.Sim:
    sim = icarus.Sim(config, seed, scenario)
    ctrl = icarus.Controller("pid", config, scenario)
    while not sim.done:
        sim.step_quantized(*ctrl.act(sim))
    return sim


def test_metadata():
    assert icarus.controllers() == ["pid"]
    assert icarus.scenarios() == [
        "nominal",
        "initial_tilt",
        "strong_gusts",
        "delayed_sensors",
        "gimbal_stuck",
        "thrust_loss",
        "out_of_distribution",
    ]
    assert icarus.Controller().name == "pid"


def test_unknown_names_raise():
    with pytest.raises(ValueError, match="controller"):
        icarus.Controller("lqr")
    with pytest.raises(ValueError, match="scenario"):
        icarus.Controller("pid", "easy", "hurricane")
    with pytest.raises(ValueError, match="scenario"):
        icarus.Sim("easy", 1, "hurricane")
    with pytest.raises(ValueError, match="configuration"):
        icarus.evaluate("pid", "nope", "nominal", [1])


def test_pid_lands_the_easy_vehicle():
    sim = fly("easy", 3)
    assert sim.outcome == "landed"
    assert sim.score > 1000


def test_python_loop_matches_rust_evaluate():
    """Stepping in Python and the Rust battery loop make identical decisions."""
    rows = icarus.evaluate("pid", "realistic", "thrust_loss", [4, 5])
    for row in rows:
        sim = fly("realistic", row["seed"], "thrust_loss")
        assert sim.outcome == row["outcome"]
        assert sim.tick == row["ticks"]
        assert sim.score == row["score"]
        assert row["act_ns"] > 0


def test_evaluate_is_deterministic_apart_from_timing():
    a = icarus.evaluate("pid", "storm", "nominal", [1, 2, 3])
    b = icarus.evaluate("pid", "storm", "nominal", [1, 2, 3])
    strip = [{k: v for k, v in r.items() if k != "act_ns"} for r in a]
    assert strip == [{k: v for k, v in r.items() if k != "act_ns"} for r in b]
    assert icarus.evaluate("pid", "easy", "nominal", []) == []


def test_controller_refuses_a_mismatched_sim():
    ctrl = icarus.Controller("pid", "realistic")
    with pytest.raises(ValueError, match="cannot fly"):
        ctrl.act(icarus.Sim("storm", 1))
    with pytest.raises(ValueError, match="cannot fly"):
        ctrl.act(icarus.Sim("realistic", 1, "delayed_sensors"))
    sim = icarus.Sim("realistic", 1, "delayed_sensors")
    assert sim.scenario == "delayed_sensors"
    assert icarus.Sim("easy", 1).scenario == "nominal"


def test_one_decision_per_tick():
    ctrl = icarus.Controller("pid", "easy")
    sim = icarus.Sim("easy", 1)
    ctrl.act(sim)
    with pytest.raises(ValueError, match="already called"):
        ctrl.act(sim)
    sim.step_quantized(127, 127, 127)
    ctrl.act(sim)
    ctrl.reset()
    ctrl.act(icarus.Sim("easy", 2))


def test_reset_makes_a_controller_reusable():
    ctrl = icarus.Controller("pid", "realistic")
    first = []
    for _ in range(2):
        ctrl.reset()
        sim = icarus.Sim("realistic", 9)
        while not sim.done:
            sim.step_quantized(*ctrl.act(sim))
        first.append((sim.outcome, sim.tick))
    assert first[0] == first[1]


def test_controller_flight_replays_exactly():
    rec = icarus.Recorder("easy", 12)
    ctrl = icarus.Controller("pid", "easy")
    while not rec.done:
        rec.step_quantized(*ctrl.act_recorder(rec))
    replayed = icarus.run_replay(rec.replay())
    assert replayed.telemetry() == rec.telemetry()
    assert replayed.outcome == rec.outcome


def test_scenarios_change_the_episode():
    nominal = icarus.Sim("easy", 5)
    tilted = icarus.Sim("easy", 5, "initial_tilt")
    theta = icarus.TELEMETRY_FIELDS.index("theta")
    assert abs(nominal.telemetry()[theta]) < math.radians(5)
    assert math.radians(30) <= tilted.telemetry()[theta] <= math.radians(35)


@pytest.mark.parametrize(
    ("successes", "n", "low", "high"),
    [(0, 10, 0.0, 0.2775), (10, 10, 0.7225, 1.0), (5, 10, 0.2366, 0.7634), (0, 0, 0.0, 1.0)],
)
def test_wilson_interval(successes, n, low, high):
    lo, hi = evaluate.wilson_interval(successes, n)
    assert lo == pytest.approx(low, abs=1e-4)
    assert hi == pytest.approx(high, abs=1e-4)


def test_wilson_rejects_impossible_counts():
    with pytest.raises(ValueError, match="outside"):
        evaluate.wilson_interval(11, 10)


def test_battery_writes_report_and_json(tmp_path):
    assert evaluate.main(["--seeds", "2", "--out", str(tmp_path)]) == 0
    report = (tmp_path / "pid.md").read_text(encoding="utf-8")
    data = json.loads((tmp_path / "pid.json").read_text(encoding="utf-8"))
    assert "## Battery on `realistic`" in report
    assert "`out_of_distribution`" in report
    assert set(data["battery"]) == {"easy", "realistic"}
    assert len(data["battery"]["easy"]) == len(icarus.scenarios())
    assert [s["config"] for s in data["campaign"]] == list(evaluate.CAMPAIGN)
    for s in data["campaign"]:
        assert s["episodes"] == 2
        assert 0.0 <= s["ci_low"] <= s["success_rate"] <= s["ci_high"] <= 1.0
