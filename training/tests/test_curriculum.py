"""The curriculum promotes on success and keeps replaying earlier stages."""

from __future__ import annotations

import pytest
from icarus_env import Curriculum, Stage


def make(replay: float = 0.0) -> Curriculum:
    return Curriculum.from_spec(
        {
            "replay_fraction": replay,
            "stages": [
                {"name": "a", "task": {"config": "training"}, "promote_at": 0.8, "window": 10},
                {"name": "b", "task": {"config": "easy"}, "promote_at": 0.5, "window": 4},
                {"name": "c", "task": {"config": "realistic"}},
            ],
        },
        seed=1,
    )


def test_promotion_needs_a_full_window_at_the_threshold() -> None:
    c = make()
    for _ in range(9):
        c.record(0, True)
    assert not c.ready(), "window not full"
    c.record(0, False)
    assert c.success_rate == 0.9
    assert c.ready()
    c.promote(step=1000)
    assert c.stage.name == "b"
    assert c.episodes == 0
    for landed in (True, False, False, False):
        c.record(1, landed)
    assert not c.ready()
    for landed in (True, True):
        c.record(1, landed)
    assert c.ready(), "the window slides: last four are F, F, T, T"
    c.promote(step=2000)
    assert c.is_last
    assert not c.ready()
    with pytest.raises(RuntimeError):
        c.promote()
    assert c.history == [(0, "a"), (1000, "b"), (2000, "c")]


def test_replayed_stages_do_not_count() -> None:
    c = make()
    c.promote()
    for _ in range(10):
        c.record(0, True)
    assert c.episodes == 0


def test_assignment_replays_earlier_stages() -> None:
    c = make(replay=0.25)
    assert c.assign(8) == [0] * 8, "nothing earlier to replay"
    c.promote()
    c.promote()
    stages = c.assign(16)
    assert stages.count(2) == 12
    assert all(s in (0, 1) for s in stages if s != 2)


def test_invalid_specs() -> None:
    with pytest.raises(ValueError, match="at least one"):
        Curriculum(stages=[])
    with pytest.raises(ValueError, match="promote_at"):
        Stage.from_spec({"name": "x", "task": {}, "promote_at": 1.5})
    with pytest.raises(ValueError, match="unique"):
        Curriculum(stages=[Stage("x", {}), Stage("x", {})])
    with pytest.raises(ValueError, match="replay_fraction"):
        Curriculum(stages=[Stage("x", {})], replay_fraction=1.0)
    with pytest.raises(ValueError, match="unknown stage keys"):
        Stage.from_spec({"name": "x", "task": {}, "promote": 0.5})


def test_stage_thresholds_must_match_their_position() -> None:
    with pytest.raises(ValueError, match="needs promote_at"):
        Curriculum(stages=[Stage("a", {}), Stage("b", {})])
    with pytest.raises(ValueError, match="cannot have promote_at"):
        Curriculum(stages=[Stage("a", {}, promote_at=0.5)])
    with pytest.raises(ValueError, match="needs"):
        Stage.from_spec({"name": "a"})
    with pytest.raises(ValueError, match="window"):
        Stage.from_spec({"name": "a", "task": {}, "window": 2.7})


def test_assignment_is_deterministic_and_rounds_up() -> None:
    a, b = make(replay=0.25), make(replay=0.25)
    for c in (a, b):
        c.promote()
        c.promote()
    assert a.assign(16) == b.assign(16)
    assert a.assign(16) == a.assign(16), "independent of how often it is called"
    small = make(replay=0.25)
    small.promote()
    assert small.assign(2).count(0) == 1, "a 25% share of 2 envs is one env"
    assert small.assign(1) == [1], "never every env"


def test_replay_can_be_restricted_to_named_stages() -> None:
    c = Curriculum.from_spec(
        {
            "replay_fraction": 0.5,
            "replay_stages": ["b"],
            "stages": [
                {"name": "a", "task": {}, "promote_at": 0.5, "window": 1},
                {"name": "b", "task": {}, "promote_at": 0.5, "window": 1},
                {"name": "c", "task": {}},
            ],
        }
    )
    assert c.assign(8) == [0] * 8, "b is not earlier than a"
    c.promote()
    c.promote()
    stages = c.assign(8)
    assert stages.count(2) == 4
    assert stages.count(1) == 4
    with pytest.raises(ValueError, match="not stages"):
        Curriculum(stages=[Stage("a", {})], replay_stages=["zz"])
