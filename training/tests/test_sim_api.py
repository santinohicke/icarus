"""The Python binding exposes the Rust core faithfully."""

from __future__ import annotations

import math
import struct

import icarus
import pytest


def bits(values: list[float]) -> list[int]:
    """IEEE-754 bit patterns, so comparisons are exact (and NaN-safe)."""
    return [struct.unpack("<Q", struct.pack("<d", v))[0] for v in values]


def scripted(tick: int) -> tuple[int, int, int]:
    return 110 + (tick * 7) % 120, (0, 127, 254)[(tick // 40) % 3], 127


def test_module_metadata() -> None:
    assert icarus.PHYSICS_VERSION >= 1
    assert icarus.REPLAY_FORMAT_VERSION >= 1
    assert icarus.configs() == ["easy", "realistic", "training", "crosswind", "storm", "engine_out"]
    assert "landed" in icarus.OUTCOMES
    assert len(icarus.TELEMETRY_FIELDS) == len(set(icarus.TELEMETRY_FIELDS))
    assert icarus.outcome_from_code(1) == "landed"
    for code in (0, -1, 255, 256, 10**12):
        assert icarus.outcome_from_code(code) is None


def test_rejects_unknown_config_and_bad_integers() -> None:
    with pytest.raises(ValueError, match="unknown vehicle configuration"):
        icarus.Sim("hard", 1)
    with pytest.raises(OverflowError):
        icarus.Sim("easy", -1)
    sim = icarus.Sim("easy", 1)
    with pytest.raises(OverflowError):
        sim.step_quantized(300, 127, 127)
    with pytest.raises(ValueError, match=r"0..=254"):
        sim.step_quantized(255, 127, 127)
    assert sim.tick == 0, "a rejected step does not advance"


def test_initial_state() -> None:
    easy = icarus.Sim("easy", 3)
    assert easy.config == "easy"
    assert easy.seed == 3
    assert easy.tick == 0
    assert easy.engine_on
    assert easy.ignitions_left == 3
    assert not easy.done
    assert easy.outcome is None
    realistic = icarus.Sim("realistic", 3)
    assert not realistic.engine_on
    assert realistic.initial_throttle_level() == 0


def test_same_seed_is_bit_identical_and_different_seeds_differ() -> None:
    a, b, c = icarus.Sim("realistic", 5), icarus.Sim("realistic", 5), icarus.Sim("realistic", 6)
    for tick in range(400):
        q = scripted(tick)
        a.step_quantized(*q)
        b.step_quantized(*q)
        c.step_quantized(*q)
        assert bits(a.telemetry()) == bits(b.telemetry())
        assert bits(a.observation()) == bits(b.observation())
    assert bits(a.telemetry()) != bits(c.telemetry())


def test_quantized_step_equals_float_step_with_dequantized_values() -> None:
    a, b = icarus.Sim("easy", 9), icarus.Sim("easy", 9)
    for tick in range(300):
        q = scripted(tick)
        a.step_quantized(*q)
        b.step(*((v - 127) / 127 for v in q))
    assert bits(a.telemetry()) == bits(b.telemetry())


def test_non_finite_controls_are_sanitized() -> None:
    sim = icarus.Sim("easy", 1)
    sim.step(math.nan, math.inf, -math.inf)
    t = sim.telemetry_dict()
    assert not sim.engine_on, "a NaN throttle means engine off"
    assert all(math.isfinite(v) for v in t.values())


def test_telemetry_dict_matches_the_field_list() -> None:
    sim = icarus.Sim("easy", 1)
    t = sim.telemetry_dict()
    assert list(t) == icarus.TELEMETRY_FIELDS, "keys follow TELEMETRY_FIELDS order"
    assert list(t.values()) == sim.telemetry()
    assert len(sim.observation()) == 6


def test_episode_end_flags_are_consistent() -> None:
    sim = icarus.Sim("realistic", 2)
    while sim.step(-1.0) is None:
        pass
    assert sim.done
    assert sim.terminated != sim.truncated
    assert sim.outcome in icarus.OUTCOMES
    assert sim.outcome_code == icarus.OUTCOMES.index(sim.outcome) + 1
    tick = sim.tick
    assert sim.step(1.0) == sim.outcome, "stepping a finished episode does nothing"
    assert sim.tick == tick


def test_injections() -> None:
    sim = icarus.Sim("easy", 1)
    assert sim.inject_thrust_loss(0.8)
    assert sim.inject_gimbal_stuck(0.01)
    assert sim.inject_gust(5.0, 2.0)
    assert not sim.inject_gust(math.nan, 2.0)
    sim.step(1.0)
    t = sim.telemetry_dict()
    assert t["thrust_factor"] == 0.8
    assert t["gimbal_stuck"] == 1.0


def test_recorder_replay_round_trip() -> None:
    rec = icarus.Recorder("realistic", 11)
    for tick in range(3_000):
        if tick == 200:
            assert rec.inject_gust(-6.0, 3.0)
        if rec.step_quantized(*scripted(tick)) is not None:
            break
    replay = rec.replay()
    assert replay.isascii()
    replayed = icarus.run_replay(replay)
    assert replayed.tick == rec.tick
    assert replayed.outcome == rec.outcome
    assert replayed.score == rec.score
    assert bits(replayed.telemetry()) == bits(rec.telemetry())


def test_recorder_stops_recording_after_the_end() -> None:
    rec = icarus.Recorder("realistic", 2)
    while rec.step_quantized(0, 127, 127) is None:
        pass
    recorded = rec.recorded_ticks
    assert rec.done
    rec.step_quantized(200, 127, 127)
    assert rec.recorded_ticks == recorded == rec.tick


def test_empty_recording_round_trips() -> None:
    rec = icarus.Recorder("easy", 8)
    replayed = icarus.run_replay(rec.replay())
    assert replayed.tick == 0
    assert replayed.outcome is None
    assert replayed.config == "easy"
    assert replayed.seed == 8


def test_replay_decoding_edge_cases() -> None:
    rec = icarus.Recorder("easy", 3)
    for tick in range(50):
        rec.step_quantized(*scripted(tick))
    text = rec.replay()
    assert icarus.run_replay(f"  {text}\n").tick == 50, "surrounding whitespace is ignored"
    with pytest.raises(ValueError):  # noqa: PT011 - padding is not URL-safe base64
        icarus.run_replay(text + "=")
    for cut in (len(text) // 2, len(text) - 2):
        with pytest.raises(ValueError):  # noqa: PT011
            icarus.run_replay(text[:cut])


def test_replay_from_another_physics_version_is_rejected() -> None:
    import base64

    raw = bytearray(base64.urlsafe_b64decode(icarus.Recorder("easy", 1).replay() + "=="))
    raw[5] ^= 0xFF  # physics version field
    tampered = base64.urlsafe_b64encode(bytes(raw)).decode().rstrip("=")
    with pytest.raises(ValueError, match="physics version"):
        icarus.run_replay(tampered)


def test_clone_and_copy_are_independent() -> None:
    import copy

    sim = icarus.Sim("easy", 12)
    for tick in range(100):
        sim.step_quantized(*scripted(tick))
    for twin in (sim.clone(), copy.copy(sim), copy.deepcopy(sim)):
        assert bits(twin.telemetry()) == bits(sim.telemetry())
        twin.step(1.0)
        assert twin.tick == sim.tick + 1
    assert sim.tick == 100


@pytest.mark.parametrize("garbage", ["", "!!!", "QUJD", "x" * 400_000])
def test_run_replay_rejects_garbage(garbage: str) -> None:
    with pytest.raises(ValueError):  # noqa: PT011 - the message varies by failure
        icarus.run_replay(garbage)
