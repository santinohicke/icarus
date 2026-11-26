"""The single-environment API (``icarus.Env``) and its Gymnasium wrapper."""

from __future__ import annotations

import math

import icarus
import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")

from gymnasium.utils.env_checker import check_env  # noqa: E402
from icarus_env import IcarusEnv, resolve_task  # noqa: E402


# Observations are unbounded on purpose (VecNormalize clips them).
@pytest.mark.filterwarnings("ignore:.*Box observation space m.*:UserWarning")
def test_gymnasium_api_compliance() -> None:
    check_env(IcarusEnv({"config": "realistic"}), skip_render_check=True)
    check_env(
        IcarusEnv(
            {"config": "storm", "randomization": {"sensor_delay": [0, 12], "wind": [0.5, 1.5]}}
        ),
        skip_render_check=True,
    )


def test_spaces() -> None:
    env = IcarusEnv()
    assert env.observation_space.shape == (env.core.obs_dim,)
    assert env.observation_space.dtype == np.float32
    assert env.action_space.shape == (3,)
    assert np.all(env.action_space.low == -1.0)
    assert np.all(env.action_space.high == 1.0)
    assert len(icarus.obs_fields()) == env.core.obs_dim == 26


def test_reset_seeding_is_reproducible_and_never_an_evaluation_seed() -> None:
    env = IcarusEnv({"config": "crosswind"})
    a, info_a = env.reset(seed=123)
    b, info_b = env.reset(seed=123)
    assert info_a["episode_seed"] == info_b["episode_seed"]
    assert info_a["episode_seed"] >= 1 << 63
    np.testing.assert_array_equal(a, b)
    c, _ = env.reset(seed=124)
    assert not np.array_equal(a, c)


def test_explicit_episode_seed_matches_the_core() -> None:
    env = IcarusEnv({"config": "realistic"})
    obs, info = env.reset(options={"episode_seed": 7})
    assert info["episode_seed"] == 7
    core = icarus.Env({"config": "realistic"})
    np.testing.assert_array_equal(obs, core.reset(7))
    for k in range(50):
        a = np.array([0.5, math.sin(k), -0.2], np.float32)
        o1, r1, t1, u1, _ = env.step(a)
        o2, r2, t2, u2, _ = core.step(a.astype(np.float64))
        np.testing.assert_array_equal(o1, o2)
        assert r1 == r2
        assert t1 == t2
        assert u1 == u2


def test_task_switch_through_options() -> None:
    env = IcarusEnv({"config": "easy"})
    obs, _ = env.reset(options={"episode_seed": 1})
    assert obs[10] == 1.0  # engine on
    obs, _ = env.reset(options={"episode_seed": 1, "task": {"config": "realistic"}})
    assert obs[10] == 0.0  # engine off


def test_step_info_and_episode_end() -> None:
    env = icarus.Env({"config": "realistic"})
    env.reset(9)
    while True:
        _obs, reward, terminated, truncated, info = env.step([-1.0, 0.0, 0.0])
        if terminated or truncated:
            break
    assert terminated
    assert not truncated
    assert info["outcome"].startswith("crashed")
    assert info["landed"] is False
    assert info["score"] == 0
    assert info["seed"] == 9
    assert reward < -50
    assert env.done
    assert env.potential < 0.0, "the terminal state keeps its potential"
    with pytest.raises(RuntimeError, match="after the episode ended"):
        env.step([0.0, 0.0, 0.0])


def test_bad_input_is_rejected() -> None:
    env = icarus.Env()
    with pytest.raises(RuntimeError, match="before reset"):
        env.step([0.0, 0.0, 0.0])
    env.reset(0)
    with pytest.raises(ValueError, match="finite"):
        env.step([float("nan"), 0.0, 0.0])
    with pytest.raises(ValueError, match="3 components"):
        env.step([0.0, 0.0])
    with pytest.raises(ValueError, match="unknown task key"):
        icarus.Env({"vehicle": "easy"})
    with pytest.raises(ValueError, match="unknown override"):
        icarus.Env({"overrides": {"start.nope": [1, 2]}})
    with pytest.raises(ValueError, match="unknown env_config key"):
        icarus.Env(None, {"repeat": 3})
    with pytest.raises(ValueError, match="unknown reward key"):
        icarus.Env(None, {"reward": {"gama": 0.99}})
    with pytest.raises(ValueError, match="crash penalty"):
        icarus.Env(None, {"reward": {"crash_penalty": 10.0}})
    with pytest.raises(ValueError, match="history_len"):
        icarus.Env({"config": "storm"}, {"action_repeat": 1})
    with pytest.raises(ValueError, match="start altitude"):
        icarus.Env({"overrides": {"start.altitude": [10, 20]}})
    with pytest.raises(ValueError, match="sensor_delay"):
        icarus.Env({"randomization": {"sensor_delay": [1, 2, 3]}})


def test_degree_overrides_are_converted() -> None:
    task = resolve_task({"config": "easy", "overrides": {"start.tilt_deg": [10, 20]}})
    assert task["overrides"]["start.tilt"] == pytest.approx([math.radians(10), math.radians(20)])
    with pytest.raises(ValueError, match="twice"):
        resolve_task({"overrides": {"start.tilt_deg": [1, 2], "start.tilt": [0, 0.1]}})
    with pytest.raises(ValueError, match="unknown task keys"):
        resolve_task({"config": "easy", "seed": 3})


def test_default_env_config_round_trips() -> None:
    cfg = icarus.default_env_config()
    assert cfg["action_repeat"] == 3
    assert cfg["history_len"] == 4
    assert cfg["reward"]["gamma"] == pytest.approx(0.999)
    env = icarus.Env(None, cfg)
    assert env.env_config() == cfg
    keys = dict(icarus.override_keys())
    assert keys["start.altitude"] == 2
    assert keys["sensors.noise_sigma"] == 6


def test_telemetry_is_available_for_debugging() -> None:
    env = icarus.Env()
    assert env.telemetry_dict() == {}
    env.reset(3)
    t = env.telemetry_dict()
    assert list(t) == icarus.TELEMETRY_FIELDS


def test_override_values_must_be_numbers() -> None:
    with pytest.raises(ValueError, match="expected a number"):
        resolve_task({"overrides": {"start.altitude": "300"}})
    with pytest.raises(ValueError, match="expected a number"):
        resolve_task({"overrides": {"start.tilt_deg": True}})
    with pytest.raises(ValueError, match="mapping"):
        resolve_task({"overrides": [1, 2]})
    task = resolve_task({"overrides": {"wind.gust_rate": np.float64(0.1)}})
    assert task["overrides"]["wind.gust_rate"] == [0.1]


def test_obs_fields_is_bounded() -> None:
    assert len(icarus.obs_fields(12)) == 14 + 36
    with pytest.raises(ValueError, match="at most"):
        icarus.obs_fields(10**7)
