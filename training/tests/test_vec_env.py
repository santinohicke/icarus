"""The vectorized environment equals N single environments, bit for bit."""

from __future__ import annotations

import icarus
import numpy as np
import pytest

TASK = {
    "config": "realistic",
    "overrides": {"limits.max_time": 4.0},  # short episodes: many auto-resets
    "randomization": {"wind": [0.5, 1.5], "dry_mass": [0.95, 1.05], "sensor_delay": [0, 9]},
}


def actions_for(step: int, n: int) -> np.ndarray:
    rng = np.random.default_rng(step)
    return rng.uniform(-1.0, 1.0, size=(n, 3))


@pytest.mark.parametrize("parallel", [False, True])
def test_vec_env_matches_single_envs(parallel: bool) -> None:
    n, base = 7, 42
    vec = icarus.VecEnv(n, seed=base, task=TASK, parallel=parallel)
    singles = [icarus.Env(TASK) for _ in range(n)]
    episode = [0] * n
    obs = vec.reset()
    for i, env in enumerate(singles):
        np.testing.assert_array_equal(obs[i], env.reset(icarus.episode_seed(base, i, 0)))
        episode[i] = 1
    ends = 0
    for step in range(200):
        acts = actions_for(step, n)
        obs, rewards, terminated, truncated, info = vec.step(acts)
        for i, env in enumerate(singles):
            o, r, te, tr, inf = env.step(acts[i])
            assert rewards[i] == r, (step, i)
            assert terminated[i] == te
            assert truncated[i] == tr
            if te or tr:
                ends += 1
                np.testing.assert_array_equal(info["terminal_obs"][i], o)
                assert info["seed"][i] == icarus.episode_seed(base, i, episode[i] - 1)
                assert icarus.outcome_from_code(int(info["outcome"][i])) == inf["outcome"]
                assert info["score"][i] == inf["score"]
                assert info["episode_return"][i] == env.episode_return
                assert info["episode_length"][i] == env.episode_length
                o = env.reset(icarus.episode_seed(base, i, episode[i]))
                episode[i] += 1
            else:
                assert info["outcome"][i] == 0
            np.testing.assert_array_equal(obs[i], o)
    assert ends > n, "auto-reset was exercised"


def test_parallel_and_serial_are_identical() -> None:
    results = []
    for parallel in (False, True):
        vec = icarus.VecEnv(32, seed=5, task=TASK, parallel=parallel)
        trace = [vec.reset()]
        for step in range(100):
            obs, rewards, te, tr, info = vec.step(actions_for(step, 32))
            trace += [obs, rewards, te, tr, *(info[k] for k in sorted(info))]
        results.append(trace)
    for a, b in zip(*results, strict=True):
        np.testing.assert_array_equal(a, b)


def test_shapes_dtypes_and_errors() -> None:
    vec = icarus.VecEnv(4, task={"config": "easy"})
    assert (vec.num_envs, vec.obs_dim, vec.action_dim) == (4, 26, 3)
    with pytest.raises(RuntimeError, match="before reset"):
        vec.step(np.zeros((4, 3)))
    obs = vec.reset()
    assert obs.shape == (4, 26)
    assert obs.dtype == np.float32
    obs, rewards, te, tr, info = vec.step(np.zeros((4, 3), np.float32))
    assert rewards.dtype == np.float64
    assert te.dtype == np.bool_
    assert tr.dtype == np.bool_
    assert info["terminal_obs"].shape == (4, 26)
    with pytest.raises(ValueError, match="shape"):
        vec.step(np.zeros((3, 3)))
    with pytest.raises(ValueError, match="finite"):
        vec.step(np.full((4, 3), np.inf))
    with pytest.raises(ValueError, match="seeds"):
        vec.reset(seeds=[1, 2])
    with pytest.raises(ValueError, match="num_envs"):
        icarus.VecEnv(0)
    with pytest.raises(ValueError, match="out of range"):
        vec.set_task({"config": "realistic"}, indices=[0, 4])
    # A rejected set_task changed nothing, even for the valid index 0.
    assert np.all(vec.reset()[:, 10] == 1.0)
    with pytest.raises(ValueError, match="history_len"):
        icarus.VecEnv(2, env_config={"action_repeat": 1}).set_task({"config": "storm"})


def test_set_task_applies_to_chosen_envs_at_their_next_reset() -> None:
    vec = icarus.VecEnv(4, task={"config": "easy"})
    vec.reset()
    vec.set_task({"config": "realistic"}, indices=[1, 3])
    obs, *_ = vec.step(np.zeros((4, 3)))
    assert np.all(obs[:, 10] == 1.0), "running episodes keep their task"
    obs = vec.reset()
    assert list(obs[:, 10]) == [1.0, 0.0, 1.0, 0.0]


def test_evaluation_mode_without_autoreset() -> None:
    vec = icarus.VecEnv(3, task={"config": "realistic"}, autoreset=False)
    vec.reset(seeds=[0, 1, 2])
    finished = np.zeros(3, bool)
    outcomes: dict[int, int] = {}
    for _ in range(5000):
        _, rewards, te, tr, info = vec.step(np.tile([-1.0, 0.0, 0.0], (3, 1)))
        for i in np.flatnonzero(te | tr):
            outcomes[int(i)] = int(info["outcome"][i])
        assert np.all(rewards[finished] == 0.0)
        assert np.all(info["idle"] == finished)
        finished |= te | tr
        if finished.all():
            break
    assert finished.all()
    results = vec.episode_results()
    assert [r["seed"] for r in results] == [0, 1, 2]
    assert all(r["done"] and not r["landed"] for r in results)
    sim = icarus.Sim("realistic", 1)
    while not sim.done:
        sim.step(-1.0)
    assert results[1]["outcome"] == sim.outcome
    assert results[1]["ticks"] == sim.tick
    assert icarus.outcome_from_code(outcomes[1]) == sim.outcome


def test_base_seed_restarts_the_sequence() -> None:
    vec = icarus.VecEnv(2, seed=1)
    a = vec.reset()
    vec.set_base_seed(1)
    np.testing.assert_array_equal(vec.reset(), a)
    vec.set_base_seed(2)
    assert not np.array_equal(vec.reset(), a)


class TestSb3Adapter:
    sb3 = pytest.importorskip("stable_baselines3")

    def test_infos_follow_the_sb3_contract(self) -> None:
        from icarus_env import IcarusVecEnv

        # Low starts crash within the 1 s limit, high ones are truncated.
        mixed = {
            "config": "realistic",
            "overrides": {"limits.max_time": 1.0, "start.altitude": [60.0, 200.0]},
        }
        venv = IcarusVecEnv(5, seed=3, task=mixed)
        obs = venv.reset()
        assert obs.shape == (5, 26)
        seen_truncation = seen_termination = False
        for step in range(200):
            obs, rewards, dones, infos = venv.step(actions_for(step, 5).astype(np.float32))
            assert rewards.dtype == np.float32
            assert dones.dtype == np.bool_
            for i, info in enumerate(infos):
                if not dones[i]:
                    assert info == {}
                    continue
                assert info["terminal_observation"].shape == (26,)
                assert set(info["episode"]) == {"r", "l", "t"}
                assert info["is_success"] == (info["outcome"] == "landed")
                if info["TimeLimit.truncated"]:
                    assert info["outcome"] == "time_limit"
                    seen_truncation = True
                else:
                    seen_termination = True
        assert seen_truncation
        assert seen_termination

    def test_seed_and_wrappers(self) -> None:
        from icarus_env import IcarusVecEnv
        from stable_baselines3.common.vec_env import VecNormalize

        venv = IcarusVecEnv(3, task={"config": "easy"})
        venv.seed(11)
        a = venv.reset()
        venv.seed(11)
        np.testing.assert_array_equal(venv.reset(), a)
        assert venv.get_attr("render_mode") == [None] * 3
        assert venv.env_is_wrapped(object) == [False] * 3  # type: ignore[arg-type]
        norm = VecNormalize(venv, norm_obs=True, norm_reward=True)
        norm.reset()
        out = norm.step(np.zeros((3, 3), np.float32))
        assert out[0].shape == (3, 26)
        norm.env_method("set_task", {"config": "realistic"})
        assert np.all(norm.venv.reset()[:, 10] == 0.0)


def test_parallel_results_do_not_depend_on_the_thread_count() -> None:
    import subprocess
    import sys

    script = (
        "import icarus, numpy as np, hashlib\n"
        "v = icarus.VecEnv(48, seed=9, task={'config': 'storm'}, parallel=True)\n"
        "h = hashlib.sha256(v.reset().tobytes())\n"
        "rng = np.random.default_rng(0)\n"
        "for _ in range(150):\n"
        "    o, r, te, tr, info = v.step(rng.uniform(-1, 1, (48, 3)))\n"
        "    for a in (o, r, te, tr, info['terminal_obs'], info['seed']):\n"
        "        h.update(a.tobytes())\n"
        "print(h.hexdigest())\n"
    )
    digests = set()
    for threads in ("1", "2", "4"):
        env = {**__import__("os").environ, "RAYON_NUM_THREADS": threads}
        out = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, env=env, check=True
        )
        digests.add(out.stdout.strip())
    assert len(digests) == 1


def test_ppo_rollout_smoke() -> None:
    """One PPO update through VecNormalize: rollout, truncation bootstrap,
    episode and success buffers."""
    pytest.importorskip("stable_baselines3")
    from icarus_env import IcarusVecEnv
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize

    task = {"config": "training", "overrides": {"limits.max_time": 2.0}}
    venv = VecNormalize(IcarusVecEnv(4, seed=1, task=task), gamma=0.999)
    model = PPO("MlpPolicy", venv, n_steps=64, batch_size=64, n_epochs=1, gamma=0.999, seed=0)
    model.learn(total_timesteps=256)
    assert len(model.ep_info_buffer) > 0
    assert len(model.ep_success_buffer) > 0
    assert all(set(e) >= {"r", "l"} for e in model.ep_info_buffer)
