"""Stable-Baselines3 ``VecEnv`` adapter over the Rust vectorized environment.

One call steps every environment in Rust, in parallel, without per-env
Python overhead. Python only builds the ``infos`` of the environments whose
episode just ended:

- ``terminal_observation``: the last observation (SB3 bootstraps from it),
- ``TimeLimit.truncated``: the episode hit the time limit,
- ``episode``: ``{"r", "l", "t"}`` like SB3's ``Monitor`` (undiscounted
  shaped return, decisions, wall-clock seconds since creation),
- ``is_success`` (landed), ``outcome``, ``score`` and ``seed``.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import gymnasium as gym
import icarus
import numpy as np
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnvIndices,
    VecEnvObs,
    VecEnvStepReturn,
)

from .tasks import resolve_task


class IcarusVecEnv(VecEnv):
    """``num_envs`` environments stepped together by ``icarus.VecEnv``.

    Args:
        num_envs: batch size.
        seed: base seed of the episode-seed sequence (``icarus.episode_seed``).
        task: task spec for every environment (see ``tasks.resolve_task``).
        env_config: the core's environment configuration.
        parallel: step on every core (default: decided by the core).
    """

    def __init__(
        self,
        num_envs: int,
        seed: int = 0,
        task: Mapping[str, Any] | None = None,
        env_config: Mapping[str, Any] | None = None,
        parallel: bool | None = None,
    ) -> None:
        self.core = icarus.VecEnv(
            num_envs,
            seed=seed,
            task=resolve_task(task),
            env_config=dict(env_config) if env_config else None,
            autoreset=True,
            parallel=parallel,
        )
        dim = self.core.obs_dim
        observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(dim,), dtype=np.float32)
        action_space = gym.spaces.Box(-1.0, 1.0, shape=(icarus.ACTION_DIM,), dtype=np.float32)
        self._actions: np.ndarray | None = None
        self._start = time.monotonic()
        super().__init__(num_envs, observation_space, action_space)

    # --- core API ---------------------------------------------------------

    def reset(self) -> VecEnvObs:
        # SB3 stores seeds from ``seed()`` in ``_seeds``; the first one is
        # the base of the per-environment seed sequences.
        if self._seeds[0] is not None:
            self.core.set_base_seed(int(self._seeds[0]))
        obs = self.core.reset()
        self._reset_seeds()
        self._reset_options()
        self.reset_infos = [{} for _ in range(self.num_envs)]
        return obs

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = np.asarray(actions, dtype=np.float64).reshape(self.num_envs, -1)

    def step_wait(self) -> VecEnvStepReturn:
        if self._actions is None:
            raise RuntimeError("step_wait() called without step_async()")
        actions, self._actions = self._actions, None
        # SB3 already clips to the action space; the core clamps anyway.
        obs, rewards, terminated, truncated, info = self.core.step(actions)
        dones = terminated | truncated
        infos: list[dict[str, Any]] = [{} for _ in range(self.num_envs)]
        if dones.any():
            elapsed = round(time.monotonic() - self._start, 6)
            for i in np.flatnonzero(dones):
                code = int(info["outcome"][i])
                infos[i] = {
                    "terminal_observation": info["terminal_obs"][i],
                    "TimeLimit.truncated": bool(truncated[i] and not terminated[i]),
                    "episode": {
                        "r": float(info["episode_return"][i]),
                        "l": int(info["episode_length"][i]),
                        "t": elapsed,
                    },
                    "is_success": code == 1,
                    "outcome": icarus.outcome_from_code(code),
                    "score": int(info["score"][i]),
                    "seed": int(info["seed"][i]),
                }
        return obs, rewards.astype(np.float32), dones, infos

    def seed(self, seed: int | None = None) -> Sequence[int | None]:
        """Sets the base seed used from the next ``reset``."""
        if seed is None:
            seed = int(np.random.SeedSequence().generate_state(1, np.uint64)[0])
        self._seeds = [seed + i for i in range(self.num_envs)]
        return self._seeds

    def close(self) -> None:
        """Nothing to release (the batch lives in Rust memory)."""

    # --- curriculum hooks -------------------------------------------------

    def set_task(self, task: Mapping[str, Any], indices: Iterable[int] | None = None) -> None:
        """Gives the environments ``indices`` (default: all) a new task from
        their next episode on."""
        idx = None if indices is None else [int(i) for i in indices]
        self.core.set_task(resolve_task(task), idx)

    # --- attribute plumbing (SB3 API) ---------------------------------------

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> list[Any]:
        n = len(self._index_list(indices))
        if attr_name == "render_mode":
            return [None] * n
        if hasattr(self.core, attr_name):
            return [getattr(self.core, attr_name)] * n
        raise AttributeError(f"IcarusVecEnv has no attribute {attr_name!r}")

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        raise AttributeError(f"IcarusVecEnv attributes are read-only ({attr_name!r})")

    def env_method(
        self,
        method_name: str,
        *method_args: Any,
        indices: VecEnvIndices = None,
        **method_kwargs: Any,
    ) -> list[Any]:
        if method_name == "set_task":
            idx = self._index_list(indices)
            self.set_task(*method_args, indices=idx, **method_kwargs)
            return [None] * len(idx)
        raise AttributeError(f"IcarusVecEnv has no per-environment method {method_name!r}")

    def env_is_wrapped(
        self, wrapper_class: type[gym.Wrapper], indices: VecEnvIndices = None
    ) -> list[bool]:
        return [False] * len(self._index_list(indices))

    def _index_list(self, indices: VecEnvIndices) -> list[int]:
        return [int(i) for i in self._get_indices(indices)]
