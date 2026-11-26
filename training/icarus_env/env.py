"""A thin single-environment Gymnasium wrapper around ``icarus.Env``.

Kept for ``gymnasium.utils.env_checker.check_env``, debugging and small
experiments; training uses the vectorized environment (``vec_env.py``).
Observation, reward and termination are computed in Rust.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import gymnasium as gym
import icarus
import numpy as np

from .tasks import resolve_task

#: Training episode seeds keep the top bit set, like ``icarus.episode_seed``,
#: so a seed drawn by the wrapper is never an evaluation seed.
TRAINING_SEED_BIT = 1 << 63


class IcarusEnv(gym.Env[np.ndarray, np.ndarray]):
    """One landing episode per ``reset``.

    Args:
        task: a task spec (see ``tasks.resolve_task``); default: ``easy``.
        env_config: the core's environment configuration (see
            ``icarus.default_env_config()``).

    ``reset(seed=s)`` seeds the wrapper's generator, which draws the episode
    seed; ``reset(options={"episode_seed": n})`` runs episode ``n`` exactly
    (for evaluation), and ``options={"task": spec}`` switches the task.
    """

    metadata = {"render_modes": []}  # noqa: RUF012 (Gymnasium's convention)

    def __init__(
        self,
        task: Mapping[str, Any] | None = None,
        env_config: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self._env = icarus.Env(resolve_task(task), dict(env_config) if env_config else None)
        dim = self._env.obs_dim
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(dim,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(icarus.ACTION_DIM,), dtype=np.float32)
        self.render_mode = None

    @property
    def core(self) -> icarus.Env:
        """The underlying Rust environment."""
        return self._env

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}
        if "task" in options:
            self._env.set_task(resolve_task(options["task"]))
        if "episode_seed" in options:
            episode_seed = int(options["episode_seed"])
        else:
            episode_seed = int(self.np_random.integers(0, 2**63)) | TRAINING_SEED_BIT
        obs = self._env.reset(episode_seed)
        return obs, {"episode_seed": episode_seed}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self._env.step(np.asarray(action, np.float64))
        info["is_success"] = info["landed"]
        return obs, float(reward), bool(terminated), bool(truncated), info
