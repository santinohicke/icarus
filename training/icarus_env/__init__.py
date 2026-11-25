"""Python side of the Icarus RL environment.

The environment itself (observation, reward, termination, domain
randomization) is implemented in Rust (``icarus_core::env``) and exposed by
the ``icarus`` extension module. This package only adapts it:

- :class:`IcarusEnv`: single-environment Gymnasium wrapper (``check_env``,
  debugging),
- :class:`IcarusVecEnv`: Stable-Baselines3 ``VecEnv`` over the Rust batch,
- :class:`Curriculum`: which task each environment trains on,
- :func:`resolve_task`: YAML-friendly task specs to the core's task dicts.
"""

from typing import TYPE_CHECKING

from .curriculum import Curriculum, Stage
from .env import IcarusEnv
from .tasks import resolve_task, validate_task

if TYPE_CHECKING:
    from .vec_env import IcarusVecEnv

__all__ = ["Curriculum", "IcarusEnv", "IcarusVecEnv", "Stage", "resolve_task", "validate_task"]


def __getattr__(name: str) -> object:
    # Imported lazily: it needs Stable-Baselines3 (and PyTorch), which the
    # Gymnasium wrapper and the curriculum do not.
    if name == "IcarusVecEnv":
        from .vec_env import IcarusVecEnv

        return IcarusVecEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
