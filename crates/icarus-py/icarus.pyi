"""Type stubs for the ``icarus`` extension module (Rust core, built with maturin)."""

from typing import Any, Final, final

import numpy as np
import numpy.typing as npt

__version__: Final[str]
PHYSICS_VERSION: Final[int]
REPLAY_FORMAT_VERSION: Final[int]
TELEMETRY_FIELDS: Final[list[str]]
OUTCOMES: Final[list[str]]

def configs() -> list[str]:
    """Names of the built-in configurations: easy, realistic, training, crosswind, storm, engine_out."""

def outcome_from_code(code: int) -> str | None:
    """Name of an outcome code (``None`` for 0 or unknown codes)."""

def run_replay(encoded: str) -> Sim:
    """Re-simulates a replay string. Raises ``ValueError`` if it is invalid."""

def scenarios() -> list[str]:
    """Names of the evaluation scenarios (``"nominal"`` first)."""

def controllers() -> list[str]:
    """Names of the built-in classical controllers."""

def evaluate(controller: str, config: str, scenario: str, seeds: list[int]) -> list[dict[str, Any]]:
    """Runs a controller over seeds in Rust; one result dict per seed.

    Keys: seed, outcome, landed, score, ticks, contact_vx, contact_vy,
    position_error, propellant_used, tilt, act_ns (wall-clock, the only
    non-deterministic field).
    """

@final
class Controller:
    """A classical controller: ``sim.step_quantized(*ctrl.act(sim))``."""

    def __new__(cls, controller: str = "pid", config: str = "easy", scenario: str = "nominal") -> Controller: ...
    def act(self, sim: Sim) -> tuple[int, int, int]:
        """Quantized commands for the sim's current tick (once per tick).

        Raises ``ValueError`` if the sim's config/scenario differs from the
        controller's, or if this tick was already decided.
        """
    def act_recorder(self, recorder: Recorder) -> tuple[int, int, int]:
        """Quantized commands for a recorder's current tick."""
    def reset(self) -> None:
        """Forgets the internal state, for a new episode."""
    @property
    def name(self) -> str: ...

@final
class Sim:
    """One landing episode. Use one object per thread."""

    def __new__(cls, config: str = "easy", seed: int = 0, scenario: str = "nominal") -> Sim: ...
    def step(self, throttle: float, gimbal: float = 0.0, rcs: float = 0.0) -> str | None:
        """Advances one tick with normalized controls in [-1, 1]."""
    def step_quantized(self, throttle: int, gimbal: int, rcs: int) -> str | None:
        """Advances one tick with quantized controls (0..=254, 127 = neutral).

        Raises ``ValueError`` above 254 and ``OverflowError`` outside 0..=255.
        """
    def inject_gimbal_stuck(self, angle: float) -> bool: ...
    def inject_thrust_loss(self, factor: float) -> bool: ...
    def inject_gust(self, amplitude: float, duration: float) -> bool: ...
    def telemetry(self) -> list[float]: ...
    def telemetry_dict(self) -> dict[str, float]: ...
    def observation(self) -> list[float]: ...
    def initial_throttle_level(self) -> int: ...
    def clone(self) -> Sim:
        """An independent copy of the episode in its current state."""
    def __copy__(self) -> Sim: ...
    def __deepcopy__(self, memo: Any) -> Sim: ...
    @property
    def config(self) -> str: ...
    @property
    def seed(self) -> int: ...
    @property
    def tick(self) -> int: ...
    @property
    def time(self) -> float: ...
    @property
    def max_ticks(self) -> int: ...
    @property
    def outcome(self) -> str | None: ...
    @property
    def outcome_code(self) -> int: ...
    @property
    def done(self) -> bool: ...
    @property
    def terminated(self) -> bool: ...
    @property
    def truncated(self) -> bool: ...
    @property
    def score(self) -> int: ...
    @property
    def engine_on(self) -> bool: ...
    @property
    def ignitions_left(self) -> int: ...
    @property
    def propellant(self) -> float: ...

@final
class Recorder:
    """A recorded episode that can be exported as a replay string."""

    def __new__(cls, config: str = "easy", seed: int = 0) -> Recorder: ...
    def step_quantized(self, throttle: int, gimbal: int, rcs: int) -> str | None: ...
    def inject_gimbal_stuck(self, angle: float) -> bool: ...
    def inject_thrust_loss(self, factor: float) -> bool: ...
    def inject_gust(self, amplitude: float, duration: float) -> bool: ...
    def telemetry(self) -> list[float]: ...
    def telemetry_dict(self) -> dict[str, float]: ...
    def observation(self) -> list[float]: ...
    def replay(self) -> str: ...
    @property
    def recorded_ticks(self) -> int: ...
    @property
    def config(self) -> str: ...
    @property
    def seed(self) -> int: ...
    @property
    def done(self) -> bool: ...
    @property
    def outcome(self) -> str | None: ...
    @property
    def score(self) -> int: ...
    @property
    def tick(self) -> int: ...

# --- Reinforcement-learning environment (stage 4) ---------------------------

ACTION_DIM: Final[int]

def episode_seed(base: int, env_index: int, episode: int) -> int:
    """Seed of episode ``episode`` of environment ``env_index`` in a batch seeded with ``base``.

    Always has the top bit set, so it never collides with evaluation seeds.
    """

def default_env_config() -> dict[str, Any]:
    """The default environment configuration, as a dict."""

def obs_fields(history_len: int = 4) -> list[str]:
    """Names of the observation components."""

def override_keys() -> list[tuple[str, int]]:
    """Names accepted in a task's ``overrides`` with their number of values (angles in radians)."""

@final
class Env:
    """One environment: observation, reward and termination computed in Rust.

    ``task``: ``{"config", "scenario", "overrides", "randomization"}`` (all optional).
    ``env_config``: ``{"action_repeat", "history_len", "action_history",
    "quantize_actions", "reward"}`` (all optional).
    """

    def __new__(
        cls, task: dict[str, Any] | None = None, env_config: dict[str, Any] | None = None
    ) -> Env: ...
    def reset(self, seed: int) -> npt.NDArray[np.float32]: ...
    def step(
        self, action: npt.ArrayLike
    ) -> tuple[npt.NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        """One decision: ``(obs, reward, terminated, truncated, info)``.

        Raises ``ValueError`` for a malformed or non-finite action and
        ``RuntimeError`` before ``reset`` or after the episode ended.
        """
    def set_task(self, task: dict[str, Any]) -> None:
        """Replaces the task from the next reset on."""
    def observation(self) -> npt.NDArray[np.float32]: ...
    def telemetry_dict(self) -> dict[str, float]: ...
    def env_config(self) -> dict[str, Any]: ...
    @property
    def obs_dim(self) -> int: ...
    @property
    def action_dim(self) -> int: ...
    @property
    def seed(self) -> int | None: ...
    @property
    def episode_return(self) -> float: ...
    @property
    def episode_length(self) -> int: ...
    @property
    def potential(self) -> float: ...
    @property
    def done(self) -> bool: ...

@final
class VecEnv:
    """A batch of environments stepped in Rust on every core (GIL released)."""

    def __new__(
        cls,
        num_envs: int,
        seed: int = 0,
        task: dict[str, Any] | None = None,
        env_config: dict[str, Any] | None = None,
        autoreset: bool = True,
        parallel: bool | None = None,
    ) -> VecEnv: ...
    def reset(self, seeds: list[int] | None = None) -> npt.NDArray[np.float32]: ...
    def step(
        self, actions: npt.ArrayLike
    ) -> tuple[
        npt.NDArray[np.float32],
        npt.NDArray[np.float64],
        npt.NDArray[np.bool_],
        npt.NDArray[np.bool_],
        dict[str, npt.NDArray[Any]],
    ]:
        """One decision per environment: ``(obs, rewards, terminated, truncated, info)``.

        ``info`` holds arrays: ``terminal_obs``, ``outcome``, ``score``,
        ``episode_return``, ``episode_length``, ``seed`` and ``idle``.
        """
    def set_task(self, task: dict[str, Any], indices: list[int] | None = None) -> None: ...
    def set_base_seed(self, seed: int) -> None: ...
    def episode_results(self) -> list[dict[str, Any]]: ...
    def telemetry_dict(self, index: int) -> dict[str, float]: ...
    def env_config(self) -> dict[str, Any]: ...
    @property
    def num_envs(self) -> int: ...
    @property
    def obs_dim(self) -> int: ...
    @property
    def action_dim(self) -> int: ...
    @property
    def parallel(self) -> bool: ...
    @parallel.setter
    def parallel(self, value: bool) -> None: ...
