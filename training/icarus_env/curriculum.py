"""Curriculum: which task each environment trains on.

The curriculum is a list of stages, from easy to hard. Training starts on
the first stage and moves to the next once the success rate over the last
``window`` episodes of the current stage reaches ``promote_at``. A share of
the environments (``replay_fraction``) keeps training on earlier stages so the
agent does not forget them.

It only ever hands task dicts to the environments (applied on their next
reset): no per-step logic lives here.
"""

from __future__ import annotations

import collections
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .tasks import resolve_task


@dataclass(frozen=True)
class Stage:
    """One curriculum stage."""

    name: str
    task: dict[str, Any]
    #: Success rate (landings) needed to move on; ``None`` for the last stage.
    promote_at: float | None = None
    #: Episodes the success rate is measured over.
    window: int = 500

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any]) -> Stage:
        unknown = set(spec) - {"name", "task", "promote_at", "window"}
        if unknown:
            raise ValueError(f"unknown stage keys {sorted(unknown)}")
        missing = {"name", "task"} - set(spec)
        if missing:
            raise ValueError(f"a stage needs {sorted(missing)}")
        promote_at = spec.get("promote_at")
        if promote_at is not None and not 0.0 < float(promote_at) <= 1.0:
            raise ValueError(f"promote_at must be in (0, 1], got {promote_at}")
        window = spec.get("window", 500)
        if isinstance(window, bool) or not isinstance(window, int) or window < 1:
            raise ValueError(f"window must be an integer >= 1, got {window!r}")
        return cls(
            name=str(spec["name"]),
            task=resolve_task(spec["task"]),
            promote_at=None if promote_at is None else float(promote_at),
            window=window,
        )


@dataclass
class Curriculum:
    """Tracks progress and decides the task of every environment.

    Args:
        stages: from first to last.
        replay_fraction: share of environments kept on earlier stages.
        replay_stages: names of the earlier stages to replay (default: all of
            them). Spreading the replay evenly over trivial early stages left
            about 1.7% of the environments on `easy`, which was forgotten.
        seed: seeds the (deterministic) choice of replayed stages.
    """

    stages: Sequence[Stage]
    replay_fraction: float = 0.0
    replay_stages: Sequence[str] | None = None
    seed: int = 0
    current: int = 0
    _results: collections.deque[bool] = field(init=False)
    #: Steps (environment decisions) at which each stage was reached.
    history: list[tuple[int, str]] = field(init=False)

    def __post_init__(self) -> None:
        if not self.stages:
            raise ValueError("a curriculum needs at least one stage")
        if not 0.0 <= self.replay_fraction < 1.0:
            raise ValueError(f"replay_fraction must be in [0, 1), got {self.replay_fraction}")
        names = [s.name for s in self.stages]
        if len(set(names)) != len(names):
            raise ValueError(f"stage names must be unique: {names}")
        # A stage without a threshold would stall the curriculum forever; a
        # threshold on the last stage would be silently ignored.
        for s in self.stages[:-1]:
            if s.promote_at is None:
                raise ValueError(f"stage {s.name!r} is not the last one and needs promote_at")
        if self.stages[-1].promote_at is not None:
            raise ValueError(f"the last stage {self.stages[-1].name!r} cannot have promote_at")
        if self.replay_stages is not None:
            unknown_replay = set(self.replay_stages) - set(names)
            if unknown_replay:
                raise ValueError(f"replay_stages {sorted(unknown_replay)} are not stages")
        if not 0 <= self.current < len(self.stages):
            raise ValueError(f"current stage {self.current} out of range")
        self._results = collections.deque(maxlen=self.stage.window)
        self.history = [(0, self.stage.name)]

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any], seed: int = 0) -> Curriculum:
        unknown = set(spec) - {"stages", "replay_fraction", "replay_stages"}
        if unknown:
            raise ValueError(f"unknown curriculum keys {sorted(unknown)}")
        return cls(
            stages=[Stage.from_spec(s) for s in spec["stages"]],
            replay_fraction=float(spec.get("replay_fraction", 0.0)),
            replay_stages=spec.get("replay_stages"),
            seed=seed,
        )

    @property
    def stage(self) -> Stage:
        return self.stages[self.current]

    @property
    def is_last(self) -> bool:
        return self.current == len(self.stages) - 1

    @property
    def success_rate(self) -> float:
        """Success rate over the episodes recorded on the current stage."""
        return sum(self._results) / len(self._results) if self._results else 0.0

    @property
    def episodes(self) -> int:
        """Episodes recorded on the current stage (up to its window)."""
        return len(self._results)

    def record(self, stage_index: int, landed: bool) -> None:
        """Records an episode that ran on stage ``stage_index`` (episodes of
        replayed stages do not count towards promotion)."""
        if stage_index == self.current:
            self._results.append(bool(landed))

    def ready(self) -> bool:
        """The current stage's promotion criterion is met."""
        threshold = self.stage.promote_at
        return (
            not self.is_last
            and threshold is not None
            and len(self._results) >= self.stage.window
            and self.success_rate >= threshold
        )

    def start_at(self, name: str) -> None:
        """Starts on stage ``name`` (continuing from a checkpoint that already
        passed the stages before it)."""
        names = [s.name for s in self.stages]
        if name not in names:
            raise ValueError(f"no stage {name!r} in {names}")
        self.current = names.index(name)
        self._results = collections.deque(maxlen=self.stage.window)
        self.history = [(0, self.stage.name)]

    def promote(self, step: int = 0) -> None:
        """Moves to the next stage."""
        if self.is_last:
            raise RuntimeError("already on the last stage")
        self.current += 1
        self._results = collections.deque(maxlen=self.stage.window)
        self.history.append((step, self.stage.name))

    def assign(self, num_envs: int) -> list[int]:
        """The stage of every environment: the current one, except a
        ``replay_fraction`` share (rounded up) spread over the earlier stages.

        Depends only on ``(seed, current stage, num_envs)``, so a resumed run
        assigns exactly as the original did."""
        stages = [self.current] * num_envs
        names = [s.name for s in self.stages]
        earlier = [
            i
            for i in range(self.current)
            if self.replay_stages is None or names[i] in self.replay_stages
        ]
        if earlier and self.replay_fraction > 0.0:
            replayed = min(num_envs - 1, math.ceil(self.replay_fraction * num_envs))
            if replayed > 0:
                rng = np.random.default_rng([self.seed, self.current])
                envs = rng.choice(num_envs, size=replayed, replace=False)
                picks = rng.integers(0, len(earlier), size=replayed)
                for e, k in zip(envs, picks, strict=True):
                    stages[int(e)] = earlier[int(k)]
        return stages
