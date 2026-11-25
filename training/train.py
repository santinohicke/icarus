"""Train a PPO landing agent on the Rust vectorized environment.

    python training/train.py --config training/configs/ppo_base.yaml --seed 0
    python training/train.py --config training/configs/ablations/no_curriculum.yaml --seed 3

Everything that defines a run (the resolved config, the stage tasks, the
seed) is written to ``<run-dir>/config.json`` before training starts.
Metrics go to TensorBoard (``<run-dir>/tb``). Checkpoints:

- ``<run-dir>/best``: highest validation success rate on the selection task
  (validation seeds never overlap evaluation seeds),
- ``<run-dir>/final``: the last policy.

Each holds ``model.zip`` (SB3), ``vecnormalize.pkl`` (frozen observation
statistics) and, once exported, ``agent.bin`` + ``agent.json``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import icarus
import numpy as np
from icarus_env.config import load_config
from icarus_env.curriculum import Curriculum, Stage
from icarus_env.tasks import resolve_task, validate_task

TRAINING_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = TRAINING_DIR / "configs" / "ppo_base.yaml"


# --- configuration -----------------------------------------------------------


#: Allowed keys per config section: a typo must fail, not silently do nothing.
CONFIG_KEYS: dict[str, set[str] | None] = {
    "": {
        "algo", "seed", "total_timesteps", "num_envs", "device", "torch_threads", "env", "ppo",
        "policy", "normalize", "randomization", "failures", "curriculum", "validation",
    },
    "ppo": {
        "n_steps", "batch_size", "n_epochs", "learning_rate", "learning_rate_final", "gamma",
        "gae_lambda", "clip_range", "ent_coef", "vf_coef", "max_grad_norm", "target_kl",
    },
    "policy": {
        "net_arch", "activation", "log_std_init", "ortho_init", "initial_action",
        "lstm_hidden_size",
        "min_std",
    },
    "normalize": {"norm_obs", "norm_reward", "clip_obs", "clip_reward", "epsilon"},
    "randomization": {"enabled", "from_stage", "ranges"},
    "failures": {"from_stage", "overrides"},
    "curriculum": {"replay_fraction", "replay_stages", "stages"},
    "validation": {"every_steps", "episodes", "seed_start", "tasks", "select_on"},
}  # fmt: skip

#: Validation seeds live between controller tuning (from 10^6) and training
#: (top bit set); evaluation seeds start at 0.
VALIDATION_SEEDS = range(2_000_000, 1 << 62)


def check_config(cfg: Mapping[str, Any]) -> None:
    """Rejects unknown keys and settings the exporter cannot handle."""
    for section, allowed in CONFIG_KEYS.items():
        data = cfg if section == "" else cfg.get(section)
        if data is None or allowed is None:
            continue
        if not isinstance(data, Mapping):
            raise ValueError(f"config section {section!r} must be a mapping")
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"unknown keys in config section {section or 'top'!r}: {unknown}")
    if cfg.get("algo", "ppo") not in ("ppo", "recurrent_ppo"):
        raise ValueError(f"unknown algo {cfg.get('algo')!r}")
    for section in ("ppo", "policy", "normalize", "curriculum", "validation"):
        if not isinstance(cfg.get(section), Mapping):
            raise ValueError(f"config section {section!r} is missing")
    policy = cfg["policy"]
    if cfg.get("algo", "ppo") == "ppo" and policy.get("activation", "tanh") != "tanh":
        raise ValueError("the exporter (and the web inference) support tanh MLPs only")
    initial = policy.get("initial_action")
    if initial is not None and (
        len(initial) != icarus.ACTION_DIM or not all(-1.0 <= float(a) <= 1.0 for a in initial)
    ):
        raise ValueError(f"policy.initial_action must be {icarus.ACTION_DIM} values in [-1, 1]")
    min_std = policy.get("min_std")
    if min_std is not None and not 0.0 < float(min_std) < 10.0:
        raise ValueError(f"policy.min_std must be in (0, 10), got {min_std}")


def env_config_of(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """The core's full environment configuration (every default resolved,
    so a later change of a core default cannot silently change what a run
    trained on), with the reward discount tied to the learner's."""
    env = dict(cfg.get("env") or {})
    reward = dict(env.get("reward") or {})
    gamma = float(cfg["ppo"]["gamma"])
    if "gamma" in reward and float(reward["gamma"]) != gamma:
        raise ValueError("env.reward.gamma must equal ppo.gamma (or be left out)")
    reward["gamma"] = gamma
    # Crash timing stays neutral at any gamma: c = (1 - gamma) * C per decision, unless
    # the config sets the time cost itself.
    if "time_weight" not in reward:
        defaults = icarus.default_env_config()
        crash = float(reward.get("crash_penalty", defaults["reward"]["crash_penalty"]))
        decisions_per_second = 60.0 / float(env.get("action_repeat", defaults["action_repeat"]))
        reward["time_weight"] = (1.0 - gamma) * crash * decisions_per_second
    env["reward"] = reward
    return icarus.Env(None, env).env_config()


def _stage_index(names: list[str], name: str | None, what: str) -> int:
    if name is None:
        return 0
    if name not in names:
        raise ValueError(f"{what}.from_stage {name!r} is not a stage ({names})")
    return names.index(name)


def stage_tasks(cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The task of every curriculum stage, with domain randomization and
    failures merged in from their starting stage on (a stage's own settings
    win)."""
    stages = cfg["curriculum"]["stages"]
    names = [s["name"] for s in stages]
    rand = cfg.get("randomization") or {}
    fails = cfg.get("failures") or {}
    rand_from = _stage_index(names, rand.get("from_stage"), "randomization")
    fail_from = _stage_index(names, fails.get("from_stage"), "failures")
    fail_overrides = resolve_task({"overrides": fails.get("overrides") or {}}).get("overrides", {})
    tasks = []
    for i, stage in enumerate(stages):
        task = resolve_task(stage["task"])
        if not rand.get("enabled", True):
            task.pop("randomization", None)  # the ablation: none anywhere
        elif rand.get("ranges") and i >= rand_from and "randomization" not in task:
            task["randomization"] = dict(rand["ranges"])
        if fail_overrides and i >= fail_from:
            task["overrides"] = {**fail_overrides, **task.get("overrides", {})}
        tasks.append(task)
    return tasks


def build_curriculum(cfg: Mapping[str, Any], seed: int) -> Curriculum:
    spec = cfg["curriculum"]
    tasks = stage_tasks(cfg)
    # resolve_task is idempotent, so the merged tasks go through the same
    # validation as hand-written stage specs.
    stages = [Stage.from_spec({**s, "task": t}) for s, t in zip(spec["stages"], tasks, strict=True)]
    return Curriculum(
        stages=stages,
        replay_fraction=float(spec.get("replay_fraction", 0.0)),
        replay_stages=spec.get("replay_stages"),
        seed=seed,
    )


def linear_schedule(start: float, end: float) -> Callable[[float], float]:
    """SB3 schedule: ``progress_remaining`` goes from 1 to 0."""

    def schedule(progress_remaining: float) -> float:
        return end + (start - end) * progress_remaining

    return schedule


# --- callbacks ---------------------------------------------------------------


def _callbacks_module() -> Any:
    from stable_baselines3.common.callbacks import BaseCallback

    class CurriculumCallback(BaseCallback):
        """Records every finished episode against the stage it ran on and
        promotes the curriculum; logs stage, success rate and outcomes."""

        def __init__(
            self,
            curriculum: Curriculum,
            venv: Any,
            log_path: Path,
            vecnorm: Any = None,
            stage_dir: Path | None = None,
        ) -> None:
            super().__init__()
            self.curriculum = curriculum
            # A checkpoint of the policy that passed each stage (to continue
            # from a later stage with --init-from/--start-stage).
            self.vecnorm = vecnorm
            self.stage_dir = stage_dir
            self.venv = venv  # the IcarusVecEnv (unwrapped)
            self.log_path = log_path
            self.n = venv.num_envs
            # Stage of each env's current episode, and from its next one on.
            self.running = [curriculum.current] * self.n
            self.pending = [curriculum.current] * self.n
            self.outcomes: dict[str, int] = {}

        def _assign(self) -> None:
            stages = self.curriculum.assign(self.n)
            for s in sorted(set(stages)):
                idx = [i for i, x in enumerate(stages) if x == s]
                self.venv.set_task(self.curriculum.stages[s].task, idx)
            self.pending = stages

        def _on_training_start(self) -> None:
            self._assign()
            # The first episodes were reset with the starting stage's task.
            self.running = [self.curriculum.current] * self.n

        def _on_step(self) -> bool:
            dones = self.locals["dones"]
            infos = self.locals["infos"]
            for i in np.flatnonzero(dones):
                info = infos[i]
                self.curriculum.record(self.running[i], bool(info.get("is_success")))
                name = info.get("outcome") or "unknown"
                self.outcomes[name] = self.outcomes.get(name, 0) + 1
                self.running[i] = self.pending[i]
            if self.curriculum.ready():
                if self.stage_dir is not None and self.vecnorm is not None:
                    passed = self.stage_dir / self.curriculum.stage.name
                    save_checkpoint(self.model, self.vecnorm, passed)
                self.curriculum.promote(self.num_timesteps)
                print(
                    f"[curriculum] step {self.num_timesteps:,}: "
                    f"promoted to {self.curriculum.stage.name}",
                    flush=True,
                )
                self._assign()
                self._save()
            return True

        def _on_rollout_end(self) -> None:
            c = self.curriculum
            self.logger.record("curriculum/stage", c.current)
            self.logger.record("curriculum/success_rate", c.success_rate)
            total = sum(self.outcomes.values())
            if total:
                for name, count in self.outcomes.items():
                    self.logger.record(f"outcomes/{name}", count / total)
            self.outcomes = {}

        def _save(self) -> None:
            self.log_path.write_text(
                json.dumps({"history": self.curriculum.history}, indent=1) + "\n",
                encoding="utf-8",
            )

        def _on_training_end(self) -> None:
            self._save()

    class ValidationCallback(BaseCallback):
        """Deterministic policy on fixed validation seeds of each validation
        task; keeps the best checkpoint on the selection task."""

        def __init__(
            self,
            vecnorm: Any,
            env_config: dict[str, Any],
            tasks: dict[str, dict[str, Any]],
            seeds: list[int],
            every: int,
            select_on: str,
            best_dir: Path,
            history_path: Path,
        ) -> None:
            super().__init__()
            self.vecnorm = vecnorm
            self.env_config = env_config
            self.tasks = tasks
            self.seeds = seeds
            self.every = every
            self.select_on = select_on
            self.best_dir = best_dir
            self.history_path = history_path
            self.next_at = every
            self.best: tuple[float, float] = (-1.0, -math.inf)
            self.history: list[dict[str, Any]] = []

        def evaluate(self, task: dict[str, Any]) -> dict[str, float]:
            vec = icarus.VecEnv(
                len(self.seeds), task=task, env_config=self.env_config, autoreset=False
            )
            obs = vec.reset(seeds=self.seeds)
            done = np.zeros(len(self.seeds), bool)
            state = None
            starts = np.ones(len(self.seeds), bool)
            while not done.all():
                action, state = self.model.predict(
                    self.vecnorm.normalize_obs(obs),
                    state=state,
                    episode_start=starts,
                    deterministic=True,
                )
                starts = np.zeros(len(self.seeds), bool)
                obs, _, te, tr, _ = vec.step(np.clip(action, -1.0, 1.0))
                done |= te | tr
            results = vec.episode_results()
            landed = [r for r in results if r["landed"]]
            return {
                "success": len(landed) / len(results),
                # None, not NaN, when nothing landed: validation.json is JSON.
                "propellant": float(np.mean([r["propellant_used"] for r in landed]))
                if landed
                else None,
                "return": float(np.mean([r["episode_return"] for r in results])),
            }

        def _on_step(self) -> bool:
            if self.num_timesteps < self.next_at:
                return True
            self.next_at += self.every
            self.run_validation()
            return True

        def run_validation(self) -> None:
            row: dict[str, Any] = {"step": int(self.num_timesteps)}
            for name, task in self.tasks.items():
                m = self.evaluate(task)
                for k, v in m.items():
                    if v is not None:
                        self.logger.record(f"validation/{name}_{k}", v)
                row[name] = m
            # Success rate first; ties (common at 100%) go to the higher return.
            selected = row[self.select_on]
            score = (selected["success"], selected["return"])
            print(
                f"[validation] step {self.num_timesteps:,}: "
                + ", ".join(f"{k} {row[k]['success']:.1%}" for k in self.tasks),
                flush=True,
            )
            if score >= self.best:
                self.best = score
                save_checkpoint(self.model, self.vecnorm, self.best_dir)
                row["best"] = True
            self.history.append(row)
            self.history_path.write_text(json.dumps(self.history, indent=1) + "\n", "utf-8")

        def _on_training_end(self) -> None:
            if not self.history or self.history[-1]["step"] != self.num_timesteps:
                self.run_validation()

    class StdFloorCallback(BaseCallback):
        """Keeps the policy's exploration std at or above ``min_std``.

        The std is state-independent in SB3's Gaussian policy and shrank to
        0.015 in a run, after which a new stage (the realistic vehicle needs
        the throttle near the cut-off instead of near hover) could never be
        discovered. The deterministic action, which is what is exported and
        evaluated, is the mean and is unaffected.
        """

        def __init__(self, min_std: float) -> None:
            super().__init__()
            self.min_log_std = math.log(min_std)

        def _clamp(self) -> None:
            import torch

            log_std = getattr(self.model.policy, "log_std", None)
            if log_std is not None:
                with torch.no_grad():
                    log_std.clamp_(min=self.min_log_std)

        def _on_training_start(self) -> None:
            self._clamp()

        def _on_rollout_start(self) -> None:
            # Called after every policy update, before the next collection.
            self._clamp()

        def _on_training_end(self) -> None:
            # The checkpoint saved after the last update respects it too.
            self._clamp()

        def _on_step(self) -> bool:
            return True

    return CurriculumCallback, ValidationCallback, StdFloorCallback


def save_checkpoint(model: Any, vecnorm: Any, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    model.save(directory / "model.zip")
    vecnorm.save(str(directory / "vecnormalize.pkl"))


# --- main --------------------------------------------------------------------


def set_initial_action(model: Any, initial: list[float] | None) -> None:
    """Sets the bias of the policy's output layer, i.e. the mean action of
    the untrained policy (its weights start near zero). Only an
    initialization: everything is learned from there."""
    if initial is None:
        return
    import torch

    bias = model.policy.action_net.bias
    if len(initial) != bias.shape[0] or not all(-1.0 <= float(a) <= 1.0 for a in initial):
        raise ValueError(f"policy.initial_action must be {bias.shape[0]} values in [-1, 1]")
    with torch.no_grad():
        bias.copy_(torch.as_tensor([float(a) for a in initial], dtype=bias.dtype))


def _has_tensorboard() -> bool:
    try:
        import torch.utils.tensorboard  # noqa: F401
    except ImportError:
        print("tensorboard is not installed: metrics go to stdout only", flush=True)
        return False
    return True


def make_model(cfg: Mapping[str, Any], venv: Any, seed: int, run_dir: Path) -> Any:
    import torch.nn as nn

    p = cfg["ppo"]
    pol = cfg["policy"]
    activations = {"tanh": nn.Tanh, "relu": nn.ReLU}
    arch = list(pol["net_arch"])
    policy_kwargs: dict[str, Any] = {
        "net_arch": {"pi": arch, "vf": arch},
        "activation_fn": activations[pol.get("activation", "tanh")],
        "log_std_init": float(pol.get("log_std_init", 0.0)),
        "ortho_init": bool(pol.get("ortho_init", True)),
    }
    lr = linear_schedule(
        float(p["learning_rate"]), float(p.get("learning_rate_final", p["learning_rate"]))
    )
    common = {
        "learning_rate": lr,
        "n_steps": int(p["n_steps"]),
        "batch_size": int(p["batch_size"]),
        "n_epochs": int(p["n_epochs"]),
        "gamma": float(p["gamma"]),
        "gae_lambda": float(p["gae_lambda"]),
        "clip_range": float(p["clip_range"]),
        "ent_coef": float(p["ent_coef"]),
        "vf_coef": float(p["vf_coef"]),
        "max_grad_norm": float(p["max_grad_norm"]),
        "target_kl": None if p.get("target_kl") is None else float(p["target_kl"]),
        "seed": seed,
        "device": cfg.get("device", "cpu"),
        "tensorboard_log": str(run_dir / "tb") if _has_tensorboard() else None,
        "verbose": 1,
    }
    algo = cfg.get("algo", "ppo")
    if algo == "ppo":
        from stable_baselines3 import PPO

        return PPO("MlpPolicy", venv, policy_kwargs=policy_kwargs, **common)
    if algo == "recurrent_ppo":
        from sb3_contrib import RecurrentPPO

        policy_kwargs["lstm_hidden_size"] = int(pol.get("lstm_hidden_size", 64))
        return RecurrentPPO("MlpLstmPolicy", venv, policy_kwargs=policy_kwargs, **common)
    raise ValueError(f"unknown algo {algo!r}")


def _versions() -> dict[str, str]:
    import platform

    out = {"python": platform.python_version(), "numpy": np.__version__}
    for name in ("torch", "stable_baselines3", "gymnasium"):
        try:
            out[name] = __import__(name).__version__
        except ImportError:
            out[name] = "missing"
    out["rayon_threads"] = os.environ.get("RAYON_NUM_THREADS", "default")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seed", type=int, help="overrides the config's seed")
    parser.add_argument("--total-timesteps", type=int, help="overrides the config")
    parser.add_argument("--num-envs", type=int, help="overrides the config")
    parser.add_argument("--run-dir", type=Path, help="default: training/runs/<config>-s<seed>")
    parser.add_argument("--no-export", action="store_true", help="skip exporting the agent")
    parser.add_argument("--force", action="store_true", help="reuse an existing run directory")
    parser.add_argument(
        "--init-from",
        type=Path,
        help="checkpoint directory (model.zip + vecnormalize.pkl) to continue training from",
    )
    parser.add_argument("--start-stage", help="curriculum stage to start on (with --init-from)")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    check_config(cfg)
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.total_timesteps is not None:
        cfg["total_timesteps"] = args.total_timesteps
    if args.num_envs is not None:
        cfg["num_envs"] = args.num_envs
    seed = int(cfg["seed"])
    config_name = args.config.stem
    run_dir = args.run_dir or TRAINING_DIR / "runs" / f"{config_name}-s{seed}"
    if (run_dir / "config.json").exists() and not args.force:
        raise SystemExit(f"{run_dir} already holds a run; pick another --run-dir or pass --force")
    run_dir.mkdir(parents=True, exist_ok=True)

    env_config = env_config_of(cfg)
    curriculum = build_curriculum(cfg, seed)
    if args.start_stage:
        if not args.init_from:
            raise SystemExit("--start-stage needs --init-from (a policy past the earlier stages)")
        curriculum.start_at(args.start_stage)
    for stage in curriculum.stages:
        validate_task(stage.task, env_config)
    val = cfg["validation"]
    val_tasks = {name: resolve_task(t) for name, t in val["tasks"].items()}
    if val["select_on"] not in val_tasks:
        raise ValueError(f"validation.select_on {val['select_on']!r} is not a validation task")
    for t in val_tasks.values():
        validate_task(t, env_config)
    seeds = list(range(int(val["seed_start"]), int(val["seed_start"]) + int(val["episodes"])))
    if seeds[0] not in VALIDATION_SEEDS or seeds[-1] not in VALIDATION_SEEDS:
        raise ValueError(
            "validation seeds must lie in [2,000,000, 2^62): apart from evaluation (from 0), "
            "controller tuning (from 1,000,000) and training (top bit set)"
        )

    record = {
        "config_name": config_name,
        "config_path": str(args.config),
        "seed": seed,
        "config": cfg,
        "env_config": env_config,
        "stage_tasks": {s.name: s.task for s in curriculum.stages},
        "physics_version": icarus.PHYSICS_VERSION,
        "icarus_version": icarus.__version__,
        "versions": _versions(),
        "init_from": str(args.init_from) if args.init_from else None,
        "start_stage": args.start_stage,
    }
    (run_dir / "config.json").write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")

    import torch
    from icarus_env.vec_env import IcarusVecEnv
    from stable_baselines3.common.callbacks import CallbackList
    from stable_baselines3.common.vec_env import VecNormalize

    torch.set_num_threads(int(cfg.get("torch_threads", os.cpu_count() or 1)))
    venv = IcarusVecEnv(
        int(cfg["num_envs"]), seed=seed, task=curriculum.stage.task, env_config=env_config
    )
    norm = cfg["normalize"]
    vecnorm = VecNormalize(
        venv,
        norm_obs=bool(norm["norm_obs"]),
        norm_reward=bool(norm["norm_reward"]),
        clip_obs=float(norm["clip_obs"]),
        clip_reward=float(norm["clip_reward"]),
        gamma=float(cfg["ppo"]["gamma"]),
        epsilon=float(norm["epsilon"]),
    )
    model = make_model(cfg, vecnorm, seed, run_dir)
    if args.init_from:
        # Continue a policy: same architecture, its weights and its
        # observation/return statistics (they keep updating from here).
        model.set_parameters(str(args.init_from / "model.zip"), exact_match=True, device="cpu")
        previous = VecNormalize.load(str(args.init_from / "vecnormalize.pkl"), venv)
        vecnorm.obs_rms = previous.obs_rms
        vecnorm.ret_rms = previous.ret_rms
    else:
        set_initial_action(model, cfg["policy"].get("initial_action"))
    curriculum_cb, validation_cb, std_floor_cb = _callbacks_module()
    callbacks = CallbackList(
        [
            curriculum_cb(
                curriculum, venv, run_dir / "curriculum.json", vecnorm, run_dir / "stages"
            ),
            std_floor_cb(float(cfg["policy"].get("min_std", 1e-6))),
            validation_cb(
                vecnorm,
                env_config,
                val_tasks,
                seeds,
                int(val["every_steps"]),
                val["select_on"],
                run_dir / "best",
                run_dir / "validation.json",
            ),
        ]
    )
    started = time.time()
    try:
        model.learn(
            total_timesteps=int(cfg["total_timesteps"]),
            callback=callbacks,
            tb_log_name="ppo",
            progress_bar=False,
        )
    finally:
        # Interrupted or not, keep the policy reached so far.
        save_checkpoint(model, vecnorm, run_dir / "final")
        elapsed = time.time() - started
        print(f"trained {model.num_timesteps:,} steps in {elapsed / 60:.1f} min", flush=True)

    failed = []
    if not args.no_export and cfg.get("algo", "ppo") == "ppo":
        from export_policy import export

        for name in ("best", "final"):
            ckpt = run_dir / name
            if not (ckpt / "model.zip").exists():
                continue
            try:
                export(ckpt / "model.zip", ckpt / "vecnormalize.pkl", ckpt)
            except (ValueError, RuntimeError) as e:
                print(f"export of {name} failed: {e}", flush=True)
                failed.append(name)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
