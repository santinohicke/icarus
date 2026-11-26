"""Export a trained PPO agent for Rust/WASM inference.

Writes ``agent.bin`` (every actor weight and bias, little-endian float32)
and ``agent.json`` (layer shapes and activations, the frozen observation
normalization with its epsilon and clip, the environment configuration, the
weights' SHA-256 and training metadata). Only the actor is exported: the
deterministic action is the Gaussian mean, clipped to ``[-1, 1]``.

The export is verified before it is written: the NumPy reference inference
(``icarus_env.policy.MlpPolicy``) must reproduce the PyTorch policy's
deterministic actions on real observations within ``--tolerance``.

    python training/export_policy.py training/runs/ppo_base-s0/best
    python training/export_policy.py RUN/best --out web/public/models
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import icarus
import numpy as np
from icarus_env.policy import FORMAT, FORMAT_VERSION, MlpPolicy, content_digest

DEFAULT_TOLERANCE = 1e-5


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parent,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def _git_commit() -> str | None:
    """HEAD, with ``-dirty`` when the working tree has uncommitted changes."""
    head = _git("rev-parse", "HEAD")
    if not head:
        return None
    status = _git("status", "--porcelain", "--untracked-files=no")
    return head + ("-dirty" if status else "")


def actor_layers(model: Any) -> list[tuple[np.ndarray, np.ndarray, str]]:
    """The actor of an SB3 ``ActorCriticPolicy`` as ``(W, b, activation)``."""
    import torch.nn as nn
    from stable_baselines3.common.torch_layers import FlattenExtractor

    policy = model.policy
    if not isinstance(policy.pi_features_extractor, FlattenExtractor):
        raise ValueError("only the default FlattenExtractor is supported")
    if getattr(policy, "squash_output", False) or getattr(policy, "use_sde", False):
        raise ValueError("squashed or state-dependent-exploration policies are not supported")
    modules = [*policy.mlp_extractor.policy_net, policy.action_net]
    layers: list[tuple[np.ndarray, np.ndarray, str]] = []
    for i, m in enumerate(modules):
        if isinstance(m, nn.Linear):
            w = m.weight.detach().cpu().numpy().astype(np.float32)
            b = m.bias.detach().cpu().numpy().astype(np.float32)
            layers.append((w, b, "identity"))
        elif isinstance(m, nn.Tanh):
            if not layers or layers[-1][2] != "identity":
                raise ValueError(f"unexpected activation at position {i}")
            w, b, _ = layers[-1]
            layers[-1] = (w, b, "tanh")
        else:
            raise ValueError(f"unsupported module {type(m).__name__} (tanh MLPs only)")
    if layers[-1][2] != "identity":
        raise ValueError("the output layer must be linear")
    return layers


def build_manifest(
    layers: list[tuple[np.ndarray, np.ndarray, str]],
    obs_rms_mean: np.ndarray,
    obs_rms_var: np.ndarray,
    epsilon: float,
    clip_obs: float,
    env_config: dict[str, Any],
    blob: bytes,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    obs_dim = int(layers[0][0].shape[1])
    return {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "physics_version": icarus.PHYSICS_VERSION,
        "obs_dim": obs_dim,
        "action_dim": int(layers[-1][0].shape[0]),
        "obs_fields": icarus.obs_fields(int(env_config["history_len"])),
        "layers": [
            {"in": int(w.shape[1]), "out": int(w.shape[0]), "activation": act}
            for w, _, act in layers
        ],
        "normalization": {
            "mean": [float(x) for x in obs_rms_mean],
            "var": [float(x) for x in obs_rms_var],
            "epsilon": float(epsilon),
            "clip_obs": float(clip_obs),
        },
        "action": {"low": -1.0, "high": 1.0, "deterministic": True},
        "env_config": env_config,
        "weights": {
            "file": "agent.bin",
            "dtype": "float32",
            "byte_order": "little",
            "layout": "per layer: weight (out x in, row-major), then bias",
            "count": len(blob) // 4,
            "sha256": hashlib.sha256(blob).hexdigest(),
        },
        "training": metadata,
    }


def finalize(manifest: dict[str, Any]) -> dict[str, Any]:
    """Adds the digest over every field inference depends on."""
    return {**manifest, "content_sha256": content_digest(manifest)}


def pack(layers: list[tuple[np.ndarray, np.ndarray, str]]) -> bytes:
    parts = []
    for w, b, _ in layers:
        parts.append(np.ascontiguousarray(w, dtype="<f4").tobytes())
        parts.append(np.ascontiguousarray(b, dtype="<f4").tobytes())
    return b"".join(parts)


def sample_observations(policy: MlpPolicy, n_random: int = 512) -> np.ndarray:
    """Observations covering what the policy will see: whole episodes flown
    by the policy itself on both vehicles (down to touchdown), the first
    seconds under random actions, and scaled noise that reaches the
    normalization clip."""
    env_config = policy.env_config
    rng = np.random.default_rng(0)
    out = []
    for config in ("training", "easy", "realistic", "storm"):
        vec = icarus.VecEnv(8, seed=7, task={"config": config}, env_config=env_config)
        obs = vec.reset()
        for k in range(600):
            out.append(obs)
            noise = rng.uniform(-1, 1, size=(8, 3)) if k < 20 else 0.0
            obs, *_ = vec.step(np.clip(policy.act(obs) + noise, -1.0, 1.0))
    real = np.concatenate(out)
    scale = np.sqrt(policy.var + policy.epsilon)
    extreme = policy.mean + rng.normal(0.0, 12.0, size=(n_random, real.shape[1])) * scale
    return np.concatenate([real, extreme.astype(np.float32)])


def export(
    model_path: Path,
    vecnormalize_path: Path,
    out_dir: Path,
    metadata: dict[str, Any] | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
) -> Path:
    """Exports and verifies; returns the manifest path."""
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize

    model = PPO.load(model_path, device="cpu")
    run_dir = model_path.parent
    config_path = run_dir / "config.json"
    if not config_path.exists():
        config_path = run_dir.parent / "config.json"
    run_config = json.loads(config_path.read_text(encoding="utf-8"))
    # Every default resolved (older runs recorded only the configured keys).
    env_config = icarus.Env(None, run_config["env_config"]).env_config()
    trained_on = run_config.get("physics_version")
    if trained_on != icarus.PHYSICS_VERSION:
        raise ValueError(
            f"the run was trained on physics version {trained_on}, but the installed core is "
            f"version {icarus.PHYSICS_VERSION}: retrain instead of exporting"
        )

    # VecNormalize.load needs a venv; the statistics are all we use.
    from icarus_env.vec_env import IcarusVecEnv

    vecnorm = VecNormalize.load(str(vecnormalize_path), IcarusVecEnv(1, env_config=env_config))
    if not vecnorm.norm_obs:
        raise ValueError("the agent was trained without observation normalization")
    layers = actor_layers(model)
    blob = pack(layers)
    meta = {
        "config": run_config.get("config_name"),
        "seed": run_config.get("seed"),
        "timesteps": int(model.num_timesteps),
        "git_commit": _git_commit(),
        "exported": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        **(metadata or {}),
    }
    manifest = finalize(
        build_manifest(
            layers,
            vecnorm.obs_rms.mean,
            vecnorm.obs_rms.var,
            vecnorm.epsilon,
            vecnorm.clip_obs,
            env_config,
            blob,
            meta,
        )
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "agent.bin").write_bytes(blob)
    manifest_path = out_dir / "agent.json"
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")

    # Verify: NumPy reference inference == PyTorch deterministic action.
    policy = MlpPolicy.load(manifest_path)
    obs = sample_observations(policy)
    with torch.no_grad():
        normalized = torch.as_tensor(vecnorm.normalize_obs(obs), dtype=torch.float32)
        reference = model.policy._predict(normalized, deterministic=True).numpy()
    reference = np.clip(reference, -1.0, 1.0)
    error = float(np.max(np.abs(policy.act(obs) - reference)))
    if not error <= tolerance:
        manifest_path.unlink()
        (out_dir / "agent.bin").unlink()
        raise RuntimeError(f"export verification failed: max action error {error:g}")
    print(f"exported {manifest_path} ({len(blob)} bytes of weights, max action error {error:.2e})")
    return manifest_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", type=Path, help="directory with model.zip + vecnormalize.pkl")
    parser.add_argument("--out", type=Path, help="output directory (default: the checkpoint)")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    args = parser.parse_args(argv)
    ckpt: Path = args.checkpoint
    export(
        ckpt / "model.zip",
        ckpt / "vecnormalize.pkl",
        args.out or ckpt,
        tolerance=args.tolerance,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
