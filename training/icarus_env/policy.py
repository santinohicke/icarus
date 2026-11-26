"""Exported policies: loading, reference inference and batch evaluation.

An exported agent is two files (written by ``training/export_policy.py``):

- ``agent.bin``: every weight and bias as little-endian ``float32``, layer
  by layer (weight matrix row-major, shape ``(out, in)``, then the bias);
- ``agent.json``: the manifest (layer shapes and activations, the frozen
  observation-normalization statistics, the environment configuration the
  agent was trained with, the SHA-256 of ``agent.bin`` and a SHA-256 over
  every field inference depends on, so tampering with either file is caught).

Inference, the same computation the Rust/WASM version performs (stage 5;
parity within float32 tolerance, since ``np.tanh`` is not ``libm::tanhf``):

1. ``z = clip((obs - mean) / sqrt(var + epsilon), -clip_obs, clip_obs)``
   computed in float64 from the float32 observation, then rounded to float32;
2. ``h = tanh(W·h + b)`` for the hidden layers, the output layer linear,
   all in float32;
3. the deterministic action (the Gaussian mean) clipped to ``[-1, 1]``.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import icarus
import numpy as np

from .tasks import resolve_task

FORMAT = "icarus-policy"
FORMAT_VERSION = 1
ACTIVATIONS = ("tanh", "identity")
#: Largest accepted weight count (a 2 x 1024 MLP is about 1.1 M): a guard
#: against reading huge files named by a malformed manifest.
MAX_WEIGHTS = 4_000_000
MAX_LAYER_WIDTH = 4096


def content_digest(manifest: Mapping[str, Any]) -> str:
    """SHA-256 over every manifest field inference depends on (canonical
    JSON), including the weights' own SHA-256."""
    content = {
        "format": manifest["format"],
        "format_version": manifest["format_version"],
        "physics_version": manifest["physics_version"],
        "obs_dim": manifest["obs_dim"],
        "action_dim": manifest["action_dim"],
        "layers": manifest["layers"],
        "normalization": manifest["normalization"],
        "action": manifest["action"],
        "env_config": manifest["env_config"],
        "weights_sha256": manifest["weights"]["sha256"],
        "weights_count": manifest["weights"]["count"],
    }
    text = json.dumps(content, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Layer:
    weight: np.ndarray  # float32, (out, in)
    bias: np.ndarray  # float32, (out,)
    activation: str


def _positive_int(value: Any, what: str, limit: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 < value <= limit:
        raise ValueError(f"{what} must be an integer in 1..={limit}, got {value!r}")
    return value


@dataclass(frozen=True)
class MlpPolicy:
    """A deterministic MLP policy with frozen observation normalization."""

    layers: tuple[Layer, ...]
    mean: np.ndarray  # float64, (obs_dim,)
    var: np.ndarray  # float64, (obs_dim,)
    epsilon: float
    clip_obs: float
    env_config: dict[str, Any]
    manifest: dict[str, Any]

    @property
    def obs_dim(self) -> int:
        return int(self.layers[0].weight.shape[1])

    @property
    def action_dim(self) -> int:
        return int(self.layers[-1].weight.shape[0])

    @classmethod
    def load(cls, manifest_path: str | Path) -> MlpPolicy:
        """Loads and validates ``agent.json`` and its weight file.

        Raises ``ValueError`` for anything malformed, tampered with, or made
        for another physics version or observation layout than the installed
        ``icarus`` core.
        """
        path = Path(manifest_path)
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
            return cls._from_manifest(path, manifest)
        except (KeyError, TypeError, IndexError, AttributeError, OSError, ValueError) as e:
            if isinstance(e, ValueError) and not isinstance(e, json.JSONDecodeError):
                raise
            raise ValueError(f"{path}: malformed manifest ({type(e).__name__}: {e})") from e

    @classmethod
    def _from_manifest(cls, path: Path, manifest: dict[str, Any]) -> MlpPolicy:
        if manifest.get("format") != FORMAT or manifest.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"{path}: not an {FORMAT} v{FORMAT_VERSION} manifest")
        if manifest.get("content_sha256") != content_digest(manifest):
            raise ValueError(f"{path}: manifest checksum mismatch (edited or corrupted)")
        if manifest["physics_version"] != icarus.PHYSICS_VERSION:
            raise ValueError(
                f"{path}: trained on physics version {manifest['physics_version']}, "
                f"but the installed core is version {icarus.PHYSICS_VERSION}"
            )
        obs_dim = _positive_int(manifest["obs_dim"], "obs_dim", MAX_LAYER_WIDTH)
        if manifest["action_dim"] != icarus.ACTION_DIM:
            raise ValueError(f"{path}: action_dim must be {icarus.ACTION_DIM}")
        env_config = dict(manifest["env_config"])
        core_dim = icarus.Env(None, env_config).obs_dim  # validates env_config too
        if core_dim != obs_dim:
            raise ValueError(f"{path}: obs_dim {obs_dim} != {core_dim} for its env_config")
        fields = manifest.get("obs_fields")
        if fields is not None and fields != icarus.obs_fields(int(env_config["history_len"])):
            raise ValueError(f"{path}: observation layout differs from the installed core")

        weights = manifest["weights"]
        name = weights["file"]
        # Only a plain file name next to the manifest: no directories,
        # absolute or UNC paths.
        if not isinstance(name, str) or Path(name).name != name or name in ("", ".", ".."):
            raise ValueError(f"{path}: weights.file must be a plain file name, got {name!r}")
        count = _positive_int(weights["count"], "weights.count", MAX_WEIGHTS)
        blob_path = path.parent / name
        if blob_path.stat().st_size != 4 * count:
            raise ValueError(f"{path}: {name} must hold exactly {count} float32 values")
        blob = blob_path.read_bytes()
        if hashlib.sha256(blob).hexdigest() != weights["sha256"]:
            raise ValueError(f"{path}: weight file checksum mismatch")
        flat = np.frombuffer(blob, dtype="<f4")
        if not np.all(np.isfinite(flat)):
            raise ValueError(f"{path}: non-finite weights")

        layers = []
        offset = 0
        previous_out = obs_dim
        for spec in manifest["layers"]:
            n_in = _positive_int(spec["in"], "layer in", MAX_LAYER_WIDTH)
            n_out = _positive_int(spec["out"], "layer out", MAX_LAYER_WIDTH)
            activation = spec["activation"]
            if n_in != previous_out:
                raise ValueError(f"{path}: layer input {n_in} does not follow {previous_out}")
            if activation not in ACTIVATIONS:
                raise ValueError(f"{path}: unknown activation {activation!r}")
            if offset + n_in * n_out + n_out > flat.size:
                raise ValueError(f"{path}: the layers need more weights than {count}")
            w = flat[offset : offset + n_in * n_out].reshape(n_out, n_in)
            offset += n_in * n_out
            b = flat[offset : offset + n_out]
            offset += n_out
            layers.append(Layer(w.astype(np.float32), b.astype(np.float32), activation))
            previous_out = n_out
        if not layers:
            raise ValueError(f"{path}: no layers")
        if offset != flat.size:
            raise ValueError(f"{path}: {flat.size - offset} unused weights")
        if previous_out != icarus.ACTION_DIM:
            raise ValueError(f"{path}: output width {previous_out} != {icarus.ACTION_DIM}")

        norm = manifest["normalization"]
        mean = np.asarray(norm["mean"], dtype=np.float64)
        var = np.asarray(norm["var"], dtype=np.float64)
        if mean.shape != (obs_dim,) or var.shape != mean.shape:
            raise ValueError(f"{path}: normalization statistics have the wrong shape")
        if not (np.all(np.isfinite(mean)) and np.all(np.isfinite(var)) and np.all(var >= 0)):
            raise ValueError(f"{path}: normalization statistics must be finite (var >= 0)")
        epsilon, clip_obs = float(norm["epsilon"]), float(norm["clip_obs"])
        valid = math.isfinite(epsilon) and math.isfinite(clip_obs)
        if not (valid and epsilon > 0 and clip_obs > 0):
            raise ValueError(f"{path}: epsilon and clip_obs must be finite and positive")
        return cls(
            layers=tuple(layers),
            mean=mean,
            var=var,
            epsilon=epsilon,
            clip_obs=clip_obs,
            env_config=env_config,
            manifest=manifest,
        )

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        """Normalized observations (float32), shape ``(..., obs_dim)``."""
        z = (np.asarray(obs, dtype=np.float64) - self.mean) / np.sqrt(self.var + self.epsilon)
        return np.clip(z, -self.clip_obs, self.clip_obs).astype(np.float32)

    def act(self, obs: np.ndarray) -> np.ndarray:
        """Deterministic actions in ``[-1, 1]`` (float32) for a batch or one
        observation."""
        h = self.normalize(obs)
        for layer in self.layers:
            h = h @ layer.weight.T + layer.bias
            if layer.activation == "tanh":
                h = np.tanh(h)
        return np.clip(h, -1.0, 1.0).astype(np.float32)


def run_agent(
    policy: MlpPolicy,
    task: Mapping[str, Any],
    seeds: Sequence[int],
    batch: int = 256,
) -> list[dict[str, Any]]:
    """Runs one episode per seed (no auto-reset, no randomization beyond the
    task's) and returns one result dict per seed, with the same keys as
    ``icarus.evaluate``. ``act_ns`` is the policy time per decision,
    amortized over the batch of still-running episodes (finished ones are not
    evaluated)."""
    rows: list[dict[str, Any]] = []
    resolved = resolve_task(task)
    for start in range(0, len(seeds), batch):
        chunk = [int(s) for s in seeds[start : start + batch]]
        vec = icarus.VecEnv(
            len(chunk), task=resolved, env_config=policy.env_config, autoreset=False
        )
        obs = vec.reset(seeds=chunk)
        done = np.zeros(len(chunk), dtype=bool)
        actions = np.zeros((len(chunk), icarus.ACTION_DIM), dtype=np.float32)
        decisions = 0
        act_time = 0.0
        while not done.all():
            running = ~done
            t0 = time.perf_counter()
            actions[running] = policy.act(obs[running])
            act_time += time.perf_counter() - t0
            decisions += int(running.sum())
            obs, _, terminated, truncated, _ = vec.step(actions)
            done |= terminated | truncated
        act_ns = 1e9 * act_time / max(decisions, 1)
        for r in vec.episode_results():
            rows.append({**r, "act_ns": act_ns})
    return rows
