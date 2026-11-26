"""Policy export: weights + manifest round-trip and match PyTorch."""

from __future__ import annotations

import json
from pathlib import Path

import icarus
import numpy as np
import pytest

pytest.importorskip("stable_baselines3")

from icarus_env.policy import MlpPolicy, run_agent


@pytest.fixture(scope="module")
def exported(tmp_path_factory: pytest.TempPathFactory) -> Path:
    import train

    run = tmp_path_factory.mktemp("run")
    train.main(
        [
            "--total-timesteps",
            "2048",
            "--num-envs",
            "8",
            "--run-dir",
            str(run),
        ]
    )
    return run


def test_training_writes_a_reproducible_run(exported: Path) -> None:
    record = json.loads((exported / "config.json").read_text(encoding="utf-8"))
    assert record["env_config"]["reward"]["gamma"] == record["config"]["ppo"]["gamma"]
    assert record["physics_version"] == icarus.PHYSICS_VERSION
    for name in ("best", "final"):
        for f in ("model.zip", "vecnormalize.pkl", "agent.bin", "agent.json"):
            assert (exported / name / f).exists(), (name, f)
    history = json.loads((exported / "validation.json").read_text(encoding="utf-8"))
    assert history
    assert history[-1]["step"] >= 2048


def test_reference_inference_matches_pytorch(exported: Path) -> None:
    import torch
    from icarus_env import IcarusVecEnv
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize

    ckpt = exported / "final"
    policy = MlpPolicy.load(ckpt / "agent.json")
    assert policy.obs_dim == 26
    assert policy.action_dim == 3
    assert [layer.activation for layer in policy.layers] == ["tanh", "tanh", "identity"]
    model = PPO.load(ckpt / "model.zip", device="cpu")
    vecnorm = VecNormalize.load(str(ckpt / "vecnormalize.pkl"), IcarusVecEnv(1))
    rng = np.random.default_rng(3)
    obs = rng.normal(0, 3, size=(256, 26)).astype(np.float32)
    with torch.no_grad():
        z = torch.as_tensor(vecnorm.normalize_obs(obs), dtype=torch.float32)
        reference = np.clip(model.policy._predict(z, deterministic=True).numpy(), -1, 1)
    np.testing.assert_allclose(policy.act(obs), reference, atol=1e-5)
    np.testing.assert_allclose(policy.normalize(obs), vecnorm.normalize_obs(obs), atol=1e-6)
    # Single observations work too.
    np.testing.assert_allclose(policy.act(obs[0]), reference[0], atol=1e-5)


def test_manifest_describes_the_weights(exported: Path) -> None:
    ckpt = exported / "final"
    m = json.loads((ckpt / "agent.json").read_text(encoding="utf-8"))
    blob = (ckpt / "agent.bin").read_bytes()
    expected = sum(layer["in"] * layer["out"] + layer["out"] for layer in m["layers"])
    assert m["weights"]["count"] == expected == len(blob) // 4
    assert m["obs_fields"] == icarus.obs_fields(m["env_config"]["history_len"])
    assert len(m["normalization"]["mean"]) == m["obs_dim"]
    assert m["normalization"]["clip_obs"] == 10.0
    assert m["physics_version"] == icarus.PHYSICS_VERSION


def test_tampering_is_detected(exported: Path, tmp_path: Path) -> None:
    src = exported / "final"
    (tmp_path / "agent.json").write_bytes((src / "agent.json").read_bytes())
    blob = bytearray((src / "agent.bin").read_bytes())
    blob[0] ^= 1
    (tmp_path / "agent.bin").write_bytes(bytes(blob))
    with pytest.raises(ValueError, match="checksum"):
        MlpPolicy.load(tmp_path / "agent.json")
    m = json.loads((src / "agent.json").read_text(encoding="utf-8"))
    m["format_version"] = 99
    (tmp_path / "agent.json").write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest"):
        MlpPolicy.load(tmp_path / "agent.json")


def test_run_agent_rows_match_the_evaluation_format(exported: Path) -> None:
    policy = MlpPolicy.load(exported / "final" / "agent.json")
    rows = run_agent(policy, {"config": "realistic"}, [0, 1, 2], batch=2)
    assert [r["seed"] for r in rows] == [0, 1, 2]
    keys = {"outcome", "landed", "score", "contact_vy", "position_error", "act_ns"}
    assert all(r["done"] and keys <= set(r) for r in rows)
    again = run_agent(policy, {"config": "realistic"}, [0, 1, 2], batch=2)
    assert [r["ticks"] for r in rows] == [r["ticks"] for r in again], "deterministic"


def _copy(src: Path, dst: Path) -> dict:
    for f in ("agent.json", "agent.bin"):
        (dst / f).write_bytes((src / f).read_bytes())
    return json.loads((dst / "agent.json").read_text(encoding="utf-8"))


def _write(dst: Path, manifest: dict, refresh_digest: bool = True) -> None:
    from icarus_env.policy import content_digest

    if refresh_digest:
        manifest["content_sha256"] = content_digest(manifest)
    (dst / "agent.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_manifest_edits_are_detected(exported: Path, tmp_path: Path) -> None:
    m = _copy(exported / "final", tmp_path)
    m["normalization"]["clip_obs"] = 5.0
    _write(tmp_path, m, refresh_digest=False)
    with pytest.raises(ValueError, match="manifest checksum"):
        MlpPolicy.load(tmp_path / "agent.json")


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda m: m["weights"].update(file="../agent.bin"), "plain file name"),
        (lambda m: m["weights"].update(file="/etc/agent.bin"), "plain file name"),
        (lambda m: m["weights"].update(count=10), "exactly"),
        (lambda m: m.update(physics_version=999), "physics version"),
        (lambda m: m.update(obs_dim=24), "obs_dim"),
        (lambda m: m["normalization"].update(epsilon=-1.0), "epsilon"),
        (lambda m: m["layers"][0].update(activation="relu"), "activation"),
        (lambda m: m.pop("normalization"), "malformed"),
    ],
)
def test_malformed_manifests_are_rejected(
    exported: Path, tmp_path: Path, edit: object, message: str
) -> None:
    m = _copy(exported / "final", tmp_path)
    edit(m)  # type: ignore[operator]
    try:
        _write(tmp_path, m)
    except KeyError:
        _write(tmp_path, m, refresh_digest=False)
    with pytest.raises(ValueError, match=message):
        MlpPolicy.load(tmp_path / "agent.json")
