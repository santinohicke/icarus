"""Training configs: base merging, stage tasks and the ablations."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("yaml")

import train
from icarus_env.config import deep_merge, load_config
from icarus_env.tasks import validate_task

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
ABLATIONS = sorted((CONFIGS / "ablations").glob("*.yaml"))


def stages(path: Path) -> list:
    return train.build_curriculum(load_config(path), seed=0).stages


def test_deep_merge() -> None:
    base = {"a": {"x": 1, "y": 2}, "b": [1, 2]}
    assert deep_merge(base, {"a": {"y": 3}, "b": [9]}) == {"a": {"x": 1, "y": 3}, "b": [9]}
    assert deep_merge(base, {"a": None}) == {"a": None, "b": [1, 2]}
    assert base == {"a": {"x": 1, "y": 2}, "b": [1, 2]}, "inputs untouched"


def test_base_config_is_valid_and_ties_gamma() -> None:
    cfg = load_config(CONFIGS / "ppo_base.yaml")
    env_config = train.env_config_of(cfg)
    assert env_config["reward"]["gamma"] == cfg["ppo"]["gamma"]
    s = stages(CONFIGS / "ppo_base.yaml")
    for stage in s:
        validate_task(stage.task, env_config)
    names = [x.name for x in s]
    rand_from = names.index(cfg["randomization"]["from_stage"])
    fail_from = names.index(cfg["failures"]["from_stage"])
    for i, stage in enumerate(s):
        assert ("randomization" in stage.task) == (i >= rand_from), stage.name
        has_faults = "faults.probability" in stage.task.get("overrides", {})
        assert has_faults == (i >= fail_from), stage.name
    with pytest.raises(ValueError, match="gamma"):
        train.env_config_of({**cfg, "env": {"reward": {"gamma": 0.9}}})


@pytest.mark.parametrize("path", ABLATIONS, ids=lambda p: p.stem)
def test_every_ablation_is_valid(path: Path) -> None:
    cfg = load_config(path)
    env_config = train.env_config_of(cfg)
    for stage in stages(path):
        validate_task(stage.task, env_config)


def test_each_ablation_changes_what_it_says() -> None:
    ab = CONFIGS / "ablations"
    assert len(stages(ab / "no_curriculum.yaml")) == 1
    assert not any("randomization" in s.task for s in stages(ab / "no_randomization.yaml"))
    assert not any(
        "faults.probability" in s.task.get("overrides", {}) for s in stages(ab / "no_failures.yaml")
    )
    assert load_config(ab / "no_history.yaml")["env"]["action_history"] is False
    assert load_config(ab / "mlp128.yaml")["policy"]["net_arch"] == [128, 128]
    assert load_config(ab / "recurrent.yaml")["algo"] == "recurrent_ppo"
    # Everything else is inherited unchanged.
    base = load_config(CONFIGS / "ppo_base.yaml")
    assert load_config(ab / "mlp128.yaml")["ppo"] == base["ppo"]


def test_disabled_randomization_ignores_its_starting_stage() -> None:
    cfg = load_config(CONFIGS / "ppo_base.yaml")
    cfg["curriculum"] = {
        "stages": [s for s in cfg["curriculum"]["stages"] if s["name"] == "touchdown"]
    }
    cfg["curriculum"]["stages"][0] = {
        k: v for k, v in cfg["curriculum"]["stages"][0].items() if k != "promote_at"
    }
    cfg["randomization"] = {"enabled": False, "from_stage": "not_a_stage_here"}
    cfg["failures"] = {"overrides": None}
    tasks = train.stage_tasks(cfg)
    assert tasks
    assert "randomization" not in tasks[0]
