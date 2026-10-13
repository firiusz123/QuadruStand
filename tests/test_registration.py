"""Tests for the project's task registrations."""

import gymnasium as gym
import pytest

import quadru_stand.tasks  # noqa: F401  (importing runs every gym.register call)

from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

TASK_ID = "QuadruStand-Velocity-Flat-Spot-v0"

EXPECTED_HIDDEN_DIMS = {
    "rsl_rl_cfg_entry_point": ("quadrustand_spot_flat_baseline", [512, 256, 128]),
    "rsl_rl_small_cfg_entry_point": ("quadrustand_spot_flat_small", [128, 128]),
    "rsl_rl_medium_cfg_entry_point": ("quadrustand_spot_flat_medium", [256, 128, 64]),
    "rsl_rl_large_cfg_entry_point": ("quadrustand_spot_flat_large", [1024, 512, 256]),
}


def test_task_registered():
    spec = gym.spec(TASK_ID)
    assert spec.entry_point == "isaaclab.envs:ManagerBasedRLEnv"
    assert (
        spec.kwargs["env_cfg_entry_point"]
        == "quadru_stand.tasks.locomotion.spot_flat.flat_env_cfg:QuadruStandSpotFlatEnvCfg"
    )


@pytest.mark.parametrize("key", list(EXPECTED_HIDDEN_DIMS))
def test_agent_entry_points(key):
    experiment_name, hidden_dims = EXPECTED_HIDDEN_DIMS[key]
    cfg = load_cfg_from_registry(TASK_ID, key)
    assert cfg.experiment_name == experiment_name
    assert list(cfg.actor.hidden_dims) == hidden_dims
    assert list(cfg.critic.hidden_dims) == hidden_dims


def _without_network_and_name(cfg):
    d = cfg.to_dict()
    d.pop("experiment_name")
    d["actor"].pop("hidden_dims")
    d["critic"].pop("hidden_dims")
    return d


def test_variants_only_differ_in_network_and_name():
    """Everything except hidden_dims / experiment_name must be inherited unchanged."""
    base = _without_network_and_name(load_cfg_from_registry(TASK_ID, "rsl_rl_cfg_entry_point"))
    for key in EXPECTED_HIDDEN_DIMS:
        assert _without_network_and_name(load_cfg_from_registry(TASK_ID, key)) == base


def test_env_cfg_identical_to_upstream_spot():
    """At this stage the env config must be the upstream Spot config, class name aside."""
    ours = load_cfg_from_registry(TASK_ID, "env_cfg_entry_point")
    upstream = load_cfg_from_registry("IsaacContrib-Velocity-Flat-Spot", "env_cfg_entry_point")
    assert type(ours).__mro__[1] is type(upstream)
    assert ours.rewards.to_dict() == upstream.rewards.to_dict()
