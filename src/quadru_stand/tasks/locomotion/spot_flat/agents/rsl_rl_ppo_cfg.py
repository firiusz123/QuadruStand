"""rsl_rl PPO runner configurations for the QuadruStand Spot flat task.

All variants inherit Isaac Lab's Spot runner config. They differ from it (and from each other) ONLY in
the MLP ``hidden_dims`` of actor and critic and in ``experiment_name`` (which sets the log folder).
"""

from isaaclab.utils import configclass

from isaaclab_tasks.contrib.velocity.config.spot.agents.rsl_rl_ppo_cfg import SpotFlatPPORunnerCfg


@configclass
class QuadruStandSpotFlatPPORunnerCfg(SpotFlatPPORunnerCfg):
    """Baseline: network [512, 256, 128], same as upstream; only the experiment name differs."""

    experiment_name = "quadrustand_spot_flat_baseline"

"""The networks from section A (BENCHMARK.md) size comparison"""

@configclass
class QuadruStandSpotFlatExtraSmallPPORunnerCfg(QuadruStandSpotFlatPPORunnerCfg):
    experiment_name = "quadrustand_spot_flat_extra_small"

    def __post_init__(self):
        super().__post_init__()
        self.actor.hidden_dims = [64, 32, 16]
        self.critic.hidden_dims = [64, 32, 16]

@configclass
class QuadruStandSpotFlatSmallPPORunnerCfg(QuadruStandSpotFlatPPORunnerCfg):
    experiment_name = "quadrustand_spot_flat_small"

    def __post_init__(self):
        super().__post_init__()
        self.actor.hidden_dims = [128, 64, 32]
        self.critic.hidden_dims = [128, 64, 32]


@configclass
class QuadruStandSpotFlatMediumPPORunnerCfg(QuadruStandSpotFlatPPORunnerCfg):
    experiment_name = "quadrustand_spot_flat_medium"

    def __post_init__(self):
        super().__post_init__()
        self.actor.hidden_dims = [256, 128, 64]
        self.critic.hidden_dims = [256, 128, 64]


@configclass
class QuadruStandSpotFlatLargePPORunnerCfg(QuadruStandSpotFlatPPORunnerCfg):
    experiment_name = "quadrustand_spot_flat_large"

    def __post_init__(self):
        super().__post_init__()
        self.actor.hidden_dims = [1024, 512, 256]
        self.critic.hidden_dims = [1024, 512, 256]

"""The networks from section B (BENCHMARK.md) depth comparison"""

@configclass
class QuadruStandSpotFlatDeep_1_PPORunnerCfg(QuadruStandSpotFlatPPORunnerCfg):
    experiment_name = "quadrustand_spot_flat_deep_1"

    def __post_init__(self):
        super().__post_init__()
        self.actor.hidden_dims = [3072]
        self.critic.hidden_dims = [3072]

@configclass
class QuadruStandSpotFlatDeep_2_PPORunnerCfg(QuadruStandSpotFlatPPORunnerCfg):
    experiment_name = "quadrustand_spot_flat_deep_2"

    def __post_init__(self):
        super().__post_init__()
        self.actor.hidden_dims = [400, 400]
        self.critic.hidden_dims = [400, 400]

@configclass
class QuadruStandSpotFlatDeep_5_PPORunnerCfg(QuadruStandSpotFlatPPORunnerCfg):
    experiment_name = "quadrustand_spot_flat_deep_5"

    def __post_init__(self):
        super().__post_init__()
        self.actor.hidden_dims = [208, 208, 208, 208, 208]
        self.critic.hidden_dims = [208, 208, 208, 208, 208]