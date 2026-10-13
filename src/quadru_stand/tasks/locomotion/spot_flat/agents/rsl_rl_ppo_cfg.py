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


@configclass
class QuadruStandSpotFlatSmallPPORunnerCfg(QuadruStandSpotFlatPPORunnerCfg):
    experiment_name = "quadrustand_spot_flat_small"

    def __post_init__(self):
        super().__post_init__()
        self.actor.hidden_dims = [128, 128]
        self.critic.hidden_dims = [128, 128]


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
