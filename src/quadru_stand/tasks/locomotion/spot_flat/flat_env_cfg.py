"""Environment configuration for the QuadruStand Spot flat-terrain velocity task."""

from isaaclab.utils import configclass

from isaaclab_tasks.contrib.velocity.config.spot.flat_env_cfg import SpotFlatEnvCfg


@configclass
class QuadruStandSpotFlatEnvCfg(SpotFlatEnvCfg):
    """Spot on flat terrain; currently identical to Isaac Lab's ``IsaacContrib-Velocity-Flat-Spot``.

    Everything (scene, observations, actions, commands, rewards, terminations, events, simulation
    settings) is inherited from :class:`SpotFlatEnvCfg`. Project-specific changes go in
    ``__post_init__`` so the difference from the upstream task stays explicit.
    """

    def __post_init__(self):
        # Run the parent's __post_init__ first: it sets decimation, sim dt, the Spot robot,
        # the terrain, the contact-sensor update period, etc.
        super().__post_init__()

        # Reward changes (front-leg balancing) will be applied here in a later plan, e.g.
        #   self.rewards.<term>.weight = ...
