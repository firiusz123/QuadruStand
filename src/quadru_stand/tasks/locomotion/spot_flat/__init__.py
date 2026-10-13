"""Registration of the QuadruStand Spot flat-terrain velocity task."""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="QuadruStand-Velocity-Flat-Spot-v0",
    # Same environment class as the upstream Spot task: only the *config* differs.
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        # Lazy "module:Class" strings: resolved only when the task is actually created.
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:QuadruStandSpotFlatEnvCfg",
        # Default agent (used when `--agent` is not given). Key name matches the upstream Spot task.
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadruStandSpotFlatPPORunnerCfg",
        # Extra agents for the network-size benchmark, selected with `--agent <key>`.
        "rsl_rl_small_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadruStandSpotFlatSmallPPORunnerCfg",
        "rsl_rl_medium_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadruStandSpotFlatMediumPPORunnerCfg",
        "rsl_rl_large_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadruStandSpotFlatLargePPORunnerCfg",
    },
)
