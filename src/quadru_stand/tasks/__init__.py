"""Task registration for the QuadruStand project.

Isaac Lab's CLI (``isaaclab train``, ``play``, ``list_envs``) imports this package because
``pyproject.toml`` declares it under the ``isaaclab.tasks`` entry-point group, so the
``gym.register(...)`` calls below run on every CLI start-up.

Files in this package:
    flat_env_cfg.py    environment config (inherits Isaac Lab's Spot flat-terrain config)
    rsl_rl_ppo_cfg.py  rsl_rl PPO runner configs: baseline and the network-size / depth variants

Adding another task: write its config file here and add another ``gym.register`` call.
"""

import gymnasium as gym

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
        "rsl_rl_cfg_entry_point": f"{__name__}.rsl_rl_ppo_cfg:QuadruStandSpotFlatPPORunnerCfg",
        # Extra agents for the network benchmark, selected with `--agent <key>`.
        "rsl_rl_extra_small_cfg_entry_point": f"{__name__}.rsl_rl_ppo_cfg:QuadruStandSpotFlatExtraSmallPPORunnerCfg",
        "rsl_rl_small_cfg_entry_point": f"{__name__}.rsl_rl_ppo_cfg:QuadruStandSpotFlatSmallPPORunnerCfg",
        "rsl_rl_medium_cfg_entry_point": f"{__name__}.rsl_rl_ppo_cfg:QuadruStandSpotFlatMediumPPORunnerCfg",
        "rsl_rl_large_cfg_entry_point": f"{__name__}.rsl_rl_ppo_cfg:QuadruStandSpotFlatLargePPORunnerCfg",
        "rsl_rl_deep_1_cfg_entry_point": f"{__name__}.rsl_rl_ppo_cfg:QuadruStandSpotFlatDeep_1_PPORunnerCfg",
        "rsl_rl_deep_2_cfg_entry_point": f"{__name__}.rsl_rl_ppo_cfg:QuadruStandSpotFlatDeep_2_PPORunnerCfg",
        "rsl_rl_deep_5_cfg_entry_point": f"{__name__}.rsl_rl_ppo_cfg:QuadruStandSpotFlatDeep_5_PPORunnerCfg",
    },
)
