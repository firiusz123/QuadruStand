"""Evaluate one trained QuadruStand policy on the benchmark inference metrics.
Runs the deterministic actor in the training scene and writes one row to a CSV.
"""
import argparse
import csv
from pathlib import Path
import gymnasium as gym
import torch
from isaaclab.app import add_launcher_args, launch_simulation
import quadru_stand.tasks  # noqa: F401  (registers the Gym task)
from isaaclab_rl.entrypoints.backends import cli_args_rsl_rl as cli_args
from isaaclab_rl.entrypoints.common import apply_env_overrides, close_env, create_isaaclab_env
from isaaclab_rl.rsl_rl import (
    RslRlVecEnvWrapper,
    check_rsl_rl_version,
    create_rsl_rl_runner,
    handle_deprecated_rsl_rl_cfg,
)
from isaaclab_tasks.utils import resolve_task_config
TASK = "QuadruStand-Velocity-Flat-Spot-v0"
LOG_ROOT = Path("logs/rsl_rl")
CHECKPOINT_NAME = "model_1499.pt"
AGENT_BY_EXPERIMENT = {
    "quadrustand_spot_flat_baseline": "rsl_rl_cfg_entry_point",
    "quadrustand_spot_flat_extra_small": "rsl_rl_extra_small_cfg_entry_point",
    "quadrustand_spot_flat_small": "rsl_rl_small_cfg_entry_point",
    "quadrustand_spot_flat_medium": "rsl_rl_medium_cfg_entry_point",
    "quadrustand_spot_flat_large": "rsl_rl_large_cfg_entry_point",
    "quadrustand_spot_flat_deep_1": "rsl_rl_deep_1_cfg_entry_point",
    "quadrustand_spot_flat_deep_2": "rsl_rl_deep_2_cfg_entry_point",
    "quadrustand_spot_flat_deep_5": "rsl_rl_deep_5_cfg_entry_point",
}
COLUMNS = [
    "variant",
    "checkpoint",
    "seed",
    "num_envs",
    "num_steps",
    "lin_vel_error",
    "yaw_error",
    "tilt_mean",
    "tilt_max",
    "fall_rate",
    "tracking_success",
]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a QuadruStand checkpoint.")
    parser.add_argument("--task", type=str, default=TASK)
    parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=f"One checkpoint. Omit to evaluate every {CHECKPOINT_NAME} under {LOG_ROOT}.",
    )
    parser.add_argument("--num_envs", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--num_steps", type=int, default=3000)
    parser.add_argument("--output", type=str, default="results/eval_metrics.csv")
    parser.add_argument("--frontend", type=str, default="torch", choices=["torch", "warp"])
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--logger", type=str, default=None)
    parser.add_argument("--log_project_name", type=str, default=None)
    add_launcher_args(parser)
    return parser.parse_args()


def evaluate(env: RslRlVecEnvWrapper, policy, num_steps: int) -> dict[str, float]:
    obs = env.get_observations()
    err_xy_sum = err_yaw_sum = tilt_sum = 0.0
    tilt_max = 0.0
    alive_steps = success_steps = fall_ends = episode_ends = 0

    with torch.inference_mode():
        for _ in range(num_steps):
            obs, _, dones, _ = env.step(policy(obs))
            policy.reset(dones)
            base = env.unwrapped
            robot = base.scene["robot"]
            cmd = base.command_manager.get_command("base_velocity")
            err_xy = torch.linalg.norm(cmd[:, :2] - robot.data.root_lin_vel_b.torch[:, :2], dim=-1)
            err_yaw = torch.abs(cmd[:, 2] - robot.data.root_ang_vel_b.torch[:, 2])
            tilt = torch.acos(torch.clamp(-robot.data.projected_gravity_b.torch[:, 2], -1.0, 1.0))
            terms = base.termination_manager
            fallen = terms.get_term("body_contact")
            ended = terms.terminated | terms.time_outs
            alive = ~fallen
            if bool(alive.any()):
                err_xy_sum += err_xy[alive].sum().item()
                err_yaw_sum += err_yaw[alive].sum().item()
                tilt_sum += tilt[alive].sum().item()
                tilt_max = max(tilt_max, tilt[alive].max().item())
                success_steps += int(((err_xy < 0.2) & (err_yaw < 0.2) & alive).sum())
            alive_steps += int(alive.sum())
            fall_ends += int(fallen.sum())
            episode_ends += int(ended.sum())

    if alive_steps == 0:
        raise RuntimeError("Every robot fell on every step; there is nothing to average.")
    return {
        "lin_vel_error": err_xy_sum / alive_steps,
        "yaw_error": err_yaw_sum / alive_steps,
        "tilt_mean": tilt_sum / alive_steps,
        "tilt_max": tilt_max,
        "fall_rate": fall_ends / episode_ends if episode_ends else float("nan"),
        "tracking_success": success_steps / alive_steps,
    }

def discover_checkpoints(root: Path) -> list[tuple[str, Path]]:
    """Return (agent key, checkpoint) for every model_1499.pt under logs/rsl_rl.

    Run directories without that file are absent from the glob and are skipped.
    """
    jobs: list[tuple[str, Path]] = []
    for path in sorted(root.glob(f"*/*/{CHECKPOINT_NAME}")):
        experiment = path.parents[1].name
        agent = AGENT_BY_EXPERIMENT.get(experiment)
        if agent is None:
            print(f"skip unknown experiment: {path}")
            continue
        jobs.append((agent, path))
    return jobs


def append_row(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

def main() -> None:
    args = parse_args()
    if args.checkpoint:
        jobs = [(args.agent, Path(args.checkpoint))]
    else:
        jobs = discover_checkpoints(LOG_ROOT)
        if not jobs:
            raise SystemExit(f"No {CHECKPOINT_NAME} under {LOG_ROOT}")
    for agent, checkpoint in jobs:
        print(f"queued {agent} {checkpoint}")

    installed_version = check_rsl_rl_version()
    # The scene is identical for every network. Resolve it once from the baseline agent.
    # overrides=[] keeps this script's own flags out of Hydra.
    env_cfg, _ = resolve_task_config(
        args.task, "rsl_rl_cfg_entry_point", play_mode=False, overrides=[]
    )
    apply_env_overrides(args, env_cfg)
    env_cfg.seed = args.seed
    with launch_simulation(env_cfg, args):
        env = create_isaaclab_env(args.task, env_cfg, args, convert_marl_to_single_agent=False)
        try:
            wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
            for agent, checkpoint in jobs:
                print(f"evaluating {checkpoint}")
                _, agent_cfg = resolve_task_config(args.task, agent, play_mode=False, overrides=[])
                args.agent = agent
                args.checkpoint = str(checkpoint)
                agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args)
                agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
                runner = create_rsl_rl_runner(wrapped, agent_cfg, log_dir=None)
                runner.load(str(checkpoint))
                policy = runner.get_inference_policy(device=wrapped.unwrapped.device)
                wrapped.unwrapped.reset(seed=args.seed)
                metrics = evaluate(wrapped, policy, args.num_steps)
                row = {
                    "variant": agent,
                    "checkpoint": str(checkpoint),
                    "seed": args.seed,
                    "num_envs": args.num_envs,
                    "num_steps": args.num_steps,
                    **metrics,
                }
                append_row(Path(args.output), row)
                for key in COLUMNS[5:]:
                    print(f"{key}: {row[key]:.4f}")
        finally:
            close_env(env)



if __name__ == "__main__":
    main()