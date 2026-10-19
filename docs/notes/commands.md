# Command lookup table

All commands run from the repo root with the environment active:

```bash
source .venv/bin/activate
```

Status column: **verified** = checked against this install's `--help` output or source, or run on this machine.
**not run** = the flags are real, but I did not run that exact command here. Check the output yourself the first time.

Shorthand used below: `TASK=QuadruStand-Velocity-Flat-Spot-v0`.

| Agent key (`--agent`) | Network (actor and critic) | Log folder `logs/rsl_rl/...` |
|---|---|---|
| *(none)*, i.e. `rsl_rl_cfg_entry_point` | `[512, 256, 128]` | `quadrustand_spot_flat_baseline` |
| `rsl_rl_small_cfg_entry_point` | `[128, 128]` | `quadrustand_spot_flat_small` |
| `rsl_rl_medium_cfg_entry_point` | `[256, 128, 64]` | `quadrustand_spot_flat_medium` |
| `rsl_rl_large_cfg_entry_point` | `[1024, 512, 256]` | `quadrustand_spot_flat_large` |

## 1. Check that everything is registered

| Goal | Command | Status |
|---|---|---|
| Is the task registered? | `isaaclab list_envs \| grep QuadruStand` | verified |
| Registration tests | `python -m pytest tests/test_registration.py -q` | verified |
| Is our package visible to Isaac Lab? | `python -c "import importlib.metadata as m; print(list(m.entry_points(group='isaaclab.tasks')))"` | verified |
| Which packages are installed | `uv pip list \| grep -i "isaaclab\|rsl\|quadru"` | verified |

## 2. Training

| Goal | Command | Status |
|---|---|---|
| Quick smoke test (10 iterations) | `isaaclab train --rl_library rsl_rl --task $TASK --num_envs 512 --max_iterations 10` | verified |
| Same, small network | add `--agent rsl_rl_small_cfg_entry_point` | verified |
| Full-size run | `isaaclab train --rl_library rsl_rl --task $TASK --num_envs 4096` | not run |
| Fix the random seed | add `--seed 1` | not run |
| Tag the run folder | add `--run_name my_test` (suffix on the timestamp folder) | not run |
| Train with a window | add `--visualizer kit` (alias `--viz kit`), usually with fewer envs, e.g. `--num_envs 64` | not run |
| Draw only some of the envs | add `--max_visible_envs 16` | not run |
| Choose the GPU | add `--device cuda:0` | not run |

Notes:

- Replace `$TASK` with the real id (the shell does not know it), or run `TASK=QuadruStand-Velocity-Flat-Spot-v0` first.
- **Headless is the default.** Without `--visualizer`, no window opens. There is no `--headless` flag in this Isaac Lab version.
- The default length is **20000 iterations**. Always add `--max_iterations N` unless you mean it. Stop a run with `Ctrl+C`.
- Plain `isaaclab train --help` shows only `--rl_library`. For the real option list use:

```bash
isaaclab train --rl_library rsl_rl --help
```

- `--num_envs` and `--max_iterations` change `params/env.yaml` and `params/agent.yaml` of the run. Compare runs with this in mind.

## 3. Resume training from a checkpoint

`--checkpoint` accepts `latest`, `best`, a path to a `.pt` file, or a run folder. It is the same flag for `train` and `play`.

| Goal | Command | Status |
|---|---|---|
| Resume from the newest run of this task and agent | `isaaclab train --rl_library rsl_rl --task $TASK --checkpoint latest --max_iterations 500` | not run |
| Resume from a specific file | `isaaclab train --rl_library rsl_rl --task $TASK --checkpoint logs/rsl_rl/<experiment>/<timestamp>/model_300.pt --max_iterations 500` | not run |

Behaviour (from the rsl_rl source in `.venv`):

- `--max_iterations` then means **additional** iterations: training runs from the checkpoint's iteration up to that
  iteration plus `max_iterations`.
- A resumed run writes to a **new** timestamp folder. The old folder is not modified.
- `latest` and `best` only consider runs of the same library, task and agent. If you used `--agent`, repeat it.
- The network shape in the checkpoint must match the agent config. A `small` checkpoint will not load into the `large` agent.

## 4. Play a trained policy (watch it)

| Goal | Command | Status |
|---|---|---|
| Play the latest run of this agent | `isaaclab play --rl_library rsl_rl --task $TASK --num_envs 16 --visualizer kit` | not run |
| Same, small agent | add `--agent rsl_rl_small_cfg_entry_point` | not run |
| Play a specific checkpoint | add `--checkpoint logs/rsl_rl/<experiment>/<timestamp>/model_300.pt` | not run |
| Play the newest or the best checkpoint | add `--checkpoint latest` or `--checkpoint best` | not run |
| Run in real time | add `--real-time` | not run |
| Use the exact training env config | add `--train_env_cfg` (skips the play-mode overrides) | not run |
| Record video | add `--video --video_length 300` | not run |

Notes:

- With **no** `--checkpoint`, play loads the latest run's latest model in the log folder of the selected agent.
  So for the small network always pass `--agent rsl_rl_small_cfg_entry_point`, otherwise play looks in the baseline folder.
- Play uses the policy's **mean** action (no exploration noise).
- I did not check what the "play-mode overrides" change. `--train_env_cfg` turns them off.

## 5. TensorBoard (training curves)

| Goal | Command | Status |
|---|---|---|
| All runs | `tensorboard --logdir logs/rsl_rl` | tensorboard 2.21.0 is installed; not run |
| One experiment | `tensorboard --logdir logs/rsl_rl/quadrustand_spot_flat_small` | not run |
| Different port | add `--port 6007` | not run |

Then open `http://localhost:6006` in a browser. Useful curves:

- `Train/mean_reward` and `Train/mean_episode_length` (overall progress; I did not check these exact tag names, browse the tag list)
- `Metrics/base_velocity/error_vel_xy` and `error_vel_yaw`: velocity tracking error vs the command
- `Episode_Reward/<term>`: the contribution of each reward term
- `Episode_Termination/body_contact`: fraction of episodes ending in a fall
- `Curriculum/terrain_levels`

The same numbers also print in the terminal every iteration.

## 6. Inspect a run

Every run folder `logs/rsl_rl/<experiment>/<timestamp>/` contains `model_<iter>.pt`, `params/env.yaml`,
`params/agent.yaml`, `git/` diffs, `run.json` and the TensorBoard event file.

| Goal | Command | Status |
|---|---|---|
| List all runs | `find logs -maxdepth 3 -mindepth 2 -type d \| sort` | verified |
| Network size of a run | `grep -n hidden_dims -A3 logs/rsl_rl/<experiment>/<timestamp>/params/agent.yaml` | verified |
| Compare two runs' env config | `diff logs/rsl_rl/<a>/params/env.yaml logs/rsl_rl/<b>/params/env.yaml` | verified |
| Compare two runs' agent config | `diff logs/rsl_rl/<a>/params/agent.yaml logs/rsl_rl/<b>/params/agent.yaml` | verified |
| Network input/output sizes from a checkpoint | see below | verified |

```bash
python -c "import torch; sd = torch.load('logs/rsl_rl/<experiment>/<timestamp>/model_9.pt', map_location='cpu', weights_only=False)['actor_state_dict']; print({k: tuple(v.shape) for k, v in sd.items()})"
```

For the baseline this shows the first layer `(512, 48)` (hidden size, input size) and the last `(12, 128)` (output size).

## 7. Debug an environment without training

| Goal | Command | Status |
|---|---|---|
| Run the env with **zero** actions (robot holds the default pose) | `isaaclab zero_agent --task $TASK --num_envs 4 --max_steps 500 --visualizer kit` | options verified; not run |
| Run the env with **random** actions | `isaaclab random_agent --task $TASK --num_envs 4 --max_steps 500 --visualizer kit` | options verified; not run |
| Pick the physics backend (presets) | add `physics=newton_mjwarp` (no dashes) | not run |

Preset options for a task appear at the end of `--help` when `--task=...` is passed. The Spot config defines the
presets `isaacsim_physx` (default), `physx`, `newton_mjwarp` and `newton_kamino`. Only the default has been used here.

## 8. Packages and environment

| Goal | Command | Status |
|---|---|---|
| Activate the environment | `source .venv/bin/activate` | verified |
| Reinstall our package (after changing `pyproject.toml` metadata) | `uv pip install -e .` | verified |
| Install another package | `uv pip install <package>` | verified |
| Record exact versions | `uv pip freeze > requirements.lock` | not run |
| Python and Isaac Lab location | `python -c "import isaaclab; print(isaaclab.__file__)"` | verified |

**Never run:** `uv sync`, `uv lock`, `sudo`, `apt`, or anything that modifies `~/tools/nvidia/IsaacLab`.

## 9. Other `isaaclab` subcommands

Seen in `isaaclab --help` (not tried here): `train_multigpu`, `benchmark`, `microbenchmark`, `demo`,
`example`, `leapp` (export or deploy a policy), `teleop`. Policy export and sim-to-real are later plans.

## 10. Quick troubleshooting

| Symptom | Likely cause / what to check |
|---|---|
| `isaaclab: command not found` | the environment is not activated (`source .venv/bin/activate`) |
| Task not listed by `list_envs` | not activated, or `uv pip install -e .` was not run after changing `pyproject.toml` |
| `Could not find configuration ... Existing agent configuration entry points` | wrong `--agent` key; the message lists the valid keys |
| Play or resume picks the wrong folder | missing `--agent` (the log folder is chosen by the agent's `experiment_name`) |
| Checkpoint does not load (size mismatch) | the agent variant differs from the one the checkpoint was trained with |
| Thousands of `Failed to find rigid body at /World/collisions/groupN` lines | PhysX warnings seen on every run so far; training still completed |
| No window with `--visualizer kit` | check your display and drivers, and read the first error lines (not diagnosed here) |
