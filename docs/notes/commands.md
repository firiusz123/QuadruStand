# Command lookup table

All commands run from the repo root with the environment active:

```bash
source .venv/bin/activate
```

Status column: **verified** = checked against this install's `--help` output or source, or run on this machine.
**not run** = the flags are real, but I did not run that exact command here. Check the output yourself the first time.

This project has one task, `QuadruStand-Velocity-Flat-Spot-v0`, so every command below uses it literally and can be
copied as is.

### The names (read this first)

One variant has **four different names**, and they are easy to mix up. The command line only ever takes the
**agent key** (second column), never the variant label.

| Variant label (our word) | **Agent key** = what you pass to `--agent` | Config class (in `rsl_rl_ppo_cfg.py`) | `experiment_name` = log folder `logs/rsl_rl/...` | Network (actor and critic) |
|---|---|---|---|---|
| baseline | `rsl_rl_cfg_entry_point` (the default; **no** `baseline` in the key) | `QuadruStandSpotFlatPPORunnerCfg` | `quadrustand_spot_flat_baseline` | `[512, 256, 128]` |
| small | `rsl_rl_small_cfg_entry_point` | `QuadruStandSpotFlatSmallPPORunnerCfg` | `quadrustand_spot_flat_small` | `[128, 128]` |
| medium | `rsl_rl_medium_cfg_entry_point` | `QuadruStandSpotFlatMediumPPORunnerCfg` | `quadrustand_spot_flat_medium` | `[256, 128, 64]` |
| large | `rsl_rl_large_cfg_entry_point` | `QuadruStandSpotFlatLargePPORunnerCfg` | `quadrustand_spot_flat_large` | `[1024, 512, 256]` |

- The **agent key** is a key in the task's registration kwargs (`spot_flat/__init__.py`). `--agent small` would fail.
- The **config class** is what the key points to. The **experiment name** is set inside that class and decides the log
  folder. The command line never takes either of these.
- The **task id** is a different thing again: `--task QuadruStand-Velocity-Flat-Spot-v0` selects the environment,
  and one task id has all four keys.
- `--experiment_name <name>` also exists and overrides the log folder name for a run. Without it the folder comes
  from the config class.
- Each run folder records the key it was started with in `run.json` (`"metadata": {"agent": "..."}`). The experiment
  name and the key are therefore both visible after the fact (see section 6).

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
| Quick smoke test (10 iterations) | `isaaclab train --rl_library rsl_rl --task QuadruStand-Velocity-Flat-Spot-v0 --num_envs 512 --max_iterations 10` | verified |
| Same, small-network variant | add `--agent rsl_rl_small_cfg_entry_point` (the key, not the word "small") | verified |
| Full-size run | `isaaclab train --rl_library rsl_rl --task QuadruStand-Velocity-Flat-Spot-v0 --num_envs 4096` | not run |
| Fix the random seed | add `--seed 1` | not run |
| Tag the run folder | add `--run_name my_test` (suffix on the timestamp folder) | not run |
| Train with a window | add `--visualizer kit` (alias `--viz kit`), usually with fewer envs, e.g. `--num_envs 64` | not run |
| Draw only some of the envs | add `--max_visible_envs 16` | not run |
| Choose the GPU | add `--device cuda:0` | not run |

Notes:

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
| Resume from the newest run of this task and agent key (baseline key if `--agent` is omitted) | `isaaclab train --rl_library rsl_rl --task QuadruStand-Velocity-Flat-Spot-v0 --checkpoint latest --max_iterations 500` | not run |
| Resume from a specific file | `isaaclab train --rl_library rsl_rl --task QuadruStand-Velocity-Flat-Spot-v0 --checkpoint logs/rsl_rl/<experiment>/<timestamp>/model_300.pt --max_iterations 500` | not run |

Behaviour (from the rsl_rl source in `.venv`):

- `--max_iterations` then means **additional** iterations: training runs from the checkpoint's iteration up to that
  iteration plus `max_iterations`.
- A resumed run writes to a **new** timestamp folder. The old folder is not modified.
- `latest` and `best` only consider runs with the same library, task id and **agent key** (matched against the
  `run.json` of each run). If you trained with `--agent <key>`, repeat that key.
- The network shape in the checkpoint must match the config class the key selects. A checkpoint from the small
  variant will not load when `--agent` selects the large variant's key.

## 4. Play a trained policy (watch it)

| Goal | Command | Status |
|---|---|---|
| Play the latest run of the selected agent key (baseline if `--agent` is omitted) | `isaaclab play --rl_library rsl_rl --task QuadruStand-Velocity-Flat-Spot-v0 --num_envs 16 --visualizer kit` | not run |
| Same, small-network variant | add `--agent rsl_rl_small_cfg_entry_point` | not run |
| Play a specific checkpoint | add `--checkpoint logs/rsl_rl/<experiment>/<timestamp>/model_300.pt` | not run |
| Play the newest or the best checkpoint | add `--checkpoint latest` or `--checkpoint best` | not run |
| Run in real time | add `--real-time` | not run |
| Use the exact training env config | add `--train_env_cfg` (skips the play-mode overrides) | not run |
| Record video | add `--video --video_length 300` | not run |

Notes:

- With **no** `--checkpoint`, play loads the latest model of the latest run in the log folder named by the selected
  config's `experiment_name`. So for the small variant always pass `--agent rsl_rl_small_cfg_entry_point`, otherwise
  play looks in the baseline folder (`quadrustand_spot_flat_baseline`).
- `--task` is always needed (it selects the environment). `--agent` is needed only for the non-baseline keys.
- Play uses the policy's **mean** action (no exploration noise).
- I did not check what the "play-mode overrides" change. `--train_env_cfg` turns them off.

## 5. TensorBoard (training curves)

| Goal | Command | Status |
|---|---|---|
| All runs | `tensorboard --logdir logs/rsl_rl` | tensorboard 2.21.0 is installed; not run |
| One experiment | `tensorboard --logdir logs/rsl_rl/quadrustand_spot_flat_small` | not run |
| Different port | add `--port 6007` | not run |

Then open `http://localhost:6006` in a browser. Useful curves:

Tag names, read from a real event file of this project:

- `Train/mean_reward`, `Train/mean_episode_length` (overall progress; the `.../time` variants plot against wall-clock time)
- `Metrics/base_velocity/error_vel_xy`, `Metrics/base_velocity/error_vel_yaw`: velocity tracking error vs the command
- `Metrics/success_rate`
- `Episode_Reward/<term>`: the contribution of each reward term (`gait`, `air_time`, `base_orientation`, ...)
- `Episode_Termination/body_contact`, `.../time_out`, `.../terrain_out_of_bounds`: why episodes ended (body contact = a fall)
- `Curriculum/terrain_levels`
- `Loss/value`, `Loss/surrogate`, `Loss/entropy`, `Loss/learning_rate`, `Policy/mean_std`: PPO internals
- `Perf/total_fps`, `Perf/collection_time`, `Perf/learning_time`: speed

The same numbers also print in the terminal every iteration.

## 6. Inspect a run

Every run folder `logs/rsl_rl/<experiment>/<timestamp>/` contains `model_<iter>.pt`, `params/env.yaml`,
`params/agent.yaml`, `git/` diffs, `run.json` and the TensorBoard event file.

| Goal | Command | Status |
|---|---|---|
| List all runs | `find logs -maxdepth 3 -mindepth 2 -type d \| sort` | verified |
| Which `--task` and `--agent` key a run was started with | `cat logs/rsl_rl/<experiment>/<timestamp>/run.json` (fields `task` and `metadata.agent`) | verified |
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
| Run the env with **zero** actions (robot holds the default pose) | `isaaclab zero_agent --task QuadruStand-Velocity-Flat-Spot-v0 --num_envs 4 --max_steps 500 --visualizer kit` | options verified; not run |
| Run the env with **random** actions | `isaaclab random_agent --task QuadruStand-Velocity-Flat-Spot-v0 --num_envs 4 --max_steps 500 --visualizer kit` | options verified; not run |
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
| Play or resume picks the wrong folder | missing `--agent` key (the log folder comes from the selected config's `experiment_name`) |
| Checkpoint does not load (size mismatch) | the `--agent` key differs from the one the checkpoint was trained with (see its `run.json`) |
| `--agent small` is rejected | the value must be the full key, `rsl_rl_small_cfg_entry_point` |
| Thousands of `Failed to find rigid body at /World/collisions/groupN` lines | PhysX warnings seen on every run so far; training still completed |
| No window with `--visualizer kit` | check your display and drivers, and read the first error lines (not diagnosed here) |
