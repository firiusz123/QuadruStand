# Plan 01 — Register a QuadruStand Spot task and benchmark agent variants

## Goal

Create the project's first Isaac Lab task, `QuadruStand-Velocity-Flat-Spot-v0`, as a thin wrapper
around Isaac Lab's built-in Spot flat-terrain locomotion task, and register three additional rsl_rl
agent configurations that differ **only** in network size. Behaviour must be identical to the
built-in task at this stage; reward changes come in a later plan.

## Context (verified on this machine)

- Repository: `~/Projects/private/QuadruStand` (this repo). Python env: `.venv` in the repo root,
  built by the Isaac Lab installer. Activate with `source .venv/bin/activate`.
- Isaac Lab source checkout (release/3.0.0): `~/tools/nvidia/IsaacLab`. **Read-only. Do not modify.**
- Isaac Lab 3.0, Isaac Sim 6.1, PyTorch 2.11 + cu130, rsl-rl-lib 5.5.1.
- Reference task (already trains successfully here):
  - Task ID: `IsaacContrib-Velocity-Flat-Spot`
  - Workflow: manager-based (`isaaclab.envs:ManagerBasedRLEnv`)
  - Env config: `isaaclab_tasks.contrib.velocity.config.spot.flat_env_cfg:SpotFlatEnvCfg`
  - Source: `~/tools/nvidia/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/contrib/velocity/config/spot/`
    (`__init__.py`, `flat_env_cfg.py`, `agents/rsl_rl_ppo_cfg.py`, `mdp/`)
  - Baseline network (from a completed run's `params/agent.yaml`): actor and critic MLP
    `[512, 256, 128]`, ELU, `num_steps_per_env=24`, `experiment_name="spot_flat"`.
- Task discovery: the `isaaclab` CLI (`train`, `play`, `list_envs`) loads every installed package
  that declares an entry point in the group `isaaclab.tasks` (see
  `source/isaaclab/isaaclab/cli/__init__.py` and `cli/commands/list_envs.py`). Isaac Lab's own
  external-project template uses:
  ```toml
  [project.entry-points."isaaclab.tasks"]
  quadru_stand = "quadru_stand.tasks"
  ```
- Official templates to mirror for structure and style:
  `~/tools/nvidia/IsaacLab/tools/template/templates/` — in particular `tasks/__init__task`
  (gym.register pattern), `agents/rsl_rl_ppo_cfg`, `extension/__init__tasks`
  (`import_packages` auto-import) and `external/test_registration`.
- Current repo state: `src/quadru_stand/` contains only empty placeholder folders (`.gitkeep`),
  including `tasks/direct/quadru_stand/`. `pyproject.toml` declares the package
  (setuptools, `packages.find where = ["src"]`) and `optuna` only.

## Constraints

1. Do not modify anything under `~/tools/nvidia/IsaacLab`.
2. Do not run `uv sync`, `uv lock`, `pip install` against the system, or any `sudo`/`apt` command.
   The only allowed install command is `uv pip install -e .` (or `uv pip install pytest`) with
   `.venv` active.
3. **Inherit, do not copy** the Spot configuration (`class ...(SpotFlatEnvCfg)`), so that the
   difference from the reference task is explicit and upstream fixes still apply. Copy code only if
   inheritance is technically impossible, and state why.
4. Only short smoke-test training runs (≤ 10 iterations, ≤ 512 envs). Do not start long runs.
5. Before writing registration code, read the reference Spot `__init__.py` and
   `agents/rsl_rl_ppo_cfg.py` and mirror their kwargs (e.g. `rsl_rl_cfg_entry_point`, any
   `default_agent`, and any play-mode config class). Do not guess names.

## Target layout

```
src/quadru_stand/
├── __init__.py
├── tasks/
│   ├── __init__.py                      # auto-import sub-packages (import_packages pattern)
│   └── locomotion/
│       ├── __init__.py
│       └── spot_flat/
│           ├── __init__.py              # gym.register(...)
│           ├── flat_env_cfg.py          # QuadruStandSpotFlatEnvCfg(SpotFlatEnvCfg) [+ PLAY variant if upstream has one]
│           └── agents/
│               ├── __init__.py
│               └── rsl_rl_ppo_cfg.py    # baseline + small / medium / large variants
tests/
└── test_registration.py
```

Remove the empty `src/quadru_stand/tasks/direct/` placeholder tree. Keep `assets/`, `mdp/`, `config/`
placeholders for later plans.

## Implementation steps

1. **Read references.** Read the reference Spot `__init__.py`, `flat_env_cfg.py`,
   `agents/rsl_rl_ppo_cfg.py`, and the templates listed above. Note exact class names and
   registration kwargs.
2. **Environment config** (`flat_env_cfg.py`): `QuadruStandSpotFlatEnvCfg(SpotFlatEnvCfg)` with a
   `__post_init__` that calls `super().__post_init__()` and changes nothing yet (leave a comment
   marking where reward changes will go). Add a play variant only if the reference defines one.
3. **Agent configs** (`agents/rsl_rl_ppo_cfg.py`): subclass the reference Spot runner config.
   - Baseline: identical network `[512, 256, 128]`; `experiment_name = "quadrustand_spot_flat_baseline"`.
   - `Small`: actor and critic `[128, 128]`; `experiment_name = "quadrustand_spot_flat_small"`.
   - `Medium`: `[256, 128, 64]`; `experiment_name = "quadrustand_spot_flat_medium"`.
   - `Large`: `[1024, 512, 256]`; `experiment_name = "quadrustand_spot_flat_large"`.
   - Change only `hidden_dims` (actor and critic) and `experiment_name`. Everything else
     (PPO hyperparameters, activation, steps per env, iterations) must stay inherited.
4. **Registration** (`spot_flat/__init__.py`): `gym.register` with
   `id="QuadruStand-Velocity-Flat-Spot-v0"`, `entry_point="isaaclab.envs:ManagerBasedRLEnv"`,
   `disable_env_checker=True`, and kwargs:
   - `env_cfg_entry_point` → `QuadruStandSpotFlatEnvCfg`
   - `rsl_rl_cfg_entry_point` → baseline runner config (use the exact key name the reference uses)
   - `rsl_rl_small_cfg_entry_point`, `rsl_rl_medium_cfg_entry_point`, `rsl_rl_large_cfg_entry_point`
   - any other kwargs the reference sets (e.g. `default_agent`). Register a `-Play-v0` id only if the
     reference registers a play variant.
5. **Package wiring**: `tasks/__init__.py` auto-imports sub-packages (mirror
   `templates/extension/__init__tasks`). Add to `pyproject.toml`:
   ```toml
   [project.entry-points."isaaclab.tasks"]
   quadru_stand = "quadru_stand.tasks"
   ```
   Then reinstall so the entry point is registered: `uv pip install -e .`
6. **Test** (`tests/test_registration.py`): mirror `templates/external/test_registration`. Assert the
   task id exists, `env_cfg_entry_point` points to our class, and all four `rsl_rl_*` agent entry
   points resolve and have the expected `hidden_dims`.
7. **Docs**: update `README.md` — repository layout (manager-based, `tasks/locomotion/spot_flat/`),
   status line, and usage examples using the new task id and `--agent` variants. Remove the stale
   last line of `.gitignore` claiming `uv.lock` is committed.

## Acceptance criteria

Run from the repo root with `.venv` active:

1. `uv pip install -e .` succeeds.
2. `python -m pytest tests/test_registration.py -q` passes (install pytest with `uv pip install pytest`
   if missing).
3. `isaaclab list_envs | grep QuadruStand` shows `QuadruStand-Velocity-Flat-Spot-v0` with `rsl_rl`.
4. Smoke test, baseline:
   `isaaclab train --rl_library rsl_rl --task QuadruStand-Velocity-Flat-Spot-v0 --num_envs 512 --max_iterations 10`
   completes and writes `logs/rsl_rl/quadrustand_spot_flat_baseline/<timestamp>/`.
5. Smoke test, variant:
   same command with `--agent rsl_rl_small_cfg_entry_point` writes to
   `logs/rsl_rl/quadrustand_spot_flat_small/...`, and that run's `params/agent.yaml` shows
   `hidden_dims: [128, 128]` for actor and critic.
6. `params/env.yaml` of the baseline smoke run matches the reference task's config (same reward terms
   and weights), apart from run-specific fields such as `log_dir` and `seed`.

## Out of scope (later plans)

- Benchmark runner script (seeds × variants, result collection) — Plan 02.
- Front-leg balancing rewards, curriculum, terminations — Plan 03.
- Policy export / sim-to-real — later.

## Report back

Summarise: files created/changed, the exact registration kwargs used and why (with reference to the
upstream Spot `__init__.py`), test output, and the two smoke-test log paths. Flag anything that
deviated from this plan.
