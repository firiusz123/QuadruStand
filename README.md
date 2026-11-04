# QuadruStand

Benchmark of neural-network architectures and hyperparameters for quadruped locomotion in Isaac Lab,
followed by a reward-shaping study in which the robot learns to stand upright on its front legs.
The long-term objective is sim-to-real transfer of the resulting policy to a physical Boston Dynamics Spot.

| | |
|---|---|
| Simulation | Isaac Sim / Isaac Lab (manager-based RL workflow, `isaaclab` CLI) |
| RL library | rsl_rl (PPO) |
| Hyperparameter optimisation | Optuna (one training run per trial, resumable study; planned, not implemented yet) |

## Status

Environment setup is complete. The first task, `QuadruStand-Velocity-Flat-Spot-v0`, is registered: it inherits
Isaac Lab's Spot flat-terrain velocity task unchanged, with eight rsl_rl agent variants that differ only in network
shape: the baseline, four sizes (`extra_small`, `small`, `medium`, `large`) and three depths at a fixed parameter budget
(`deep_1`, `deep_2`, `deep_5`). The benchmark protocol is in `docs/lab_required/BENCHMARK.md`; training runs, a
learning-speed analysis and a policy-evaluation script exist (`scripts/`). The reward-shaping study (standing on the
front legs) and the Optuna study are not implemented yet.

## Documentation

Start with `docs/README.md`: it lists every document and says which one to read for what.

## Repository layout

```
QuadruStand/
├── src/quadru_stand/           # our Python package (installed in editable mode)
│   └── tasks/
│       ├── __init__.py             #   gym.register("QuadruStand-Velocity-Flat-Spot-v0", ...) and all --agent keys
│       ├── flat_env_cfg.py         #   QuadruStandSpotFlatEnvCfg (inherits Isaac Lab's SpotFlatEnvCfg)
│       └── rsl_rl_ppo_cfg.py       #   PPO runner configs: baseline, size variants, depth variants
├── scripts/                    # command-line tools, run from the repo root
│   ├── evaluate_policy.py          #   deterministic evaluation of trained policies -> results/*.csv
│   ├── learning_speed.py           #   learning speed of every run in logs/
│   ├── make_learning_speed_plots.py#   figures for the presentation
│   └── configure_pyrefly.sh        #   editor helper (see "Editor (Pyrefly)")
├── tests/                      # registration tests (pytest)
├── notebooks/                  # exploratory analysis of the evaluation results
├── docs/                       # documentation (start at docs/README.md)
├── logs/                       # training runs (contents gitignored)
└── results/                    # evaluation CSVs (gitignored)
```

## Environment

### Reference configuration

| Component | Version |
|---|---|
| OS / hardware | Ubuntu, NVIDIA GPU with a driver supporting CUDA 13 |
| Python | 3.12 (managed by uv) |
| Isaac Lab | `release/3.0.0` |
| Isaac Sim | 6.1.0.0 (pinned by Isaac Lab) |
| PyTorch | 2.11.0 + CUDA 13.0 (`cu130`) |
| rsl-rl-lib | 5.5.1 |

### Installation

Prerequisites: `git`, `uv`, `cmake`. The Isaac Lab installer runs `sudo apt-get install cmake
build-essential` if `cmake` is not found.

```bash
# 1. Isaac Lab source checkout (any location; it must remain in place, see "Design notes")
git clone https://github.com/isaac-sim/IsaacLab.git --branch release/3.0.0 ~/tools/nvidia/IsaacLab

# 2. Project virtual environment
cd QuadruStand
uv venv --python 3.12 --seed .venv
source .venv/bin/activate

# 3. Isaac Lab, Isaac Sim, PyTorch and rsl_rl (several GB; allow 10+ minutes)
~/tools/nvidia/IsaacLab/isaaclab.sh -i 'isaacsim,rl[rsl-rl]'

# 4. This project (editable) and Optuna
uv pip install -e .

# 5. Pyrefly (Cursor language server): expose Isaac Lab source paths
./scripts/configure_pyrefly.sh
```

Then reload the Cursor window (Command Palette → **Developer: Reload Window**) so Pyrefly rereads `.venv`.

### Verification

```bash
python -c "import isaaclab, isaacsim, torch; print(torch.__version__, torch.cuda.is_available())"
# expected output: 2.11.0+cu130 True
```

The first simulator launch unpacks Isaac Sim extensions (10+ minutes) and requires accepting the
NVIDIA Omniverse EULA.

### Design notes

- **Isaac Lab is installed from a source checkout, not from PyPI.** The packages published on PyPI
  are 3.0 pre-releases with dependency pins that conflict with Isaac Sim 6.x. The release branch
  and its installer are the supported route.
- **The installer is the source of truth for dependency versions.** It reads the pins in the
  checkout's `pyproject.toml` and installs matching PyTorch (cu130), Isaac Sim and Newton builds.
  This repository's `pyproject.toml` therefore declares only the project package and Optuna.
- **Editable links.** Isaac Lab packages in `.venv` point to the checkout. Do not move or delete it.
- **Do not run `uv sync` in this repository.** It would reconcile the environment to the minimal
  `pyproject.toml` and remove Isaac Lab. Add packages with `uv pip install`.
- **Reproducibility.** After a verified installation, record the exact package set with
  `uv pip freeze > requirements.lock` and commit it.

### Editor (Pyrefly)

Cursor uses **Pyrefly** as the Python language server. `[tool.pyrefly]` in `pyproject.toml`
points it at `.venv/bin/python`, which is enough for packages that actually live in
`.venv` (`torch`, `isaacsim`, `optuna`).

Isaac Lab is different: the installer registers it with setuptools *import-hook* `.pth`
files. Python can run those hooks; Pyrefly cannot, so imports like `isaaclab.utils` look
missing even though `python -c "import isaaclab"` works.

`scripts/configure_pyrefly.sh` reads the hook mappings already in `.venv` (so it
follows *this* machine's Isaac Lab checkout) and writes a path-based
`isaaclab_editable_paths.pth` next to them. Rerun the script after reinstalling Isaac Lab
or moving the checkout. The generated file stays in `.venv` and is not committed.

If the editor still reports `Cannot find module isaaclab...`, reload the window.

## Usage

With the environment activated (`source .venv/bin/activate`), from the repo root:

```bash
isaaclab list_envs | grep QuadruStand                # confirm the task is registered
python -m pytest tests/test_registration.py -q       # registration tests

# single training run, baseline network [512, 256, 128] (default agent)
# add --viz kit to open the simulator window while it trains
isaaclab train --rl_library rsl_rl --task QuadruStand-Velocity-Flat-Spot-v0

# other networks, selected with --agent <key>
isaaclab train --rl_library rsl_rl --task QuadruStand-Velocity-Flat-Spot-v0 \
    --agent rsl_rl_small_cfg_entry_point             # [128, 64, 32]
#   --agent rsl_rl_extra_small_cfg_entry_point       # [64, 32, 16]
#   --agent rsl_rl_medium_cfg_entry_point            # [256, 128, 64]
#   --agent rsl_rl_large_cfg_entry_point             # [1024, 512, 256]
#   --agent rsl_rl_deep_1_cfg_entry_point            # [3072]
#   --agent rsl_rl_deep_2_cfg_entry_point            # [400, 400]
#   --agent rsl_rl_deep_5_cfg_entry_point            # [208, 208, 208, 208, 208]

# analysis
python scripts/learning_speed.py                     # learning speed of every run in logs/
python scripts/evaluate_policy.py                    # evaluate trained policies (starts the simulator)
```

Each agent variant logs to its own folder: `logs/rsl_rl/quadrustand_spot_flat_<variant>/<timestamp>/`.
All commands, with the exact flags and what each name means: `docs/commands.md`.
