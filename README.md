# QuadruStand

Benchmark of neural-network architectures and hyperparameters for quadruped locomotion in Isaac Lab,
followed by a reward-shaping study in which the robot learns to stand upright on its front legs.
The long-term objective is sim-to-real transfer of the resulting policy to a physical Boston Dynamics Spot.

| | |
|---|---|
| Simulation | Isaac Sim / Isaac Lab (manager-based RL workflow, `isaaclab` CLI) |
| RL library | rsl_rl (PPO) |
| Hyperparameter optimisation | Optuna (one training run per trial, resumable study) |

## Status

Environment setup is complete. The first task, `QuadruStand-Velocity-Flat-Spot-v0`, is registered:
it inherits Isaac Lab's Spot flat-terrain velocity task unchanged, with four rsl_rl agent variants
that differ only in network size (baseline / small / medium / large). Reward changes, the benchmark
runner and the Optuna study are not implemented yet.

## Repository layout

```
QuadruStand/
├── src/quadru_stand/               # Isaac Lab extension (environment code)
│   ├── config/                     #   extension.toml (Isaac Sim extension metadata)
│   ├── assets/robots/              #   custom robot USD/URDF, if a built-in asset is not used
│   ├── mdp/                        #   custom reward / observation / termination terms
│   └── tasks/
│       ├── __init__.py             #   auto-imports sub-packages so every gym.register runs
│       └── locomotion/spot_flat/
│           ├── __init__.py         #   gym.register("QuadruStand-Velocity-Flat-Spot-v0", ...)
│           ├── flat_env_cfg.py     #   QuadruStandSpotFlatEnvCfg (inherits Isaac Lab's SpotFlatEnvCfg)
│           └── agents/
│               └── rsl_rl_ppo_cfg.py  # baseline / small / medium / large PPO runner configs
├── scripts/
│   ├── hpo/                        #   Optuna study, objective, search space, metric parsing
│   ├── analysis/                   #   network-shape comparison, Optuna plots, run summaries
│   ├── eval/                       #   deterministic policy evaluation and video rendering
│   └── tools/                      #   policy export (JIT / ONNX)
├── configs/
│   ├── network_shapes/             #   architecture comparison matrix (YAML)
│   └── hpo/                        #   study settings: trials, sampler, pruner, metric, timeout
├── tests/                          #   registration tests (pytest)
├── logs/  outputs/  hpo_studies/   #   run artefacts (contents gitignored)
├── notebooks/                      #   exploratory analysis
├── docs/                           #   reward design and HPO methodology notes
└── results/                        #   final figures and tables
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
```

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

## Usage

With the environment activated (`source .venv/bin/activate`):

```bash
isaaclab list_envs | grep QuadruStand                # confirm the task is registered
python -m pytest tests/test_registration.py -q       # registration tests

# single training run, baseline network [512, 256, 128] (default agent)

#to turn on the visulazation put alongside the command --viz flag 
isaaclab train --rl_library rsl_rl --task QuadruStand-Velocity-Flat-Spot-v0 

# other network sizes, selected with --agent
isaaclab train --rl_library rsl_rl --task QuadruStand-Velocity-Flat-Spot-v0  \
    --agent rsl_rl_small_cfg_entry_point             # [128, 128]
#   --agent rsl_rl_medium_cfg_entry_point            # [256, 128, 64]
#   --agent rsl_rl_large_cfg_entry_point             # [1024, 512, 256]

python scripts/hpo/optimize.py                       # Optuna study (not implemented yet)
```

Each agent variant logs to its own folder: `logs/rsl_rl/quadrustand_spot_flat_<variant>/<timestamp>/`.
