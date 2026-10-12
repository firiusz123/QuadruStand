# QuadruStand

Project regarding comparison of different network shapes and tuning of hyperparameters affecting the policy learning of quadrupeds trying to walk in the Isaac Lab environment.

- **Simulation:** Isaac Sim / Isaac Lab (direct RL workflow, trained via the `isaaclab` CLI)
- **HPO:** Optuna (one training run per trial, resumable study)
- **RL library:** rsl_rl (PPO)

## Repository layout

```
QuadruStand/
├── src/quadru_stand/               # Isaac Lab extension — the environment code
│   ├── config/                     #   extension.toml — Isaac Sim extension metadata
│   ├── assets/robots/              #   custom quadruped USD/URDF (skip if using built-in A1/Go2/Anymal)
│   ├── mdp/                        #   custom reward / observation / termination functions
│   └── tasks/direct/quadru_stand/
│       ├── __init__.py             #   gym.register entry point, e.g. "QuadruStand-Velocity-Flat-v0"
│       ├── quadru_stand_env_cfg.py #   scene, actions, observations, rewards, terminations, events
│       └── agents/                 #   rsl_rl PPO configs per network shape (small/medium/large), selected with --agent
├── scripts/
│   ├── hpo/                        #   Optuna: study setup, objective (launches one train run/trial), search space, config overrides, metric parsing
│   ├── analysis/                   #   compare network shapes, plot Optuna results, summarize runs
│   ├── eval/                       #   deterministic policy evaluation + video rendering
│   └── tools/                      #   export trained policies (JIT/ONNX)
├── configs/
│   ├── network_shapes/             #   architecture comparison matrix (YAML)
│   └── hpo/                        #   study settings: n_trials, sampler, pruner, metric, timeout
├── logs/                           #   TensorBoard + run artifacts                        [contents gitignored]
├── outputs/                        #   trained models / exported policies                 [contents gitignored]
│   ├── rsl_rl/<experiment>/<run>/  #   manual/experiment runs
│   └── hpo/<study>/trial_<n>/      #   per-trial training runs
├── hpo_studies/                    #   Optuna sqlite DB + trial reports                   [contents gitignored]
├── notebooks/                      #   scratch result analysis
├── docs/                           #   reward design + HPO methodology notes
└── results/                        #   final figures/tables for the writeup
```

Root-level files: `pyproject.toml` (uv-managed deps), `README.md`, `.gitignore`, `.flake8`, `.pre-commit-config.yaml`.

## Commands

```bash
uv sync                                              # install environment
uv run isaaclab list_envs                            # list registered tasks
uv run isaaclab train --rl_library rsl_rl \
    --task QuadruStand-Velocity-Flat-v0 \
    --agent rsl_rl_ppo_cfg_entry_point --headless    # train a single run
uv run python scripts/hpo/optimize.py                # launch Optuna study
```

