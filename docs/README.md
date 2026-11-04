# Documentation index

Which document to read for what. The repository root `README.md` covers installation and the first commands.

## Working with the project

| Document | Read it when you want to... |
|---|---|
| [commands.md](commands.md) | look up a command: train, resume, play, TensorBoard, evaluate, analyse; also the naming of tasks and `--agent` keys |
| [project_structure.md](project_structure.md) | understand how the repo, the `pyproject.toml`, the virtual environment, Isaac Lab and the task registration fit together |

## Course deliverables and benchmark definition (`lab_required/`)

| Document | Content |
|---|---|
| [project_plan.md](lab_required/project_plan.md) | the project plan and the division of work between the two students |
| [lab1_st2.md](lab_required/lab1_st2.md) | Laboratory 1, student 2: the observed states and the reward terms with their weights |
| [network_input_output.md](lab_required/network_input_output.md) | exactly what the network receives (48 numbers) and outputs (12 numbers) |
| [BENCHMARK.md](lab_required/BENCHMARK.md) | the network-shape benchmark protocol: questions, variants, seeds, stages, metrics, status |
| [METRICS.md](lab_required/METRICS.md) | the evaluation metrics |

## Plans and presentation

| Path | Content |
|---|---|
| [plans/](plans/) | step-by-step implementation plans; `01_spot_task_registration.md` is completed (historical) |
| [presentation/presentation.md](presentation/presentation.md) | draft of the results presentation; figures are in `presentation/images/` |

## Where the code and data are

| Path | Content |
|---|---|
| `src/quadru_stand/tasks/` | task registration, environment config, PPO configs of all variants |
| `scripts/` | `evaluate_policy.py`, `learning_speed.py`, `make_learning_speed_plots.py`, `configure_pyrefly.sh` |
| `tests/` | registration tests |
| `notebooks/` | analysis of the evaluation CSVs |
| `logs/` | training runs (not in git) |
| `results/` | evaluation CSVs (not in git) |

## Suggested reading order for a newcomer

1. `README.md` (setup), then [project_structure.md](project_structure.md) (how it fits together).
2. [network_input_output.md](lab_required/network_input_output.md) (what the policy sees and does).
3. [BENCHMARK.md](lab_required/BENCHMARK.md) (what is being measured and why).
4. [commands.md](commands.md) whenever you need a command.
