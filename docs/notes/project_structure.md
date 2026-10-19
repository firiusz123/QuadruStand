# How this project works (notes)

A reference for "how is this put together and why". It is written for someone new to Isaac Lab and to
Python packaging. Everything here was checked against this repository and the Isaac Lab source at
`~/tools/nvidia/IsaacLab` (release/3.0.0). Where something was not verified, it says so.

## 1. The 30-second overview

```
You type:  isaaclab train --rl_library rsl_rl --task QuadruStand-Velocity-Flat-Spot-v0 ...
             │
             ├─ the CLI imports every package that is registered under the entry-point group "isaaclab.tasks"
             │     └─ ours: quadru_stand.tasks  → gym.register("QuadruStand-Velocity-Flat-Spot-v0", ...)
             ├─ looks the task id up in gymnasium's registry → gets two config classes (env + agent)
             ├─ builds the environment (ManagerBasedRLEnv) from the env config
             └─ trains the policy with rsl_rl (PPO) and writes logs/rsl_rl/<experiment_name>/<timestamp>/
```

Three separate things have to be true for that to work, and they are explained below:

1. Python can import our code (the `src/` folder, the editable install).
2. Isaac Lab discovers our code (the entry point).
3. Our code describes a task (the registration and the config classes).

## 2. Repository layout

```
QuadruStand/
├── pyproject.toml            # package description + entry point (section 4)
├── requirements.lock         # exact package versions of the working environment (uv pip freeze)
├── .venv/                    # the project's private Python environment (not in git)
├── src/quadru_stand/         # our code; the only folder that Python imports from (section 5)
│   ├── __init__.py
│   └── tasks/
│       ├── __init__.py       # auto-imports sub-packages so every gym.register runs
│       └── locomotion/spot_flat/
│           ├── __init__.py       # gym.register(...)  ← the task is registered here
│           ├── flat_env_cfg.py   # QuadruStandSpotFlatEnvCfg: inherits Isaac Lab's Spot config
│           └── agents/rsl_rl_ppo_cfg.py   # baseline / small / medium / large PPO runner configs
├── tests/test_registration.py
├── docs/                     # plans/, notes/ (this file), network_input_output.md
├── logs/                     # training output (contents are gitignored)
├── scripts/  configs/  results/  notebooks/  hpo_studies/  outputs/   # placeholders for later plans
└── README.md
```

Files you will edit regularly: `flat_env_cfg.py` and `agents/rsl_rl_ppo_cfg.py`. Files you should
almost never touch: every `__init__.py`, `pyproject.toml`.

## 3. The virtual environment and how Isaac Lab lives inside it

`.venv/` is a private Python installation for this project, created with `uv venv`. Activating it
(`source .venv/bin/activate`) puts its `python`, `pip`/`uv pip` and the `isaaclab` command first on the PATH.

Important facts, all verified on this machine:

- **Isaac Lab's code is not inside `.venv`.** It is a source checkout at `~/tools/nvidia/IsaacLab`.
  The Isaac Lab installer (`isaaclab.sh -i`) installed each of its packages into `.venv` in **editable
  mode**: the venv holds only small link files that point to the checkout. For example
  `isaaclab.__file__` resolves to `~/tools/nvidia/IsaacLab/source/isaaclab/isaaclab/__init__.py`.
  Consequently: **do not move or delete the checkout, and do not modify it** (this project's rule).
- **Our package is installed the same way**, with `uv pip install -e .`. So Isaac Lab and quadru_stand are
  equal citizens of the same environment, which is why they can find each other.
- **Isaac Sim, PyTorch and rsl-rl-lib are real installed packages** in `.venv`, chosen by the Isaac Lab
  installer (see the table in `README.md`).
- **Never run `uv sync` here.** It would reconcile the environment with this repo's minimal
  `pyproject.toml` and could remove Isaac Lab. Add packages with `uv pip install <pkg>`.

## 4. Why `pyproject.toml` looks the way it does

TOML is a plain config-file format (`[section]` headers, `key = value`). `pyproject.toml` is the **standard
Python project file** (PEP 518 / 621). It is not specific to Isaac Lab or to uv. It only *declares*; it does
nothing until `uv pip install -e .` reads it.

```toml
[build-system]
requires = ["setuptools>=77"]          # which tool builds/installs this package
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]                        # setuptools-specific: "my packages are under src/"

[project]
name = "quadru-stand"
version = "0.1.0"
requires-python = ">=3.12,<3.13"       # matches the Python the Isaac Lab stack is installed with
dependencies = ["optuna>=4.0"]         # ONLY our own extra dependency (see below)

[project.entry-points."isaaclab.tasks"]
quadru_stand = "quadru_stand.tasks"    # the one line that makes Isaac Lab find our tasks
```

- **Why `dependencies` lists only `optuna`.** The Isaac Lab installer decides the versions of Isaac Sim,
  PyTorch and the rest. Listing them here would risk conflicts (the comment at the top of the file says the
  same). This file describes *our* package only.
- **`[tool.X]` sections** belong to tool X. `[tool.setuptools...]` is read by setuptools only. Isaac Lab's own
  project template uses a different build tool (`uv_build`) and `[tool.uv.build-backend] module-name`, which
  does the same job. There is no single "correct" file.
- **The entry-point table** is a standard feature. Only the group name `isaaclab.tasks` is Isaac Lab's choice.
- `setuptools>=77`: a minimum version for modern `pyproject.toml` metadata. I did not check why exactly 77.
- **You can edit this file by hand.** After changing *metadata* (name, dependencies, entry points) rerun
  `uv pip install -e .`. Changing source code needs no reinstall.

Three things that are easy to confuse:

| Thing | Role |
|---|---|
| `.venv/` | **where** packages get installed |
| `pyproject.toml` | **what** our package is |
| `uv pip install -e .` | the **link** that reads the second and installs it into the first |

`requirements.lock` is just a snapshot (`uv pip freeze`) of exact versions for reproducibility. It is not
used for installing here (`uv.lock` is not used in this project).

## 5. Why the `src/` folder

`where = ["src"]` means the importable packages live in `src/`, not in the repo root. What `uv pip install -e .`
actually does is write one file into the venv:

```
.venv/lib/python3.12/site-packages/__editable__.quadru_stand-0.1.0.pth   →  contains:  /…/QuadruStand/src
```

Python reads every `.pth` file at start-up and adds each line to `sys.path`, the list of folders searched for
`import`. That is how `import quadru_stand` works from anywhere, while the code stays in the repository
(editable: edits are live, no reinstall).

Why a `src/` folder instead of putting `quadru_stand/` in the repo root: the common reason is that the repo
root (with `tests/`, `scripts/`, `docs/`) is not accidentally importable, so tests always run against the
*installed* package and packaging mistakes show up early. For this project it is mostly convention, and
Isaac Lab's templates use it.

## 6. How Isaac Lab finds our code (entry points)

When `uv pip install -e .` ran, it also created
`.venv/lib/python3.12/site-packages/quadru_stand-0.1.0.dist-info/entry_points.txt`:

```
[isaaclab.tasks]
quadru_stand = quadru_stand.tasks
```

Every installed package has such a `*.dist-info` metadata folder in the same `site-packages`. The Isaac Lab
CLI does, in essence:

```python
for ep in importlib.metadata.entry_points(group="isaaclab.tasks"):
    ep.load()          # imports "quadru_stand.tasks"
```

`entry_points(...)` scans **all `.dist-info` folders of the active environment**, wherever the package code
lives. That is the entire link between `~/tools/nvidia/IsaacLab` (reader) and this repo (writer): a shared
environment. Neither points to the other's path. If you do not activate `.venv`, Isaac Lab cannot see our
task. (Isaac Lab's own tasks are imported directly in its code, so on this machine ours is the only row in that group.)

## 7. How the task gets registered

1. `import quadru_stand.tasks` runs `tasks/__init__.py`, which calls
   `import_packages(__name__, ["utils", ".mdp"])`.
2. That walker imports every sub-package (a folder with `__init__.py`) below it and recurses. Plain `.py`
   files (`flat_env_cfg.py`, `rsl_rl_ppo_cfg.py`) are **not** imported at this stage.
3. Importing `spot_flat/__init__.py` executes its `gym.register(...)` call. Importing runs the file, and that is
   the whole trick.
4. The skip list `["utils", ".mdp"]` is copied from Isaac Lab's template. Any sub-package whose dotted name
   contains those strings is skipped. It does nothing for us today (our `mdp/` is outside `tasks/`).
5. Empty `__init__.py` files exist only to mark folders as packages. The walker only sees folders that have one.

`__name__` is a variable Python sets in every module to its dotted import name (for example
`quadru_stand.tasks.locomotion.spot_flat`). `from . import agents` is a relative import: "from my own
package, import the sub-package `agents`".

### What `gym.register` stores

`gym.register` adds a **recipe** to gymnasium's global registry. It builds nothing. Our recipe:

```python
gym.register(
    id="QuadruStand-Velocity-Flat-Spot-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",       # environment class, same as upstream
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point":          f"{__name__}.flat_env_cfg:QuadruStandSpotFlatEnvCfg",
        "rsl_rl_cfg_entry_point":       f"{agents.__name__}.rsl_rl_ppo_cfg:QuadruStandSpotFlatPPORunnerCfg",
        "rsl_rl_small_cfg_entry_point": "...:QuadruStandSpotFlatSmallPPORunnerCfg",
        "rsl_rl_medium_cfg_entry_point": "...", "rsl_rl_large_cfg_entry_point": "...",
    },
)
```

- The values are **strings** of the form `"dotted.module.path:ClassName"`. They are resolved lazily with
  `importlib.import_module` + `getattr` only when the task is used. The reason: registration runs at CLI
  start-up for every task, and importing every config eagerly would be slow (Isaac Lab's own walker docstring
  says the same).
- `f"{agents.__name__}.rsl_rl_ppo_cfg:..."` is an f-string: `{...}` is replaced by the value, giving
  `quadru_stand.tasks.locomotion.spot_flat.agents.rsl_rl_ppo_cfg:...`.
- The key names are our choice but must end in `_cfg_entry_point`. `--agent <key>` looks up that key.
  Without `--agent`, the train script uses `rsl_rl_cfg_entry_point`.
- The upstream Spot task has no `default_agent` and no play variant, so none was added.

## 8. Config classes and inheritance

`QuadruStandSpotFlatEnvCfg(SpotFlatEnvCfg)` is almost empty. It inherits everything (scene, observations,
actions, commands, rewards, terminations, events, simulation settings) from Isaac Lab's Spot config. Its
only code is:

```python
def __post_init__(self):
    super().__post_init__()      # 1. parent sets decimation, sim.dt, robot, terrain ... FIRST
    # 2. our overrides go here, e.g.   self.rewards.gait.weight = 2.0
```

- **`@dataclass`**: a normal class where Python generates `__init__` (and a readable `repr`) from a list of
  fields such as `x: int = 0`. **`@configclass`** is Isaac Lab's wrapper around it that also deep-copies the
  fields per object (so one config cannot leak changes into another) and adds helpers like `to_dict()`
  (used to write `params/env.yaml`).
- **`__post_init__`** is the hook the generated `__init__` calls after the fields are filled. It plays the
  role of "the rest of the constructor".
- **`super().__post_init__()` must be called first.** `@configclass` does *not* call the parent's version
  for you. Without it, parent-computed values stay `MISSING` (verified: `decimation` and `episode_length_s`
  were missing, `sim.dt` fell back to a default).
- The call chain runs top-down through five classes (ours → `SpotFlatEnvCfg` → `LocomotionVelocityRoughEnvCfg`
  → `ManagerBasedRLEnvCfg` → `ManagerBasedEnvCfg`), so **our lines run last and win**.
- The agent configs follow the same idea: they subclass `SpotFlatPPORunnerCfg`, set `experiment_name` as a
  class attribute (plain string), and change `hidden_dims` of the nested `actor`/`critic` objects in
  `__post_init__`. Everything else (PPO settings, activation, steps per env) stays inherited.

## 9. What runs at training time

`gym.make(task, cfg=env_cfg)` builds one `ManagerBasedRLEnv`, vectorised over `--num_envs` robots
(every tensor has `num_envs` rows). The policy talks to it through `RslRlVecEnvWrapper`.

Each `env.step(action)` (`isaaclab/envs/manager_based_rl_env.py`):

1. `action_manager` turns the network output into joint targets.
2. Physics runs `decimation` (10) times with `sim.dt` (0.002 s), so one step is 0.02 s (50 Hz).
3. `termination_manager` checks for falls and timeouts.
4. `reward_manager` computes the weighted sum of reward terms.
5. Terminated environments are reset.
6. `observation_manager` builds the next observation.

"Manager-based" means exactly this: the config sections (`rewards`, `observations`, `actions`, `commands`,
`terminations`, `events`) each configure one manager. Overriding `self.rewards.gait.weight` changes what step 4
computes. Network input/output details are in `docs/network_input_output.md`.

## 10. Everyday commands

Run from the repo root.

```bash
source .venv/bin/activate

isaaclab list_envs | grep QuadruStand
python -m pytest tests/test_registration.py -q

# train (headless is the default; add --visualizer kit for a window)
isaaclab train --rl_library rsl_rl --task QuadruStand-Velocity-Flat-Spot-v0 --num_envs 512 --max_iterations 10
isaaclab train --rl_library rsl_rl --task QuadruStand-Velocity-Flat-Spot-v0 --agent rsl_rl_small_cfg_entry_point ...
isaaclab train --rl_library rsl_rl --help        # full option list (plain `isaaclab train --help` shows only --rl_library)
```

Logs go to `logs/rsl_rl/<experiment_name>/<timestamp>/`: `model_*.pt` checkpoints, `params/env.yaml` and
`params/agent.yaml` (the exact configs used), TensorBoard event files.

## 11. When to do what

| You want to... | Do |
|---|---|
| Change a reward, termination, command range | Edit `__post_init__` in `flat_env_cfg.py` (after `super()`) |
| Change network size / PPO settings | Edit `agents/rsl_rl_ppo_cfg.py` |
| Add a new agent variant | New subclass there + a new `rsl_rl_<name>_cfg_entry_point` key in `spot_flat/__init__.py` |
| Add a new task | New folder under `tasks/locomotion/` with an `__init__.py` containing `gym.register(...)` (the walker finds it, no other edits) |
| Add a Python dependency | `uv pip install <pkg>`, and add it to `dependencies` in `pyproject.toml` if the code needs it |
| Changed `pyproject.toml` metadata | `uv pip install -e .` again |
| Verify nothing broke | `python -m pytest tests/test_registration.py -q`, then a short smoke run |

## 12. Pitfalls

- **Forgetting `super().__post_init__()`**: parent values are missing or wrong (section 8).
- **Running outside `.venv`**: `isaaclab` or our task is not found (section 6).
- **`uv sync`**: can break the environment (section 3).
- **Modifying `~/tools/nvidia/IsaacLab`**: project rule: inherit, never edit.
- **Changing observations or actions**: the network input/output sizes change and old checkpoints stop loading.
- **Changing the action scale (0.2)**: needs retraining. The scale is part of what the network's output means.
- **Long runs**: the default is 20000 iterations. Always cap with `--max_iterations` unless you mean it.

## 13. Glossary

| Term | Meaning |
|---|---|
| module / package | a `.py` file / a folder with `__init__.py` |
| `sys.path` | the list of folders Python searches when you `import` |
| `site-packages` | the venv folder holding installed packages and their `.dist-info` metadata |
| editable install (`-e`) | install by linking to the source folder instead of copying it |
| `.pth` file | a text file in `site-packages`; each line is added to `sys.path` |
| entry point | a row `group: name = module` in a package's metadata; other programs can query a group |
| gym registry | gymnasium's dictionary from an id string to an environment recipe |
| env cfg / agent cfg | Isaac Lab config object describing the environment / the RL algorithm and network |
| manager | a component of `ManagerBasedRLEnv` (action, observation, reward, termination, command, event) |
| vectorised env | one env object simulating many robots at once; all tensors have a `num_envs` dimension |
| `decimation` | physics steps per policy step (10 here) |
| `configclass` | Isaac Lab's dataclass wrapper used for all configs |

## 14. Known loose ends

- The order of the 12 joints in observations/actions was not verified from the source (print
  `env.unwrapped.scene["robot"].joint_names` from a created environment to see it).
- The `--visualizer kit` window command was verified against the CLI help but not run on this machine.
