# Network-shape benchmark: protocol

Applies to `QuadruStand-Velocity-Flat-Spot-v0` (Spot, flat terrain, velocity tracking, rsl_rl PPO).
Metric definitions extend `docs/METRICS.md`. Network input/output layout: `docs/network_input_output.md`.

## 1. Questions

| ID | Question | Experiment | Scope |
|---|---|---|---|
| Q1 | How does network **size** affect final performance and learning speed? | A | Minimum |
| Q2 | At a fixed parameter budget, does **depth vs. width** matter? | B | Maximum |
| Q3 | Can a **small actor** with a large critic match the baseline? (Relevant for on-robot inference.) | C | Maximum |

- **Minimum plan:** Experiment A only.
- **Maximum plan:** Experiments A, B and C.

## 2. Fixed conditions

Everything except the network shape (and the seed) is identical across runs. Any deviation invalidates the comparison.

| Item | Value |
|---|---|
| Task | `QuadruStand-Velocity-Flat-Spot-v0` |
| Physics backend | `isaacsim_physx` (the CLI default; shown in the run header). Never mix backends. |
| Parallel environments | 4096 (task default; do not pass `--num_envs` for benchmark runs) |
| Training iterations | `N`, fixed once in Stage 0 (expected 1000–1500) |
| PPO hyperparameters | Inherited from upstream `SpotFlatPPORunnerCfg`, unchanged |
| Activation | ELU (upstream) |
| Software | Isaac Lab `release/3.0.0`, Isaac Sim 6.1.0.0, torch 2.11.0+cu130; package set in `requirements.lock` |
| Hardware | The two identical laptops. Record the machine for each run (wall-clock time is only comparable on the same machine). |

## 3. Variants

The parameter counts are for the actor (48 inputs, 12 outputs). The critic is the same size minus about 1–7k, since it has a single output.

| Key (`--agent rsl_rl_<key>_cfg_entry_point`) | Actor `hidden_dims` | Critic `hidden_dims` | Actor params | Exp. | Status |
|---|---|---|---|---|---|
| baseline (default, no `--agent`) | [512, 256, 128] | same | 191k | A, B | registered |
| `extra_small` | [64, 32, 16] | same | 6k | A | registered (see §10, bug) |
| `small` | [128, 64, 32] | same | 17k | A | registered |
| `medium` | [256, 128, 64] | same | 54k | A | registered |
| `large` | [1024, 512, 256] | same | 709k | A | registered |
| `depth_1` | [3072] | same | 187k | B | to add |
| `depth_2` | [400, 400] | same | 185k | B | to add |
| `depth_5` | [208, 208, 208, 208, 208] | same | 187k | B | to add |
| `asym` | [128, 64, 32] | [512, 256, 128] | 17k | C | to add |

Rationale:
- **Experiment A** scales Isaac Lab's tapering shape by roughly 3–4× in parameters per step, i.e. on a logarithmic scale.
- Isaac Lab's own flat-terrain locomotion tasks (ANYmal-D, Go2) use [128, 128, 128], about 41k parameters. The [512, 256, 128] baseline is their rough-terrain size, so it is likely larger than flat walking needs.
- **Experiment B** keeps the baseline's parameter budget (about 190k) and varies only the depth.
- **Experiment C** follows Andrychowicz et al. (ICLR 2021): a wide value network does not hurt, while the policy width should be tuned.

## 4. Seeds

**What a seed is.** The seed initialises every random number generator used in a run:
- the network's initial weights
- the robots' initial states and the reset poses
- the randomisation (friction, added mass, pushes)
- the velocity commands
- PPO's action sampling and mini-batch shuffling

Two runs that differ only in the seed are equally valid samples of "how this network learns". The spread between them is the noise floor against which network differences are judged.

**Current behaviour.** The upstream runner config sets `seed: 42`, which is visible in every run's `params/agent.yaml`. Without `--seed`, every run uses 42. Running the same command three times therefore does **not** give three independent samples: they are near-identical, differing only through GPU physics non-determinism.

**Rules:**
- Always pass the seed explicitly: `--seed 1`, `--seed 2`, `--seed 3` (and `4`, `5` if needed).
- Use the **same seed list for every variant**. Seed 1 of `medium` is then paired with seed 1 of the baseline.
- Never use the default 42 for benchmark runs, so that benchmark runs cannot be confused with earlier test runs.
- Read the seed for each run from its `params/agent.yaml` (`seed:`). Do not rely on folder names, because the run folder is named only by timestamp.
- Identical seeds on GPU are **not** bit-for-bit reproducible. Expect small differences on a rerun; this is why seeds are compared statistically, not run-for-run.

Evaluation uses its own fixed **evaluation seed** (§6.2). It is identical for every policy, so that all policies are tested on the same commands and disturbances.

## 5. Procedure

### Stage 0: fix the iteration budget N

1. Train the baseline with seed 1 for 1500 iterations.
2. In TensorBoard, find the iteration at which `Train/mean_reward` (smoothed) reaches its plateau: the point where the last 20% of the run adds less than about 5%.
3. Set `N` to that iteration, rounded up to a multiple of 50.

Checkpoints are saved every 50 iterations, so this run's `model_<N>.pt` serves as baseline seed 1 in Stage 1, and no rerun is needed.

### Stage 1: measure the noise floor

Train the baseline with seeds 2 and 3 for `N` iterations. Together with Stage 0, this gives 3 baseline seeds.

The **noise band** of a metric is the min–max range of these 3 baseline runs.

### Stage 2: screening (1 seed per variant)

Train each variant of the chosen scope with seed 1 for `N` iterations.

### Stage 3: confirmation (only where needed)

For each variant, compare its seed-1 primary metric (§6.3) with the baseline noise band:

| Seed-1 result | Action |
|---|---|
| Outside the noise band | Difference accepted as real. No more seeds required. |
| Inside the noise band | Inconclusive. Train seeds 2 and 3, then compare the 3-seed mean and range with the baseline's. |

This is a screening rule for a limited compute budget, not a formal significance test. Report it as such.

### Stage 4: evaluation

Run the evaluation (§6.2) on the final checkpoint `model_<N>.pt` of every run.

### Stage 5: analysis

- Aggregate per variant: the mean and min–max over seeds.
- Plot the training curves with the seed range shaded.
- Plot each metric against parameter count (Exp. A) and against depth (Exp. B).

## 6. Metrics

### 6.1 Training-curve metrics (from TensorBoard logs, no extra code)

| Metric | Source tag | Use |
|---|---|---|
| Mean reward | `Train/mean_reward` | Learning curve; final value = mean of the last 5% of iterations |
| Mean episode length | `Train/mean_episode_length` | Survival during training |
| Velocity error (training) | `Metrics/base_velocity/error_vel_xy`, `.../error_vel_yaw` | Tracking during training (with exploration noise) |
| Fall share (training) | `Episode_Termination/body_contact` vs `.../time_out` | Stability during training |
| **Learning speed** | derived | First iteration at which the smoothed mean reward reaches 90% of the variant's own final value. Also report the iteration at which it reaches 90% of the **baseline's** final value; "never" is a valid result. |
| Wall-clock time | run duration | Training cost (compare only on the same machine) |

### 6.2 Evaluation metrics (`docs/METRICS.md`, computed by an evaluation script)

The training metrics include exploration noise. The definitive comparison uses a separate evaluation of each trained policy under identical conditions:
- **Deterministic policy:** the mean action, with no sampling.
- **Fixed evaluation seed** (for example 1000), the same for every policy.
- Fixed number of environments (for example 1024) and steps (for example 3000 steps = 60 s).
- Same command ranges and randomisation as in training.

| # | Metric | Definition |
|---|---|---|
| 1 | Linear velocity error | ‖cmd_xy − base_lin_vel_xy‖ in m/s, averaged over steps |
| 2 | Yaw rate error | ‖cmd_yaw − base_ang_vel_z‖ in rad/s, averaged over steps |
| 3 | Tilt | arccos(−g_z) from `projected_gravity`; report the mean and the max |
| 4 | Fall rate | Fraction of episodes ending in `body_contact` (vs. `time_out`) |
| 5 | Tracking success rate | Fraction of steps with velocity error < 0.2 m/s **and** yaw error < 0.2 rad/s |

Exclude steps of fallen robots from metrics 1, 2, 3 and 5, and always report metric 4 next to them.

### 6.3 Primary metric and cost metrics

- **Primary metric (for the Stage 3 rule):** tracking success rate (#5). It is fixed in advance to avoid choosing the metric after seeing the results.
- **Gate:** a variant with a clearly higher fall rate than the baseline is reported as worse, regardless of tracking.
- **Cost:** report parameter count (actor, critic), wall-clock training time, and actor inference time per step on CPU (an indication of the on-robot cost).

## 7. Run matrix and time budget

Measured on these laptops: about 1.5 s per iteration plus about 2 min start-up. At N = 1000, one run takes about 27 min; at N = 1500, about 40 min.

| Stage | Minimum plan (A) | Maximum plan (A + B + C) |
|---|---|---|
| 0 + 1: baseline × 3 seeds | 3 runs | 3 runs |
| 2: screening, seed 1 | 4 (`extra_small`, `small`, `medium`, `large`) | 8 (+ `depth_1`, `depth_2`, `depth_5`, `asym`) |
| 3: confirmation (estimate) | ~2 variants × 2 seeds = 4 | ~4 variants × 2 seeds = 8 |
| **Total runs** | **~11** | **~19** |
| **Time at N = 1000** | **~5 h** | **~8.5 h** |
| Per laptop (split in two) | ~2.5 h | ~4.5 h |

Evaluation adds a few minutes per run.

**Split between the two laptops.** Split by seed, not by variant. For example, laptop 1 runs every seed-1 run and laptop 2 runs seeds 2 and 3. Any machine difference then affects all variants equally.

## 8. Commands

From the repo root, with `.venv` active:

```bash
# Stage 0 — baseline, seed 1, long run
isaaclab train --rl_library rsl_rl --task QuadruStand-Velocity-Flat-Spot-v0 --seed 1 --max_iterations 1500

# Stage 1 — baseline, seeds 2 and 3
isaaclab train --rl_library rsl_rl --task QuadruStand-Velocity-Flat-Spot-v0 --seed 2 --max_iterations N
isaaclab train --rl_library rsl_rl --task QuadruStand-Velocity-Flat-Spot-v0 --seed 3 --max_iterations N

# Stage 2 — one variant, seed 1
isaaclab train --rl_library rsl_rl --task QuadruStand-Velocity-Flat-Spot-v0 \
    --agent rsl_rl_medium_cfg_entry_point --seed 1 --max_iterations N
```

Logs land in `logs/rsl_rl/quadrustand_spot_flat_<variant>/<timestamp>/`. The seed is in `params/agent.yaml`.

## 9. Record keeping

- Log every benchmark run in `results/benchmark_runs.csv` with these columns: `variant, seed, N, machine, run_dir, started, duration_min, status, notes`.
- Mark failed or interrupted runs as such and rerun them with the same seed. Never silently drop them.
- Commit `results/` (CSV, final figures and tables), not `logs/`.
- Do not change code, configs or the Isaac Lab checkout during a benchmark. Each run's `git/*.diff` records any uncommitted change, so check it is empty.

## 10. Open items before the benchmark starts

1. **Bug in `extra_small`.** In `agents/rsl_rl_ppo_cfg.py`, `QuadruStandSpotFlatExtraSmallPPORunnerCfg.__post_init__` calls `super().__post_init()`, which is missing the trailing underscores. Python treats the name as private and raises `AttributeError` when the config is created. Fix it to `super().__post_init__()`, and extend `tests/test_registration.py` to cover this variant.
2. **README is stale.** Its usage section still lists `small` as [128, 128]; the registered value is [128, 64, 32]. It also does not mention `extra_small` or `--seed`.
3. **Experiments B and C** (maximum plan only): add the variants `depth_1`, `depth_2`, `depth_5` and `asym` (§3) and register them.
4. **Evaluation script.** §6.2 requires `scripts/eval/evaluate_policy.py`, which does not exist yet.
5. **Analysis script.** Stage 5 requires `scripts/analysis/compare_runs.py` to read the TensorBoard logs and evaluation outputs and produce the tables and figures.
6. **Scope alignment.** `docs/lab_required/project_plan.md` describes **rough** terrain and Hydra overrides for model size and environment count. This benchmark uses **flat** terrain (the only Spot locomotion task in Isaac Lab 3.0) and registered agent variants, with a fixed environment count. Agree with the supervisor, or update the plan, before Laboratory 2.

## References

- Andrychowicz et al., *What Matters in On-Policy Reinforcement Learning? A Large-Scale Empirical Study*, ICLR 2021. https://arxiv.org/abs/2006.05990
- Isaac Lab `release/3.0.0` locomotion agent configs (ANYmal-D, Go2): `isaaclab_tasks/core/velocity/config/*/agents/rsl_rl_ppo_cfg.py`
- NVIDIA, *Closing the Sim-to-Real Gap: Training Spot Quadruped Locomotion with NVIDIA Isaac Lab* (developer blog).
