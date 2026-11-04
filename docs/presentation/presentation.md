

# Network-shape benchmark for quadruped locomotion in Isaac Lab

**Boston Dynamics Spot, flat terrain, velocity tracking, PPO (rsl_rl)**

- Eryk Pawełek, Michał Domański

---

## 1. Goal and research questions

**Goal:** recommend a network size for a quadruped locomotion policy trained with PPO.

| ID | Question | Experiment |
|---|---|---|
| Q1 | How does network **size** affect final performance and learning speed? | A (size) |
| Q2 | At a fixed parameter budget, does **depth vs. width** matter? | B (depth) |


---

## 2. Setup

| Item | Value |
|---|---|
| Simulator / framework | Isaac Sim 6.1, Isaac Lab 3.0 (manager-based workflow) |
| Task | `QuadruStand-Velocity-Flat-Spot-v0` (Spot, flat terrain, velocity-command tracking) |
| Algorithm | PPO, rsl_rl 5.5.1 (upstream hyperparameters, unchanged) |
| Parallel environments | 4096 |
| Training length | N = 1500 |
| Activation | ELU |
| Hardware | RTX5090 laptop variant |

![Spot in the simulator](images/Spot_title.png)


---

## 3. What the network sees and does

![placeholder](images/network_input_output.png)

- **Input, 48 numbers:** body linear velocity (3), body angular velocity (3), projected gravity (3),
  velocity command (3), joint positions (12), joint velocities (12), previous action (12).
- **Output, 12 numbers:** one per leg joint. Target joint position = default pose + 0.2 × output, followed by a PD actuator.
- **Rate:** one decision every 0.02 s (50 Hz; 10 physics steps of 2 ms).
- Actor and critic are separate MLPs with the same hidden layers; only the actor runs on the robot.

---

## 4. Compared variants

The parameter counts are for the actor (48 inputs, 12 outputs); the critic has the same hidden layers.

**Experiment A: size** (tapering shape)

| Variant | Hidden layers | Actor params |
|---|---|---|
| extra_small | [64, 32, 16] | 6k |
| small | [128, 64, 32] | 17k |
| medium | [256, 128, 64] | 54k |
| baseline | [512, 256, 128] | 191k |
| large | [1024, 512, 256] | 709k |

**Experiment B: depth at a fixed budget (about 190k parameters)**

| Variant | Hidden layers | Actor params |
|---|---|---|
| deep_1 | [3072] | 187k |
| deep_2 | [400, 400] | 185k |
| baseline (3 layers) | [512, 256, 128] | 191k |
| deep_5 | [208, 208, 208, 208, 208] | 187k |

Only `hidden_dims` differs between runs. Everything else is inherited from Isaac Lab's Spot configuration.

---

## 5. Protocol

- **Same conditions for every run:** same task, 4096 envs, same PPO settings, same number of iterations.
- **Seeds:** baseline trained with seeds 1, 2, 3. Each other variant with seed 1 (screening).
- **Noise band:** the min–max range of a metric over the 3 baseline seeds. A variant counts as different from the
  baseline only if it falls **outside** this band.
- This is a screening rule for a limited compute budget, **not** a formal significance test.

[0.9 smoothing factor] ![Protocol overview](images/baseline_mean_reward.png)

---

## 6. Metrics

**From training logs (TensorBoard):**
- Mean reward (`Train/mean_reward`)
- Velocity tracking error (`Metrics/base_velocity/error_vel_xy`, `.../error_vel_yaw`)
- Fall share (`Episode_Termination/body_contact`)
- **Learning speed:** first iteration where the smoothed reward reaches 90% of a target value
  - target 1: the run's own final reward
  - target 2: the baseline's final reward
  - final reward = mean of the last 5% of iterations; smoothing = 20-iteration moving average

**From a separate deterministic evaluation (planned):** linear velocity error, yaw rate error, tilt, fall rate,
tracking success rate. The primary metric is the **tracking success rate**.


---

## 7. Results: learning curves

 ![Experiment A: mean reward](images/Experiment_A_mean_reward.png)

![Experiment B: mean reward](images/Experiment_B_mean_reward.png)

---

## 8. Results: learning speed (PRELIMINARY, 1 seed per variant)

Iterations until the smoothed reward reaches 90% of the target; **beyond noise** = distance outside the baseline noise
band (0 = inside the band, `+` = slower than the slowest baseline seed, `-` = faster than the fastest).
Source: `scripts/learning_speed.py`, window 20.

Noise band: `own_90` 275 .. 472, `baseline_90` 274 .. 438 (3 baseline seeds).

| Variant | Seed | Final reward | to 90% of own final | beyond noise | to 90% of baseline final | beyond noise |
|---|---|---|---|---|---|---|
| baseline | 1 | 264.0 | 397 | 0 | 438 | 0 |
| baseline | 2 | 269.6 | 275 | 0 | 274 | 0 |
| baseline | 3 | 274.6 | 472 | 0 | 388 | 0 |
| extra_small | 1 | 257.9 | 468 | 0 | 806 | +368 |
| small | 1 | 268.6 | 394 | 0 | 395 | 0 |
| medium | 1 | 277.6 | 317 | 0 | 272 | -2 |
| large | 1 | 270.5 | 532 | +60 | 529 | +91 |
| deep_1 | 1 | 274.8 | 829 | +357 | 757 | +319 |
| deep_2 | 1 | 244.0 | 536 | +64 | 1162 | +724 |
| deep_5 | 1 | 269.0 | 312 | 0 | 312 | 0 |

---

## 8a. Learning speed vs network size (Experiment A)

Lower = faster. Grey band = range of the 3 baseline seeds. Only points **outside** the band are labelled.

![Learning speed vs parameters](images/learning_speed_vs_params.png)

---

## 8b. Learning speed vs depth at a fixed budget (Experiment B)

![Learning speed vs depth](images/learning_speed_vs_depth.png)

---

## 8c. Where each curve reaches the threshold

The marker on each curve is the iteration at which its smoothed reward first reaches 90% of the baseline's final reward
(the numbers in the legend are the table's last "to 90% of baseline final" column).

![Experiment A: learning speed curves](images/learning_speed_curves_expA.png)

![Experiment B: learning speed curves](images/learning_speed_curves_expB.png)

*Figures are generated by `scripts/make_learning_speed_plots.py`; rerun it after new runs.*

---

## 9. Results: evaluation of the trained policies
### 9.1. Inference script
The Evaluation of inference was carried out by running all the model checkpoints that were trained and recording following parameters:
1. **Network variant**
2. **Checkpoint of network**
3. **Seed**
4. **Number of environments running in parallel**
5. **Absolute linear velocity error**
6. **Yaw error**
7. **Mean tilt of robot**
8. **Maximal tilt of robot**
9. **Fall rate of robot across the parallel environments**

The script used to recording those parameters take into the count:
- number of parallel environments
- Amount of physical steps limit
- seed
- and few more to run the isaaclab hydra properly

The script was run with:
1. **Hard limit of 1000 physical steps**
2. **16 parallel environments**

### 9.2. Resulting data
From the aforementioned parameters the most interesting are: Linear velocity error, yaw rate error and fall rate.
The Graph below shows the mean value of Linear velocity error , mean yaw rate error and mean fall rate. It is the mean value because 16 environments are running parallel. Also the plots are limited to half of the physical steps due to the robots starting the simulation by spawning again and thus introduce noise from starting the tasks again.
![Protocol overview](images/error_curves.png)
At the figure it is clearly visible that besides the diffrence in the Linear velocity there is not much of a diffrence in models in **Yaw rate** and **fall rate**. At first the fall rate might seem like a simple bug because it is just a ssingle horizontal line, but fear not. Simply all the models were trained enough not to fall. Another key factor occuring from the plot is that the key diffrence in them is the Linear Velocity.


Following scatter plot shows the **mediana of Yaw rate error** at Y axis and **mediana of Linear Velocity error** at X axis per each of the models. **So the closer the model is to point (0,0) the better it get**
![Protocol overview](images/linear_yaw_error_scatter.png)

Thus for ease of assesing which model is easier the following succes score function is constructed:
### Evaluation Metric Formulation

1. **Composite Error per Run:**
   $$f(x_i, y_i) = 10x_i + 5y_i$$

2. **Maximum Observed Error:**
   $$e_{\max} = \max_{j \in \{1, \dots, N\}} f(x_j, y_j)$$

3. **Inverted Success Score:**
   $$S_i = (e_{\max} + 2) - f(x_i, y_i)$$

**Combined Form:**
$$S(x_i, y_i) = \left( \max_{j=1}^{N} (10x_j + 5y_j) + 2 \right) - (10x_i + 5y_i)$$

The weights for Linear velocity error are twice the weights for yaw rate error due to the importance of the Linear velocity error. The scores are also reversed so it is easier to check for best model as the one with most score. Below is the Bar plot with the Succes metric calculated per each of the models.

![Protocol overview](images/succes_comparison.png)

---

## 10. Conclusions

*Fill in after the final results. Each conclusion should name its evidence and its caveat.*


**Q1: network size**
- Finding: Network size matters only on the extrimes. 
Small -> to small capacity
Large -> hard to optimize

Final recomendation would be to allways start wit the medium size network.

**Q2: depth at a fixed budget**
- Finding: At fixed parameter budget, deeper networks learn faster. 
The mean reward doesn't follow that rule.

**Q3: Defining reward for agent doesnt always lead to expected result**
- even with defining many rewards for agent and thus softly constraining it to certain learning direction. One might susspect that with linear velocity error at some point it will start walking or running as it is fastest way of locomotion for such robot right? **WRONG** it started jumping.


---

## 11. Limitations

- Flat terrain only (the only Spot locomotion task in Isaac Lab 3.0)
- Most variants have a **single seed**; conclusions rest on the baseline noise band from 3 seeds.
- Training metrics include exploration noise; the definitive comparison is the deterministic evaluation.
- Simulation only; no hardware test.
- Wall-clock times are comparable only on the same machine.

---
