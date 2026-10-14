# Project Plan

**Students:**
- Eryk Pawełek
- Michał Domański

---

## Task Planning

### 1. Task Definition
The engineering objective is to implement and evaluate a reinforcement learning policy for motion control of a quadruped in generated rough terrain.

### 2. Data / System Representation and Preparation
In RL, data is provided in real time via parallel simulation from simulation software.

### 3. Input Representation / Features
The model receives:
- Joint positions
- Joint velocities
- Gravity projection
- Velocity commands (desired forward/lateral velocity and yaw)
- Previous actions
- Angular velocities (IMU gyroscope readings)

### 4. Model(s) / Solution Method
The solution utilizes the **Proximal Policy Optimization (PPO)** algorithm. The baseline approach will be a standard, lightweight **Multi-Layer Perceptron (MLP)** network with an optimizing framework.

### 5. Experimental Setup and Comparative Variants
The variants compared will be:
- Network architecture size (small vs. large MLP)
- Shape of the network

### 6. Results, Validation, and Interpretation
- **Logs:** Used to visualize reward progression.
- **Validation:** Performed by:
  - Observing trained policies.
  - Tracking velocity commands across rough terrain variants.
  - Identifying failure modes such as falling or instability.
  - Plotting comparisons between tested variants.

### 7. Conclusions, Limitations, and Recommendations
The final analysis will recommend the optimal network size.

---

## Workflow and Division of Labor (2 Persons)

The project executes over 12 tasks spanning the planning session, three laboratories, and the defense. Both students must understand every part of the codebase and be able to explain all decisions.

### Planning Session (Tasks 1–6)
- **Scope:** Planning the project scope and setup.

### Laboratory 1 (Tasks 7–8)
- **Student 1:** Configures the environment generation using the quadruped robot of choice and ensures the simulation constraints for the Spot robot are correct.
- **Student 2:** Verifies observation extraction and reward scaling.
- *Milestone:* Completes **Checkpoint C2** for the first lab.

### Laboratory 2 (Tasks 9–10)
- **Student 1:** Runs the baseline model smoke test training runs.
- **Student 2:** Configures the Hydra override parameters for the varied model size and parallel environment count, launching the comparative runs.
- *Milestone:* Completes **Checkpoint C2** for the second lab.

### Laboratory 3 (Tasks 10–11)
- **Both Students:** Gather final metrics, compile TensorBoard graphs, and analyze convergence differences between the two model sizes.
- *Milestone:* Completes **Checkpoint C2** for the third lab.

### Defense (Task 12)
- **Both Students:** Write the final conclusion, submit the final documentation (**Checkpoint C3**), and defend their findings to the teacher.

---

*Note: The document was aided by GenAI.*