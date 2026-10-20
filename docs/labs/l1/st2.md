
## Observed states
| Slice | Signal | Size | Frame |
| --- | --- | --- | --- |
| `base_lin_vel` | Linear velocity | 3 | body |
| `base_ang_vel` | Angular velocity | 3 | body |
| `projected_gravity` | Gravity direction | 3 | body |
| `velocity_commands` | Commanded `vx`, `vy`, yaw rate | 3 | body |
| `joint_pos` | Joint position relative to the default pose | 12 | joint |
| `joint_vel` | Joint velocity relative to the default | 12 | joint |
| `actions` | Previous action | 12 | joint |

## Per-state Weights of rewards 

| Term | Weight |
| --- | --- |
| `air_time` | 5.0 |
| `base_angular_velocity` | 5.0 |
| `base_linear_velocity` | 5.0 |
| `foot_clearance` | 0.5 |
| `gait` | 10.0 |
| `action_smoothness` | −1.0 |
| `air_time_variance` | −1.0 |
| `base_motion` | −2.0 |
| `base_orientation` | −3.0 |
| `foot_slip` | −0.5 |
| `joint_acc` | −0.0001 |
| `joint_pos` | −0.7 |
| `joint_torques` | −0.0005 |
| `joint_vel` | −0.01 |