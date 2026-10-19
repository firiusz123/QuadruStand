Proposed metrics
#	Metric	How to compute	Why
1	Linear velocity error	‖cmd_xy − base_lin_vel_xy‖ in m/s, averaged over steps	Core tracking quality
2	Yaw rate error	‖cmd_yaw − base_ang_vel_z‖ in rad/s	Turning quality
3	Tilt	Angle between the body’s up axis and the world’s up axis, from projected_gravity: arccos(−g_z). Report the mean and the maximum.	Orientation stability. For the stand-up task, the pitch matters most.
4	Fall rate	Fraction of episodes ending in body_contact termination vs time_out	Errors mean little if the robot fell. Always report this next to 1–3.
5	Tracking success rate	Fraction of steps where velocity error < 0.2 m/s and yaw error < 0.2 rad/s (thresholds are my guess, tune them)	One easy-to-read percentage