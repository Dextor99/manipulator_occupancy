| scenario | method | solver | accepted | D_min dense/m | J_risk full | J_smooth | goal error | nearest link | time/ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | MINCO-base | True | False | 0.000221789 | 0.00348607 | 1.25076 | 1.47523e-16 | gripper_base_link | 0 |
| A | MINCO-risk | True | True | 0.11135 | 1.36674e-08 | 3.72493 | 5.77556e-16 | wrist2_Link | 985.505 |
| A | RRT-Connect + smoothing | True | True | 0.0849028 | 1.77726e-05 | 0.985696 | 2.5172e-16 | wrist2_Link | 1362.22 |
| B | MINCO-base | True | False | 0.000414818 | 0.000191164 | 1.25076 | 1.47523e-16 | foreArm_Link | 0 |
| B | MINCO-risk | True | True | 0.110023 | 6.23391e-10 | 3.72493 | 5.77556e-16 | foreArm_Link | 1006.94 |
| B | RRT-Connect + smoothing | True | True | 0.0924599 | 1.72226e-06 | 2.20123 | 5.41767e-16 | foreArm_Link | 1423.65 |
| C | MINCO-base | True | False | 0.000243485 | 0.00299255 | 1.25076 | 1.47523e-16 | wrist2_Link | 0 |
| C | MINCO-risk | True | True | 0.18447 | 0 | 4.0709 | 7.39049e-16 | wrist1_Link | 1406.82 |
| C | RRT-Connect + smoothing | True | False | 0.0557618 | 3.1843e-05 | 3.08066 | 1.59444e-16 | foreArm_Link | 2791.63 |
