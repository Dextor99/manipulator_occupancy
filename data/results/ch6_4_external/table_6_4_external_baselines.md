| scenario | method | solver | accepted | D_min dense/m | J_risk | J_smooth | goal error | nearest link | time/ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | minco_base | True | False | 0.000221789 | 0.00348607 | 1.25076 | 1.47523e-16 | gripper_base_link | 0 |
| A | minco_risk | True | True | 0.11135 | 1.36674e-08 | 3.72493 | 5.77556e-16 | wrist2_Link | 968.417 |
| A | rrt_connect_smooth | True | False | 0.111131 | 3.48799e-05 | 2.10975 | 1.07128e-14 | foreArm_Link | 1248.39 |
| B | minco_base | True | False | 0.000414818 | 0.000191164 | 1.25076 | 1.47523e-16 | foreArm_Link | 0 |
| B | minco_risk | True | True | 0.110023 | 6.23391e-10 | 3.72493 | 5.77556e-16 | foreArm_Link | 944.653 |
| B | rrt_connect_smooth | True | True | 0.0968866 | 2.08225e-06 | 1.19122 | 2.48253e-16 | wrist1_Link | 1149.54 |
| C | minco_base | True | False | 0.000243485 | 0.00299255 | 1.25076 | 1.47523e-16 | wrist2_Link | 0 |
| C | minco_risk | True | True | 0.18447 | 0 | 4.0709 | 7.39049e-16 | wrist1_Link | 1319.76 |
| C | rrt_connect_smooth | True | False | 0.0921105 | 1.08969e-05 | 2.26856 | 2.498e-16 | wrist1_Link | 2498.98 |
