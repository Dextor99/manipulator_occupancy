| scenario | name | expected | passive D_min | active D_min | gain | replans | accepted | hold/s | safety events | finished |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | dynamic_body_replan | accepted_replan | 0.0498404 | 0.0617409 | 0.0119005 | 1 | 1 | 0 | 0 | True |
| B | far_obstacle_no_trigger | no_trigger | 1.08211 | 1.08214 | 3.07067e-05 | 0 | 0 | 0 | 0 | True |
| C | dynamic_ee_replan | accepted_replan | 0.0425275 | 0.0827564 | 0.0402289 | 1 | 1 | 0 | 0 | True |
| D | immediate_high_safety_hold | safety_takeover | 0 | 0 | 0 | 0 | 0 | 0.05 | 1 | False |
