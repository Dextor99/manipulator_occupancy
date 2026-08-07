# 6.5.2 Candidate Family Selection

Selection rule:

1. Candidates must first pass solver, dense safety, joint limits, continuity, and goal checks.
2. TCP height is not a hard preference in the general static-avoidance setting.
3. Feasible candidates are Pareto-filtered over L_TCP, L_q, jerk energy, near-boundary clearance penalty, and duration.
4. Only Pareto non-dominated candidates receive a normalized weighted score.
5. Rejected or dominated candidates may be used for analysis/figures, not as the selected execution candidate.

## Ranked Candidates

| rank | candidate | route | feasible | dominated | selected score | min dist / m | L_TCP ratio | L_q / rad | jerk | J_clear | T / s | max z dev / m | reasons |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `ccro_nubs_jointspace_plan` | overpass | True | False | 0.0000 | 0.1168 | 1.2122 | 1.8108 | 0.1197 | 0.0000 | 8.00 | 0.2144 | - |
| 2 | `ccro_nubs_jointspace_plan_flatZ_300iter` | planar/lateral | True | True | NA | 0.0845 | 1.4117 | 2.4707 | 0.5298 | 0.0548 | 8.00 | 0.0154 | - |
| 3 | `ccro_nubs_jointspace_plan_flatZ_160iter` | planar/lateral | False | None | NA | 0.0845 | 1.4214 | 2.4706 | 0.5357 | 0.0547 | 8.00 | 0.0154 | solver_ok |
| 4 | `ccro_nubs_jointspace_plan_flatZ` | planar/lateral | False | None | NA | 0.0824 | 1.4309 | 2.4220 | 0.5305 | 0.0671 | 8.00 | 0.0156 | solver_ok |

## Recommendation

- Selected execution candidate: `ccro_nubs_jointspace_plan`.
- Feasible candidate count: `2`.
- Pareto non-dominated count: `1`.

If no feasible candidate exists, the trial status is `NO_EXECUTABLE_CANDIDATE` and the robot must hold.
