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
| 1 | `ccro_nubs_jointspace_plan_from_flatZ_original_objective` | overpass | True | False | 0.3500 | 0.1085 | 1.3290 | 1.9060 | 0.1595 | 0.0001 | 8.00 | 0.2966 | - |
| 2 | `ccro_nubs_jointspace_plan` | overpass | True | False | 0.3500 | 0.1085 | 1.3290 | 1.9060 | 0.1595 | 0.0001 | 8.00 | 0.2966 | - |
| 3 | `ccro_nubs_jointspace_plan_flatZ_300iter` | planar/lateral | True | False | 0.6500 | 0.1056 | 1.0928 | 2.4594 | 0.4743 | 0.0006 | 8.00 | 0.0151 | - |
| 4 | `ccro_nubs_jointspace_plan_flatZ` | planar/lateral | False | None | NA | 0.1053 | 1.0937 | 2.4641 | 0.4778 | 0.0007 | 8.00 | 0.0152 | solver_ok |
| 5 | `ccro_nubs_jointspace_plan_minchange` | hybrid | False | None | NA | 0.0232 | 1.1965 | 2.4476 | 2.3553 | 1.3742 | 8.00 | 0.0299 | distance_ok |
| 6 | `ccro_nubs_jointspace_plan_minchange_from_flatZ` | hybrid | False | None | NA | 0.0678 | 1.2060 | 2.4245 | 1.4937 | 0.2278 | 8.00 | 0.0443 | distance_ok |

## Recommendation

- Selected execution candidate: `ccro_nubs_jointspace_plan_from_flatZ_original_objective`.
- Feasible candidate count: `3`.
- Pareto non-dominated count: `3`.

If no feasible candidate exists, the trial status is `NO_EXECUTABLE_CANDIDATE` and the robot must hold.
