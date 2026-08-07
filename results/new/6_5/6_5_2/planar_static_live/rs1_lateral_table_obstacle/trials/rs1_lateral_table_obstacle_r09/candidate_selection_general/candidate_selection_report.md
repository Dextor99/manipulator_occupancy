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
| 1 | `ccro_nubs_jointspace_plan` | overpass | True | False | 0.3500 | 0.1045 | 1.3732 | 2.1723 | 0.2952 | 0.0010 | 8.00 | 0.2887 | - |
| 2 | `ccro_nubs_jointspace_plan_flatZ_300iter` | planar/lateral | True | False | 0.6500 | 0.0991 | 1.0808 | 2.4822 | 0.4890 | 0.0073 | 8.00 | 0.0154 | - |

## Recommendation

- Selected execution candidate: `ccro_nubs_jointspace_plan`.
- Feasible candidate count: `2`.
- Pareto non-dominated count: `2`.

If no feasible candidate exists, the trial status is `NO_EXECUTABLE_CANDIDATE` and the robot must hold.
