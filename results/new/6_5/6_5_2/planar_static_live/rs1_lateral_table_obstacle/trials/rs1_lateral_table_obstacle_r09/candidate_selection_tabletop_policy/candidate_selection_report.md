# 6.5.2 Candidate Family Selection

Selection rule:

1. Candidates must first pass solver, dense safety, joint limits, continuity, and goal checks.
2. For tabletop high obstacles, vertical overpass routes are rejected as task-inappropriate before scoring.
3. Remaining feasible candidates are Pareto-filtered over L_TCP, L_q, jerk energy, near-boundary clearance penalty, and duration.
4. Only Pareto non-dominated candidates receive a normalized weighted score.
5. Rejected or dominated candidates may be used for analysis/figures, not as the selected execution candidate.

Tabletop overpass policy:

- Policy: `reject_high_obstacle_overpass`.
- High obstacle threshold: `0.220 m` above the table.
- Vertical overpass threshold: `0.090 m` TCP z deviation.

## Ranked Candidates

| rank | candidate | route | feasible | dominated | score | min dist / m | L_TCP ratio | L_q / rad | jerk | J_clear | T / s | max z dev / m | obs height / m | tabletop ok | reasons |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `ccro_nubs_jointspace_plan_flatZ_300iter` | planar/lateral | True | False | 0.0000 | 0.0991 | 1.0808 | 2.4822 | 0.4890 | 0.0073 | 8.00 | 0.0154 | 0.4840 | True | - |
| 2 | `ccro_nubs_jointspace_plan` | overpass | False | None | NA | 0.1045 | 1.3732 | 2.1723 | 0.2952 | 0.0010 | 8.00 | 0.2887 | 0.4840 | False | high_tabletop_obstacle_vertical_overpass_rejected |

## Recommendation

- Selected execution candidate: `ccro_nubs_jointspace_plan_flatZ_300iter`.
- Feasible candidate count: `1`.
- Pareto non-dominated count: `1`.

If no feasible candidate exists, the trial status is `NO_EXECUTABLE_CANDIDATE` and the robot must hold.
