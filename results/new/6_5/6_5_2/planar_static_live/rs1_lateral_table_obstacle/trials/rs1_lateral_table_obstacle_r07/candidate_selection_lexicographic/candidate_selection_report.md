# 6.5.2 Candidate Family Selection

Selection rule:

1. Strict real execution may use only candidates with `strict_execution_ok=true`.
2. Dense safety is a hard gate, not a score.
3. TCP height/orientation task corridors are hard gates for tabletop execution.
4. Among hard-feasible candidates, use lexicographic selection: J_xy, J_R, smooth energy, L_q, saturated clearance margin.
5. Rejected candidates may be used only for analysis/figures, not for robot execution.

## Ranked Candidates

| rank | candidate | route | hard feasible | strict | dense | task | min dist / m | max z dev / m | mean xy dev / m | orient / deg | energy | joint length / rad | margin / m | reasons |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `ccro_nubs_jointspace_plan_flatZ_300iter` | planar/lateral | True | True | True | True | 0.1130 | 0.0151 | 0.0557 | 18.851 | 0.1462 | 1.8714 | 0.0300 | - |
| 2 | `ccro_nubs_jointspace_plan` | overpass | False | True | True | False | 0.1170 | 0.1413 | 0.0370 | 17.177 | 0.0905 | 1.7413 | 0.0300 | tcp_z_corridor_ok |
| 3 | `ccro_nubs_jointspace_plan_flatZ` | planar/lateral | False | False | True | True | 0.1130 | 0.0151 | 0.0557 | 18.685 | 0.1462 | 1.8707 | 0.0300 | solver_ok |

## Recommendation

- Hard-feasible execution candidate: `ccro_nubs_jointspace_plan_flatZ_300iter`.
- Minimal-change dense geometric candidate for analysis: `ccro_nubs_jointspace_plan_flatZ_300iter`.

If no hard-feasible candidate exists, the trial status is `NO_EXECUTABLE_CANDIDATE` and the robot must hold.
