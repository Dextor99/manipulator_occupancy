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
| 1 | `ccro_nubs_jointspace_plan_flatZ_300iter` | planar/lateral | True | True | True | True | 0.0845 | 0.0154 | 0.0828 | 23.674 | 0.5298 | 2.4707 | 0.0045 | - |
| 2 | `ccro_nubs_jointspace_plan` | overpass | False | True | True | False | 0.1168 | 0.2144 | 0.0630 | 24.869 | 0.1197 | 1.8108 | 0.0300 | tcp_z_corridor_ok |
| 3 | `ccro_nubs_jointspace_plan_flatZ_160iter` | planar/lateral | False | False | True | True | 0.0845 | 0.0154 | 0.0837 | 22.386 | 0.5357 | 2.4706 | 0.0045 | solver_ok |
| 4 | `ccro_nubs_jointspace_plan_flatZ` | planar/lateral | False | False | True | True | 0.0824 | 0.0156 | 0.0858 | 18.904 | 0.5305 | 2.4220 | 0.0024 | solver_ok |

## Recommendation

- Hard-feasible execution candidate: `ccro_nubs_jointspace_plan_flatZ_300iter`.
- Minimal-change dense geometric candidate for analysis: `ccro_nubs_jointspace_plan_flatZ_300iter`.

If no hard-feasible candidate exists, the trial status is `NO_EXECUTABLE_CANDIDATE` and the robot must hold.
