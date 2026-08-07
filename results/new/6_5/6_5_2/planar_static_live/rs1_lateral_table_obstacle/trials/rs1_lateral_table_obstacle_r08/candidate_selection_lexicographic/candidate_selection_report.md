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
| 1 | `ccro_nubs_jointspace_plan_flatZ_300iter` | planar/lateral | True | True | True | True | 0.1056 | 0.0151 | 0.1237 | 18.818 | 0.4743 | 2.4594 | 0.0256 | - |
| 2 | `ccro_nubs_jointspace_plan_from_flatZ_original_objective` | overpass | False | True | True | False | 0.1085 | 0.2966 | 0.0675 | 35.541 | 0.1595 | 1.9060 | 0.0285 | tcp_z_corridor_ok |
| 3 | `ccro_nubs_jointspace_plan` | overpass | False | True | True | False | 0.1085 | 0.2966 | 0.0675 | 35.540 | 0.1595 | 1.9060 | 0.0285 | tcp_z_corridor_ok |
| 4 | `ccro_nubs_jointspace_plan_flatZ` | planar/lateral | False | False | True | True | 0.1053 | 0.0152 | 0.1245 | 21.395 | 0.4778 | 2.4641 | 0.0253 | solver_ok |
| 5 | `ccro_nubs_jointspace_plan_minchange` | hybrid | False | False | False | True | 0.0232 | 0.0299 | 0.0682 | 14.958 | 2.3553 | 2.4476 | 0.0000 | distance_ok |
| 6 | `ccro_nubs_jointspace_plan_minchange_from_flatZ` | hybrid | False | False | False | False | 0.0678 | 0.0443 | 0.0917 | 13.905 | 1.4937 | 2.4245 | 0.0000 | distance_ok,tcp_z_corridor_ok |

## Recommendation

- Hard-feasible execution candidate: `ccro_nubs_jointspace_plan_flatZ_300iter`.
- Minimal-change dense geometric candidate for analysis: `ccro_nubs_jointspace_plan_flatZ_300iter`.

If no hard-feasible candidate exists, the trial status is `NO_EXECUTABLE_CANDIDATE` and the robot must hold.
