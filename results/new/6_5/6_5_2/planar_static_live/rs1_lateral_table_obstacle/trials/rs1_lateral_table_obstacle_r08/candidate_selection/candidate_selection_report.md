# 6.5.2 Candidate Family Selection

Selection rule:

1. Strict real execution may use only candidates with `strict_execution_ok=true`.
2. Among strict candidates, rank by minimal task-space change after the safety gate.
3. Rejected candidates may be used only for analysis/figures, not for robot execution.

## Ranked Candidates

| rank | candidate | route | strict | geom. dense | min dist / m | max TCP z dev / m | max TCP xyz dev / m | joint length / rad | score | reasons |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `ccro_nubs_jointspace_plan_flatZ_300iter` | planar/lateral | True | True | 0.1056 | 0.0151 | 0.2794 | 2.4594 | 0.4528 | - |
| 2 | `ccro_nubs_jointspace_plan_flatZ` | planar/lateral | False | True | 0.1053 | 0.0152 | 0.2819 | 2.4641 | 0.4556 | solver_ok |
| 3 | `ccro_nubs_jointspace_plan_from_flatZ_original_objective` | overpass | True | True | 0.1085 | 0.2966 | 0.3399 | 1.9060 | 1.3306 | - |
| 4 | `ccro_nubs_jointspace_plan` | overpass | True | True | 0.1085 | 0.2966 | 0.3399 | 1.9060 | 1.3306 | - |
| 5 | `ccro_nubs_jointspace_plan_minchange` | hybrid | False | False | 0.0232 | 0.0299 | 0.1601 | 2.4476 | 0.3722 | distance_ok |
| 6 | `ccro_nubs_jointspace_plan_minchange_from_flatZ` | hybrid | False | False | 0.0678 | 0.0443 | 0.2119 | 2.4245 | 0.4661 | distance_ok |

## Recommendation

- Strict execution candidate: `ccro_nubs_jointspace_plan_flatZ_300iter`.
- Minimal-change geometric candidate for analysis: `ccro_nubs_jointspace_plan_flatZ_300iter`.

If these are different, the result means the current optimizer can find a safer/executable route, but the lower-spatial-change route still needs either stricter convergence or an explicit hard-constrained planner before real execution.
