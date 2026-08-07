# 6.5.2 Candidate Family Selection

Selection rule:

1. Candidates must first pass solver, dense safety, joint limits, continuity, and goal checks.
2. Route class is determined from the robot swept surface relative to the inflated obstacle XY footprint.
3. A route is a true overpass only if the swept surface enters the inflated obstacle footprint and clears the robust obstacle top.
4. Feasible candidates are time-scaled using fixed velocity/acceleration limits.
5. Candidates within the near-best execution-time set are ranked lexicographically by TCP path length, joint path length, jerk energy, and near-boundary clearance penalty.

Route geometry:

- Inflated obstacle XY margin: `0.080 m`.
- Required vertical margin above robust obstacle top: `d_accept + 0.020 m`.
- Near-best time ratio: `1.050`.

## Ranked Candidates

| rank | candidate | route | feasible | route geom | family ok | near T | selected rank | min dist / m | T_req / s | L_TCP / m | L_TCP ratio | L_q / rad | jerk | J_clear | max z dev / m | p99 height / m | footprint pts | reasons |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `base_side` | lateral | True | True | True | True | 0.0000 | 0.1032 | 8.000 | 0.8589 | 1.0804 | 2.5701 | 0.5145 | 0.0027 | 0.0428 | 0.4989 | 645 | - |
| 2 | `seed_free` | hybrid_vertical_lateral | True | True | True | True | 1.0000 | 0.1053 | 8.000 | 1.0225 | 1.2861 | 2.2068 | 0.2777 | 0.0008 | 0.2586 | 0.4989 | 1141 | - |
| 3 | `overpass` | hybrid_vertical_lateral | False | True | False | None | NA | 0.1096 | 8.000 | 1.0616 | 1.3353 | 2.2190 | 0.2844 | 0.0000 | 0.2835 | 0.4989 | 1205 | optimizer_not_successful,dense_solver_not_ok,route_family_not_preserved |
| 4 | `outer_side` | lateral | False | True | False | None | NA | 0.0004 | 8.000 | 1.8707 | 2.3531 | 4.7911 | 6.3928 | 1.7049 | 0.0508 | 0.4989 | 5617 | dense_distance_below_acceptance,route_family_not_preserved |

## Recommendation

- Selected execution candidate: `base_side`.
- Feasible candidate count: `2`.
- Near-best time candidate count: `2`.

If no feasible candidate exists, the trial status is `NO_EXECUTABLE_CANDIDATE` and the robot must hold.
