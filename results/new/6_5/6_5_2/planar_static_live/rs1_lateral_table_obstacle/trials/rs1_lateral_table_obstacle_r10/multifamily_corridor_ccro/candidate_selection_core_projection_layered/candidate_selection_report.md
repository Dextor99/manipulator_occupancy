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
| 1 | `seed_free` | hybrid_vertical_lateral | True | True | True | True | 0.0000 | 0.1139 | 8.000 | 1.0450 | 1.3145 | 1.9963 | 0.1653 | 0.0000 | 0.2941 | 0.2732 | 7956 | - |
| 2 | `overpass` | hybrid_vertical_lateral | True | True | True | True | 1.0000 | 0.1139 | 8.000 | 1.0450 | 1.3145 | 1.9963 | 0.1653 | 0.0000 | 0.2941 | 0.2732 | 7956 | - |
| 3 | `base_side` | hybrid_vertical_lateral | True | True | True | True | 2.0000 | 0.1177 | 8.000 | 1.0518 | 1.3230 | 2.0830 | 0.2138 | 0.0000 | 0.2901 | 0.2732 | 3143 | - |
| 4 | `outer_side` | hybrid_vertical_lateral | False | True | False | None | NA | 0.0009 | 8.000 | 3.2365 | 4.0710 | 5.5434 | 22.1937 | 2.5392 | 0.7547 | 0.2732 | 22128 | dense_distance_below_acceptance,route_family_not_preserved |

## Recommendation

- Selected execution candidate: `seed_free`.
- Feasible candidate count: `3`.
- Near-best time candidate count: `3`.

If no feasible candidate exists, the trial status is `NO_EXECUTABLE_CANDIDATE` and the robot must hold.
