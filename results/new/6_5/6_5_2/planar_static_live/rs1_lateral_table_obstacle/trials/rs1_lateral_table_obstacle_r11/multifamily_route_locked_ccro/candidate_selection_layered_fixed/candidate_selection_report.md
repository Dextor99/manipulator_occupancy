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
| 1 | `base_side` | lateral | True | True | True | True | 0.0000 | 0.1007 | 8.000 | 0.8376 | 1.0536 | 2.4983 | 0.4618 | 0.0050 | 0.0263 | 0.4838 | 1320 | - |
| 2 | `seed_free` | hybrid_vertical_lateral | True | True | True | True | 1.0000 | 0.1068 | 8.000 | 1.0603 | 1.3337 | 2.2503 | 0.3015 | 0.0005 | 0.2793 | 0.4838 | 229 | - |
| 3 | `overpass` | hybrid_vertical_lateral | False | True | False | None | NA | 0.1068 | 8.000 | 1.0603 | 1.3337 | 2.2503 | 0.3015 | 0.0005 | 0.2793 | 0.4838 | 229 | route_family_not_preserved |
| 4 | `outer_side` | lateral | False | True | False | None | NA | 0.0002 | 8.000 | 3.0569 | 3.8451 | 5.8278 | 26.7347 | 4.6376 | 0.0525 | 0.4838 | 15651 | dense_distance_below_acceptance,route_family_not_preserved |

## Recommendation

- Selected execution candidate: `base_side`.
- Feasible candidate count: `2`.
- Near-best time candidate count: `2`.

If no feasible candidate exists, the trial status is `NO_EXECUTABLE_CANDIDATE` and the robot must hold.
