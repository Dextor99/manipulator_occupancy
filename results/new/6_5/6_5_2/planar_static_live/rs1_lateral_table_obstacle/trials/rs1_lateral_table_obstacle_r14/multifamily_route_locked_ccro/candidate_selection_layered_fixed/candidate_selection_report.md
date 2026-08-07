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
| 1 | `seed_free` | true_overpass | True | True | True | True | 0.0000 | 0.1159 | 8.000 | 0.9374 | 1.1791 | 1.8887 | 0.1160 | 0.0000 | 0.2077 | 0.2201 | 19171 | - |
| 2 | `overpass` | true_overpass | True | True | True | True | 1.0000 | 0.1159 | 8.000 | 0.9374 | 1.1791 | 1.8887 | 0.1160 | 0.0000 | 0.2077 | 0.2201 | 19170 | - |
| 3 | `base_side` | lateral | False | True | False | None | NA | 0.0469 | 8.744 | 2.0047 | 2.5216 | 8.9466 | 44.4642 | 0.4640 | 0.0566 | 0.2201 | 4426 | dense_distance_below_acceptance,velocity_violation,route_family_not_preserved |
| 4 | `outer_side` | lateral | False | True | False | None | NA | 0.0002 | 10.413 | 2.7640 | 3.4767 | 8.3261 | 93.8617 | 7.7430 | 0.0845 | 0.2201 | 31769 | dense_distance_below_acceptance,velocity_violation,route_family_not_preserved |

## Recommendation

- Selected execution candidate: `seed_free`.
- Feasible candidate count: `2`.
- Near-best time candidate count: `2`.

If no feasible candidate exists, the trial status is `NO_EXECUTABLE_CANDIDATE` and the robot must hold.
