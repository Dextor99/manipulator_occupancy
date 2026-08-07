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

| rank | candidate | route | feasible | near T | selected rank | min dist / m | T_req / s | L_TCP / m | L_TCP ratio | L_q / rad | jerk | J_clear | max z dev / m | p99 height / m | footprint pts | reasons |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `base_side_from_flatZ` | lateral | True | True | 0.0000 | 0.1068 | 8.000 | 0.8150 | 1.0251 | 2.2656 | 0.3144 | 0.0003 | 0.0454 | 0.4786 | 1896 | - |
| 2 | `seed_free` | footprint_crossing_vertical | True | True | 1.0000 | 0.1107 | 8.000 | 0.9753 | 1.2268 | 2.1460 | 0.2455 | 0.0000 | 0.2325 | 0.4786 | 74 | - |

## Recommendation

- Selected execution candidate: `base_side_from_flatZ`.
- Feasible candidate count: `2`.
- Near-best time candidate count: `2`.

If no feasible candidate exists, the trial status is `NO_EXECUTABLE_CANDIDATE` and the robot must hold.
