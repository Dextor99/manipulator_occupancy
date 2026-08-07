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
| 1 | `base_side` | lateral | True | True | True | True | 0.0000 | 0.1076 | 8.000 | 0.9501 | 1.1951 | 2.5390 | 0.6411 | 0.0001 | 0.0504 | 0.2004 | 2339 | - |
| 2 | `seed_free` | footprint_crossing_vertical | False | False | True | None | NA | 0.1159 | 8.000 | 0.9309 | 1.1709 | 1.8820 | 0.1124 | 0.0000 | 0.2029 | 0.2004 | 13843 | route_geometry_overpass_margin_not_satisfied |
| 3 | `overpass` | footprint_crossing_vertical | False | False | False | None | NA | 0.1159 | 8.000 | 0.9309 | 1.1709 | 1.8820 | 0.1124 | 0.0000 | 0.2029 | 0.2004 | 13844 | route_geometry_overpass_margin_not_satisfied,route_family_not_preserved |
| 4 | `outer_side` | lateral | False | True | False | None | NA | 0.0011 | 9.449 | 2.2681 | 2.8529 | 6.7100 | 55.5814 | 4.8012 | 0.0677 | 0.2004 | 26449 | dense_distance_below_acceptance,velocity_violation,route_family_not_preserved |

## Recommendation

- Selected execution candidate: `base_side`.
- Feasible candidate count: `1`.
- Near-best time candidate count: `1`.

If no feasible candidate exists, the trial status is `NO_EXECUTABLE_CANDIDATE` and the robot must hold.
