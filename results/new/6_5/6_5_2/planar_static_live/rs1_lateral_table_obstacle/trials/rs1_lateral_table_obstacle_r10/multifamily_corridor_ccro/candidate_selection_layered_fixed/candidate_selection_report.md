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
| 1 | `seed_free` | footprint_crossing_vertical | False | False | True | None | NA | 0.1139 | 8.000 | 1.0450 | 1.3145 | 1.9963 | 0.1653 | 0.0000 | 0.2941 | 0.2732 | 7956 | route_geometry_overpass_margin_not_satisfied |
| 2 | `overpass` | footprint_crossing_vertical | False | False | True | None | NA | 0.1139 | 8.000 | 1.0450 | 1.3145 | 1.9963 | 0.1653 | 0.0000 | 0.2941 | 0.2732 | 7956 | route_geometry_overpass_margin_not_satisfied |
| 3 | `base_side` | footprint_crossing_vertical | False | False | True | None | NA | 0.1177 | 8.000 | 1.0518 | 1.3230 | 2.0830 | 0.2138 | 0.0000 | 0.2901 | 0.2732 | 3143 | route_geometry_overpass_margin_not_satisfied |
| 4 | `outer_side` | footprint_crossing_vertical | False | False | False | None | NA | 0.0009 | 8.000 | 3.2365 | 4.0710 | 5.5434 | 22.1937 | 2.5392 | 0.7547 | 0.2732 | 22128 | distance_ok,route_geometry_overpass_margin_not_satisfied,route_family_not_preserved |

## Recommendation

- Selected execution candidate: `NONE`.
- Feasible candidate count: `0`.
- Near-best time candidate count: `0`.

If no feasible candidate exists, the trial status is `NO_EXECUTABLE_CANDIDATE` and the robot must hold.
