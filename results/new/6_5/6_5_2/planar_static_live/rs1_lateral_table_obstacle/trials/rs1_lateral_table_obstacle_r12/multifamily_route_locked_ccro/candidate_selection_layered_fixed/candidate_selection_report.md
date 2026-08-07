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
| 1 | `overpass` | footprint_crossing_vertical | False | False | False | None | NA | 0.0674 | 8.000 | 0.9212 | 1.1588 | 1.9600 | 0.2410 | 0.3901 | 0.1945 | 0.1884 | 35862 | dense_distance_below_acceptance,route_geometry_overpass_margin_not_satisfied,route_family_not_preserved |
| 2 | `seed_free` | footprint_crossing_vertical | False | False | True | None | NA | 0.0674 | 8.000 | 0.9213 | 1.1588 | 1.9600 | 0.2410 | 0.3901 | 0.1945 | 0.1884 | 35861 | dense_distance_below_acceptance,route_geometry_overpass_margin_not_satisfied |
| 3 | `outer_side` | lateral | False | True | False | None | NA | 0.0003 | 10.254 | 2.8505 | 3.5855 | 8.1671 | 89.2913 | 7.5074 | 0.0822 | 0.1884 | 45454 | dense_distance_below_acceptance,velocity_violation,route_family_not_preserved |
| 4 | `base_side` | hybrid_vertical_lateral | False | True | False | None | NA | 0.0099 | 12.003 | 3.0446 | 3.8296 | 10.6075 | 89.7187 | 1.5716 | 0.0910 | 0.1884 | 15849 | dense_distance_below_acceptance,velocity_violation,route_family_not_preserved |

## Recommendation

- Selected execution candidate: `NONE`.
- Feasible candidate count: `0`.
- Near-best time candidate count: `0`.

If no feasible candidate exists, the trial status is `NO_EXECUTABLE_CANDIDATE` and the robot must hold.
