# 6.5.2 Objective-Term Audit

This audit explains why the original CCRO-NUBS objective can prefer an overpass route.

| candidate | accepted | approx original objective | smooth term | risk term | dense min / m | margin / m | max z dev / m | z violation / m | mean xy dev / m | orient / deg | orient violation / deg | TCP xy length / m | joint length / rad |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ccro_nubs_jointspace_plan` | True | 0.004545 | 0.004525 | 0.000020 | 0.1170 | 0.0370 | 0.1413 | 0.1113 | 0.0370 | 17.177 | 0.000 | 0.8161 | 1.7413 |
| `ccro_nubs_jointspace_plan_flatZ_300iter` | True | 0.007387 | 0.007310 | 0.000076 | 0.1130 | 0.0330 | 0.0151 | 0.0000 | 0.0557 | 18.851 | 0.000 | 0.9351 | 1.8714 |

Interpretation:

- The original objective is dominated by joint-space smooth energy once both candidates are outside the hard acceptance distance.
- It has no direct penalty for lifting the TCP or deviating from the intended tabletop path.
- Therefore an overpass route can be mathematically optimal under the original objective, even when a planar/lateral route is more appropriate for the real tabletop task.
- The corrected 6.5.2 policy is: dense safety gate first, TCP height/orientation task corridor second, then lexicographic minimal task-space change.
