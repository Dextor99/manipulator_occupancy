# 6.5.2 Objective-Term Audit

This audit explains why the original CCRO-NUBS objective can prefer an overpass route.

| candidate | accepted | approx original objective | smooth term | risk term | dense min / m | margin / m | max z dev / m | z violation / m | mean xy dev / m | orient / deg | orient violation / deg | TCP xy length / m | joint length / rad |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ccro_nubs_jointspace_plan` | True | 0.008070 | 0.007974 | 0.000096 | 0.1085 | 0.0285 | 0.2966 | 0.2966 | 0.0675 | 35.540 | 0.000 | 0.8038 | 1.9060 |
| `ccro_nubs_jointspace_plan_flatZ_300iter` | True | 0.024218 | 0.023714 | 0.000504 | 0.1056 | 0.0256 | 0.0151 | 0.0151 | 0.1237 | 18.818 | 0.000 | 0.8654 | 2.4594 |

Interpretation:

- The original objective is dominated by joint-space smooth energy once both candidates are outside the hard acceptance distance.
- It has no direct penalty for lifting the TCP or deviating from the intended tabletop path.
- Therefore an overpass route can be mathematically optimal under the original objective, even when a planar/lateral route is more appropriate for the real tabletop task.
- The corrected 6.5.2 policy is: dense safety gate first, TCP height/orientation task corridor second, then lexicographic minimal task-space change.
