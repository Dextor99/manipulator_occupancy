# 6.5.2 Objective-Term Audit

This audit explains why the original CCRO-NUBS objective can prefer an overpass route.

| candidate | accepted | approx original objective | smooth term | risk term | dense min / m | margin / m | max z dev / m | z violation / m | mean xy dev / m | orient / deg | orient violation / deg | TCP xy length / m | joint length / rad |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ccro_nubs_jointspace_plan` | True | 0.006019 | 0.005986 | 0.000033 | 0.1168 | 0.0368 | 0.2144 | 0.1844 | 0.0630 | 24.869 | 0.000 | 0.8090 | 1.8108 |
| `ccro_nubs_jointspace_plan_flatZ_300iter` | True | 0.048673 | 0.026488 | 0.022185 | 0.0845 | 0.0045 | 0.0154 | 0.0000 | 0.0828 | 23.674 | 0.000 | 1.1182 | 2.4707 |

Interpretation:

- The original objective is dominated by joint-space smooth energy once both candidates are outside the hard acceptance distance.
- It has no direct penalty for lifting the TCP or deviating from the intended tabletop path.
- Therefore an overpass route can be mathematically optimal under the original objective, even when a planar/lateral route is more appropriate for the real tabletop task.
- The corrected 6.5.2 policy is: dense safety gate first, TCP height/orientation task corridor second, then lexicographic minimal task-space change.
