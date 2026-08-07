# 6.5.2 Objective-Term Audit

This audit compares original CCRO-NUBS objective terms with the general candidate-selection metrics.

| candidate | accepted | approx original objective | smooth term | risk term | dense min / m | margin / m | max z dev / m | z violation / m | mean xy dev / m | orient / deg | orient violation / deg | TCP xy length / m | joint length / rad |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ccro_nubs_jointspace_plan` | True | 0.015102 | 0.014762 | 0.000339 | 0.1045 | 0.0245 | 0.2887 | 0.2887 | 0.1305 | 24.244 | 0.000 | 0.8836 | 2.1723 |
| `ccro_nubs_jointspace_plan_flatZ_300iter` | True | 0.025392 | 0.024451 | 0.000941 | 0.0991 | 0.0191 | 0.0154 | 0.0154 | 0.1196 | 18.057 | 0.000 | 0.8547 | 2.4822 |

Interpretation:

- Dense safety, joint limits, continuity, and goal reaching are hard feasibility gates.
- TCP height is not a hard preference in the general static-avoidance setting; vertical motion contributes through the 3D TCP path length.
- Candidate selection should compare feasible path families using 3D TCP path length, joint path length, jerk energy, duration, and near-boundary clearance penalty.
- The selected path may therefore be an overpass or a lateral route depending on the unified objective values.
