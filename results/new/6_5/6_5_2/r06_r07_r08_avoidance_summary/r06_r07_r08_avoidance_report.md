# R06-R08 Static Avoidance Reanalysis

Policy: multi-candidate generation -> dense safety hard gate -> TCP height corridor -> lexicographic minimal-change selection. TCP orientation is reported but not used as a hard gate in the current tabletop lateral-avoidance task.

## Selected Candidates

| trial | selected candidate | status | dense min / m | nearest link | max TCP z dev / m | mean TCP xy dev / m | max orient / deg | joint path / rad | top view |
|---|---|---|---:|---|---:|---:|---:|---:|---|
| r06 | `ccro_nubs_jointspace_plan_flatZ_300iter` | EXECUTABLE_CANDIDATE_SELECTED | 0.0845 | foreArm_Link | 0.0154 | 0.0828 | 23.67 | 2.4707 | [top](/home/hzy/Code/manipulator_occupancy/results/new/6_5/6_5_2/r06_r07_r08_avoidance_summary/figures/r06_selected_planar_top_view.png) |
| r07 | `ccro_nubs_jointspace_plan_flatZ_300iter` | EXECUTABLE_CANDIDATE_SELECTED | 0.1130 | right_link | 0.0151 | 0.0557 | 18.85 | 1.8714 | [top](/home/hzy/Code/manipulator_occupancy/results/new/6_5/6_5_2/r06_r07_r08_avoidance_summary/figures/r07_selected_planar_top_view.png) |
| r08 | `ccro_nubs_jointspace_plan_flatZ_300iter` | EXECUTABLE_CANDIDATE_SELECTED | 0.1056 | wrist3_Link | 0.0151 | 0.1237 | 18.82 | 2.4594 | [top](/home/hzy/Code/manipulator_occupancy/results/new/6_5/6_5_2/r06_r07_r08_avoidance_summary/figures/r08_selected_planar_top_view.png) |

## Original vs Selected

| trial | candidate | route | hard feasible | dense min / m | max TCP z dev / m | z gate | original objective approx | smooth term | risk term | top view |
|---|---|---|---:|---:|---:|---|---:|---:|---:|---|
| r06 | `ccro_nubs_jointspace_plan` | overpass | False | 0.1168 | 0.2144 | FAIL | 0.006019 | 0.005986 | 0.000033 | [top](/home/hzy/Code/manipulator_occupancy/results/new/6_5/6_5_2/r06_r07_r08_avoidance_summary/figures/r06_original_overpass_top_view.png) |
| r06 | `ccro_nubs_jointspace_plan_flatZ_300iter` | planar/lateral | True | 0.0845 | 0.0154 | PASS | 0.048673 | 0.026488 | 0.022185 | [top](/home/hzy/Code/manipulator_occupancy/results/new/6_5/6_5_2/r06_r07_r08_avoidance_summary/figures/r06_selected_planar_top_view.png) |
| r07 | `ccro_nubs_jointspace_plan` | overpass | False | 0.1170 | 0.1413 | FAIL | 0.004545 | 0.004525 | 0.000020 | [top](/home/hzy/Code/manipulator_occupancy/results/new/6_5/6_5_2/r06_r07_r08_avoidance_summary/figures/r07_original_overpass_top_view.png) |
| r07 | `ccro_nubs_jointspace_plan_flatZ_300iter` | planar/lateral | True | 0.1130 | 0.0151 | PASS | 0.007387 | 0.007310 | 0.000076 | [top](/home/hzy/Code/manipulator_occupancy/results/new/6_5/6_5_2/r06_r07_r08_avoidance_summary/figures/r07_selected_planar_top_view.png) |
| r08 | `ccro_nubs_jointspace_plan` | overpass | False | 0.1085 | 0.2966 | FAIL | 0.008070 | 0.007974 | 0.000096 | [top](/home/hzy/Code/manipulator_occupancy/results/new/6_5/6_5_2/r06_r07_r08_avoidance_summary/figures/r08_original_overpass_top_view.png) |
| r08 | `ccro_nubs_jointspace_plan_flatZ_300iter` | planar/lateral | True | 0.1056 | 0.0151 | PASS | 0.024218 | 0.023714 | 0.000504 | [top](/home/hzy/Code/manipulator_occupancy/results/new/6_5/6_5_2/r06_r07_r08_avoidance_summary/figures/r08_selected_planar_top_view.png) |

## Interpretation

- The original CCRO-NUBS candidates are dense-safe, but they are not necessarily tabletop-task feasible because the TCP may lift substantially above the reference height.
- The selected `flatZ_300iter` candidates all satisfy the 0.03 m TCP height corridor and dense safety gate.
- r06 is close to the 0.08 m dense threshold, so it is usable as an accepted result but should be described with its smaller margin.
- Extra clearance is not used as a direct reward; after the safety threshold is passed, candidate choice is driven by task consistency and smoothness.

## Source Reports

- r06 candidate selection: [/home/hzy/Code/manipulator_occupancy/results/new/6_5/6_5_2/planar_static_live/rs1_lateral_table_obstacle/trials/rs1_lateral_table_obstacle_r06/candidate_selection_lexicographic/candidate_selection_report.md](/home/hzy/Code/manipulator_occupancy/results/new/6_5/6_5_2/planar_static_live/rs1_lateral_table_obstacle/trials/rs1_lateral_table_obstacle_r06/candidate_selection_lexicographic/candidate_selection_report.md)
- r06 objective audit: [/home/hzy/Code/manipulator_occupancy/results/new/6_5/6_5_2/planar_static_live/rs1_lateral_table_obstacle/trials/rs1_lateral_table_obstacle_r06/objective_audit_lexicographic/objective_terms_report.md](/home/hzy/Code/manipulator_occupancy/results/new/6_5/6_5_2/planar_static_live/rs1_lateral_table_obstacle/trials/rs1_lateral_table_obstacle_r06/objective_audit_lexicographic/objective_terms_report.md)
- r07 candidate selection: [/home/hzy/Code/manipulator_occupancy/results/new/6_5/6_5_2/planar_static_live/rs1_lateral_table_obstacle/trials/rs1_lateral_table_obstacle_r07/candidate_selection_lexicographic/candidate_selection_report.md](/home/hzy/Code/manipulator_occupancy/results/new/6_5/6_5_2/planar_static_live/rs1_lateral_table_obstacle/trials/rs1_lateral_table_obstacle_r07/candidate_selection_lexicographic/candidate_selection_report.md)
- r07 objective audit: [/home/hzy/Code/manipulator_occupancy/results/new/6_5/6_5_2/planar_static_live/rs1_lateral_table_obstacle/trials/rs1_lateral_table_obstacle_r07/objective_audit_lexicographic/objective_terms_report.md](/home/hzy/Code/manipulator_occupancy/results/new/6_5/6_5_2/planar_static_live/rs1_lateral_table_obstacle/trials/rs1_lateral_table_obstacle_r07/objective_audit_lexicographic/objective_terms_report.md)
- r08 candidate selection: [/home/hzy/Code/manipulator_occupancy/results/new/6_5/6_5_2/planar_static_live/rs1_lateral_table_obstacle/trials/rs1_lateral_table_obstacle_r08/candidate_selection_lexicographic/candidate_selection_report.md](/home/hzy/Code/manipulator_occupancy/results/new/6_5/6_5_2/planar_static_live/rs1_lateral_table_obstacle/trials/rs1_lateral_table_obstacle_r08/candidate_selection_lexicographic/candidate_selection_report.md)
- r08 objective audit: [/home/hzy/Code/manipulator_occupancy/results/new/6_5/6_5_2/planar_static_live/rs1_lateral_table_obstacle/trials/rs1_lateral_table_obstacle_r08/objective_audit_lexicographic/objective_terms_report.md](/home/hzy/Code/manipulator_occupancy/results/new/6_5/6_5_2/planar_static_live/rs1_lateral_table_obstacle/trials/rs1_lateral_table_obstacle_r08/objective_audit_lexicographic/objective_terms_report.md)
