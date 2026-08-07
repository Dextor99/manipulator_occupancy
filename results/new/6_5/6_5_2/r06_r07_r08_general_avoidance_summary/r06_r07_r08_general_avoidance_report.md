# R06-R08 General Static Avoidance Reanalysis

Policy: multi-direction candidates -> hard dense/kinematic verification -> Pareto filtering -> normalized cost over 3D TCP path length, joint path length, jerk energy, near-boundary clearance penalty, and duration. No TCP height or planar-motion preference is used.

## Selected Candidates

| trial | selected candidate | route | feasible | Pareto non-dominated | score | dense min / m | L_TCP ratio | L_q / rad | jerk | J_clear | max TCP z dev / m | top view |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| r06 | `ccro_nubs_jointspace_plan` | overpass | True | True | 0.0000 | 0.1168 | 1.2122 | 1.8108 | 0.1197 | 0.000000 | 0.2144 | [top](/home/hzy/Code/manipulator_occupancy/results/new/6_5/6_5_2/r06_r07_r08_general_avoidance_summary/figures/r06_selected_ccro_nubs_jointspace_plan_top_view.png) |
| r07 | `ccro_nubs_jointspace_plan` | overpass | True | True | 0.0000 | 0.1170 | 1.1265 | 1.7413 | 0.0905 | 0.000000 | 0.1413 | [top](/home/hzy/Code/manipulator_occupancy/results/new/6_5/6_5_2/r06_r07_r08_general_avoidance_summary/figures/r07_selected_ccro_nubs_jointspace_plan_top_view.png) |
| r08 | `ccro_nubs_jointspace_plan_from_flatZ_original_objective` | overpass | True | True | 0.3500 | 0.1085 | 1.3290 | 1.9060 | 0.1595 | 0.000081 | 0.2966 | [top](/home/hzy/Code/manipulator_occupancy/results/new/6_5/6_5_2/r06_r07_r08_general_avoidance_summary/figures/r08_selected_ccro_nubs_jointspace_plan_from_flatZ_original_objective_top_view.png) |

## Original vs Flat Candidate

| trial | candidate | route | feasible | dominated | score | dense min / m | L_TCP ratio | L_q / rad | jerk | J_clear | max z / m | top view |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| r06 | `ccro_nubs_jointspace_plan` | overpass | True | False | 0.0000 | 0.1168 | 1.2122 | 1.8108 | 0.1197 | 0.000000 | 0.2144 | [top](/home/hzy/Code/manipulator_occupancy/results/new/6_5/6_5_2/r06_r07_r08_general_avoidance_summary/figures/r06_original_ccro_nubs_jointspace_plan_top_view.png) |
| r06 | `ccro_nubs_jointspace_plan_flatZ_300iter` | planar/lateral | True | True | NA | 0.0845 | 1.4117 | 2.4707 | 0.5298 | 0.054777 | 0.0154 | [top](/home/hzy/Code/manipulator_occupancy/results/new/6_5/6_5_2/r06_r07_r08_general_avoidance_summary/figures/r06_flatZ_ccro_nubs_jointspace_plan_flatZ_300iter_top_view.png) |
| r07 | `ccro_nubs_jointspace_plan` | overpass | True | False | 0.0000 | 0.1170 | 1.1265 | 1.7413 | 0.0905 | 0.000000 | 0.1413 | [top](/home/hzy/Code/manipulator_occupancy/results/new/6_5/6_5_2/r06_r07_r08_general_avoidance_summary/figures/r07_original_ccro_nubs_jointspace_plan_top_view.png) |
| r07 | `ccro_nubs_jointspace_plan_flatZ_300iter` | planar/lateral | True | True | NA | 0.1130 | 1.1805 | 1.8714 | 0.1462 | 0.000000 | 0.0151 | [top](/home/hzy/Code/manipulator_occupancy/results/new/6_5/6_5_2/r06_r07_r08_general_avoidance_summary/figures/r07_flatZ_ccro_nubs_jointspace_plan_flatZ_300iter_top_view.png) |
| r08 | `ccro_nubs_jointspace_plan` | overpass | True | False | 0.3500 | 0.1085 | 1.3290 | 1.9060 | 0.1595 | 0.000081 | 0.2966 | [top](/home/hzy/Code/manipulator_occupancy/results/new/6_5/6_5_2/r06_r07_r08_general_avoidance_summary/figures/r08_original_ccro_nubs_jointspace_plan_top_view.png) |
| r08 | `ccro_nubs_jointspace_plan_flatZ_300iter` | planar/lateral | True | False | 0.6500 | 0.1056 | 1.0928 | 2.4594 | 0.4743 | 0.000606 | 0.0151 | [top](/home/hzy/Code/manipulator_occupancy/results/new/6_5/6_5_2/r06_r07_r08_general_avoidance_summary/figures/r08_flatZ_ccro_nubs_jointspace_plan_flatZ_300iter_top_view.png) |

## Interpretation

- Under the general static-avoidance metric, no path direction is preferred in advance.
- r06 and r07 select the original overpass candidate because it Pareto-dominates the planar candidate over the evaluated objectives.
- r08 keeps both overpass and planar candidates on the Pareto front: planar has shorter 3D TCP path, while overpass has lower joint path length and jerk. With the frozen weights, overpass has the lower normalized score.
- Therefore the current general criterion supports executing the original overpass candidate for these three trials. The planar `flatZ_300iter` candidates remain useful alternative path-family evidence, not the automatically preferred formal result.

## Source Reports

- r06 general selection: [/home/hzy/Code/manipulator_occupancy/results/new/6_5/6_5_2/planar_static_live/rs1_lateral_table_obstacle/trials/rs1_lateral_table_obstacle_r06/candidate_selection_general/candidate_selection_report.md](/home/hzy/Code/manipulator_occupancy/results/new/6_5/6_5_2/planar_static_live/rs1_lateral_table_obstacle/trials/rs1_lateral_table_obstacle_r06/candidate_selection_general/candidate_selection_report.md)
- r06 general audit: [/home/hzy/Code/manipulator_occupancy/results/new/6_5/6_5_2/planar_static_live/rs1_lateral_table_obstacle/trials/rs1_lateral_table_obstacle_r06/objective_audit_general/objective_terms_report.md](/home/hzy/Code/manipulator_occupancy/results/new/6_5/6_5_2/planar_static_live/rs1_lateral_table_obstacle/trials/rs1_lateral_table_obstacle_r06/objective_audit_general/objective_terms_report.md)
- r07 general selection: [/home/hzy/Code/manipulator_occupancy/results/new/6_5/6_5_2/planar_static_live/rs1_lateral_table_obstacle/trials/rs1_lateral_table_obstacle_r07/candidate_selection_general/candidate_selection_report.md](/home/hzy/Code/manipulator_occupancy/results/new/6_5/6_5_2/planar_static_live/rs1_lateral_table_obstacle/trials/rs1_lateral_table_obstacle_r07/candidate_selection_general/candidate_selection_report.md)
- r07 general audit: [/home/hzy/Code/manipulator_occupancy/results/new/6_5/6_5_2/planar_static_live/rs1_lateral_table_obstacle/trials/rs1_lateral_table_obstacle_r07/objective_audit_general/objective_terms_report.md](/home/hzy/Code/manipulator_occupancy/results/new/6_5/6_5_2/planar_static_live/rs1_lateral_table_obstacle/trials/rs1_lateral_table_obstacle_r07/objective_audit_general/objective_terms_report.md)
- r08 general selection: [/home/hzy/Code/manipulator_occupancy/results/new/6_5/6_5_2/planar_static_live/rs1_lateral_table_obstacle/trials/rs1_lateral_table_obstacle_r08/candidate_selection_general/candidate_selection_report.md](/home/hzy/Code/manipulator_occupancy/results/new/6_5/6_5_2/planar_static_live/rs1_lateral_table_obstacle/trials/rs1_lateral_table_obstacle_r08/candidate_selection_general/candidate_selection_report.md)
- r08 general audit: [/home/hzy/Code/manipulator_occupancy/results/new/6_5/6_5_2/planar_static_live/rs1_lateral_table_obstacle/trials/rs1_lateral_table_obstacle_r08/objective_audit_general/objective_terms_report.md](/home/hzy/Code/manipulator_occupancy/results/new/6_5/6_5_2/planar_static_live/rs1_lateral_table_obstacle/trials/rs1_lateral_table_obstacle_r08/objective_audit_general/objective_terms_report.md)
