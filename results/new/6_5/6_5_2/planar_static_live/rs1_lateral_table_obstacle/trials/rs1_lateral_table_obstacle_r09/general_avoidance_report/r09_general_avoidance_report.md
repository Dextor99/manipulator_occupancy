# R09 General Static Avoidance Analysis

Policy: hard dense/kinematic verification, Pareto filtering, then normalized score over 3D TCP path length, joint path length, jerk energy, near-boundary clearance penalty, and duration. No planar or height preference is imposed.

## Capture Check

| item | value |
|---|---:|
| perception frames | 37 |
| obstacle points | 2715 |
| reference min clearance / m | 0.0035 |
| preview candidate min clearance / m | 0.1192 |

## Candidate Selection

| candidate | route | feasible | Pareto dominated | score | dense min / m | L_TCP ratio | L_q / rad | jerk | J_clear | max z / m | top view |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `ccro_nubs_jointspace_plan` | overpass | True | False | 0.3500 | 0.1045 | 1.3732 | 2.1723 | 0.2952 | 0.000982 | 0.2887 | [top](/home/hzy/Code/manipulator_occupancy/results/new/6_5/6_5_2/planar_static_live/rs1_lateral_table_obstacle/trials/rs1_lateral_table_obstacle_r09/general_avoidance_report/figures/r09_original_ccro_nubs_jointspace_plan_top_view.png) |
| `ccro_nubs_jointspace_plan_flatZ_300iter` | planar/lateral | True | False | 0.6500 | 0.0991 | 1.0808 | 2.4822 | 0.4890 | 0.007288 | 0.0154 | [top](/home/hzy/Code/manipulator_occupancy/results/new/6_5/6_5_2/planar_static_live/rs1_lateral_table_obstacle/trials/rs1_lateral_table_obstacle_r09/general_avoidance_report/figures/r09_flatZ_ccro_nubs_jointspace_plan_flatZ_300iter_top_view.png) |

## Result

Selected execution candidate: `ccro_nubs_jointspace_plan`.

- Both candidates pass dense and kinematic verification.
- The planar/lateral candidate has shorter 3D TCP path length.
- The original overpass candidate has lower joint path length, lower jerk energy, and lower near-boundary clearance penalty.
- Under the frozen general weights, the original overpass candidate has lower total score and is selected.

## Source Files

- Selection report: [/home/hzy/Code/manipulator_occupancy/results/new/6_5/6_5_2/planar_static_live/rs1_lateral_table_obstacle/trials/rs1_lateral_table_obstacle_r09/candidate_selection_general/candidate_selection_report.md](/home/hzy/Code/manipulator_occupancy/results/new/6_5/6_5_2/planar_static_live/rs1_lateral_table_obstacle/trials/rs1_lateral_table_obstacle_r09/candidate_selection_general/candidate_selection_report.md)
- Objective audit: [/home/hzy/Code/manipulator_occupancy/results/new/6_5/6_5_2/planar_static_live/rs1_lateral_table_obstacle/trials/rs1_lateral_table_obstacle_r09/objective_audit_general/objective_terms_report.md](/home/hzy/Code/manipulator_occupancy/results/new/6_5/6_5_2/planar_static_live/rs1_lateral_table_obstacle/trials/rs1_lateral_table_obstacle_r09/objective_audit_general/objective_terms_report.md)
- Original top-view: [/home/hzy/Code/manipulator_occupancy/results/new/6_5/6_5_2/planar_static_live/rs1_lateral_table_obstacle/trials/rs1_lateral_table_obstacle_r09/general_avoidance_report/figures/r09_original_ccro_nubs_jointspace_plan_top_view.png](/home/hzy/Code/manipulator_occupancy/results/new/6_5/6_5_2/planar_static_live/rs1_lateral_table_obstacle/trials/rs1_lateral_table_obstacle_r09/general_avoidance_report/figures/r09_original_ccro_nubs_jointspace_plan_top_view.png)
- FlatZ top-view: [/home/hzy/Code/manipulator_occupancy/results/new/6_5/6_5_2/planar_static_live/rs1_lateral_table_obstacle/trials/rs1_lateral_table_obstacle_r09/general_avoidance_report/figures/r09_flatZ_ccro_nubs_jointspace_plan_flatZ_300iter_top_view.png](/home/hzy/Code/manipulator_occupancy/results/new/6_5/6_5_2/planar_static_live/rs1_lateral_table_obstacle/trials/rs1_lateral_table_obstacle_r09/general_avoidance_report/figures/r09_flatZ_ccro_nubs_jointspace_plan_flatZ_300iter_top_view.png)
