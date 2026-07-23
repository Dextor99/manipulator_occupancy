# 6.4 dynamic obstacle virtual closed-loop rerun status

Run directory: `results/new/6_4`

This rerun addresses the latest review notes and replaces the previous 6.4 formal data:

- Critical-point-NUBS now uses a full-body sparse critical-point geometry with equivalent radii, rather than target-link mesh filtering.
- CCRO-NUBS and Critical-point-NUBS share the same trigger, optimization variables, asynchronous pending state, planned switch slot, and acceptance workflow; the geometric risk representation is the intended difference.
- The planned switch delay is unified as `PLANNED_SWITCH_DELAY = 3.0 s`; the candidate deadline is the same planned switch timestamp.
- Replanning interval control now uses `first_replan_time` and `last_replan_time` separately.
- Candidate events remain pending until the planned switch slot, including candidates that eventually exceed the planning budget, so pending-interval execution is recorded.
- Bridge distance is split into `bridge_min_distance_obs_predicted` and `bridge_min_distance_gt_executed`.
- The main table reports `task safe` separately from `replan success`.
- Speed, trigger-lead-time, candidate-validation, dense-GT, and initial-high-risk audit tables are regenerated.

Current interpretation:

- `body_crossing` is the main positive evidence. CCRO-NUBS reaches 0.93 task-safe success and 0.73 replan success, with dense-GT violation rate 0.07. Critical-point-NUBS reaches 0.67 task-safe success but 0.00 replan success and dense-GT violation rate 0.20, showing that the sparse representation can miss or fail to generate acceptable candidate switches even when it sometimes becomes safe by waiting.
- `ee_crossing` should be written as an operating-domain boundary. CCRO-NUBS improves over Critical-point-NUBS, SSM, and SSM+APF in some low-speed samples, but its overall task-safe success is 0.27 and dense-GT violation rate is 0.73. Reference-only remains safe in 0.80 of these samples, so this scenario must not be used to claim unconditional superiority over the reference trajectory.
- `far_safe` remains a non-trigger sanity check.
- `initial_high_risk` remains a safety-hold sanity check; negative Dmin and zero finish are expected.

Paper-facing tables:

- `paper/table_6_4_dynamic_replanning.md`
- `paper/table_6_4_by_speed.md`
- `paper/table_6_4_by_lead_time.md`
- `paper/table_6_4_candidate_validation_audit.md`
- `paper/table_6_4_gt_dense_recheck.md`
- `paper/table_6_4_initial_high_risk_hold.md`
