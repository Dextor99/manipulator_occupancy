# 6.4 dynamic obstacle virtual closed-loop rerun status

Run directory: `results/new/6_4`

This rerun implements the stricter experiment definition requested by the latest review notes:

- asynchronous planning is decoupled from immediate slowdown; the scene clock, obstacle motion, and reference execution continue while a candidate is pending;
- candidate switches are delayed to the planned switch time, then revalidated against online predicted occupancy, continuity limits, and a reference-vs-candidate safety gate;
- optimizer convergence is reported separately from candidate acceptance;
- SSM+APF and Critical-point-NUBS baselines are included in addition to Reference-only, SSM, and CCRO-NUBS;
- summary tables report task-safe success separately from replan-switch success;
- speed and trigger-lead-time stratified tables are generated;
- dense GT audit tables are generated after the formal run.

Current interpretation:

- `body_crossing` is the strongest positive evidence for the proposed CCRO-NUBS loop: CCRO-NUBS reaches 0.93 task-safe success and 0.87 replan success, clearly above Critical-point-NUBS, SSM, SSM+APF, and Reference-only.
- `ee_crossing` should be written as a time-coupled capability-boundary case, not as a uniformly successful benchmark. CCRO-NUBS still improves over Critical-point-NUBS, SSM, and SSM+APF, but only reaches 0.40 task-safe/replan success under the current strict 3 s asynchronous switch model, while Reference-only remains safe in many samples because those samples are not all initially unsafe.
- `far_safe` is a non-trigger sanity check.
- `initial_high_risk` is a safety-hold sanity check. Its negative Dmin and zero finish are expected and should not be mixed with ordinary task-completion success.

Paper-facing tables:

- `paper/table_6_4_dynamic_replanning.md`
- `paper/table_6_4_by_speed.md`
- `paper/table_6_4_by_lead_time.md`
- `paper/table_6_4_candidate_validation_audit.md`
- `paper/table_6_4_gt_dense_recheck.md`
- `paper/table_6_4_initial_high_risk_hold.md`
