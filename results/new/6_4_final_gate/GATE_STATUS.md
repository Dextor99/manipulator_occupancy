# 6.4 near-final gate status

This directory records the gate checks after the near-final 6.4 modifications.

Completed mechanism changes:

- Archived the previous formal 170-trial run to `results/new/6_4_async_v3_boundary`.
- Replaced the Critical-point-NUBS evaluator with fixed local-frame critical points and fixed equivalent radii.
- Removed per-query mesh resampling/reselection from Critical-point-NUBS.
- Unified final online candidate acceptance to the medium-density Mesh verifier for both Critical-point-NUBS and CCRO-NUBS.
- Added an online candidate acceptance margin: `D_ONLINE_ACCEPT = 0.09 m` while keeping GT stop/safety judgment at `D_STOP = 0.08 m`.
- Replaced the old 2% pending near-stop rule with staged pending slowdown.
- Added D1-main, D2-main, and D2-stress dataset structure.
- Changed the primary comparison methods to Reference-only, SSM+APF, Critical-point-NUBS, and CCRO-NUBS.
- Added switch-outcome summary counts: safe with accepted switch, safe without switch, unsafe or unfinished.
- Added frozen pending-slot execution (`alpha_slot`) so the planned switch state matches the actual pending execution state unless safety hold aborts the slot.
- Added reference-trajectory warm-start through `DynamicRiskNUBSOptimizer.optimize(p_inner_initial=...)`.
- Added local candidate segment support and resume-to-reference-tail logic.
- Extended `gate_64.py` with an automatic mechanism gate summary.
- Finalized the main operating domain as a 6.0 s switch slot with a 5.5 s optimization budget.
- Kept D2-stress as the strict 3.0 s switch-slot boundary case.
- Increased the local candidate segment horizon to 4.0 s.
- Added one deterministic clearance-guided warm-start seed and selected it against the reference warm-start by coarse predicted clearance.
- Final solver-level shaping after the last review:
  - switched CCRO mesh clearance from clamped distance to signed clearance;
  - changed the CCRO risk objective from full-surface average hinge to top-k soft-min clearance hinge;
  - cached the online-verified seed candidate and falls back to it if the optimized candidate loses feasibility.

Passed component gate:

- Critical-point gradient query: about 6.87 ms in the latest gate run.
- CCRO mesh gradient query: about 10.07 ms in the latest gate run.
- Critical-point count: 16.
- Critical-point query is faster than CCRO mesh query.

Blocked mechanism gate:

- The final automatic D1 mechanism gate ran 6 CCRO-NUBS trials after the solver-level shaping above.
- `gate_pass=false`.
- Task-safe success was 4/6, below the required 5/6.
- Replan success was 0/6, below the required 4/6.
- Deadline misses were 0, so the 6 s operating-domain budget is adequate for this implementation.
- Continuity rejections occurred in 2 candidate events.
- Online distance rejections occurred in 6 candidate events.
- Planner P95 was about 5.39 s, within the 5.5 s planning budget but close to the limit.
- Safe-with-switch count was 0/6.
- This means the final formal run should not be launched. The system can often remain safe by slowed execution, but it does not demonstrate stable asynchronous candidate switching under the frozen 6.4 mechanism.

G0 consistency audit after replanning review:

- Added `experiments/new/6_4/g0_audit_64.py` as the pre-formal implementation audit.
- Latest G0 output: `results/new/6_4_final_gate/mechanism_d1/paper/table_6_4_g0_consistency_audit.md`.
- The current 6-trial pilot contains 12 CCRO-NUBS trigger events.
- Submission-time online safety passed in 7/12 candidate events.
- Switch-time online safety passed in 2/12 candidate events.
- No deadline misses were observed.
- Trial-level medium-vs-dense GT recheck error was small: p95 medium-minus-dense about 0.001 m.
- Interpretation: the pilot failure should not be treated as simple optimizer failure or threshold tuning target. The main unresolved issue is candidate validity at the planned switch state/time, so formal work must proceed through candidate replay and capability-boundary testing.

Stage-A candidate replay scaffold:

- Added `experiments/new/6_4/candidate_replay_64.py`.
- Smoke output: `results/new/6_4_candidate_replay_d1_smoke/paper/table_6_4_candidate_replay.md`.
- The smoke run is an entry-point validation only, not formal evidence; it confirms that fixed-state candidate replay can produce Critical-NUBS vs CCRO-NUBS funnel rows across Long/Medium/Short lead-time groups.

Stage-A smoke v2 after P0 replay fixes:

- Updated replay semantics:
  - conflict time uses `first_accept_violation_time` when available;
  - switch delay uses `min(6.0, lead_time - 1.0)` instead of a fixed 6.0 s for every lead group;
  - local forecasts use the actual switch delay;
  - replay samples with clipped local horizon or zero motion are marked `invalid_replay_window`;
  - Critical-NUBS and CCRO-NUBS use different planning risk evaluators but the same medium-density Mesh online gate;
  - dense feasibility is separated from solver success, and `usable` requires ready-before-switch, solver success, and dense geometric feasibility;
  - candidate benefit uses common dense Mesh reference/candidate distances.
- Smoke v2 output: `results/new/6_4_candidate_replay_d1_smoke_v2/paper/table_6_4_candidate_replay.md`.
- All current D1 smoke replay windows are invalid under the non-clipped 4.0 s local horizon rule.
- Interpretation: the previous Long/Medium/Short smoke result was not a valid capability boundary. It included terminal clipping and endpoint-hold artifacts. The next step is not full D1/D2 Stage-A, but a G1/simple-feasible development set or a method-independent replay-instance generator whose reference trajectory leaves a valid 4.0 s local segment after the planned switch point.

Realtime redesign decision:

- The seconds-level CCRO-NUBS candidate optimizer is no longer treated as a valid online dynamic-obstacle replanner.
- Dynamic realtime claims require a fast local repair layer with an end-to-end P95 target of roughly 100--150 ms and a hard maximum below 200 ms.
- Added `experiments/new/6_4/fast_local_repair_64.py` as the new 1 s local-repair experiment scaffold.
- Fast local repair uses:
  - 1.0 s local horizon;
  - fixed time allocation;
  - 5 local NUBS segments;
  - at most 3 repair steps;
  - one worst-risk gradient query per repair step;
  - shared medium Mesh online gate and dense Mesh geometric recheck.
- D1 fast smoke output: `results/new/6_4_fast_local_repair_d1_smoke/paper/table_6_4_fast_local_repair.md`.
- D1 fast smoke timing:
  - Critical-fast-repair total time was about 82--97 ms across smoke cases.
  - CCRO-fast-repair total time was about 104--119 ms across smoke cases.
  - Validation P95 was about 40--52 ms depending on case.
- D1 fast smoke feasibility:
  - dense feasible was 0 for both methods;
  - usable candidate rate was 0 for both methods;
  - CCRO often improved dense clearance by a few millimeters but did not restore the 0.08 m dense safety threshold.
- Interpretation: the time scale is now compatible with online local repair, but the current one-shot gradient-bump repair is not yet a valid dynamic avoidance algorithm. The next algorithmic work should improve the fast repair law, not revive the seconds-level optimizer.

Fast repair v2 and G1 check:

- Added a constrained-acceptance repair update:
  - repair direction is based on finite-difference minimum-clearance gradient rather than risk-cost gradient;
  - candidate step sizes are searched over `[0.008, 0.015, 0.025, 0.035]` rad;
  - an update is accepted only if medium clearance improves and position/velocity/acceleration limits remain satisfied;
  - online medium-gate latency is separated from offline dense GT recheck latency;
  - `usable_candidate` now requires online accepted, dense GT safe, and online latency under 150 ms.
- Added method-independent G1 generation by dense Mesh filtering:
  - reference dense minimum is constrained to `[0.04, 0.08)` m;
  - generated G1 instances are shallow-risk rather than deep-penetration smoke cases.
- G1 CCRO v2 output: `results/new/6_4_fast_local_repair_g1_ccro_v2/paper/table_6_4_fast_local_repair.md`.
- G1 CCRO v2 result:
  - valid shallow-risk scenarios: 10/10;
  - online P95: about 141 ms;
  - hard max: about 142 ms;
  - acceleration gate pass: 9/10;
  - dense feasible: 0/10;
  - usable candidate: 0/10;
  - median dense clearance improvement: 0 m, max improvement about 0.005 m.
- Interpretation: v2 fixed the worst dynamic-feasibility issue but still does not produce enough clearance recovery. Do not run full fast Stage-A. The next algorithmic step is a stronger constrained local repair law, likely a small QP/SQP over local interpolation points or an analytic nearest-point Jacobian, not parameter tuning or larger fixed steps.

Final stop-rule decision:

- Do not continue adjusting individual failed D1 pilot cases.
- Do not lower `D_ONLINE_ACCEPT`, retune weights, or reshape obstacle trajectories to make the pilot pass.
- Do not run a new formal closed-loop dataset until G0 plus a simple feasible validation set and Stage-A candidate replay show acceptable candidate feasibility.
- Reframe 6.4 as an asynchronous local replanning effectiveness and capability-boundary experiment.
- Use the archived `results/new/6_4_async_v3_boundary` and current final-gate data only as pilot/diagnostic material, not as final proof of stable asynchronous replanning.
- Do not treat invalid replay windows as failures of CCRO-NUBS or Critical-NUBS. They are experiment-design exclusions that must be reported separately.
- Do not claim realtime trajectory replanning from any 2--5 s candidate-generation result.
- Keep seconds-level CCRO-NUBS in 6.3/static benchmark or 6.4 diagnostic discussion only.
- Use Fast CCRO-NUBS local repair as the only path toward realtime 6.4 claims; it must pass G1 before any formal dynamic closed-loop claim.
- Current fast repair v2 has not passed G1; it may be reported only as a development diagnostic showing that latency is feasible while geometric repair remains unsolved.

Fast CCRO-NUBS v3 scaffold:

- Added modular v3 repair code under `experiments/new/6_4/repair/`:
  - `active_distance.py`: active distance extraction and finite-difference distance gradient;
  - `nubs_linearization.py`: local NUBS interpolation-point sensitivity;
  - `local_qp_solver.py`: small SLSQP-backed linearized repair step with safety slack;
  - `repair_v3.py`: sequential convex repair loop.
- Added `ccro_fast_v3` and `critical_fast_v3` methods to `fast_local_repair_64.py`.
- Added G1-near generation with method-independent dense Mesh filtering: `0.07 <= Dmin_reference_dense < 0.08`.
- G1-near CCRO v3 slack output: `results/new/6_4_fast_local_repair_g1_near_ccro_v3_slack/paper/table_6_4_fast_local_repair.md`.
- G1-near CCRO v3 slack result:
  - valid near-risk scenarios: 10/10;
  - dense feasible: 0/10;
  - usable candidate: 0/10;
  - acceleration gate pass: 9/10;
  - online mean: about 123.5 ms;
  - online P95: about 245.4 ms, above the 150 ms target;
  - median dense clearance improvement: 0 m, max improvement about 0.001 m.
- Interpretation: the v3 modular structure is in place, but the temporary finite-difference/SLSQP scaffold is not yet the desired Fast CCRO-NUBS solver. It should be treated as an implementation scaffold. The next real implementation step is OSQP-style QP with analytic nearest-point Jacobians and a better active-set model.

Fast CCRO-NUBS v4 active-set scaffold:

- Added analytic point-Jacobian support:
  - `robot/urdf_model.py` now exposes world-frame joint origins/axes;
  - `planning/robot_surface_model.py` now exposes a link-local surface-point translational Jacobian.
- Replaced risk-sample active constraints with dense nearest Mesh active constraints in `repair/active_distance.py`.
- Removed SLSQP from the v4 path. The current environment does not include OSQP, so `repair/local_qp_solver.py` now uses a deterministic projected half-space QP fallback with the same linearized constraint interface.
- Added `ccro_fast_v4` and `critical_fast_v4` method entries.
- G1-near CCRO v4 output: `results/new/6_4_fast_local_repair_g1_near_ccro_v4/paper/table_6_4_fast_local_repair.md`.
- G1-near CCRO v4 result:
  - valid near-risk scenarios: 10/10;
  - dense feasible: 0/10;
  - usable candidate: 0/10;
  - acceleration gate pass: 10/10;
  - online mean: about 128.4 ms;
  - online P95: about 183.2 ms, above the 150 ms target;
  - QP fallback median time: about 3.4 ms;
  - median dense clearance improvement: 0 m, max improvement about 0.00024 m.
- Interpretation: v4 moves the implementation closer to the intended active-set CCRO-QP structure and removes SLSQP from the repair step, but it still does not pass G1-near. The QP fallback is fast, yet the accepted linearized updates do not translate into meaningful dense clearance recovery. The remaining blocker is the active-set/Jacobian/local-DOF effectiveness, not merely solver runtime.

Fast CCRO-NUBS v4 controllability audit and gated update:

- Added `experiments/new/6_4/v4_linearization_audit.py` for a single-instance audit of:
  - dense active points;
  - analytic point Jacobian versus finite difference;
  - NUBS interpolation-point sensitivity;
  - QP predicted clearance gain versus actual dense clearance gain;
  - applied delta norm and online verifier outcome.
- The first audit exposed an invalid G1-near design issue: the dominant dense risk lay at the fixed 1.0 s local tail, where local interpolation variables had zero effective sensitivity.
- Updated G1/G1-near generation:
  - G1 conflict time is now 0.55 s instead of the previous 0.90 s tail-adjacent conflict;
  - generated G1 samples must pass a dense active-set controllability filter;
  - endpoint-dominated risks and active rows with near-zero NUBS sensitivity are rejected;
  - clearance candidates are selected by dense Mesh filtering rather than one-shot random clearance.
- Controllable single-instance audit output:
  - `results/new/6_4_v4_linearization_audit_d1_00_controllable/paper/table_6_4_v4_linearization_audit.md`;
  - reference dense min: about 0.0774 m;
  - candidate dense min: about 0.0853 m;
  - actual dense gain: about 0.0080 m;
  - QP predicted min distance: about 0.0867 m;
  - dominant active row time: 0.4 s;
  - dominant active-row norm: about 0.212;
  - analytic point-Jacobian relative error was effectively zero.
- Updated v4 online acceptance:
  - v4 uses one QP iteration in the online path;
  - accepted steps are backtracked against 0.04 s motion samples;
  - v4 targets 0.095 m clearance while keeping the online acceptance threshold at 0.09 m and dense GT threshold at 0.08 m.
- Current G1-near CCRO-fast-v4 gated output:
  - `results/new/6_4_fast_local_repair_g1_near_ccro_v4_controllable_target095/paper/table_6_4_fast_local_repair.md`;
  - valid near-risk scenarios: 10/10;
  - dense geometry feasible: 10/10;
  - acceleration gate pass: 10/10;
  - hard real-time pass under 200 ms: 10/10;
  - online P95: about 136.1 ms;
  - online max: about 142.6 ms;
  - usable candidate rate: 5/10;
  - mean dense clearance improvement: about 0.0125 m.
- Interpretation: this is the first v4 result where the linearized active-set repair produces meaningful dense clearance gain within the realtime budget. However, it still does not justify launching a formal full 6.4 dataset: only 5/10 near-risk cases satisfy the stricter 0.09 m online acceptance margin, and the paper claim should remain at the G1 development/capability-boundary level until a larger gated validation set passes.

G1 acceptance audit and Critical-point baseline:

- Added the acceptance-audit funnel to `experiments/new/6_4/fast_local_repair_64.py`.
- The fast repair output now records:
  - `reference_online_min_distance`;
  - `candidate_online_min_distance`;
  - `candidate_medium_dense_gap`;
  - `online_threshold_margin`;
  - `dense_threshold_margin`;
  - per-gate booleans for QP solved, dense repair success, online distance pass, acceleration pass, and time pass.
- The generated audit table is `paper/table_6_4_fast_acceptance_audit.md`.
- Fixed the Critical-fast-v4 baseline semantics:
  - Critical-fast-v4 now uses the same v4 one-QP/backtracking solver settings;
  - but its active constraints come from the fixed critical-point evaluator rather than the CCRO dense Mesh active set.
- CCRO-fast-v4 G1-near acceptance audit:
  - output: `results/new/6_4_fast_local_repair_g1_near_ccro_v4_acceptance_audit/paper/table_6_4_fast_acceptance_audit.md`;
  - QP solved: 10/10;
  - dense repair success: 10/10;
  - online acceptance: 5/10;
  - verified safety among online-accepted candidates: 5/5;
  - time pass under 150 ms: 10/10;
  - acceleration pass: 10/10;
  - mean medium-minus-dense candidate distance gap: about 0.0004 m;
  - all 5 CCRO online rejections were `dense_safe_but_online_margin_rejected`.
- Critical-fast-v4 G1-near baseline:
  - output: `results/new/6_4_fast_local_repair_g1_near_v4_acceptance_audit/paper/table_6_4_fast_acceptance_audit.md`;
  - QP solved: 8/10;
  - dense repair success: 6/10;
  - online acceptance: 1/10;
  - verified safety among online-accepted candidates: 1/1;
  - time pass under 150 ms: 9/10;
  - acceleration pass: 10/10;
  - mean dense clearance improvement: about 0.0077 m.
- Interpretation: the current CCRO result should no longer be described as "5/10 algorithm success." A more precise conclusion is that Fast CCRO-NUBS generated dense-safe candidates for all G1-near cases within the real-time budget, while the predefined conservative online margin accepted half of them. The Critical-point baseline supports the CCRO contribution: under the same solver shell and instances, the sparse critical-point active set repaired fewer dense GT cases and produced only 1/10 online-accepted candidates.

Margin-stability sweep and calibration:

- Added explicit v4 development parameters:
  - `FAST_V4_TARGET_CLEARANCE`;
  - `FAST_V4_MAX_ITERATIONS`;
  - `FAST_V4_CLEARANCE_REWARD`;
  - CLI overrides in `fast_local_repair_64.py` for transparent development sweeps.
- Added `experiments/new/6_4/fast_v4_sweep_64.py` for the G1-near margin sweep.
- Added `experiments/new/6_4/margin_calibration_64.py` for online-vs-dense calibration tables.
- No safety threshold was relaxed:
  - online acceptance remains `D_ONLINE_ACCEPT = 0.09 m`;
  - dense GT safety remains `D_STOP = 0.08 m`.
- Reward sweep output:
  - `results/new/6_4_fast_v4_margin_sweep_reward03/paper/table_6_4_fast_v4_margin_sweep.md`;
  - clearance reward `0.3` did not improve online acceptance;
  - several cases introduced dense repair loss or timing tails.
- Target/iteration sweep output:
  - `results/new/6_4_fast_v4_margin_sweep_reward00/paper/table_6_4_fast_v4_margin_sweep.md`;
  - `target=0.095, iterations=1, reward=0.0` remained Pareto-best on the current 10-case development set:
    - repair success: 10/10;
    - online acceptance: 5/10;
    - verified safety: 5/5;
    - time pass: 10/10;
    - acceleration pass: 10/10;
    - online P95: about 130.3 ms.
  - two QP iterations increased dense clearance but exceeded the realtime budget: P95 about 215.7--393.2 ms depending on target.
  - target clearances `0.100` and `0.105` did not improve online acceptance and introduced dense repair failures or timing tails.
- Margin calibration output:
  - `results/new/6_4_margin_calibration_g1_near_v4/paper/table_6_4_margin_calibration.md`;
  - CCRO candidate medium-minus-dense gap mean was about 0.0004 m;
  - CCRO P95 absolute gap was about 0.0020 m;
  - this shows that current online rejection is dominated by the 0.09 m conservative margin rather than a large medium-vs-dense evaluator discrepancy.
- Parameter-freeze decision for the next stage:
  - keep `target_clearance=0.095`;
  - keep one online QP iteration;
  - keep `clearance_reward=0.0`;
  - keep online and dense thresholds unchanged.
- Interpretation: the attempted margin maximization and sequential refinement are useful negative controls. They improve neither online acceptance nor paper credibility on the current G1-near development set. The next credibility step should be scenario stratification and reporting the three-rate funnel (`repair success`, `online acceptance`, `verified safety`) rather than further local solver tuning.

G1-band capability boundary after parameter freeze:

- Added `experiments/new/6_4/fast_v4_band_study_64.py`.
- Frozen Fast CCRO-NUBS v4 parameters:
  - `target_clearance=0.095`;
  - one online QP iteration;
  - `clearance_reward=0.0`;
  - online threshold remains 0.09 m;
  - dense GT threshold remains 0.08 m.
- Output:
  - `results/new/6_4_fast_v4_g1_band_study/paper/table_6_4_fast_v4_g1_band_study.md`.
- D1 G1-band results:
  - Easy band `[0.075, 0.080)`:
    - Critical-fast-v4: repair success 9/10, online acceptance 4/10, verified safety 4/4, P95 about 113.1 ms;
    - CCRO-fast-v4: repair success 10/10, online acceptance 8/10, verified safety 8/8, P95 about 129.4 ms.
  - Main band `[0.065, 0.075)`:
    - Critical-fast-v4: repair success 8/10, online acceptance 3/10, verified safety 3/3, P95 about 123.4 ms;
    - CCRO-fast-v4: repair success 9/10, online acceptance 9/10, verified safety 9/9, P95 about 130.3 ms.
  - Hard band `[0.055, 0.065)`:
    - Critical-fast-v4: repair success 1/10, online acceptance 1/10, verified safety 1/1, P95 about 150.6 ms;
    - CCRO-fast-v4: repair success 3/10, online acceptance 2/10, verified safety 2/2, P95 about 146.7 ms.
- Interpretation:
  - The band study gives a credible capability curve rather than a single success number.
  - CCRO's dense whole-body active set consistently outperforms the sparse Critical-point active set under the same solver shell.
  - The main band is a strong candidate for the paper's primary G1 result: CCRO-fast-v4 reached 9/10 dense repair success and 9/10 online acceptance with 100% verified safety among accepted candidates.
  - The hard band should be presented as the capability boundary, not as the main operating regime.
  - This confirms the current freeze decision: further solver tuning is lower priority than moving to formal D1/D2 fast experiments with the frozen v4 configuration and three-rate reporting.

Final paper-table consolidation:

- Added `experiments/new/6_4/fast_v4_paper_tables_64.py`.
- Generated paper-ready tables from the frozen G1-band study:
  - `results/new/6_4_fast_v4_g1_band_study/paper/table_6_4_fast_v4_main_performance.md`;
  - `results/new/6_4_fast_v4_g1_band_study/paper/table_6_4_fast_v4_clearance_gain.md`;
  - `results/new/6_4_fast_v4_g1_band_study/paper/table_6_4_fast_v4_runtime_decomposition.md`.
- Clearance-gain evidence:
  - in the main band, Critical-fast-v4 improved dense clearance by about 0.0128 m on average;
  - CCRO-fast-v4 improved dense clearance by about 0.0193 m on average.
- Runtime evidence:
  - in the main band, CCRO-fast-v4 online P95 was about 130.3 ms;
  - mean QP time was about 8.87 ms;
  - mean online gate time was about 22.72 ms;
  - dense recheck was reported separately as offline GT validation.
- Final state:
  - Fast CCRO-NUBS v4 is frozen for paper writing and D1/D2 extension.
  - Further solver tuning is explicitly out of scope unless new D1/D2 evidence contradicts the G1-band capability curve.
