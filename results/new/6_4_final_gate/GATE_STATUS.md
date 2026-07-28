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

Final stop-rule decision:

- Do not continue adjusting individual failed D1 pilot cases.
- Do not lower `D_ONLINE_ACCEPT`, retune weights, or reshape obstacle trajectories to make the pilot pass.
- Do not run a new formal closed-loop dataset until G0 plus a simple feasible validation set and Stage-A candidate replay show acceptable candidate feasibility.
- Reframe 6.4 as an asynchronous local replanning effectiveness and capability-boundary experiment.
- Use the archived `results/new/6_4_async_v3_boundary` and current final-gate data only as pilot/diagnostic material, not as final proof of stable asynchronous replanning.
- Do not treat invalid replay windows as failures of CCRO-NUBS or Critical-NUBS. They are experiment-design exclusions that must be reported separately.
