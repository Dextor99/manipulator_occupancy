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

Passed component gate:

- Critical-point gradient query: about 6.80 ms in the latest gate run.
- CCRO mesh gradient query: about 9.19 ms in the latest gate run.
- Critical-point count: 16.
- Critical-point query is faster than CCRO mesh query.

Blocked mechanism gate:

- The latest automatic D1 mechanism gate ran 6 CCRO-NUBS trials.
- `gate_pass=false`.
- Task-safe success was 4/6, below the required 5/6.
- Replan success was 0/6, below the required 4/6.
- Deadline misses were 0, so the 6 s operating-domain budget is adequate for this implementation.
- Continuity rejections occurred in 2 candidate events.
- Online distance rejections occurred in 6 candidate events.
- Planner P95 was about 5.48 s, within the 5.5 s planning budget but close to the limit.
- Safe-with-switch count was 0/6.
- This means the final formal run should not be launched. The system can often remain safe by slowed execution, but it does not yet demonstrate stable asynchronous candidate switching.

Final stop-rule decision:

- Do not continue adding more optimization modules inside 6.4.
- Do not run the final D1-main/D2-main/D2-stress formal dataset.
- Use the archived `results/new/6_4_async_v3_boundary` data only as a boundary/diagnostic result, not as final proof of stable asynchronous replanning.
- Write 6.4 in the reduced form: dynamic risk trigger, safety slowdown/hold behavior, candidate-generation diagnostics, and current operating-domain boundary.
- Keep the stronger asynchronous replanning claim for future work or for a later implementation with a faster/lower-level optimizer.
