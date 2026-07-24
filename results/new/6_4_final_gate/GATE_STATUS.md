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

Passed component gate:

- Critical-point gradient query: about 6.95 ms in the latest gate run.
- CCRO mesh gradient query: about 9.32 ms in the latest gate run.
- Critical-point count: 16.
- Critical-point query is faster than CCRO mesh query.

Passed mechanism sub-gate:

- Pending switch-state consistency is now essentially fixed in the D1 mechanism gate.
- Mean `tau_prediction_error_at_switch` is about `2.8e-4 s`.
- There are no systematic continuity rejections.

Blocked mechanism gate:

- The latest automatic D1 mechanism gate ran 3 CCRO-NUBS smoke trials.
- Task-safe success was 1/3.
- Replan success was 0/3.
- Deadline misses occurred in 4 candidate events.
- Online distance rejections occurred in 2 candidate events.
- Planner P95 was about 5.34 s.
- This means the next formal run should not be launched yet. The current code now exposes the remaining local candidate optimization limitation instead of hiding it with near-stop behavior.

Recommended next step:

- Improve the local candidate segment itself: reduce variables further, add a collision-side detour initialization, or add a stronger geometric clearance objective so local candidates pass the medium Mesh distance gate within the 3 s slot.
- After D1 reaches at least 5/6 task-safe and 4/6 accepted switches in the automatic mechanism gate, add D2-main to the 12-case mechanism gate before producing the final D1-main/D2-main/D2-stress formal dataset.
