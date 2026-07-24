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

Passed component gate:

- Critical-point gradient query: about 6.63 ms.
- CCRO mesh gradient query: about 8.84 ms.
- Critical-point count: 16.
- Critical-point query is faster than CCRO mesh query.

Blocked mechanism gate:

- A D1-main single-sample gate showed that, after removing the 2% pending near-stop behavior, CCRO-NUBS can remain safe but fail to finish or fail to produce an accepted candidate under the current 3 s switch slot.
- This means the next formal run should not be launched yet. The current code now exposes the real planning/execution limitation instead of hiding it with near-stop behavior.

Recommended next step:

- Add the full execution-layer risk correction for CCRO-NUBS, or introduce a local candidate segment/warm-start optimization so CCRO candidates pass the 3 s planning gate without relying on near-stop waiting.
- After that, rerun the 12-case mechanism gate before producing the final D1-main/D2-main/D2-stress formal dataset.
