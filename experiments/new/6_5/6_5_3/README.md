# 6.5.3 Dynamic Obstacle Fast CCRO-NUBS Real-System Trial

This folder contains the staged implementation for Section 6.5.3. D1 r02-r15
are development logs and are not formal paper evidence.

The experiment reuses the safe 6.5.2 tabletop reference:

- home joints: `[0, 0, 90, 0, 90, 0] deg`
- TCP X: home TCP + `0.10 m`
- TCP Y: `+0.40 m -> -0.40 m`
- TCP Z / orientation: fixed from the home TCP pose

## Modes

`run_653_dynamic_repair_trial.py` supports:

- `shadow`: RealSense + AUBO feedback only, no robot command.
- `dynamic-track-audit`: stationary robot; clustering/tracking audit only. It
  never sends motion commands, never triggers STRO/Fast repair, and writes
  `clusters.csv` plus `tracks.csv`.
- `moving-shadow-stop`: command the recorded one-way low-speed reference line, trigger Fast
  CCRO-NUBS, then stop; perform post-planning authorization in shadow only.
- `live-stop-replan-execute`: currently stops and plans but is fail-closed before
  execution while the shadow-authorization protocol is being validated.

Use `moving-shadow-stop` as the required pilot before enabling true online
trajectory switching.

The moving modes fail closed unless `--reference-feedback-csv` is supplied.
The same recorded reference drives future STRO evaluation and supplies the
candidate endpoint at `t + H_local`; `q + qd * lookahead` is no longer used.

Before the next dynamic trial, run an empty-workspace alignment audit. The
point-cloud hard guard remains active, while `--reference-audit-only` prevents
all STRO/Fast triggers:

```bash
/home/hzy/miniconda3/envs/py310/bin/python experiments/new/6_5/6_5_3/run_653_dynamic_repair_trial.py \
  --scene D1 --repeat 1 --mode moving-shadow-stop --reference-audit-only \
  --operator-phrase CCRO_653_DYNAMIC_SHADOW_APPROVED \
  --reference-feedback-csv results/new/6_5/6_5_3/reference_xp10_line/reference_feedback.csv \
  --x-offset 0.10 --y-start 0.40 --y-goal -0.40 \
  --line-velocity-m-s 0.020 --line-acc-m-s2 0.05 --duration-s 45 \
  --output results/new/6_5/6_5_3/reference_alignment_validation
```

Proceed only when the summary status is `REFERENCE_ALIGNMENT_PASS`.

## Dynamic-track audit before r18

Keep the robot stationary and perform one continuous foam crossing per run:

```bash
/home/hzy/miniconda3/envs/py310/bin/python experiments/new/6_5/6_5_3/run_653_dynamic_repair_trial.py \
  --scene D1 --repeat 1 --mode dynamic-track-audit --duration-s 15 \
  --output results/new/6_5/6_5_3/dynamic_track_audit
```

The dynamic tracker uses a five-sample timestamped net-motion window, enters
dynamic state at 0.08 m/s, and exits only after three samples below 0.04 m/s.
The first prediction-ready frame (age/history/association/dynamic-state checks)
may enter STRO immediately; the stricter two-frame `dynamic_valid` state remains
an audit-quality label. STRO and Fast repair use the same frozen window velocity
vector. Because prediction-ready tracks are already classified by the window
hysteresis, STRO does not reapply a scalar static-speed threshold and always
emits future samples over 0.1--0.5 s. Every robot link may trigger predicted risk or the 0.12 m
current-distance fallback stop; scene metadata has no control authority.
It may retain identity across two missed frames, but missed tracks are never
returned for STRO prediction. The audit passes when one ID lasts at least five
tracked frames, reaches 0.08 m/s, and is dynamically valid for at least two
consecutive frames. The safety and dynamic trackers both receive all external
clusters remaining after workspace, plane, point-count, and volume filtering.
Radius is geometry, not an object-identity gate: a connected foam-plus-support
component is conservatively treated as one obstacle. `tracks.csv` records `cluster_radius_raw_m`,
`tracked_radius_m`, and conservative `risk_radius_m` separately.

The perception settings are frozen at `cluster_eps=0.05` with temporal denoise
enabled. Run one final stationary audit; one credible passing moving track ends
the audit phase. A thin or camera-occluded support is preferred but is not a
PASS condition.

For fixture diagnosis, the non-commanding audit/shadow modes support an
Open3D view without changing the formal perception protocol:

```bash
/home/hzy/miniconda3/envs/py310/bin/python experiments/new/6_5/6_5_3/run_653_dynamic_repair_trial.py \
  --scene D1 --repeat 1 --mode dynamic-track-audit --duration-s 8 \
  --visualize-audit --show-filtered --show-noise \
  --output results/new/6_5/6_5_3/fixture_visual_diagnostic
```

The view separates robot, plane, valid-cluster, filtered-cluster, and DBSCAN
noise points, and overlays each valid cluster's OBB, 90-percent raw-radius
sphere, and center. Prediction-ready clusters are highlighted. The option is
rejected in every robot-commanding mode. Independently of visualization,
audit/shadow runs save each cluster with `max(bbox)>0.20 m` or raw radius
`>0.12 m` under `anomalous_clusters/` as a compressed NPZ containing the exact
points and frame/cluster/track metadata. These are diagnostic outputs only;
they do not filter, resize, or otherwise alter an obstacle.

## 1. Optional Reference Recording

Dry run:

```bash
/home/hzy/miniconda3/envs/py310/bin/python experiments/new/6_5/6_5_3/prepare_653_reference.py \
  --output results/new/6_5/6_5_3/reference_xp10_line
```

Real low-speed reference recording:

```bash
/home/hzy/miniconda3/envs/py310/bin/python experiments/new/6_5/6_5_3/prepare_653_reference.py \
  --execute \
  --operator-phrase CCRO_653_REFERENCE_APPROVED \
  --reference-shape line \
  --x-offset 0.10 \
  --y-start 0.4 \
  --y-goal -0.4 \
  --line-velocity-m-s 0.025 \
  --line-acc-m-s2 0.06 \
  --record-duration-s 40 \
  --output results/new/6_5/6_5_3/reference_xp10_line
```

## 2. D1/D2 Shadow Trial

This mode does not move the robot:

```bash
/home/hzy/miniconda3/envs/py310/bin/python experiments/new/6_5/6_5_3/run_653_dynamic_repair_trial.py \
  --scene D1 \
  --repeat 1 \
  --mode shadow \
  --duration-s 18 \
  --output results/new/6_5/6_5_3/dynamic_repair_pilot
```

For D2:

```bash
/home/hzy/miniconda3/envs/py310/bin/python experiments/new/6_5/6_5_3/run_653_dynamic_repair_trial.py \
  --scene D2 \
  --repeat 1 \
  --mode shadow \
  --duration-s 18 \
  --output results/new/6_5/6_5_3/dynamic_repair_pilot
```

## 3. Moving Shadow With Stop

This commands the low-speed reference line and stops after risk trigger. It
generates and validates a Fast CCRO-NUBS candidate, but does not switch to it.

```bash
/home/hzy/miniconda3/envs/py310/bin/python experiments/new/6_5/6_5_3/run_653_dynamic_repair_trial.py \
  --scene D1 \
  --repeat 1 \
  --mode moving-shadow-stop \
  --reference-feedback-csv results/new/6_5/6_5_3/reference_xp10_line/reference_feedback.csv \
  --operator-phrase CCRO_653_DYNAMIC_SHADOW_APPROVED \
  --x-offset 0.10 \
  --y-start 0.4 \
  --y-goal -0.4 \
  --line-velocity-m-s 0.020 \
  --line-acc-m-s2 0.05 \
  --duration-s 18 \
  --output results/new/6_5/6_5_3/dynamic_repair_pilot
```

After the single stationary audit passes, run one D1 `moving-shadow-stop`
pilot. It should trigger predicted risk at 0.14 m while current all-link
clearance remains above the 0.12 m fallback stop and the independent raw-cloud
guard remains above 0.10 m. The candidate criterion remains `accepted_steps >
0`, at least 3 mm clearance gain, candidate clearance at least 0.09 m, and no
safety-gate failure. This produces `LOCAL_REPAIR_READY`; it does not by itself
authorize execution.

The stop interface is frozen based on `stop_interface_validation/r03`; do not
repeat that validation merely to tune Fast-repair thresholds. Formal settings
remain `replan=0.14 m`, `H_local=1.0 s`, `online_accept=0.09 m`, and
`fast_budget=150 ms`; current-distance fallback stop is 0.12 m and raw-cloud
hard guard is 0.10 m.

The 150 ms Fast limit is a fail-closed online deadline. The repair loop checks
the deadline before each expensive stage and scale distance scan. Candidate
acceptance uses complete online pipeline time (repair, candidate verification,
required reference verification, and final comparison), rather than reporting
repair-only time as the online latency.

Fast repair optimizes `z=[Delta Q, delta q_T]`. The six elastic tail-position
variables are part of the finite NUBS sensitivity and QP, while terminal
velocity and acceleration remain equal to the recorded reference. The Fast
stage validates only the 1 s local segment and reports either
`FAST_REPAIR_FAILED` or `LOCAL_REPAIR_READY`. Rejoin is deliberately excluded
from this first-stage result.

Scale candidates use linearized active-distance screening plus exact motion
checks. Only the selected candidate receives full online geometric
verification, avoiding repeated dense scans without weakening the final gate.
The candidate and reference profiles are written separately as
`fast_candidate_risk_profile.csv` and `fast_reference_risk_profile.csv`.

After a trigger, robot motion is stopped before planning. The system then
captures a fresh 0.6 s RGB-D window and associates observations against the
trigger track's predicted position. At least three associated frames spanning
0.25 s are required. The latest center, conservative maximum observed radius,
and a timestamped linear-regression velocity replace the trigger-time state
for repair and bounded-rejoin validation. Missing, stale, or unassociated data
produces `REJECTED_FRESH_RECHECK` and keeps the robot stopped. The acquisition
time is logged separately; the unchanged 150 ms budget starts with Fast repair.
`post_stop_fresh_recheck.json` records every association decision.

The latest associated fresh cluster (only the final frame, never an
uncompensated multi-frame union) is decomposed along its first PCA axis into
one to four local spheres. Every observed point must be covered by the sphere
union; the 5 mm fit margin is geometric only, while the existing Fast forecast
continues to apply its unchanged margin and uncertainty. The audit is saved as
`fresh_multisphere.json`, and the exact source points as
`fresh_latest_cluster_points.npy`. STRO retains its fast conservative
object-level sphere; multi-sphere refinement is used only after stopping.

After `LOCAL_REPAIR_READY`, a second post-planning RGB-D acquisition requires
at least three associated frames spanning 0.25 s. It rebuilds the multisphere
geometry and velocity with the candidate execution start as the new forecast
time origin. The unchanged 1.25--2.00 s rejoin window is searched, a C2 bridge
is appended, and the entire repair-plus-bridge trajectory is verified again
from tau zero. Success is recorded as `EXECUTION_AUTHORIZED` and emits the
`EXECUTION_AUTHORIZED_SHADOW` event; this pilot still sends no execution
command. Any association, geometry, rejoin, motion, or full-verification
failure produces `POST_PLAN_RECHECK_FAILED` and holds the robot stopped. No
automatic second Fast repair is attempted. Fresh #2 acquisition and
authorization latency are logged separately from the frozen 150 ms Fast
budget.

A 20-repeat r26 A/B found exact agreement between serial and threaded paired
verification, but threading was slower (67.75 ms versus 55.18 ms median).
Therefore the production Fast path retains serial candidate/reference
verification; `paired_verifier_audit.json` records the rejected optimization.
The retained serial verifier uses an exact clearance-only geometry kernel:
it keeps the same 0.04 s samples, medium surface density, forecast occupancy,
joint checks, and thresholds, while omitting risk-cost/contact fields that the
verifier never consumes. On the same r26 26-sample trajectory it matched every
minimum distance/link/object result and reduced one verification median from
22.74 ms to 12.20 ms. The audit is stored in
`verifier_clearance_kernel_audit.json`.
Ten complete r26 offline replays then produced `LOCAL_REPAIR_READY` in every
run, with a 127.92 ms median and 123.56--129.45 ms range; all ten met the
unchanged 150 ms deadline. See `offline_fast_budget_replay.json`.

In r27 the dense active scan exhausted the deadline before QP because it built
Jacobians for every time/link/sphere candidate before discarding all but eight.
The implementation now sorts the exact vertex distances first and evaluates
Jacobians only until the same first eight valid constraints are obtained. Ten
r27 offline replays reduced the risk-scan median from the recorded 145.03 ms to
11.91 ms and the complete Fast median to 59.14 ms, with accepted steps in all
ten runs. The unchanged 3 mm gain gate still rejected this particular geometry
(representative gain 1.88 mm), so it is not labeled `LOCAL_REPAIR_READY`.

If repair returns `accepted_steps=0`, the online decision ends immediately.
At most one reference verification may run afterwards as diagnostics and is
reported as `diagnostic_reference_verification_ms`, outside
`online_pipeline_elapsed_ms`. Every trial summary also records `git_commit`
and `git_dirty` for source/result traceability.

For D1, move the obstacle laterally across a future robot swept region,
approximately perpendicular or oblique to the robot reference motion.
Do not move it head-on along the reference line toward the current gripper.
Complete the main crossing naturally in about 2--3 s after `REFERENCE_ARMED`.
The desired evidence is `D_predicted < 0.14 m` while `D_current > 0.12 m` and
`D_guard > 0.10 m`. D2 uses an opposing or oblique approaching path. The
actual most-at-risk link is measured rather than prescribed.

## Unified D1/D2 formal protocol

D1 and D2 differ only in obstacle motion geometry. Both use protocol ID
`653_unified_d1_d2_v1`, including the same perception, window-motion, STRO,
all-link safety, Fast-repair, and candidate-acceptance parameters. Every moving
trial writes the complete `formal_protocol` signature to `summary.json`.
Changing any frozen algorithm/safety value through the CLI causes
`BLOCKED_NONFORMAL_PROTOCOL` before a robot connection is opened. Scene,
repeat, duration, output path, and operator/reference inputs remain ordinary
experimental metadata rather than algorithm parameters.

## Outputs

Each trial writes:

- `frames.csv`
- `summary.json`
- `candidate/candidate_summary.json` if triggered
- `candidate/fast_ccro_nubs_candidate.csv` if triggered

Plot one trial:

```bash
/home/hzy/miniconda3/envs/py310/bin/python experiments/new/6_5/6_5_3/plot_653_dynamic_trial.py \
  --trial-dir results/new/6_5/6_5_3/dynamic_repair_pilot/trials/D1_crossing_body_r01
```

Summarize a result folder:

```bash
/home/hzy/miniconda3/envs/py310/bin/python experiments/new/6_5/6_5_3/summarize_653_dynamic_results.py \
  --input results/new/6_5/6_5_3/dynamic_repair_pilot
```

Official 6.5.3 live switching should only be enabled after D1/D2
`moving-shadow-stop` logs show stable trigger timing and accepted candidates.

## Stop Interface Validation

After any contact in moving dynamic pilots, do not continue obstacle trials.
First validate the controller stop API in an empty workspace:

```bash
/home/hzy/miniconda3/envs/py310/bin/python experiments/new/6_5/6_5_3/validate_653_stop_interface.py \
  --execute \
  --operator-phrase CCRO_653_STOP_VALIDATION_APPROVED \
  --x-offset 0.10 \
  --y-start 0.4 \
  --y-goal -0.4 \
  --line-velocity-m-s 0.010 \
  --line-acc-m-s2 0.025 \
  --stop-after-s 3.0 \
  --record-after-stop-s 4.0 \
  --output results/new/6_5/6_5_3/stop_interface_validation/r01
```
