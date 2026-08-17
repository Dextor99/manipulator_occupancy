# 6.5.3 Dynamic Obstacle Fast CCRO-NUBS Real-System Trial

This folder contains the staged implementation for Section 6.5.3. D1 r02-r15
are development logs and are not formal paper evidence.

The experiment reuses the safe 6.5.2 tabletop reference:

- home joints: `[0, 0, 90, 0, 90, 0] deg`
- TCP X: home TCP, with `x-offset = 0.0 m`
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
- `live-stop-replan-execute`: after explicit opt-in and phrase checks, executes
  the v2 fail-closed state machine. It prefers a Fresh #2-authorized full
  repair+rejoin. If only the local repair is authorized, it executes that
  segment, holds at its repaired tail, and requires Fresh #3 to authorize a
  separate C2 bridge to a later reference state before guarded Cartesian resume.

Use `moving-shadow-stop` as the required pilot before enabling true online
trajectory switching.

## Empty-scene v2 local-first state-machine calibration

Before the first v2 D2 live trial, validate the fallback as one complete
empty-scene sequence: Fresh #2-authorized local repair, hold, Fresh #3,
verified delayed C2 bridge, and guarded Cartesian resume. The dedicated tool
defaults to a non-commanding artifact audit:

```bash
/home/hzy/miniconda3/envs/py310/bin/python \
  experiments/new/6_5/6_5_3/calibrate_653_local_delayed_rejoin.py \
  --repeat 1
```

Real execution additionally requires `--execute`, the exact phrase
`CCRO_653_EMPTY_SCENE_LOCAL_DELAYED_REJOIN_APPROVED`, a clean worktree, three
clear hard-guard preview frames, and a start-joint match. A mismatch returns
`BLOCKED_START_MISMATCH` without commanding the robot. During local and bridge
Offline Track execution the existing 0.10 m RGB-D hard guard remains active.
Fresh #3 must authorize the bridge; otherwise the robot remains at the local
repair tail. This calibration never invokes a second Fast solve.

If the start does not match, use `align_653_authorized_start.py`. Its v2
default is the same formal D1 r01 local candidate, and it traverses only the
matched recorded-reference segment rather than issuing a free-space movej.
New files under `results/` do not block this positioning audit; any uncommitted
source or configuration change still does.

The Offline Track timing audit uses the executor's valid completion time
(`goal_check.elapsed_s`), which already requires goal tolerance, at least 90%
of the authorized duration, nonzero observed motion, and no hard-guard stop.
The first entry into goal tolerance remains logged as diagnostic evidence but
does not decide timing acceptance; this avoids false failures for short,
small-amplitude bridges without changing the frozen +/-20% timing band.

If an older run physically reached its authorized delayed-rejoin endpoint but
was stopped only by the former first-tolerance audit, use the one-shot
`resume_653_from_delayed_rejoin.py`. It checks the current joints against the
saved authorized bridge endpoint and performs three hard-guard previews before
allowing the existing guarded Cartesian resume. Such a run is archived as a
recovery/completion record, not relabeled as an uninterrupted formal trial.

## Empty-scene local Offline Track calibration

Before the first live dynamic execution, calibrate only the already authorized
r35 local trajectory. The calibration utility never moves the robot to the
candidate start: a start mismatch is logged as `BLOCKED_START_MISMATCH` with no
motion command. Its default mode is also non-commanding. Inspect the setup with:

```bash
/home/hzy/miniconda3/envs/py310/bin/python \
  experiments/new/6_5/6_5_3/calibrate_653_local_offline_track.py \
  --playback-duration-s 1.0 \
  --repeat 1
```

Real empty-scene playback additionally requires `--execute` and the exact
phrase `CCRO_653_EMPTY_SCENE_LOCAL_TRACK_APPROVED`. It uses the authorized 1 s
CSV, checks three hard-guard preview frames, keeps the 0.10 m guard active
during Offline Track, and records requested/observed duration, goal error,
nonzero motion, tracking RMSE, and maximum tracking error. Run 1.00 s three
times; test the separately authorized 1.25 s version only if 1.00 s is unstable.
The generic candidate playback default is `0`, meaning the authorized CSV's
native time axis, while formal commands still state the frozen duration.
The default D1/D2 formal duration remains 1.0 s. A shorter playback is never
implicit: use `--allow-experimental-playback-duration` with a duration in the
bounded 0.80--1.00 s range, and require the same latest-state, dynamics,
tabletop, and raw-guard authorization before execution.

If calibration reports `BLOCKED_START_MISMATCH`, do not relax the tolerance or
use a free-space movej. `align_653_authorized_start.py` matches the current and
authorized-start joints to the recorded reference, then (with explicit execute
gates) traverses only that reference segment in the required direction. It
uses three empty-scene guard previews and retains the 0.10 m hard guard during
alignment. The default 4 s alignment is positioning only and does not calibrate
the 1 s candidate playback.

After one empty-scene candidate playback, the elastic endpoint need not lie on
the recorded reference. Before a repeat, `return_653_local_candidate_start.py`
may traverse the exact authorized waypoint geometry in reverse. It refuses to
start unless the robot is already within the frozen start tolerance of the
authorized endpoint, and it retains the same empty-scene preview, explicit
phrase, Offline Track checks, and 0.10 m execution hard guard. This return is
calibration positioning only; it is not the automatic task-rejoin method.

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
  --reference-feedback-csv results/new/6_5/6_5_3/reference_xp00_line/reference_feedback.csv \
  --x-offset 0.0 --y-start 0.40 --y-goal -0.40 \
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
  --output results/new/6_5/6_5_3/reference_xp00_line
```

Real low-speed reference recording:

```bash
/home/hzy/miniconda3/envs/py310/bin/python experiments/new/6_5/6_5_3/prepare_653_reference.py \
  --execute \
  --operator-phrase CCRO_653_REFERENCE_APPROVED \
  --reference-shape line \
  --x-offset 0.0 \
  --y-start 0.4 \
  --y-goal -0.4 \
  --line-velocity-m-s 0.020 \
  --line-acc-m-s2 0.05 \
  --record-duration-s 45 \
  --output results/new/6_5/6_5_3/reference_xp00_line
```

## 2. D1/D2 Shadow Trial

The final representative D2 end-effector demonstration uses the explicit
audit label `D2_END_EFFECTOR_OPPOSING_XP00`, the recorded
`reference_xp00_line`, and `--x-offset 0.0`.  The obstacle is placed on a
fixed physical lane approximately 0.13--0.16 m laterally from the TCP lane
(nominally `X_obs ~= 0.64 m` for the home-TCP line); this is a scene setup
quantity, not the robot reference `--x-offset`.  The label and reference
offset are written to `summary.json`; they do not change planning or control
behavior.

For the final one-segment real execution, use
`run_653_simple_dynamic_nubs_live.py`.  It keeps the whole-body CCRO risk
link selection, generates bounded 0.04/0.06/0.08 m lateral seeds, invokes the
unchanged Fast verifier, executes at most one authorized 1 s local segment,
and then holds at the measured local tail.  It does not force an EEF-only
risk link, automatically rejoin the reference, or continue to the goal.

`live-stop-replan-execute` uses a bounded stopped-state rolling phase after
the strict initial Fresh observation.  If Fast fails, or a later Fresh update
invalidates the old candidate, the robot remains stopped and replans from
short updated observations for at most 3 s.  The Fresh-initialized
multisphere shape is held rigid and translated with the latest tracked center.
Each new candidate is checked against a forecast propagated over its measured
planning latency and is executed only after the unchanged 0.09 m verifier
accepts it.  The 0.12 m current-distance stop and 0.10 m raw hard guard remain
active; timeout or either distance gate produces a fail-closed safe hold.
This is stopped-state receding-horizon repair, not trajectory switching while
the robot is moving.

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
`fast_target=150 ms`, `fast_max=250 ms`; current-distance fallback stop is
0.12 m and raw-cloud hard guard is 0.10 m.

The 150 ms Fast value is a preferred realtime target, not a geometric safety
gate. Candidate acceptance uses complete online pipeline time (repair, candidate
verification, required reference verification, and final comparison), rather
than reporting repair-only time as the online latency. The 250 ms absolute
ceiling remains fail-closed because a slower result risks becoming stale before
command time; the latest-state Fresh authorization and raw/dynamics guards are
still mandatory for every execution.

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
captures a fresh 0.6 s RGB-D window. Fresh Association v2 bootstraps its first
frame from the trigger frame's actual associated raw-cluster center, testing
both constant-velocity and stopped/decelerated hypotheses under the unchanged
0.12 m gate. After bootstrap, each frame is associated to the previous fresh
center under the unchanged 0.08 m continuity gate; trigger velocity is not
reused. At least three associated frames spanning 0.25 s are required. The
latest center, conservative maximum observed radius, and a timestamped
linear-regression velocity replace the trigger-time state for repair and
bounded-rejoin validation. Missing, stale, or unassociated data produces
`REJECTED_FRESH_RECHECK` and keeps the robot stopped. The acquisition time is
logged separately; the unchanged 150 ms budget starts with Fast repair.
`post_stop_fresh_recheck.json` records the bootstrap model, CV/hold errors,
selected error, and every subsequent continuity decision.

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
time origin. A local-only gate first time-scales the 1 s repair to the exact
requested playback duration and revalidates that same physical-time trajectory.
Only success writes `local_execution_authorization/authorized_local_repair.csv`
and records `LOCAL_EXECUTION_AUTHORIZED`; the executor never reads the raw Fast
CSV. During Offline Track playback, the existing 0.10 m RGB-D hard guard remains
active and immediately invokes the validated stop interface on violation.

The full repair-plus-rejoin gate searches the unchanged 1.25--2.00 s rejoin
window and verifies a C2 bridge. It remains the preferred v2 execution path.
If this full gate fails while the independent local-only Fresh #2 gate passes,
v2 may execute only `authorized_local_repair.csv` and then holds at its safe
elastic tail. Fresh #3 must then authorize a newly generated C2 bridge from
that fixed tail to the earliest safe state in the same bounded reference
window. This is not a second Fast solve and does not relax the 0.09 m online
acceptance threshold. The bridge is checked against either the associated
Fresh #3 multisphere forecast or, after a strict scene-clear decision, a
stationary conservative union of every external cluster in the three audit
frames. Failure leaves the robot holding at the local tail.

After either the preferred full rejoin or the Fresh #3-authorized delayed
bridge executes, Fresh #3 permits resume through either
the tracked-obstacle risk check or `FRESH3_SCENE_CLEAR`.  Scene-clear requires
three consecutive valid RGB-D frames, failure to associate the original target,
all-cluster current clearance above 0.12 m, raw guard clearance above 0.10 m,
and an all-cluster scan of the next 0.5 s of reference above 0.14 m.  Missing
association alone never authorizes motion.  The nominal task is a Cartesian
Y-line, so resume uses a guarded 0.020 m/s `movel_line` rather than replaying
noisy timestamped joint feedback through an untimed Offline Track interface.
Any Fresh, time-axis, geometry, motion, distance, or overspeed failure keeps the
robot stopped. Fresh #2/#3 acquisition and authorization latency remain
separate from the frozen 150 ms Fast budget.

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

D1 and D2 differ only in obstacle motion geometry. New trials use protocol ID
`653_unified_d1_d2_v2`, including the same perception, window-motion, STRO,
all-link safety, Fast-repair, and candidate-acceptance parameters. Every moving
trial writes the complete `formal_protocol` signature to `summary.json`.
Changing any frozen algorithm/safety value through the CLI causes
`BLOCKED_NONFORMAL_PROTOCOL` before a robot connection is opened. Scene,
repeat, duration, output path, and operator/reference inputs remain ordinary
experimental metadata rather than algorithm parameters.

The archived v1 D1/D2 runs remain immutable development evidence. They must
not be relabeled as v2: v2 specifically denotes the added
`full-first -> local-first/hold -> Fresh #3 delayed rejoin` execution state
machine, while all frozen numeric safety and Fast parameters remain unchanged.

## Frozen D2 geometry-feasibility calibration

The deep head-on D2 v2 r02 is retained as a fail-closed negative example: Fast
produced a nonzero repair and improved clearance, but the candidate remained
below 0.09 m and was not executed. Before fixing the final physical D2 lane,
`offline_d2_geometry_sweep.py` replays that saved trigger/Fresh geometry and
the production Fast implementation while translating only the obstacle's
physical X or Z position. It never initializes a camera or robot:

```bash
/home/hzy/miniconda3/envs/py310/bin/python \
  experiments/new/6_5/6_5_3/offline_d2_geometry_sweep.py
```

A point is feasible only when predicted risk still triggers below 0.14 m, the
unmodified reference remains below 0.09 m, the repaired candidate reaches at
least 0.09 m, clearance gain remains at least 3 mm, and the Fast pipeline stays
within the 250 ms absolute ceiling. A miss of the 150 ms target is retained as
a realtime diagnostic. The script reports contiguous intervals rather than
selecting an isolated threshold-crossing sample. The r02 X sweep found a
continuous +0.13--+0.20 m interval; its midpoint, +0.165 m relative to the r02
observed obstacle centerline, is the fixed physical-lane target. This offset is
an experimental fixture calibration, not a controller parameter. Once marked
physically, it is not changed across the final three D2 repeats.

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
