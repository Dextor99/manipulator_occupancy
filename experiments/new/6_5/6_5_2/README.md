# 6.5.2 Static Table-Obstacle Avoidance

This folder contains the revised Chapter 6.5.2 real static-obstacle experiment.

The experiment verifies:

```text
static RGB-D capture
-> self-filtered obstacle model
-> reference trajectory risk audit
-> Full CCRO-NUBS planning
-> dense candidate validation
-> pre-execution recheck
-> guarded execution decision
```

Current robot-command status: **no robot motion is sent by this script**.  The
repository does not yet expose a bounded AUBO joint/NUBS trajectory queue or
batch execution API, so `--mode execute` writes an execution guard report and
refuses to move the arm.

Important tabletop revision: the old `config/ccro_stage2.yaml` joint-space
reference has a shoulder/elbow nodding motion and is rejected by default for
6.5.2 tabletop trials.  See
[`PLANAR_TABLETOP_REVISION.md`](PLANAR_TABLETOP_REVISION.md).  Real 6.5.2
execution should use a horizontal Cartesian straight/curved path above the
desk, not the historical non-planar joint trajectory.

## Scenarios

| ID | Name | Layout |
|---|---|---|
| `R-S1` | lateral table obstacle | Low foam box near the end-effector reference path side. |
| `R-S2` | central high table obstacle | Taller foam column or stacked foam boxes near the middle of the tabletop path. |

Do not claim that `R-S2` is a forearm/middle-link case before checking the logs.
Report the actual nearest risk link from `reference_risk.json` and
`candidate_audit.json`.

## Smoke Run

Start with one short planning-only trial.  This connects RealSense and AUBO
state feedback, but does not move the robot.

For old non-planar reference debugging only, add `--allow-nonplanar-reference`.
Do not use that flag for real tabletop execution.

```bash
/home/hzy/miniconda3/envs/py310/bin/python experiments/new/6_5/6_5_2/run_652_static_avoidance.py \
  --mode capture-plan \
  --scenario R-S1 \
  --repeat 1 \
  --capture-duration-s 4 \
  --self-filter-threshold 0.08 \
  --cluster-min-points 30 \
  --cluster-min-volume 0.001 \
  --allow-nonplanar-reference \
  --output results/new/6_5/6_5_2
```

Check:

```bash
sed -n '1,160p' results/new/6_5/6_5_2/rs1_lateral_table_obstacle/trials/rs1_lateral_table_obstacle_r01/summary.json
```

Useful files in each trial:

```text
detected_obstacle.json
reference_risk.json
candidate_audit.json
dense_verification.json
optimized_trajectory.csv
reference_trajectory.csv
figures/distance_risk_curve.png
figures/joint_trajectory_preview.png
snapshots/*/rgb_overlay_nearest.png
```

## Offline URDF Preview

After a trial has produced `reference_trajectory.csv`,
`optimized_trajectory.csv`, `trajectories.npz`, and `obstacle_points.npz`, render
an offline URDF preview before considering any real motion:

```bash
/home/hzy/miniconda3/envs/py310/bin/python experiments/new/6_5/6_5_2/preview_652_trajectory_urdf.py \
  --trial-dir results/new/6_5/6_5_2/rs1_lateral_table_obstacle/trials/rs1_lateral_table_obstacle_r01 \
  --density coarse
```

This preview is offline only and does not connect to the robot.  It writes:

```text
urdf_preview/link_center_paths.png
urdf_preview/reference_min_clearance_pose.png
urdf_preview/candidate_min_clearance_pose.png
urdf_preview/candidate_pose_sequence.png
urdf_preview/preview_summary.json
```

## Tabletop Planar Preview

For real 6.5.2 tabletop design, prefer the fixed-height Cartesian preview below.
It reads the detected obstacle and current recorded posture, then renders a
horizontal straight reference path and a horizontal curved detour path.  It is
offline only and does not command the robot.

```bash
/home/hzy/miniconda3/envs/py310/bin/python experiments/new/6_5/6_5_2/preview_652_planar_tabletop.py \
  --trial-dir results/new/6_5/6_5_2/rs1_lateral_table_obstacle/trials/rs1_lateral_table_obstacle_r01 \
  --axis y \
  --distance-m 0.20 \
  --density coarse
```

For the intended tabletop stroke where the home posture
`[0, 0, 90, 0, 90, 0] deg` is the middle of the motion and the TCP moves from
`Y=+0.4 m` to `Y=-0.4 m` with fixed `X/Z`, use:

```bash
/home/hzy/miniconda3/envs/py310/bin/python experiments/new/6_5/6_5_2/preview_652_planar_tabletop.py \
  --trial-dir results/new/6_5/6_5_2/rs1_lateral_table_obstacle/trials/rs1_lateral_table_obstacle_r01 \
  --posture-source home \
  --home-joints-deg 0,0,90,0,90,0 \
  --y-start 0.4 \
  --y-goal -0.4 \
  --density coarse \
  --output results/new/6_5/6_5_2/planar_home_y_0p4_to_m0p4_preview
```

Outputs:

```text
planar_tabletop_preview/top_view_tcp_paths.png
planar_tabletop_preview/clearance_curves.png
planar_tabletop_preview/reference_pose_sequence.png
planar_tabletop_preview/candidate_pose_sequence.png
planar_tabletop_preview/planar_path_samples.csv
planar_tabletop_preview/preview_summary.json
```

Use this preview to confirm that the task path is horizontal above the table.
Only proceed to execution design if the candidate obstacle clearance and table
clearance pass the thresholds in `preview_summary.json`.

## Guarded Planar Y Execution

After the tabletop preview is acceptable, use the guarded executor.  Dry-run
does not connect to or command the robot:

```bash
/home/hzy/miniconda3/envs/py310/bin/python experiments/new/6_5/6_5_2/execute_652_planar_y_guarded.py \
  --output results/new/6_5/6_5_2/planar_y_guarded_execution_dryrun
```

Actual low-speed execution requires the explicit flag and phrase:

```bash
/home/hzy/miniconda3/envs/py310/bin/python experiments/new/6_5/6_5_2/execute_652_planar_y_guarded.py \
  --execute \
  --operator-phrase CCRO_652_PLANAR_Y_APPROVED \
  --y-start 0.4 \
  --y-goal -0.4 \
  --line-velocity-m-s 0.025 \
  --line-acc-m-s2 0.06 \
  --output results/new/6_5/6_5_2/planar_y_guarded_execution_smoke
```

The script then waits at each stage.  Press `Enter` or `Space` to continue;
press `q` or `Ctrl-C` to abort.

```text
home posture
start pose
goal pose
return home
```

It prefers `movel_line` when the compiled AUBO SDK module exposes it.  If the
current module lacks `movel_line`, it refuses motion by default.  Only use
`--allow-movel-fallback` for a very slow smoke test if you accept that the SDK
will solve IK and execute each Cartesian target through `movel` rather than a
true controller line primitive.

## Preflight Run

After the candidate passes dense validation, run preflight mode.  It performs a
fresh 1-2 s RGB-D capture after planning and checks that the obstacle and robot
start state are still valid.

```bash
/home/hzy/miniconda3/envs/py310/bin/python experiments/new/6_5/6_5_2/run_652_static_avoidance.py \
  --mode preflight \
  --scenario R-S1 \
  --repeat 2 \
  --capture-duration-s 4 \
  --recheck-duration-s 2 \
  --self-filter-threshold 0.08 \
  --cluster-min-points 30 \
  --cluster-min-volume 0.001 \
  --output results/new/6_5/6_5_2
```

The preflight result is saved as:

```text
pre_execution_recheck.json
```

## Formal Planning/Preflight Trials

For each scenario, use one debug run first, then collect five formal trials.
Every repeat is independent: each trial re-captures the obstacle, re-plans,
validates, and rechecks.  Failed trials must remain in the result directory.

```bash
for s in R-S1 R-S2; do
  for r in 1 2 3 4 5; do
    /home/hzy/miniconda3/envs/py310/bin/python experiments/new/6_5/6_5_2/run_652_static_avoidance.py \
      --mode preflight \
      --scenario $s \
      --repeat $r \
      --capture-duration-s 4 \
      --recheck-duration-s 2 \
      --self-filter-threshold 0.08 \
      --cluster-min-points 30 \
      --cluster-min-volume 0.001 \
      --output results/new/6_5/6_5_2
  done
done
```

## Execution Guard

`execute` mode currently refuses real motion even when the operator phrase is
provided:

```bash
/home/hzy/miniconda3/envs/py310/bin/python experiments/new/6_5/6_5_2/run_652_static_avoidance.py \
  --mode execute \
  --scenario R-S1 \
  --repeat 99 \
  --operator-phrase CCRO_652_STATIC_LOW_SPEED_APPROVED \
  --allow-real-robot-commands \
  --output results/new/6_5/6_5_2
```

Expected result:

```text
status: EXECUTION_BLOCKED_BY_GUARD
robot_commanded: false
```

Real execution should only be enabled after adding and separately validating a
bounded AUBO low-speed joint trajectory execution interface.

## Metrics Naming

Use these names in the thesis/manuscript:

| Metric | Meaning |
|---|---|
| `D_min_ref_obs_m` | Reference trajectory minimum clearance under the observed obstacle model. |
| `D_min_cand_val_m` | Candidate trajectory minimum clearance in dense validation. |
| `D_min_exec_obs_m` | Executed trajectory observed-estimate minimum clearance, available only after real execution logging exists. |

The third metric is not ground-truth clearance; it is reconstructed from robot
feedback and the RGB-D observed obstacle model.

## Candidate Selection Policy

For general static-obstacle execution, the selected trajectory is not simply the
one with the largest possible clearance or the lowest original joint-space
objective, and it is not forced to remain planar or near-constant height.  The
formal 6.5.2 policy is frozen in
`candidate_selection_policy.yaml` and follows:

1. Reject every candidate that fails dense verification, joint limits,
   trajectory continuity, endpoint checks, or optimizer success.
2. Generate or retain multiple path-family candidates, such as overpass,
   base-side lateral, outer-side lateral, and free local optimization seeds.
3. Pareto-filter feasible candidates over 3D TCP path length, joint-space path
   length, NUBS jerk/smooth energy, near-boundary clearance penalty, and
   duration.
4. Score only Pareto non-dominated candidates using normalized objective terms:
   `w_tcp L_TCP + w_joint L_q + w_jerk J_jerk + w_clear J_clear + w_time T`.
5. Never execute a candidate that is only visually plausible but not hard
   feasible.

The near-boundary clearance term penalizes candidates that stay close to the
acceptance threshold, but stops rewarding clearance after the preferred margin is
reached.  TCP height is reported as a diagnostic only; vertical and horizontal
motion both contribute naturally through the 3D TCP path length.  Use
`select_652_candidate_family.py` and `audit_652_objective_terms.py` to document
the selected candidate, Pareto status, normalized score, and reasons for
rejecting or deprioritizing alternatives.
