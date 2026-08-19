#!/usr/bin/env python3
"""Diagnose return-to-goal after the external obstacle has been removed.

This is deliberately separate from the D2 production state machine.  It
requires repeated strict empty-scene observations, verifies a smooth 10 s
16-segment NUBS against an empty-scene forecast, and only executes when
``--execute`` is explicitly supplied.  Robot, tabletop, dynamics, raw-guard,
split-SDK and emergency-stop protections remain active.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import importlib
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

event = importlib.import_module("experiments.new.6_5.6_5_3.run_653_simple_dynamic_nubs_event_replan_live")
trial = importlib.import_module("experiments.new.6_5.6_5_3.run_653_dynamic_repair_trial")
v3 = importlib.import_module("experiments.new.6_5.6_5_3.dynamic_nubs_v3")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-feedback-csv", type=Path, required=True)
    p.add_argument("--output", type=Path,
                   default=ROOT / "results/new/6_5/6_5_3/empty_scene_return_to_goal")
    p.add_argument("--duration-s", type=float, default=10.0)
    p.add_argument("--segments", type=int, default=16)
    p.add_argument("--empty-confirm-frames", type=int, default=3)
    p.add_argument("--operator-confirmed-obstacle-safe-away", action="store_true",
                   help="allow visible but operator-confirmed distant clusters; raw guard remains active")
    p.add_argument("--safe-away-raw-guard-m", type=float, default=0.20)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--robot-ip", default="192.168.123.96")
    return p


def empty_forecast():
    return v3.v3_execution_multisphere_forecast(
        np.asarray([[100.0, 100.0, 100.0]], dtype=np.float64),
        np.asarray([1.0e-6], dtype=np.float64),
        np.zeros(3, dtype=np.float64),
    )


def make_args():
    args = trial.build_parser().parse_args(["--scene", "D2", "--mode", "shadow"])
    args.candidate_execute_confirm = False
    args.require_split_offline_track = True
    args.candidate_playback_duration_s = None
    return args


def build_empty_observation(processor, args):
    frame = processor.process_frame()
    scene = np.asarray(frame.scene_points, dtype=np.float64)
    robot_points = np.asarray(frame.robot_points, dtype=np.float64)
    rois = trial.apply_two_layer_roi(scene, args, need_planning=False)
    plane_removal = None
    if args.remove_planes:
        plane_removal = {"enabled": True, "distance_threshold": args.plane_dist,
                         "max_planes": args.max_planes}
    clustered = trial.FastClusteringFilter(
        rois["safety_points"], robot_points,
        workspace=getattr(processor, "_workspace", None),
        plane_removal=plane_removal, eps=args.cluster_eps,
        min_samples=args.cluster_min_samples, min_points=args.cluster_min_points,
        min_volume=args.cluster_min_volume,
    )
    clusters = trial.filter_guard_clusters(list(clustered.clusters), args)
    raw_guard = trial.execution_hard_guard_distance(processor, None, args)
    safe_away = bool(
        args.operator_confirmed_obstacle_safe_away
        and raw_guard > float(args.safe_away_raw_guard_m)
    )
    return {
        "frame_valid": True,
        "external_cluster_count": len(clusters),
        "raw_guard_distance_m": float(raw_guard),
        "timestamp": float(getattr(frame, "timestamp", time.time())),
        "strict_empty": bool(not clusters and raw_guard > float(args.guided_hard_stop_m)),
        "operator_safe_away": safe_away,
    }


def confirm_empty(processor, args, count: int):
    samples = [build_empty_observation(processor, args) for _ in range(int(count))]
    empty_ok = bool(samples and all(row["strict_empty"] for row in samples))
    safe_away_ok = bool(samples and all(row["operator_safe_away"] for row in samples))
    ready = bool(empty_ok or safe_away_ok)
    return {
        "samples": samples,
        "required_frames": int(count),
        "mode": "STRICT_EMPTY" if empty_ok else ("OPERATOR_CONFIRMED_SAFE_AWAY" if safe_away_ok else None),
        "reason": "strict_empty_scene_confirmed" if empty_ok else ("operator_confirmed_obstacle_safe_away" if safe_away_ok else "external_cluster_or_raw_guard_not_clear"),
    }


def make_empty_monitor(processor, args, count):
    class EmptyMonitor:
        def _check(self):
            return confirm_empty(processor, args, count)

        def prearm(self):
            return self._check()

        def command_time_revalidate(self, *, actual_q):
            result = self._check()
            result["action"] = "execute" if result["ready"] else "hold"
            return result

        def final_precommand_barrier(self, *, actual_q):
            result = self._check()
            result["action"] = "execute" if result["ready"] else "hold"
            return result

        def __call__(self, *, elapsed_s, actual_q, obstacle_snapshot):
            result = self._check()
            return {
                "motion_safe": bool(result["ready"]),
                "reason": None if result["ready"] else "external_obstacle_reappeared",
                "empty_scene_audit": result,
            }

    return EmptyMonitor()


def make_trajectory(q_start, q_goal, duration_s, segments):
    durations = np.full(int(segments), float(duration_s) / int(segments), dtype=np.float64)
    inner = trial.NUBSTrajectory6D.linear_inner_points(q_start, q_goal, durations)
    head = trial.NUBSTrajectory6D.make_boundary_state(q_start, np.zeros(6), np.zeros(6))
    tail = trial.NUBSTrajectory6D.make_boundary_state(q_goal, np.zeros(6), np.zeros(6))
    return trial.NUBSTrajectory6D().generate(inner, head, tail, durations)


def run(ns):
    args = make_args()
    args.operator_confirmed_obstacle_safe_away = bool(ns.operator_confirmed_obstacle_safe_away)
    args.safe_away_raw_guard_m = float(ns.safe_away_raw_guard_m)
    reference = trial.RecordedReference.load(ns.reference_feedback_csv.resolve())
    config = trial.load_stage4_config(args.stage4_config)
    model = trial.load_stage4_surface_model(config)
    processor = trial.SceneProcessor(config_dir=str(args.config_dir), urdf_path=str(args.urdf),
                                     width=args.width, height=args.height,
                                     threshold=args.self_filter_threshold,
                                     voxel_size=args.voxel_size, use_real_robot=True,
                                     use_mock_camera=False)
    output = ns.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    robot = getattr(getattr(processor, "_state_reader", None), "sdk_module", None)
    if robot is None:
        processor.stop()
        raise RuntimeError("real AUBO SDK robot interface is required")
    try:
        q_start = np.asarray(robot.get_joint(), dtype=np.float64)
        q_goal = np.asarray(reference.q[-1], dtype=np.float64)
        empty_audit = confirm_empty(processor, args, ns.empty_confirm_frames)
        result = {"status": "EMPTY_SCENE_NOT_CONFIRMED", "authorized": False,
                  "robot_commanded": False, "empty_scene_audit": empty_audit,
                  "q_start": q_start.tolist(), "q_goal": q_goal.tolist(),
                  "duration_s": float(ns.duration_s), "segments": int(ns.segments)}
        if not empty_audit["ready"]:
            (output / "diagnostic_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            return result
        trajectory = make_trajectory(q_start, q_goal, ns.duration_s, ns.segments)
        forecast = empty_forecast()
        _, verifier, _ = trial.make_risk_stack(config, model, None)
        verification = verifier.verify(trajectory, forecast, current_q=q_start,
                                       current_qd=np.zeros(6), current_qdd=np.zeros(6),
                                       q_goal=q_goal, solver_success=True)
        tabletop = trial.gripper_base_workspace_guard(
            trajectory, model, min_z_m=float(args.gripper_base_min_z_m))
        result.update({"status": "EMPTY_SCENE_RETURN_AUTHORIZED" if verification.accepted and tabletop["passed"] else "EMPTY_SCENE_RETURN_REJECTED",
                       "scene_clearance_mode": empty_audit.get("mode"),
                       "authorized": bool(verification.accepted and tabletop["passed"]),
                       "verification_min_distance_m": float(verification.min_distance),
                       "verification_checks": verification.checks,
                       "tabletop_workspace_guard": tabletop})
        csv_path = output / "authorized_empty_scene_return.csv"
        trial.save_trajectory_csv(csv_path, trajectory, dt=0.01)
        result["authorized_trajectory_csv"] = str(csv_path)
        if ns.execute and result["authorized"]:
            trial.require_confirmation(True, "障碍物已移走且空场景已确认。确认急停可用后按 Enter 执行回终点诊断。")
            monitor = make_empty_monitor(processor, args, ns.empty_confirm_frames)
            execution = trial.execute_authorized_trajectory_offline_track(
                robot, csv_path, args, processor=processor, denoiser=None,
                execution_label="empty-scene return-to-goal diagnostic",
                guard_provider=lambda: {"distance_m": trial.execution_hard_guard_distance(processor, None, args),
                                         "timestamp": time.time()},
                motion_monitor_provider=monitor,
            )
            result["execution"] = execution
            result["robot_commanded"] = bool(execution.get("robot_commanded", False))
            result["status"] = execution.get("status", result["status"])
        (output / "diagnostic_summary.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        return result
    finally:
        processor.stop()


if __name__ == "__main__":
    print(json.dumps(run(parser().parse_args()), indent=2, ensure_ascii=False, default=str))
