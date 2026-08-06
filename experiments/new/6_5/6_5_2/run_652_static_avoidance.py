#!/usr/bin/env python3
"""Run Chapter 6.5.2 real static-obstacle avoidance trials.

This script implements the conservative 6.5.2 chain:

    live RGB-D capture
    -> robot self-filtering
    -> static obstacle modeling
    -> reference trajectory risk audit
    -> Full CCRO-NUBS planning
    -> dense candidate validation
    -> pre-execution recheck with fresh RGB-D data
    -> guarded execution decision

Robot command policy
--------------------
The default modes never command the robot.  The current repository does not yet
provide a bounded AUBO joint/NUBS trajectory queue interface; therefore
``--mode execute`` also refuses to move the arm after writing a complete
preflight report.  This is intentional: do not replace it with point-by-point
Python streaming for formal hardware trials.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
EXP651 = ROOT / "experiments" / "new" / "6_5" / "6_5_1"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(EXP651) not in sys.path:
    sys.path.insert(0, str(EXP651))

from experiments.exp_ccro_stage2 import _baseline, _limits, _load, _risk_optimizer, _states  # noqa: E402
from planning.mesh_risk import MeshRiskEvaluator, StaticObstacleField  # noqa: E402
from planning.nubs_trajectory import NUBSTrajectory6D  # noqa: E402
from planning.robot_surface_model import RobotSurfaceModel  # noqa: E402
from planning.verifier import TrajectoryVerifier  # noqa: E402
from risk.safety_policy import SafetyPolicy  # noqa: E402
from test_clustering_filtering import FastClusteringFilter, TemporalDenoiser  # noqa: E402
from test_remove_robot_points_fast import SceneProcessor  # noqa: E402
from utils.config import load_config_dir  # noqa: E402

from capture_651_visual_snapshots import (  # noqa: E402
    FramePacket,
    clone_packet,
    process_visual_frame,
    render_pointcloud_snapshot,
    save_snapshot,
)
from run_651_perception_capture import (  # noqa: E402
    JOINT_NAMES,
    nearest_cluster_to_links,
    q_from_reader,
    risk_color_level,
)


DEFAULT_CONFIG = ROOT / "config" / "ccro_stage2.yaml"
DEFAULT_OUTPUT = ROOT / "results" / "new" / "6_5" / "6_5_2"
REQUIRED_OPERATOR_PHRASE = "CCRO_652_STATIC_LOW_SPEED_APPROVED"

SCENARIOS = {
    "R-S1": {
        "slug": "rs1_lateral_table_obstacle",
        "title": "R-S1 lateral table obstacle",
        "prompt": "放置较低泡沫盒到末端参考路径侧方或近旁，保持起点和目标安全",
    },
    "R-S2": {
        "slug": "rs2_central_high_table_obstacle",
        "title": "R-S2 central high table obstacle",
        "prompt": "放置较高泡沫柱或叠放泡沫盒到桌面中央区域，确保仍有明显绕行通道",
    },
}


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_surface_model(config: dict[str, Any]) -> RobotSurfaceModel:
    robot = config["robot"]
    surface = config["surface"]
    return RobotSurfaceModel(
        ROOT / robot["urdf_path"],
        robot["joint_names"],
        surface["density_totals"],
        seed=surface["random_seed"],
        min_points_per_link=surface["min_points_per_link"],
        cache_dir=surface["cache_dir"],
        geometry=surface["geometry"],
    )


def make_evaluator_and_verifier(config: dict[str, Any], surface_model: RobotSurfaceModel):
    risk_cfg = config["risk"]
    validation = config["validation"]
    limits = _limits(config)
    evaluator = MeshRiskEvaluator(
        surface_model,
        d_safe=risk_cfg["d_safe"],
        d_activate=risk_cfg["d_activate"],
        fd_epsilon_q=risk_cfg["fd_epsilon_q"],
        density=risk_cfg["optimizer_density"],
    )
    verifier = TrajectoryVerifier(
        evaluator,
        limits,
        d_stop=validation["d_accept"],
        time_step=validation["dense_time_step"],
        density=risk_cfg["validation_density"],
        epsilon_goal=validation["epsilon_goal"],
        epsilon_continuity_q=validation["epsilon_continuity_q"],
        epsilon_continuity_qd=validation["epsilon_continuity_qd"],
        epsilon_continuity_qdd=validation["epsilon_continuity_qdd"],
        limit_tolerance=validation["limit_tolerance"],
    )
    return evaluator, verifier, limits


def prompt(message: str, no_prompt: bool) -> None:
    if not no_prompt:
        input(f"\n{message}。按 Enter 开始...")


def trajectory_rows(name: str, trajectory: NUBSTrajectory6D, dt: float) -> list[dict[str, Any]]:
    count = max(2, int(math.ceil(trajectory.total_duration / dt)) + 1)
    times = np.linspace(0.0, trajectory.total_duration, count)
    samples = trajectory.sample(times)
    rows = []
    for i, t in enumerate(times):
        row: dict[str, Any] = {"trajectory": name, "t_s": f"{float(t):.6f}"}
        row.update({f"q{j+1}_rad": f"{samples.q[i, j]:.9f}" for j in range(6)})
        row.update({f"qd{j+1}_rad_s": f"{samples.qd[i, j]:.9f}" for j in range(6)})
        row.update({f"qdd{j+1}_rad_s2": f"{samples.qdd[i, j]:.9f}" for j in range(6)})
        rows.append(row)
    return rows


def sample_trajectory_distances(
    trajectory: NUBSTrajectory6D,
    evaluator: MeshRiskEvaluator,
    obstacle: StaticObstacleField,
    *,
    dt: float,
    density: str,
) -> dict[str, Any]:
    count = max(2, int(math.ceil(trajectory.total_duration / dt)) + 1)
    times = np.linspace(0.0, trajectory.total_duration, count)
    risk = evaluator.trajectory(trajectory, obstacle, times, density=density, with_gradient=False)
    min_idx = int(np.argmin(risk.sample_distances))
    return {
        "min_distance_m": float(risk.min_distance),
        "nearest_link": risk.nearest_link,
        "active_sample_count": int(risk.active_sample_count),
        "min_time_s": float(times[min_idx]),
        "sample_times_s": times,
        "sample_distances_m": risk.sample_distances,
        "sample_costs": risk.sample_costs,
    }


def save_distance_curve(path: Path, ref: dict[str, Any], cand: dict[str, Any], d_accept: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(8.6, 6.0), sharex=True)
    axes[0].plot(ref["sample_times_s"], ref["sample_distances_m"], label="reference")
    axes[0].plot(cand["sample_times_s"], cand["sample_distances_m"], label="CCRO-NUBS candidate")
    axes[0].axhline(d_accept, color="#d62728", linestyle="--", linewidth=1.2, label="accept threshold")
    axes[0].set_ylabel("D_min / m")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].plot(ref["sample_times_s"], ref["sample_costs"], label="reference")
    axes[1].plot(cand["sample_times_s"], cand["sample_costs"], label="candidate")
    axes[1].set_ylabel("R_body")
    axes[1].set_xlabel("time / s")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_joint_preview(path: Path, ref: NUBSTrajectory6D, cand: NUBSTrajectory6D) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times = np.linspace(0.0, ref.total_duration, 180)
    ref_q = ref.sample(times, max_derivative=0).q
    cand_q = cand.sample(times, max_derivative=0).q
    fig, axes = plt.subplots(3, 2, figsize=(10, 7), sharex=True)
    for j, ax in enumerate(axes.ravel()):
        ax.plot(times, ref_q[:, j], color="#777777", linestyle="--", label="reference" if j == 0 else None)
        ax.plot(times, cand_q[:, j], color="#1f77b4", label="candidate" if j == 0 else None)
        ax.set_ylabel(f"q{j+1} / rad")
        ax.grid(True, alpha=0.25)
    axes[-1, 0].set_xlabel("time / s")
    axes[-1, 1].set_xlabel("time / s")
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def choose_static_cluster(
    surface_model: RobotSurfaceModel,
    q: np.ndarray,
    clusters: list[Any],
    *,
    density: str,
    min_points: int,
) -> tuple[int | None, dict[str, Any]]:
    original_indices = {id(cluster): index for index, cluster in enumerate(clusters)}
    filtered = [c for c in clusters if len(c.points) >= min_points]
    if not filtered:
        return None, {"distance": math.inf, "link": None, "cluster_center": None, "cluster_points": 0}
    best = nearest_cluster_to_links(surface_model, q, filtered, density=density)
    if best["cluster_index"] is None:
        return None, best
    original = original_indices[id(filtered[int(best["cluster_index"])])]
    best["cluster_index"] = original
    return original, best


def collect_static_model(args: argparse.Namespace, trial_dir: Path, surface_model: RobotSurfaceModel) -> dict[str, Any]:
    config_live = load_config_dir(args.config_dir)
    safety = config_live["safety"]
    policy = SafetyPolicy(
        d_safe=float(safety.get("d_safe", 0.15)),
        d_slow=float(safety.get("d_slow", 0.10)),
        d_stop=float(safety.get("d_stop", 0.05)),
    )
    processor = SceneProcessor(
        config_dir=str(args.config_dir),
        urdf_path=str(args.urdf),
        width=args.width,
        height=args.height,
        threshold=args.self_filter_threshold,
        voxel_size=args.voxel_size,
        use_real_robot=True,
        use_mock_camera=False,
    )
    state_reader = getattr(processor, "_state_reader", None)
    if state_reader is None or type(state_reader).__name__ != "RealRobotStateReader":
        processor.stop()
        raise RuntimeError("real AUBO state reader is required; no robot command was sent")

    denoiser = TemporalDenoiser(
        voxel_size=args.denoise_voxel,
        confidence_threshold=args.denoise_conf,
        decay=args.denoise_decay,
    ) if args.temporal_denoise else None

    packets: list[FramePacket] = []
    rows: list[dict[str, Any]] = []
    q_samples: list[np.ndarray] = []
    prompt(f"[{args.scenario}] {SCENARIOS[args.scenario]['prompt']}", args.no_prompt)
    start = time.time()
    frame_index = 0
    try:
        while time.time() - start < args.capture_duration_s:
            raw_frame, scene_points, robot_points = process_visual_frame(processor, args)
            timestamp = float(getattr(raw_frame, "timestamp", time.time()))
            if denoiser is not None:
                scene_points = denoiser.filter(scene_points)
            joints, q = q_from_reader(state_reader)
            q_samples.append(q)
            plane_removal = None
            if args.remove_planes:
                plane_removal = {
                    "enabled": True,
                    "distance_threshold": args.plane_dist,
                    "max_planes": args.max_planes,
                }
            cluster_result = FastClusteringFilter(
                scene_points,
                robot_points,
                workspace=getattr(processor, "_workspace", None),
                plane_removal=plane_removal,
                eps=args.cluster_eps,
                min_samples=args.cluster_min_samples,
                min_points=args.cluster_min_points,
                min_volume=args.cluster_min_volume,
            )
            clusters = list(cluster_result.clusters)
            cluster_idx, best = choose_static_cluster(
                surface_model,
                q,
                clusters,
                density=args.surface_density,
                min_points=args.obstacle_min_points,
            )
            center = best["cluster_center"]
            row = {
                "frame": frame_index,
                "timestamp": f"{timestamp:.6f}",
                "scene_points": int(len(scene_points)),
                "robot_points": int(len(robot_points)),
                "cluster_count": int(len(clusters)),
                "nearest_distance_m": "" if math.isinf(best["distance"]) else f"{best['distance']:.6f}",
                "nearest_link": best["link"] or "",
                "nearest_cluster_index": "" if cluster_idx is None else int(cluster_idx),
                "nearest_cluster_points": int(best["cluster_points"]),
                "nearest_cluster_x": "" if center is None else f"{center[0]:.6f}",
                "nearest_cluster_y": "" if center is None else f"{center[1]:.6f}",
                "nearest_cluster_z": "" if center is None else f"{center[2]:.6f}",
                "risk_state_current": risk_color_level(policy, best["distance"]),
                **{f"q{j+1}_rad": "" if not np.isfinite(q[j]) else f"{q[j]:.8f}" for j in range(6)},
            }
            rows.append(row)
            packet = FramePacket(
                index=frame_index,
                phase="static_obstacle",
                raw_frame=raw_frame,
                scene_points=scene_points,
                robot_points=robot_points,
                cluster_result=cluster_result,
                clusters=clusters,
                q=q,
                joints=dict(joints),
                row={
                    **row,
                    "stable_track_count": int(cluster_idx is not None),
                    "risk_state_predicted": row["risk_state_current"],
                    "predicted_distance_m": row["nearest_distance_m"],
                    "predicted_nearest_link": row["nearest_link"],
                    "risk_sphere_count": 0,
                    "max_track_speed_m_s": "",
                    "mean_track_speed_m_s": "",
                },
            )
            packets.append(clone_packet(packet))
            frame_index += 1
    finally:
        processor.stop()

    write_csv(
        trial_dir / "perception_frames.csv",
        rows,
        [
            "frame", "timestamp", "scene_points", "robot_points", "cluster_count",
            "nearest_distance_m", "nearest_link", "nearest_cluster_index", "nearest_cluster_points",
            "nearest_cluster_x", "nearest_cluster_y", "nearest_cluster_z", "risk_state_current",
            *[f"q{j+1}_rad" for j in range(6)],
        ],
    )

    valid_packets = [p for p in packets if p.row.get("nearest_cluster_index", "") != ""]
    if not valid_packets:
        return {
            "accepted": False,
            "reason": "no stable obstacle cluster selected",
            "frame_count": len(rows),
            "q_mean": np.mean(q_samples, axis=0).tolist() if q_samples else None,
            "obstacle": None,
        }

    # Prefer a clear, low-distance frame for figures, but aggregate obstacle
    # geometry over several stable detections.
    ranked = sorted(
        valid_packets,
        key=lambda p: float(p.row["nearest_distance_m"]) if p.row["nearest_distance_m"] != "" else math.inf,
    )
    representative = ranked[0]
    snap_dir = trial_dir / "snapshots"
    save_snapshot(representative, "static_obstacle_model", args.scenario, args.repeat, snap_dir, processor)

    chunks = []
    for packet in ranked[: args.obstacle_model_frames]:
        idx = int(packet.row["nearest_cluster_index"])
        if idx < len(packet.clusters):
            chunks.append(np.asarray(packet.clusters[idx].points, dtype=np.float64))
    obstacle_points = np.vstack(chunks) if chunks else np.empty((0, 3))
    if len(obstacle_points) > args.max_obstacle_points:
        rng = np.random.default_rng(args.seed + args.repeat)
        choice = rng.choice(len(obstacle_points), args.max_obstacle_points, replace=False)
        obstacle_points = obstacle_points[choice]
    center = np.mean(obstacle_points, axis=0)
    radius = float(np.max(np.linalg.norm(obstacle_points - center, axis=1))) if len(obstacle_points) else 0.0
    np.savez_compressed(trial_dir / "obstacle_points.npz", points=obstacle_points)
    render_pointcloud_snapshot(
        trial_dir / "figures" / "obstacle_model_pointcloud.png",
        representative.scene_points,
        representative.robot_points,
        [obstacle_points],
        0,
    )
    obstacle_payload = {
        "accepted": bool(len(obstacle_points) >= args.obstacle_min_points),
        "reason": "ok" if len(obstacle_points) >= args.obstacle_min_points else "too few obstacle points",
        "frame_count": len(rows),
        "selected_frames": [int(p.index) for p in ranked[: args.obstacle_model_frames]],
        "point_count": int(len(obstacle_points)),
        "center": center.tolist(),
        "radius_estimate_m": radius,
        "nearest_link_representative": representative.row.get("nearest_link", ""),
        "nearest_distance_representative_m": float(representative.row["nearest_distance_m"]),
        "q_mean": np.mean(q_samples, axis=0).tolist() if q_samples else None,
        "snapshot_dir": str(snap_dir),
    }
    write_json(trial_dir / "detected_obstacle.json", obstacle_payload)
    return {"accepted": obstacle_payload["accepted"], "obstacle": obstacle_payload, "points": obstacle_points}


def pre_execution_recheck(
    args: argparse.Namespace,
    trial_dir: Path,
    surface_model: RobotSurfaceModel,
    evaluator: MeshRiskEvaluator,
    verifier: TrajectoryVerifier,
    candidate: NUBSTrajectory6D,
    obstacle_before: dict[str, Any],
    head: np.ndarray,
    tail: np.ndarray,
) -> dict[str, Any]:
    if args.skip_recheck:
        payload = {"accepted": True, "skipped": True, "reason": "operator requested --skip-recheck"}
        write_json(trial_dir / "pre_execution_recheck.json", payload)
        return payload
    prompt(f"[{args.scenario}] 执行前二次复核：保持障碍物不动，机械臂仍应在候选起点", args.no_prompt)
    recheck_args = argparse.Namespace(**vars(args))
    recheck_args.capture_duration_s = args.recheck_duration_s
    recheck_args.repeat = args.repeat
    recheck = collect_static_model(recheck_args, trial_dir / "pre_execution_recheck", surface_model)
    checks: dict[str, bool] = {"recheck_obstacle_detected": bool(recheck.get("accepted"))}
    reasons: list[str] = []
    if not recheck.get("accepted"):
        reasons.append("recheck_obstacle_detected")
        payload = {"accepted": False, "checks": checks, "reasons": reasons, "detail": recheck}
        write_json(trial_dir / "pre_execution_recheck.json", payload)
        return payload

    new_obstacle = recheck["obstacle"]
    old_center = np.asarray(obstacle_before["center"], dtype=np.float64)
    new_center = np.asarray(new_obstacle["center"], dtype=np.float64)
    center_shift = float(np.linalg.norm(new_center - old_center))
    radius_shift = float(abs(float(new_obstacle["radius_estimate_m"]) - float(obstacle_before["radius_estimate_m"])))
    obstacle_field = StaticObstacleField.from_points(recheck["points"])
    verification = verifier.verify(
        candidate,
        obstacle_field,
        current_q=head[:, 0],
        current_qd=head[:, 1],
        current_qdd=head[:, 2],
        q_goal=tail[:, 0],
        solver_success=True,
    )
    q_live = np.asarray(new_obstacle["q_mean"], dtype=np.float64)
    q_start_error = float(np.linalg.norm(q_live - head[:, 0]))
    checks.update(
        {
            "center_shift_ok": center_shift <= args.recheck_center_tolerance_m,
            "radius_shift_ok": radius_shift <= args.recheck_radius_tolerance_m,
            "candidate_still_dense_accepted": bool(verification.accepted),
            "robot_start_q_ok": q_start_error <= args.start_q_tolerance_rad,
        }
    )
    reasons = [name for name, ok in checks.items() if not ok]
    payload = {
        "accepted": bool(all(checks.values())),
        "checks": checks,
        "reasons": reasons,
        "center_shift_m": center_shift,
        "radius_shift_m": radius_shift,
        "q_start_l2_error_rad": q_start_error,
        "dense_verification_on_recheck_obstacle": asdict(verification),
        "detail": new_obstacle,
    }
    write_json(trial_dir / "pre_execution_recheck.json", payload)
    return payload


def run_trial(args: argparse.Namespace) -> dict[str, Any]:
    if args.scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario: {args.scenario}")
    scenario_cfg = SCENARIOS[args.scenario]
    output_root = args.output.resolve() / scenario_cfg["slug"]
    trial_dir = output_root / "trials" / f"{scenario_cfg['slug']}_r{args.repeat:02d}"
    if trial_dir.exists() and not args.allow_overwrite:
        raise FileExistsError(f"{trial_dir} already exists; use a new --repeat or pass --allow-overwrite")
    trial_dir.mkdir(parents=True, exist_ok=True)
    (trial_dir / "figures").mkdir(exist_ok=True)
    shutil.copy2(args.config, trial_dir / "config_used.yaml")

    config = _load(args.config)
    surface_model = make_surface_model(config)
    evaluator, verifier, limits = make_evaluator_and_verifier(config, surface_model)
    head, tail, durations = _states(config)

    status = "INIT"
    obstacle_model: dict[str, Any] | None = None
    baseline_result = None
    candidate_result = None
    reference_audit: dict[str, Any] | None = None
    candidate_audit: dict[str, Any] | None = None
    dense_verification: dict[str, Any] | None = None
    preflight: dict[str, Any] | None = None
    execution: dict[str, Any] = {
        "robot_commanded": False,
        "attempted": False,
        "executed": False,
        "reason": "not reached",
    }

    started = time.perf_counter()
    try:
        capture = collect_static_model(args, trial_dir, surface_model)
        if not capture.get("accepted"):
            status = "PERCEPTION_REJECTED"
            raise RuntimeError(capture.get("reason", "perception rejected"))
        obstacle_model = capture["obstacle"]
        obstacle = StaticObstacleField.from_points(capture["points"])
        if args.use_current_start:
            head = NUBSTrajectory6D.make_boundary_state(np.asarray(obstacle_model["q_mean"], dtype=np.float64))

        planar_gate = check_planar_reference_gate(config, head, tail, args)
        write_json(trial_dir / "reference_planar_gate.json", planar_gate)
        if not planar_gate["accepted"]:
            status = "REFERENCE_REJECTED_NONPLANAR_DESK_MOTION"
            raise RuntimeError(planar_gate["reason"])

        baseline_result = _baseline(config, head, tail, durations)
        if not baseline_result.success:
            status = "REFERENCE_GENERATION_FAILED"
            raise RuntimeError(f"reference NUBS failed: {baseline_result.message}")
        reference = baseline_result.trajectory
        reference_validation = verifier.verify(
            reference,
            obstacle,
            current_q=head[:, 0],
            current_qd=head[:, 1],
            current_qdd=head[:, 2],
            q_goal=tail[:, 0],
            solver_success=baseline_result.success,
        )
        reference_dist = sample_trajectory_distances(
            reference,
            evaluator,
            obstacle,
            dt=args.audit_dt,
            density=config["risk"]["validation_density"],
        )
        reference_audit = {
            "D_min_ref_obs_m": reference_dist["min_distance_m"],
            "nearest_link": reference_dist["nearest_link"],
            "min_time_s": reference_dist["min_time_s"],
            "dense_verification_if_reference_were_executed": asdict(reference_validation),
            "risk_condition_ok": reference_dist["min_distance_m"] < args.reference_risk_threshold_m,
            "note": "Reference trajectory is evaluated only; it must not be executed in this risky layout.",
        }
        write_json(trial_dir / "reference_risk.json", reference_audit)

        optimizer = _risk_optimizer(config, head, tail, durations, limits, evaluator, obstacle, None)
        candidate_result = optimizer.optimize(baseline_result.p_inner)
        candidate = candidate_result.trajectory
        verification = verifier.verify(
            candidate,
            obstacle,
            current_q=head[:, 0],
            current_qd=head[:, 1],
            current_qdd=head[:, 2],
            q_goal=tail[:, 0],
            solver_success=candidate_result.success,
        )
        dense_verification = asdict(verification)
        write_json(trial_dir / "dense_verification.json", dense_verification)
        candidate_dist = sample_trajectory_distances(
            candidate,
            evaluator,
            obstacle,
            dt=args.audit_dt,
            density=config["risk"]["validation_density"],
        )
        candidate_audit = {
            "D_min_cand_val_m": candidate_dist["min_distance_m"],
            "nearest_link": candidate_dist["nearest_link"],
            "min_time_s": candidate_dist["min_time_s"],
            "optimization": {
                "success": bool(candidate_result.success),
                "message": candidate_result.message,
                "elapsed_ms": candidate_result.elapsed_ms,
                "iterations": candidate_result.iterations,
                "function_evaluations": candidate_result.function_evaluations,
                "initial_min_distance_m": candidate_result.initial_min_distance,
                "final_min_distance_m": candidate_result.final_min_distance,
                "initial_risk": candidate_result.initial_risk,
                "final_risk": candidate_result.final_risk,
            },
        }
        write_json(trial_dir / "candidate_audit.json", candidate_audit)

        fields = ["trajectory", "t_s", *[f"q{j+1}_rad" for j in range(6)], *[f"qd{j+1}_rad_s" for j in range(6)], *[f"qdd{j+1}_rad_s2" for j in range(6)]]
        write_csv(
            trial_dir / "reference_trajectory.csv",
            trajectory_rows("reference", reference, args.trajectory_dt),
            fields,
        )
        write_csv(
            trial_dir / "optimized_trajectory.csv",
            trajectory_rows("ccro_nubs_candidate", candidate, args.trajectory_dt),
            fields,
        )
        np.savez_compressed(
            trial_dir / "trajectories.npz",
            reference_inner=baseline_result.p_inner,
            candidate_inner=candidate_result.p_inner,
            durations=durations,
            q_start=head[:, 0],
            q_goal=tail[:, 0],
        )
        save_distance_curve(
            trial_dir / "figures" / "distance_risk_curve.png",
            reference_dist,
            candidate_dist,
            config["validation"]["d_accept"],
        )
        save_joint_preview(trial_dir / "figures" / "joint_trajectory_preview.png", reference, candidate)

        validation_ready = bool(candidate_result.success and verification.accepted)
        if args.mode in {"preflight", "execute"} and validation_ready:
            preflight = pre_execution_recheck(
                args,
                trial_dir,
                surface_model,
                evaluator,
                verifier,
                candidate,
                obstacle_model,
                head,
                tail,
            )
        elif args.mode in {"preflight", "execute"}:
            preflight = {"accepted": False, "reasons": ["candidate_not_dense_accepted"], "skipped": False}
            write_json(trial_dir / "pre_execution_recheck.json", preflight)
        else:
            preflight = {"accepted": False, "skipped": True, "reason": "mode capture-plan does not perform pre-execution recheck"}

        if args.mode == "execute":
            execution = guarded_execute_decision(args, trial_dir, validation_ready, preflight)
        else:
            execution = {
                "robot_commanded": False,
                "attempted": False,
                "executed": False,
                "reason": f"mode {args.mode} does not command the robot",
            }
        status = classify_status(reference_audit, dense_verification, preflight, execution, args)
    except Exception as exc:
        if status == "INIT":
            status = "FAILED"
        failure = {"status": status, "error": str(exc), "robot_commanded": False}
        write_json(trial_dir / "failure.json", failure)

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scenario": args.scenario,
        "scenario_title": scenario_cfg["title"],
        "repeat": args.repeat,
        "mode": args.mode,
        "status": status,
        "robot_commanded": bool(execution.get("robot_commanded", False)),
        "output_dir": str(trial_dir),
        "elapsed_s": time.perf_counter() - started,
        "obstacle_model": obstacle_model,
        "reference_risk": reference_audit,
        "candidate_audit": candidate_audit,
        "dense_verification": dense_verification,
        "pre_execution_recheck": preflight,
        "execution": execution,
        "parameters": {
            "D_min_ref_risk_threshold_m": args.reference_risk_threshold_m,
            "D_min_exec_obs_threshold_m": args.execution_clearance_threshold_m,
            "start_q_tolerance_rad": args.start_q_tolerance_rad,
            "required_operator_phrase": REQUIRED_OPERATOR_PHRASE,
            "use_current_start": args.use_current_start,
        },
    }
    write_json(trial_dir / "summary.json", summary)
    update_index(output_root)
    return summary


def classify_status(
    reference_audit: dict[str, Any] | None,
    dense_verification: dict[str, Any] | None,
    preflight: dict[str, Any] | None,
    execution: dict[str, Any],
    args: argparse.Namespace,
) -> str:
    if reference_audit is None:
        return "REFERENCE_NOT_EVALUATED"
    if not reference_audit.get("risk_condition_ok", False):
        return "REFERENCE_NOT_RISKY_ENOUGH"
    if dense_verification is None or not dense_verification.get("accepted", False):
        return "VALIDATION_REJECTED"
    if args.mode == "capture-plan":
        return "PLANNED_AND_DENSE_ACCEPTED"
    if not preflight or not preflight.get("accepted", False):
        return "PRE_EXECUTION_REJECTED"
    if args.mode == "execute" and execution.get("executed", False):
        return "EXECUTED_REQUIRES_OFFLINE_AUDIT"
    if args.mode == "execute":
        return "EXECUTION_BLOCKED_BY_GUARD"
    return "PREFLIGHT_PASS_NOT_EXECUTED"


def check_planar_reference_gate(
    config: dict[str, Any],
    head: np.ndarray,
    tail: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Reject the old stage-2 joint trajectory for tabletop real-robot trials.

    The historical ccro_stage2 trajectory moves shoulder/elbow joints strongly,
    causing a nodding motion that is unsuitable above a desk.  6.5.2 hardware
    trials must use a dedicated tabletop planar reference/candidate path.
    """
    q_delta = tail[:, 0] - head[:, 0]
    shoulder_elbow_wrist_pitch = np.abs(q_delta[[1, 2, 3, 4]])
    max_pitch_family_delta = float(np.max(shoulder_elbow_wrist_pitch))
    checks = {
        "allow_nonplanar_reference": bool(args.allow_nonplanar_reference),
        "pitch_family_delta_le_limit": max_pitch_family_delta <= args.max_pitch_family_delta_rad,
    }
    accepted = bool(checks["allow_nonplanar_reference"] or checks["pitch_family_delta_le_limit"])
    reason = "ok" if accepted else (
        "Rejected non-planar desk reference: q2/q3/q4/q5 change is too large for a tabletop "
        "real-robot trial. Use a planar Cartesian/waypoint reference instead of config/ccro_stage2.yaml."
    )
    return {
        "accepted": accepted,
        "reason": reason,
        "checks": checks,
        "q_start": head[:, 0].tolist(),
        "q_goal": tail[:, 0].tolist(),
        "q_delta": q_delta.tolist(),
        "q_delta_deg": np.rad2deg(q_delta).tolist(),
        "max_pitch_family_delta_rad": max_pitch_family_delta,
        "max_pitch_family_delta_deg": float(np.rad2deg(max_pitch_family_delta)),
        "limit_rad": args.max_pitch_family_delta_rad,
        "note": (
            "For 6.5.2 tabletop execution, the reference motion should keep TCP height and attitude "
            "approximately constant and move in the table plane by straight or gently curved Cartesian waypoints."
        ),
    }


def guarded_execute_decision(
    args: argparse.Namespace,
    trial_dir: Path,
    validation_ready: bool,
    preflight: dict[str, Any] | None,
) -> dict[str, Any]:
    phrase_ok = args.operator_phrase == REQUIRED_OPERATOR_PHRASE
    checks = {
        "allow_real_robot_commands": bool(args.allow_real_robot_commands),
        "operator_phrase_ok": bool(phrase_ok),
        "candidate_dense_accepted": bool(validation_ready),
        "pre_execution_recheck_ok": bool(preflight and preflight.get("accepted")),
        "supported_aubo_nubs_trajectory_api_available": False,
    }
    reasons = [name for name, ok in checks.items() if not ok]
    payload = {
        "robot_commanded": False,
        "attempted": False,
        "executed": False,
        "checks": checks,
        "reasons": reasons,
        "blocking_reason": (
            "No robot command was sent. The repository currently lacks a bounded AUBO "
            "joint/NUBS trajectory queue or batch execution API; point-by-point Python "
            "streaming is intentionally not used for 6.5.2."
        ),
    }
    write_json(trial_dir / "execution_guard.json", payload)
    return payload


def aggregate_scene(scene_dir: Path) -> dict[str, Any]:
    trials = []
    for path in sorted((scene_dir / "trials").glob("*/summary.json")):
        try:
            trials.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            continue
    statuses: dict[str, int] = {}
    for trial in trials:
        statuses[trial.get("status", "UNKNOWN")] = statuses.get(trial.get("status", "UNKNOWN"), 0) + 1
    dense_ok = sum(bool(t.get("dense_verification", {}).get("accepted")) for t in trials)
    preflight_ok = sum(bool(t.get("pre_execution_recheck", {}).get("accepted")) for t in trials)
    return {
        "trial_count": len(trials),
        "statuses": statuses,
        "dense_accepted_count": dense_ok,
        "preflight_accepted_count": preflight_ok,
        "robot_commanded_count": sum(bool(t.get("robot_commanded")) for t in trials),
        "D_min_ref_obs_m": [t["reference_risk"]["D_min_ref_obs_m"] for t in trials if t.get("reference_risk")],
        "D_min_cand_val_m": [t["candidate_audit"]["D_min_cand_val_m"] for t in trials if t.get("candidate_audit")],
        "trials": [str(Path(t["output_dir"]).name) for t in trials],
    }


def update_index(output_root: Path) -> None:
    metrics = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "robot_commanded": False,
        "scenes": {},
    }
    for scene_dir in sorted(output_root.parent.glob("rs*_table_obstacle")):
        metrics["scenes"][scene_dir.name] = aggregate_scene(scene_dir)
    write_json(output_root.parent / "metrics.json", metrics)

    lines = [
        "# 6.5.2 Static Avoidance Summary",
        "",
        "Robot commanded by current scripts: **false**",
        "",
        "| scene | trials | dense accepted | preflight accepted | commanded | statuses |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for scene, row in metrics["scenes"].items():
        lines.append(
            f"| {scene} | {row['trial_count']} | {row['dense_accepted_count']} | "
            f"{row['preflight_accepted_count']} | {row['robot_commanded_count']} | "
            f"{row['statuses']} |"
        )
    (output_root.parent / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["capture-plan", "preflight", "execute"], default="capture-plan")
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), required=True)
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config-dir", type=Path, default=ROOT / "config")
    parser.add_argument("--urdf", type=Path, default=ROOT / "urdf" / "aubo_i16_gripper.urdf")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--capture-duration-s", type=float, default=4.0)
    parser.add_argument("--recheck-duration-s", type=float, default=2.0)
    parser.add_argument("--no-prompt", action="store_true")
    parser.add_argument("--allow-overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=20260652)

    parser.add_argument("--self-filter-threshold", type=float, default=0.08)
    parser.add_argument("--voxel-size", type=float, default=0.02)
    parser.add_argument("--cluster-eps", type=float, default=0.06)
    parser.add_argument("--cluster-min-samples", type=int, default=15)
    parser.add_argument("--cluster-min-points", type=int, default=30)
    parser.add_argument("--cluster-min-volume", type=float, default=0.001)
    parser.add_argument("--obstacle-min-points", type=int, default=80)
    parser.add_argument("--obstacle-model-frames", type=int, default=8)
    parser.add_argument("--max-obstacle-points", type=int, default=6000)
    parser.add_argument("--remove-planes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plane-dist", type=float, default=0.02)
    parser.add_argument("--max-planes", type=int, default=1)
    parser.add_argument("--temporal-denoise", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--denoise-voxel", type=float, default=0.02)
    parser.add_argument("--denoise-conf", type=int, default=2)
    parser.add_argument("--denoise-decay", type=float, default=0.92)
    parser.add_argument("--surface-density", choices=["coarse", "medium", "dense"], default="coarse")

    parser.add_argument("--audit-dt", type=float, default=0.04)
    parser.add_argument("--trajectory-dt", type=float, default=0.04)
    parser.add_argument("--reference-risk-threshold-m", type=float, default=0.08)
    parser.add_argument("--execution-clearance-threshold-m", type=float, default=0.08)
    parser.add_argument("--start-q-tolerance-rad", type=float, default=0.02)
    parser.add_argument("--recheck-center-tolerance-m", type=float, default=0.035)
    parser.add_argument("--recheck-radius-tolerance-m", type=float, default=0.03)
    parser.add_argument("--skip-recheck", action="store_true")
    parser.add_argument("--use-current-start", action="store_true", help="Use live mean q during obstacle capture as NUBS start.")
    parser.add_argument(
        "--allow-nonplanar-reference",
        action="store_true",
        help="Allow the historical ccro_stage2 joint reference for offline analysis only; not recommended for tabletop hardware.",
    )
    parser.add_argument("--max-pitch-family-delta-rad", type=float, default=0.20)

    parser.add_argument("--allow-real-robot-commands", action="store_true")
    parser.add_argument("--operator-phrase", default="")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_trial(args)
    print(
        json.dumps(
            {
                "scenario": summary["scenario"],
                "repeat": summary["repeat"],
                "mode": summary["mode"],
                "status": summary["status"],
                "robot_commanded": summary["robot_commanded"],
                "output_dir": summary["output_dir"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
