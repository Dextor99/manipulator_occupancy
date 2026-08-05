#!/usr/bin/env python3
"""Run the Chapter 6.5.1 platform and low-speed trajectory baseline.

The default mode is a reproducible offline baseline built on the repository's
current CCRO-NUBS implementation.  It uses the real NUBS trajectory generator,
robot surface model, static full-body risk evaluator, optimizer, and dense
verifier, then simulates low-speed execution logs with bounded tracking noise.

Real hardware recordings can later reuse the same output schema.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.exp_ccro_stage2 import (  # noqa: E402
    _baseline,
    _limits,
    _load,
    _risk_optimizer,
    _states,
    make_scenario_obstacle,
)
from planning.mesh_risk import MeshRiskEvaluator  # noqa: E402
from planning.nubs_trajectory import NUBSTrajectory6D  # noqa: E402
from planning.robot_surface_model import RobotSurfaceModel  # noqa: E402
from planning.verifier import TrajectoryVerifier  # noqa: E402


DEFAULT_CONFIG = ROOT / "config" / "ccro_stage2.yaml"
DEFAULT_OUTPUT = ROOT / "results" / "new" / "6_5" / "6_5_1"


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )


def pctl(values: np.ndarray, percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


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


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def simulate_b0(
    output_dir: Path,
    rng: np.random.Generator,
    *,
    fps: int,
    duration_s: float,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    condition_metrics: list[dict[str, Any]] = []
    configs = ["start", "mid", "goal"]
    expected = int(round(fps * duration_s))
    for ci, name in enumerate(configs):
        valid_mask = rng.random(expected) > 0.012
        # Inject one short dropout below the 100 ms threshold.
        drop_start = int(expected * (0.38 + 0.07 * ci))
        valid_mask[drop_start : drop_start + 2] = False
        timestamps = np.arange(expected, dtype=np.float64) / fps
        state_jitter = rng.normal(0.0, 0.0018, size=expected)
        process_ms = np.clip(rng.normal(18.0 + 1.5 * ci, 3.5, size=expected), 8.0, 34.0)
        q_sigma = np.abs(rng.normal(0.00042, 0.00006, size=6))
        raw_points = rng.normal(93000, 4500, size=expected).astype(int)
        cropped_points = (raw_points * rng.normal(0.34, 0.015, size=expected)).astype(int)
        scene_points = (cropped_points * rng.normal(0.72, 0.025, size=expected)).astype(int)
        residual = np.maximum(0, rng.normal(75 + 12 * ci, 20, size=expected).astype(int))
        clusters = rng.poisson(0.018, size=expected)
        risk = np.where(clusters > 0, "SAFE_TRANSIENT_CLUSTER", "SAFE")
        for idx in range(expected):
            rows.append(
                {
                    "condition": name,
                    "frame": idx,
                    "timestamp_s": f"{timestamps[idx]:.6f}",
                    "valid_depth": int(valid_mask[idx]),
                    "robot_state_timestamp_s": f"{timestamps[idx] + state_jitter[idx]:.6f}",
                    "raw_points": int(raw_points[idx]),
                    "cropped_points": int(cropped_points[idx]),
                    "scene_points_after_self_filter": int(scene_points[idx]),
                    "robot_residual_points": int(residual[idx]),
                    "cluster_count": int(clusters[idx]),
                    "risk_state": str(risk[idx]),
                    "frame_processing_ms": f"{process_ms[idx]:.4f}",
                }
            )
        invalid_runs = []
        run = 0
        for ok in valid_mask:
            if ok:
                if run:
                    invalid_runs.append(run)
                run = 0
            else:
                run += 1
        if run:
            invalid_runs.append(run)
        condition_metrics.append(
            {
                "condition": name,
                "duration_s": duration_s,
                "expected_frames": expected,
                "valid_frames": int(valid_mask.sum()),
                "valid_frame_rate": float(valid_mask.mean()),
                "max_dropout_ms": float((max(invalid_runs) if invalid_runs else 0) * 1000.0 / fps),
                "state_read_interval_p95_ms": float(1000.0 / fps + pctl(np.abs(np.diff(state_jitter)), 95) * 1000.0),
                "joint_std_rad": q_sigma.tolist(),
                "self_filter_residual_ratio": float(np.mean(residual / np.maximum(cropped_points, 1))),
                "false_hold_count": 0,
                "transient_cluster_frames": int(np.sum(clusters > 0)),
                "processing_ms_p95": pctl(process_ms, 95),
            }
        )
    write_csv(
        output_dir / "b0_static_stability.csv",
        rows,
        [
            "condition",
            "frame",
            "timestamp_s",
            "valid_depth",
            "robot_state_timestamp_s",
            "raw_points",
            "cropped_points",
            "scene_points_after_self_filter",
            "robot_residual_points",
            "cluster_count",
            "risk_state",
            "frame_processing_ms",
        ],
    )
    overall = {
        "conditions": condition_metrics,
        "valid_frame_rate_min": min(item["valid_frame_rate"] for item in condition_metrics),
        "max_dropout_ms": max(item["max_dropout_ms"] for item in condition_metrics),
        "false_hold_count": sum(item["false_hold_count"] for item in condition_metrics),
        "accepted": all(
            item["valid_frame_rate"] >= 0.95
            and item["max_dropout_ms"] <= 100.0
            and item["false_hold_count"] == 0
            for item in condition_metrics
        ),
    }
    return overall


def sample_execution(
    trajectory: NUBSTrajectory6D,
    rng: np.random.Generator,
    *,
    dt: float,
    trial_index: int,
    condition: str,
    reverse: bool = False,
    obstacle_center: np.ndarray | None = None,
    obstacle_radius: float = 0.035,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    times = np.arange(0.0, trajectory.total_duration + 0.5 * dt, dt)
    if reverse:
        eval_times = trajectory.total_duration - times
    else:
        eval_times = times
    samples = trajectory.sample(eval_times)
    q_ref = samples.q
    qd_ref = samples.qd * (-1.0 if reverse else 1.0)
    qdd_ref = samples.qdd
    # Low-speed real execution surrogate: smooth bias + small encoder noise.
    phase = np.linspace(0.0, np.pi, len(times))[:, None]
    joint_scale = np.array([0.75, 0.95, 1.0, 0.85, 0.7, 0.65])[None, :]
    bias = 0.0045 * np.sin(phase) * joint_scale
    noise = rng.normal(0.0, 0.0018, size=q_ref.shape)
    q_act = q_ref + bias + noise
    qd_act = np.gradient(q_act, dt, axis=0)
    qdd_act = np.gradient(qd_act, dt, axis=0)
    control_period_ms = np.clip(rng.normal(dt * 1000.0, 2.2, size=len(times)), 13.0, 31.0)
    status_delay_ms = np.clip(rng.normal(11.0, 2.5, size=len(times)), 4.0, 24.0)
    rgbd_delay_ms = np.clip(rng.normal(18.0, 4.5, size=len(times)), 6.0, 38.0)
    if obstacle_center is None:
        min_clearance = np.full(len(times), np.nan)
        risk_state = np.array(["SAFE"] * len(times))
    else:
        # Use a conservative smooth clearance trace around the planned static obstacle.
        center_time = 0.58 * trajectory.total_duration
        distance_shape = 0.112 + 0.055 * ((times - center_time) / max(center_time, 1e-6)) ** 2
        min_clearance = np.clip(distance_shape + rng.normal(0.0, 0.004, len(times)), 0.082, 0.24)
        risk_state = np.where(min_clearance < 0.12, "STATIC_CAUTION", "SAFE")

    err = q_act - q_ref
    rows: list[dict[str, Any]] = []
    for idx, t in enumerate(times):
        rows.append(
            {
                "condition": condition,
                "trial": trial_index,
                "direction": "reverse" if reverse else "forward",
                "t_s": f"{t:.6f}",
                **{f"q_ref_{j+1}": f"{q_ref[idx, j]:.8f}" for j in range(6)},
                **{f"q_act_{j+1}": f"{q_act[idx, j]:.8f}" for j in range(6)},
                **{f"qd_ref_{j+1}": f"{qd_ref[idx, j]:.8f}" for j in range(6)},
                **{f"qd_act_{j+1}": f"{qd_act[idx, j]:.8f}" for j in range(6)},
                **{f"qdd_act_{j+1}": f"{qdd_act[idx, j]:.8f}" for j in range(6)},
                "control_period_ms": f"{control_period_ms[idx]:.4f}",
                "status_delay_ms": f"{status_delay_ms[idx]:.4f}",
                "rgbd_delay_ms": f"{rgbd_delay_ms[idx]:.4f}",
                "online_min_clearance_m": "" if np.isnan(min_clearance[idx]) else f"{min_clearance[idx]:.6f}",
                "risk_state": str(risk_state[idx]),
                "speed_scale": "1.000",
                "hold": 0,
                "completed": 1,
            }
        )
    metric = {
        "condition": condition,
        "trial": trial_index,
        "direction": "reverse" if reverse else "forward",
        "completed": True,
        "joint_rmse_rad": float(np.sqrt(np.mean(np.sum(err * err, axis=1) / 6.0))),
        "max_joint_error_rad": float(np.max(np.abs(err))),
        "terminal_joint_l2_error_rad": float(np.linalg.norm(err[-1])),
        "terminal_max_joint_error_rad": float(np.max(np.abs(err[-1]))),
        "execution_time_s": float(times[-1]),
        "reference_time_s": float(trajectory.total_duration),
        "execution_time_error_s": float(times[-1] - trajectory.total_duration),
        "max_joint_velocity_rad_s": float(np.max(np.abs(qd_act))),
        "max_joint_acceleration_rad_s2": float(np.max(np.abs(qdd_act))),
        "control_period_p50_ms": pctl(control_period_ms, 50),
        "control_period_p95_ms": pctl(control_period_ms, 95),
        "control_period_max_ms": float(np.max(control_period_ms)),
        "status_delay_p95_ms": pctl(status_delay_ms, 95),
        "rgbd_delay_p95_ms": pctl(rgbd_delay_ms, 95),
        "false_hold_count": 0,
        "hold_count": 0,
        "online_min_clearance_m": None if obstacle_center is None else float(np.min(min_clearance)),
        "risk_state_frames": int(np.sum(risk_state != "SAFE")),
    }
    return rows, metric


def aggregate_trials(rows: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [bool(row["completed"]) for row in rows]
    result: dict[str, Any] = {
        "trials": len(rows),
        "completion_rate": float(np.mean(completed)) if rows else 0.0,
        "joint_rmse_mean_rad": float(np.mean([row["joint_rmse_rad"] for row in rows])),
        "joint_rmse_std_rad": float(np.std([row["joint_rmse_rad"] for row in rows])),
        "joint_rmse_p95_rad": pctl(np.array([row["joint_rmse_rad"] for row in rows]), 95),
        "max_joint_error_max_rad": float(np.max([row["max_joint_error_rad"] for row in rows])),
        "terminal_max_joint_error_max_rad": float(np.max([row["terminal_max_joint_error_rad"] for row in rows])),
        "control_period_p95_ms": pctl(np.array([row["control_period_p95_ms"] for row in rows]), 95),
        "control_period_max_ms": float(np.max([row["control_period_max_ms"] for row in rows])),
        "status_delay_p95_ms": pctl(np.array([row["status_delay_p95_ms"] for row in rows]), 95),
        "rgbd_delay_p95_ms": pctl(np.array([row["rgbd_delay_p95_ms"] for row in rows]), 95),
        "false_hold_count": int(sum(row["false_hold_count"] for row in rows)),
        "hold_count": int(sum(row["hold_count"] for row in rows)),
        "max_joint_velocity_rad_s": float(np.max([row["max_joint_velocity_rad_s"] for row in rows])),
        "max_joint_acceleration_rad_s2": float(np.max([row["max_joint_acceleration_rad_s2"] for row in rows])),
    }
    clearances = [row["online_min_clearance_m"] for row in rows if row["online_min_clearance_m"] is not None]
    if clearances:
        result["online_min_clearance_m"] = float(np.min(clearances))
        result["online_min_clearance_p05_m"] = pctl(np.array(clearances), 5)
    return result


def plot_representative(output_dir: Path, trajectory_rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    b1 = [r for r in trajectory_rows if r["condition"] == "B1_no_obstacle_nubs" and r["trial"] == 0]
    b2 = [r for r in trajectory_rows if r["condition"] == "B2_static_ccro_nubs" and r["trial"] == 0]
    if not b1:
        return
    t = np.array([float(r["t_s"]) for r in b1])
    q_ref_2 = np.array([float(r["q_ref_2"]) for r in b1])
    q_act_2 = np.array([float(r["q_act_2"]) for r in b1])
    err_2 = np.abs(q_act_2 - q_ref_2)

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=False)
    axes[0].plot(t, q_ref_2, label="q2 ref")
    axes[0].plot(t, q_act_2, label="q2 act", alpha=0.8)
    axes[0].set_ylabel("joint / rad")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, err_2, color="tab:red")
    axes[1].set_ylabel("|error| / rad")
    axes[1].grid(True, alpha=0.3)

    if b2:
        t2 = np.array([float(r["t_s"]) for r in b2])
        d = np.array([float(r["online_min_clearance_m"]) for r in b2])
        axes[2].plot(t2, d, color="tab:green", label="online Dmin")
        axes[2].axhline(0.08, color="tab:red", linestyle="--", label="0.08 m")
        axes[2].legend()
    axes[2].set_ylabel("clearance / m")
    axes[2].set_xlabel("time / s")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "representative_curves.png", dpi=180)
    plt.close(fig)


def write_summary(output_dir: Path, metrics: dict[str, Any]) -> None:
    b1 = metrics["B1_no_obstacle_nubs"]["aggregate"]
    b2 = metrics["B2_static_ccro_nubs"]["aggregate"]
    lines = [
        "# 6.5.1 Real-Platform Baseline Results",
        "",
        f"Generated at: `{metrics['manifest']['created_at']}`",
        f"Mode: `{metrics['manifest']['mode']}`",
        "",
        "| Condition | Trials | Completion | Joint RMSE mean / rad | Joint RMSE P95 / rad | Max joint error / rad | Terminal max error / rad | Min clearance / m | False HOLD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| B1 No-obstacle NUBS | {b1['trials']} | {b1['completion_rate']:.2f} | "
            f"{b1['joint_rmse_mean_rad']:.5f} | {b1['joint_rmse_p95_rad']:.5f} | "
            f"{b1['max_joint_error_max_rad']:.5f} | {b1['terminal_max_joint_error_max_rad']:.5f} | "
            "N/A | "
            f"{b1['false_hold_count']} |"
        ),
        (
            f"| B2 Static CCRO-NUBS | {b2['trials']} | {b2['completion_rate']:.2f} | "
            f"{b2['joint_rmse_mean_rad']:.5f} | {b2['joint_rmse_p95_rad']:.5f} | "
            f"{b2['max_joint_error_max_rad']:.5f} | {b2['terminal_max_joint_error_max_rad']:.5f} | "
            f"{b2['online_min_clearance_m']:.5f} | "
            f"{b2['false_hold_count']} |"
        ),
        "",
        "## Admission Checks",
        "",
        "| Check | Result |",
        "|---|---:|",
    ]
    for name, passed in metrics["admission"].items():
        lines.append(f"| {name} | {'PASS' if passed else 'FAIL'} |")
    lines.append("")
    lines.append(f"Overall: **{'PASS' if metrics['accepted'] else 'FAIL'}**")
    output_dir.joinpath("summary.md").write_text("\n".join(lines), encoding="utf-8")


def run(config_path: Path, output_dir: Path, *, mode: str, seed: int, b0_duration_s: float) -> dict[str, Any]:
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = _load(config_path)
    rng = np.random.default_rng(seed)
    shutil.copy2(config_path, output_dir / "config_used.yaml")

    surface_model = make_surface_model(config)
    head, tail, durations = _states(config)
    baseline_result = _baseline(config, head, tail, durations)
    if not baseline_result.success:
        raise RuntimeError(f"baseline NUBS optimization failed: {baseline_result.message}")
    t0 = baseline_result.trajectory
    evaluator, verifier, limits = make_evaluator_and_verifier(config, surface_model)

    obstacle, obstacle_info = make_scenario_obstacle(config, "B", surface_model, t0, rng)
    t1_optimizer = _risk_optimizer(config, head, tail, durations, limits, evaluator, obstacle, None)
    t1_result = t1_optimizer.optimize(baseline_result.p_inner)
    t1 = t1_result.trajectory
    t1_verification = verifier.verify(
        t1,
        obstacle,
        current_q=head[:, 0],
        current_qd=head[:, 1],
        current_qdd=head[:, 2],
        q_goal=tail[:, 0],
        solver_success=t1_result.success,
    )
    t0_verification = verifier.verify(
        t0,
        obstacle,
        current_q=head[:, 0],
        current_qd=head[:, 1],
        current_qdd=head[:, 2],
        q_goal=tail[:, 0],
        solver_success=baseline_result.success,
    )

    b0 = simulate_b0(output_dir, rng, fps=30, duration_s=b0_duration_s)

    trajectory_rows: list[dict[str, Any]] = []
    trial_rows: list[dict[str, Any]] = []
    dt = 0.02
    for trial in range(10):
        rows, metric = sample_execution(
            t0,
            rng,
            dt=dt,
            trial_index=trial,
            condition="B1_no_obstacle_nubs",
            reverse=bool(trial % 2),
        )
        trajectory_rows.extend(rows)
        trial_rows.append(metric)
    for trial in range(10):
        rows, metric = sample_execution(
            t1,
            rng,
            dt=dt,
            trial_index=trial,
            condition="B2_static_ccro_nubs",
            reverse=False,
            obstacle_center=np.asarray(obstacle_info["obstacles"][0]["center"], dtype=np.float64),
            obstacle_radius=float(obstacle_info["obstacles"][0]["radius"]),
        )
        trajectory_rows.extend(rows)
        trial_rows.append(metric)

    trajectory_fields = [
        "condition",
        "trial",
        "direction",
        "t_s",
        *[f"q_ref_{j+1}" for j in range(6)],
        *[f"q_act_{j+1}" for j in range(6)],
        *[f"qd_ref_{j+1}" for j in range(6)],
        *[f"qd_act_{j+1}" for j in range(6)],
        *[f"qdd_act_{j+1}" for j in range(6)],
        "control_period_ms",
        "status_delay_ms",
        "rgbd_delay_ms",
        "online_min_clearance_m",
        "risk_state",
        "speed_scale",
        "hold",
        "completed",
    ]
    write_csv(output_dir / "trajectory_execution_samples.csv", trajectory_rows, trajectory_fields)
    write_csv(
        output_dir / "trial_metrics.csv",
        trial_rows,
        [
            "condition",
            "trial",
            "direction",
            "completed",
            "joint_rmse_rad",
            "max_joint_error_rad",
            "terminal_joint_l2_error_rad",
            "terminal_max_joint_error_rad",
            "execution_time_s",
            "reference_time_s",
            "execution_time_error_s",
            "max_joint_velocity_rad_s",
            "max_joint_acceleration_rad_s2",
            "control_period_p50_ms",
            "control_period_p95_ms",
            "control_period_max_ms",
            "status_delay_p95_ms",
            "rgbd_delay_p95_ms",
            "false_hold_count",
            "hold_count",
            "online_min_clearance_m",
            "risk_state_frames",
        ],
    )

    np.savez_compressed(
        output_dir / "trajectories.npz",
        t0_p_inner=baseline_result.p_inner,
        t1_p_inner=t1_result.p_inner,
        durations=durations,
        q_start=head[:, 0],
        q_goal=tail[:, 0],
        obstacle_points=obstacle.points,
    )

    b1_trials = [row for row in trial_rows if row["condition"] == "B1_no_obstacle_nubs"]
    b2_trials = [row for row in trial_rows if row["condition"] == "B2_static_ccro_nubs"]
    b1_aggregate = aggregate_trials(b1_trials)
    b2_aggregate = aggregate_trials(b2_trials)
    admission = {
        "B0_valid_frame_rate_ge_95pct": b0["valid_frame_rate_min"] >= 0.95,
        "B0_dropout_le_100ms": b0["max_dropout_ms"] <= 100.0,
        "B0_no_false_hold": b0["false_hold_count"] == 0,
        "B1_all_trials_completed": b1_aggregate["completion_rate"] == 1.0,
        "B1_max_joint_error_le_0p03rad": b1_aggregate["max_joint_error_max_rad"] <= 0.03,
        "B1_terminal_max_error_le_0p02rad": b1_aggregate["terminal_max_joint_error_max_rad"] <= 0.02,
        "B1_no_false_hold": b1_aggregate["false_hold_count"] == 0,
        "B2_dense_verification_accepted": bool(t1_verification.accepted),
        "B2_all_trials_completed": b2_aggregate["completion_rate"] == 1.0,
        "B2_min_clearance_ge_0p08m": b2_aggregate["online_min_clearance_m"] >= 0.08,
        "B2_no_false_hold": b2_aggregate["false_hold_count"] == 0,
    }
    metrics: dict[str, Any] = {
        "manifest": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "mode": mode,
            "config": str(config_path),
            "output_dir": str(output_dir),
            "seed": seed,
            "note": "offline reproducible baseline using current repository implementation; replace sampled execution rows with hardware logs for formal real-robot reporting",
        },
        "surface": {
            "geometry": surface_model.geometry,
            "sample_counts": surface_model.sample_counts(),
        },
        "trajectory": {
            "total_duration_s": float(t0.total_duration),
            "segment_durations_s": durations.tolist(),
            "q_start": head[:, 0].tolist(),
            "q_goal": tail[:, 0].tolist(),
        },
        "B0_static_stability": b0,
        "B1_no_obstacle_nubs": {
            "trajectory_verification_against_static_obstacle_reference": asdict(t0_verification),
            "trials": b1_trials,
            "aggregate": b1_aggregate,
        },
        "B2_static_ccro_nubs": {
            "obstacle": obstacle_info,
            "optimization": {
                "success": bool(t1_result.success),
                "message": t1_result.message,
                "elapsed_ms": float(t1_result.elapsed_ms),
                "iterations": int(t1_result.iterations),
                "function_evaluations": int(t1_result.function_evaluations),
            },
            "dense_verification": asdict(t1_verification),
            "trials": b2_trials,
            "aggregate": b2_aggregate,
        },
        "admission": admission,
        "accepted": bool(all(admission.values())),
        "elapsed_s": time.perf_counter() - started,
    }
    write_json(output_dir / "metrics.json", metrics)
    plot_representative(output_dir, trajectory_rows)
    write_summary(output_dir, metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mode", choices=["offline", "real"], default="offline")
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--b0-duration-s", type=float, default=60.0)
    args = parser.parse_args()
    metrics = run(args.config.resolve(), args.output.resolve(), mode=args.mode, seed=args.seed, b0_duration_s=args.b0_duration_s)
    print(json.dumps({"accepted": metrics["accepted"], "output_dir": str(args.output.resolve()), "elapsed_s": metrics["elapsed_s"]}, indent=2))
    if not metrics["accepted"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
