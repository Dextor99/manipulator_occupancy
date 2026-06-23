"""CCRO-NUBS stage-one validation and result generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time
from typing import Any

import numpy as np
import yaml

from planning.nubs_trajectory import NUBSTrajectory6D
from planning.optimizer import FixedTimeNUBSOptimizer, JointLimits


ROOT = Path(__file__).resolve().parents[1]


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _boundary_states(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    trajectory = config["trajectory"]
    head = NUBSTrajectory6D.make_boundary_state(
        trajectory["q_start"], trajectory["qd_start"], trajectory["qdd_start"]
    )
    tail = NUBSTrajectory6D.make_boundary_state(
        trajectory["q_goal"], trajectory["qd_goal"], trajectory["qdd_goal"]
    )
    return head, tail


def _durations(config: dict[str, Any]) -> np.ndarray:
    trajectory = config["trajectory"]
    count = int(trajectory["segment_count"])
    total = float(trajectory["total_duration"])
    if count <= 0 or total <= 0.0:
        raise ValueError("segment_count and total_duration must be positive")
    explicit = trajectory.get("segment_durations")
    if explicit is not None:
        durations = np.asarray(explicit, dtype=np.float64)
        if durations.shape != (count,) or not np.all(np.isfinite(durations)):
            raise ValueError("segment_durations must match segment_count and be finite")
        if np.any(durations <= 1.0e-8):
            raise ValueError("all segment_durations must be greater than 1e-8")
        if not np.isclose(float(np.sum(durations)), total, rtol=1.0e-9, atol=1.0e-12):
            raise ValueError("segment_durations must sum to total_duration")
        return durations
    return np.full(count, total / count, dtype=np.float64)


def _energy_for(
    points: np.ndarray,
    head: np.ndarray,
    tail: np.ndarray,
    durations: np.ndarray,
) -> float:
    return NUBSTrajectory6D().generate(points, head, tail, durations).energy()


def check_energy_gradient(
    points: np.ndarray,
    head: np.ndarray,
    tail: np.ndarray,
    durations: np.ndarray,
    epsilon: float,
) -> dict[str, Any]:
    trajectory = NUBSTrajectory6D().generate(points, head, tail, durations)
    energy, analytic, _ = trajectory.energy_and_gradient()
    numeric = np.zeros_like(points)
    for row in range(points.shape[0]):
        for col in range(points.shape[1]):
            plus = points.copy()
            minus = points.copy()
            plus[row, col] += epsilon
            minus[row, col] -= epsilon
            numeric[row, col] = (
                _energy_for(plus, head, tail, durations)
                - _energy_for(minus, head, tail, durations)
            ) / (2.0 * epsilon)
    difference = analytic - numeric
    denominator = max(float(np.linalg.norm(numeric)), 1.0e-12)
    relative_error = float(np.linalg.norm(difference) / denominator)
    norm_product = float(np.linalg.norm(analytic) * np.linalg.norm(numeric))
    cosine = 1.0 if norm_product < 1.0e-16 else float(
        np.dot(analytic.ravel(), numeric.ravel()) / norm_product
    )
    return {
        "energy": float(energy),
        "relative_error": relative_error,
        "cosine_similarity": cosine,
        "max_absolute_error": float(np.max(np.abs(difference), initial=0.0)),
        "analytic_norm": float(np.linalg.norm(analytic)),
        "numeric_norm": float(np.linalg.norm(numeric)),
    }


def _limit_metrics(samples, limits: JointLimits) -> dict[str, float]:
    q_v = np.maximum(limits.q_min[None, :] - samples.q, 0.0)
    q_v = np.maximum(q_v, np.maximum(samples.q - limits.q_max[None, :], 0.0))
    qd_v = np.maximum(np.abs(samples.qd) - limits.qd_max[None, :], 0.0)
    qdd_v = np.maximum(np.abs(samples.qdd) - limits.qdd_max[None, :], 0.0)
    return {
        "max_abs_q": float(np.max(np.abs(samples.q))),
        "max_abs_qd": float(np.max(np.abs(samples.qd))),
        "max_abs_qdd": float(np.max(np.abs(samples.qdd))),
        "max_q_violation": float(np.max(q_v)),
        "max_qd_violation": float(np.max(qd_v)),
        "max_qdd_violation": float(np.max(qdd_v)),
    }


def _benchmark(
    points: np.ndarray,
    head: np.ndarray,
    tail: np.ndarray,
    durations: np.ndarray,
    runs: int,
) -> dict[str, float]:
    construct: list[float] = []
    sample: list[float] = []
    times = np.linspace(0.0, float(np.sum(durations)), 101)
    for _ in range(max(runs, 1)):
        start = time.perf_counter()
        trajectory = NUBSTrajectory6D().generate(points, head, tail, durations)
        construct.append((time.perf_counter() - start) * 1000.0)
        start = time.perf_counter()
        trajectory.sample(times)
        sample.append((time.perf_counter() - start) * 1000.0)
    return {
        "runs": int(max(runs, 1)),
        "construct_mean_ms": float(np.mean(construct)),
        "construct_p95_ms": float(np.percentile(construct, 95)),
        "construct_max_ms": float(np.max(construct)),
        "sample_101_mean_ms": float(np.mean(sample)),
        "sample_101_p95_ms": float(np.percentile(sample, 95)),
        "sample_101_max_ms": float(np.max(sample)),
    }


def _make_plot(initial, final, output_path: Path, joint_names: list[str]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    for joint in range(6):
        label = joint_names[joint]
        axes[0].plot(initial.times, initial.q[:, joint], "--", alpha=0.45)
        axes[0].plot(final.times, final.q[:, joint], label=label)
        axes[1].plot(final.times, final.qd[:, joint], label=label)
        axes[2].plot(final.times, final.qdd[:, joint], label=label)
    axes[0].set_ylabel("q / rad")
    axes[1].set_ylabel("qd / rad s$^{-1}$")
    axes[2].set_ylabel("qdd / rad s$^{-2}$")
    axes[2].set_xlabel("time / s")
    axes[0].set_title("Dashed: initial interpolation; solid: optimized NUBS")
    axes[0].legend(ncol=3, fontsize=8)
    for axis in axes:
        axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _write_table(metrics: dict[str, Any], output_path: Path) -> None:
    boundary = metrics["boundary_errors"]
    limits = metrics["limits"]
    gradient = metrics["gradient_check"]
    optimization = metrics["optimization"]
    rows = [
        ("start position error", boundary["q_start"]),
        ("goal position error", boundary["q_goal"]),
        ("start velocity error", boundary["qd_start"]),
        ("goal velocity error", boundary["qd_goal"]),
        ("start acceleration error", boundary["qdd_start"]),
        ("goal acceleration error", boundary["qdd_goal"]),
        ("waypoint error", metrics["waypoint_error"]),
        ("gradient relative error", gradient["relative_error"]),
        ("gradient cosine", gradient["cosine_similarity"]),
        ("initial energy", optimization["initial_energy"]),
        ("final energy", optimization["final_energy"]),
        ("max q violation", limits["max_q_violation"]),
        ("max qd violation", limits["max_qd_violation"]),
        ("max qdd violation", limits["max_qdd_violation"]),
        ("optimization time / ms", optimization["elapsed_ms"]),
    ]
    lines = ["| metric | value |", "|---|---:|"]
    lines.extend(f"| {name} | {value:.8g} |" for name, value in rows)
    lines.append("")
    lines.append(f"Overall acceptance: **{'PASS' if metrics['accepted'] else 'FAIL'}**")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(config_path: str | Path, output_override: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    config = _load_yaml(config_path)
    if int(config["trajectory"]["system_order"]) != 3:
        raise ValueError("stage one currently supports only system_order=3")
    np.random.seed(int(config["experiment"]["random_seed"]))
    output_dir = Path(output_override or config["experiment"]["output_dir"])
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output_dir / "config.yaml")

    head, tail = _boundary_states(config)
    durations = _durations(config)
    inner_initial = NUBSTrajectory6D.linear_inner_points(
        head[:, 0], tail[:, 0], durations
    )
    initial_trajectory = NUBSTrajectory6D().generate(
        inner_initial, head, tail, durations
    )

    robot = config["robot"]
    limits = JointLimits.from_arrays(
        robot["q_min"], robot["q_max"], robot["qd_max"], robot["qdd_max"]
    )
    optimizer = FixedTimeNUBSOptimizer(
        head,
        tail,
        durations,
        limits,
        **config["optimizer"],
    )
    optimization = optimizer.optimize(inner_initial)
    final_trajectory = optimization.trajectory

    validation = config["validation"]
    initial_samples = initial_trajectory.dense_sample(validation["dense_time_step"])
    final_samples = final_trajectory.dense_sample(validation["dense_time_step"])
    gradient = check_energy_gradient(
        inner_initial,
        head,
        tail,
        durations,
        float(config["optimizer"]["finite_difference_epsilon"]),
    )
    boundary = final_trajectory.boundary_errors()
    waypoint_error = final_trajectory.waypoint_error()
    limit_metrics = _limit_metrics(final_samples, limits)
    numerical_jerk_energy = float(
        np.trapezoid(np.sum(final_samples.jerk**2, axis=1), final_samples.times)
    )
    benchmark = _benchmark(
        optimization.p_inner,
        head,
        tail,
        durations,
        int(config["experiment"]["benchmark_runs"]),
    )

    checks = {
        "optimizer_success": bool(optimization.success),
        "start_position": bool(
            boundary["q_start"] <= validation["boundary_position_tolerance"]
        ),
        "goal_position": bool(
            boundary["q_goal"] <= validation["boundary_position_tolerance"]
        ),
        "start_velocity": bool(
            boundary["qd_start"] <= validation["boundary_velocity_tolerance"]
        ),
        "goal_velocity": bool(
            boundary["qd_goal"] <= validation["boundary_velocity_tolerance"]
        ),
        "start_acceleration": bool(
            boundary["qdd_start"] <= validation["boundary_acceleration_tolerance"]
        ),
        "goal_acceleration": bool(
            boundary["qdd_goal"] <= validation["boundary_acceleration_tolerance"]
        ),
        "waypoint_interpolation": bool(
            waypoint_error <= validation["waypoint_tolerance"]
        ),
        "gradient": bool(
            gradient["relative_error"]
            <= validation["gradient_relative_tolerance"]
        ),
        "energy_nonincrease": bool(
            optimization.final_energy
            <= optimization.initial_energy + validation["energy_increase_tolerance"]
        ),
        "joint_limits": bool(
            max(
                limit_metrics["max_q_violation"],
                limit_metrics["max_qd_violation"],
                limit_metrics["max_qdd_violation"],
            )
            <= validation["limit_tolerance"]
        ),
    }
    metrics = {
        "accepted": bool(all(checks.values())),
        "checks": checks,
        "joint_names": robot["joint_names"],
        "durations": durations.tolist(),
        "boundary_errors": boundary,
        "waypoint_error": waypoint_error,
        "gradient_check": gradient,
        "limits": limit_metrics,
        "energy_consistency": {
            "analytic": optimization.final_energy,
            "sampled_jerk_integral": numerical_jerk_energy,
            "absolute_error": abs(optimization.final_energy - numerical_jerk_energy),
        },
        "optimization": {
            "success": optimization.success,
            "status": optimization.status,
            "message": optimization.message,
            "iterations": optimization.iterations,
            "function_evaluations": optimization.function_evaluations,
            "initial_energy": optimization.initial_energy,
            "final_energy": optimization.final_energy,
            "final_cost": optimization.final_cost,
            "penalty_cost": optimization.penalty_cost,
            "gradient_norm": optimization.gradient_norm,
            "elapsed_ms": optimization.elapsed_ms,
        },
        "benchmark": benchmark,
        "p_inner_initial": inner_initial.tolist(),
        "p_inner_optimized": optimization.p_inner.tolist(),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    _write_table(metrics, output_dir / "table_stage1.md")
    _make_plot(
        initial_samples,
        final_samples,
        output_dir / "trajectory_stage1.png",
        list(robot["joint_names"]),
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default=str(ROOT / "config" / "ccro_stage1.yaml")
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    metrics = run(args.config, args.output)
    print(json.dumps(metrics, indent=2, ensure_ascii=False, default=_json_default))
    if not metrics["accepted"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
