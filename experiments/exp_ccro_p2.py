"""P2 validation: fixed, total-time and segment-time NUBS optimization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import yaml

from planning.nubs_trajectory import NUBSTrajectory6D
from planning.optimizer import FixedTimeNUBSOptimizer, JointLimits
from planning.time_optimizer import VariableTimeNUBSOptimizer

ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _json_default(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    raise TypeError(type(value).__name__)


def _gradient_check(optimizer: VariableTimeNUBSOptimizer, variables: np.ndarray, epsilon: float) -> dict:
    cost, analytic = optimizer.objective(variables)
    numeric = np.zeros_like(variables)
    for index in range(len(variables)):
        plus, minus = variables.copy(), variables.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        numeric[index] = (optimizer.objective(plus)[0] - optimizer.objective(minus)[0]) / (2*epsilon)
    difference = analytic - numeric
    relative = float(np.linalg.norm(difference) / max(np.linalg.norm(numeric), 1e-12))
    product = float(np.linalg.norm(analytic)*np.linalg.norm(numeric))
    cosine = 1.0 if product < 1e-16 else float(np.dot(analytic, numeric)/product)
    return {"cost": cost, "relative_error": relative, "cosine_similarity": cosine, "max_absolute_error": float(np.max(np.abs(difference))), "analytic_norm": float(np.linalg.norm(analytic)), "numeric_norm": float(np.linalg.norm(numeric))}


def _result_metrics(result) -> dict:
    trajectory = result.trajectory
    samples = trajectory.dense_sample(0.01)
    return {
        "success": result.success,
        "message": result.message,
        "durations": result.durations,
        "total_duration": float(np.sum(result.durations)),
        "initial_cost": result.initial_cost,
        "final_cost": result.final_cost,
        "cost_reduction": result.initial_cost-result.final_cost,
        "jerk_energy": result.final_energy,
        "time_cost": result.time_cost,
        "penalty_cost": result.penalty_cost,
        "iterations": result.iterations,
        "function_evaluations": result.function_evaluations,
        "gradient_norm": result.gradient_norm,
        "elapsed_ms": result.elapsed_ms,
        "boundary_errors": trajectory.boundary_errors(),
        "waypoint_error": trajectory.waypoint_error(),
        "max_abs_qd": float(np.max(np.abs(samples.qd))),
        "max_abs_qdd": float(np.max(np.abs(samples.qdd))),
        "max_q_violation": result.max_q_violation,
        "max_qd_violation": result.max_qd_violation,
        "max_qdd_violation": result.max_qdd_violation,
        "p_inner": result.p_inner,
    }


def _plot(trajectories: dict[str, NUBSTrajectory6D], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    styles = {"fixed": "--", "total": "-", "segment": ":"}
    for name, trajectory in trajectories.items():
        samples = trajectory.dense_sample(0.02)
        axes[0].plot(samples.times, samples.q[:, 0], styles[name], label=f"{name}: q1")
        axes[1].plot(samples.times, np.linalg.norm(samples.jerk, axis=1), styles[name], label=name)
    axes[0].set_ylabel("q1 / rad"); axes[1].set_ylabel("||jerk||"); axes[1].set_xlabel("physical time / s")
    for axis in axes: axis.grid(True, alpha=.3); axis.legend()
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def run(config_path: str | Path = ROOT/"config/ccro_p2.yaml", output_override: str | Path | None = None) -> dict:
    config_path = Path(config_path).resolve(); config = _load(config_path)
    source_path = ROOT/config["source_config"]; source = _load(source_path)
    output = Path(output_override or config["output_dir"]); output = output if output.is_absolute() else ROOT/output
    output.mkdir(parents=True, exist_ok=True); shutil.copy2(config_path, output/"config.yaml")
    np.random.seed(int(config["random_seed"]))
    tr = source["trajectory"]
    head = NUBSTrajectory6D.make_boundary_state(tr["q_start"], tr["qd_start"], tr["qdd_start"])
    tail = NUBSTrajectory6D.make_boundary_state(tr["q_goal"], tr["qd_goal"], tr["qdd_goal"])
    durations = np.asarray(tr["segment_durations"], dtype=float)
    limits = JointLimits.from_arrays(source["robot"]["q_min"], source["robot"]["q_max"], source["robot"]["qd_max"], source["robot"]["qdd_max"])
    linear = NUBSTrajectory6D.linear_inner_points(head[:, 0], tail[:, 0], durations)
    fixed = FixedTimeNUBSOptimizer(head, tail, durations, limits, **source["optimizer"]).optimize(linear)
    common = config["optimizer"].copy()
    total_optimizer = VariableTimeNUBSOptimizer(head, tail, durations, limits, mode="total", **common)
    segment_optimizer = VariableTimeNUBSOptimizer(head, tail, durations, limits, mode="segment", **common)
    epsilon = float(common["finite_difference_epsilon"])
    total_gradient = _gradient_check(total_optimizer, total_optimizer.encode(fixed.p_inner, durations), epsilon)
    segment_gradient = _gradient_check(segment_optimizer, segment_optimizer.encode(fixed.p_inner, durations), epsilon)
    total = total_optimizer.optimize(fixed.p_inner)
    segment = segment_optimizer.optimize(fixed.p_inner)
    fixed_traj = fixed.trajectory
    fixed_cost = fixed.final_energy + float(common["lambda_time"])*float(np.sum(durations))
    metrics = {
        "stage": "P2", "source_config": str(source_path.relative_to(ROOT)),
        "fixed": {"success": fixed.success, "durations": durations, "total_duration": float(np.sum(durations)), "jerk_energy": fixed.final_energy, "final_cost": fixed_cost, "elapsed_ms": fixed.elapsed_ms, "boundary_errors": fixed_traj.boundary_errors(), "waypoint_error": fixed_traj.waypoint_error()},
        "total": _result_metrics(total), "segment": _result_metrics(segment),
        "gradient_check": {"total": total_gradient, "segment": segment_gradient},
    }
    v = config["validation"]
    def valid(item):
        return item["success"] and max(item["boundary_errors"].values()) <= v["boundary_tolerance"] and item["waypoint_error"] <= v["waypoint_tolerance"] and max(item["max_q_violation"], item["max_qd_violation"], item["max_qdd_violation"]) <= v["limit_tolerance"] and item["cost_reduction"] > v["cost_decrease_tolerance"]
    metrics["checks"] = {
        "total_gradient": total_gradient["relative_error"] <= v["gradient_relative_tolerance"],
        "segment_gradient": segment_gradient["relative_error"] <= v["gradient_relative_tolerance"],
        "total_valid": valid(metrics["total"]), "segment_valid": valid(metrics["segment"]),
    }
    metrics["accepted"] = bool(all(metrics["checks"].values()))
    (output/"metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False, default=_json_default)+"\n", encoding="utf-8")
    rows = []
    for name in ("fixed", "total", "segment"):
        item=metrics[name]; rows.append(f"| {name} | {item['total_duration']:.6f} | {item['jerk_energy']:.8f} | {item['final_cost']:.8f} | {item['elapsed_ms']:.2f} |")
    table = "| method | duration / s | jerk energy | objective | optimize / ms |\n|---|---:|---:|---:|---:|\n"+"\n".join(rows)+f"\n\nTotal gradient relative error: `{total_gradient['relative_error']:.3e}`  \nSegment gradient relative error: `{segment_gradient['relative_error']:.3e}`  \n\nOverall acceptance: **{'PASS' if metrics['accepted'] else 'FAIL'}**\n"
    (output/"table_p2.md").write_text(table, encoding="utf-8")
    _plot({"fixed": fixed_traj, "total": total.trajectory, "segment": segment.trajectory}, output/"trajectory_p2.png")
    return metrics


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--config", default=str(ROOT/"config/ccro_p2.yaml")); parser.add_argument("--output-dir")
    args=parser.parse_args(); metrics=run(args.config,args.output_dir); print(json.dumps({"accepted": metrics["accepted"], "checks": metrics["checks"], "total_duration": metrics["total"]["total_duration"], "segment_duration": metrics["segment"]["total_duration"]}, indent=2)); raise SystemExit(0 if metrics["accepted"] else 2)


if __name__ == "__main__": main()

