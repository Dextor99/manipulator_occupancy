"""CCRO-NUBS stage-three dynamic spatio-temporal risk experiment."""

from __future__ import annotations

import argparse
from dataclasses import asdict, fields
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np
from scipy.spatial import cKDTree
import yaml

from planning.dynamic_optimizer import DynamicRiskNUBSOptimizer
from planning.nubs_trajectory import NUBSTrajectory6D
from planning.obstacle_forecast import (
    CompositeForecast,
    ConstantVelocitySphereForecast,
    FrozenSphereForecast,
)
from planning.optimizer import FixedTimeNUBSOptimizer, JointLimits
from planning.robot_surface_model import RobotSurfaceModel
from planning.spatiotemporal_risk import SpatioTemporalRiskEvaluator
from planning.verifier import DynamicTrajectoryVerifier


ROOT = Path(__file__).resolve().parents[1]


def _load(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def _states(config):
    cfg = config["trajectory"]
    head = NUBSTrajectory6D.make_boundary_state(
        cfg["q_start"], cfg["qd_start"], cfg["qdd_start"]
    )
    tail = NUBSTrajectory6D.make_boundary_state(
        cfg["q_goal"], cfg["qd_goal"], cfg["qdd_goal"]
    )
    durations = np.asarray(cfg["segment_durations"], dtype=np.float64)
    if len(durations) != cfg["segment_count"] or not np.isclose(
        np.sum(durations), cfg["total_duration"]
    ):
        raise ValueError("invalid segment durations")
    return head, tail, durations


def _limits(config):
    cfg = config["robot"]
    return JointLimits.from_arrays(
        cfg["q_min"], cfg["q_max"], cfg["qd_max"], cfg["qdd_max"]
    )


def _baseline(config, head, tail, durations, limits):
    cfg = config["optimizer"]
    optimizer = FixedTimeNUBSOptimizer(
        head,
        tail,
        durations,
        limits,
        lambda_smooth=cfg["lambda_smooth"],
        lambda_position=cfg["lambda_position"],
        lambda_velocity=cfg["lambda_velocity"],
        lambda_acceleration=cfg["lambda_acceleration"],
        samples_per_segment=cfg["samples_per_segment"],
        finite_difference_epsilon=cfg["finite_difference_epsilon"],
        max_iterations=cfg["max_iterations"],
        gradient_tolerance=cfg["gradient_tolerance"],
    )
    initial = NUBSTrajectory6D.linear_inner_points(head[:, 0], tail[:, 0], durations)
    return optimizer.optimize(initial)


def _select_sweep_point(model, trajectory, links, time_range, excluded_times):
    start = model.surface(trajectory.evaluate(0.0), "dense")
    goal = model.surface(trajectory.evaluate(trajectory.total_duration), "dense")
    start_tree, goal_tree = cKDTree(start), cKDTree(goal)
    best = None
    for fraction in np.linspace(time_range[0], time_range[1], 17):
        tau = float(fraction * trajectory.total_duration)
        if any(abs(tau - used) < 0.15 * trajectory.total_duration for used in excluded_times):
            continue
        q = trajectory.evaluate(tau)
        for link, points in model.surface_by_link(q, "medium", set(links)).items():
            d0, _ = start_tree.query(points)
            d1, _ = goal_tree.query(points)
            score = np.minimum(d0, d1)
            index = int(np.argmax(score))
            outward = points[index] - np.mean(points, axis=0)
            outward /= max(float(np.linalg.norm(outward)), 1.0e-12)
            item = {
                "surface_point": points[index].copy(),
                "outward": outward,
                "time": tau,
                "fraction": float(fraction),
                "link": link,
                "endpoint_clearance": float(score[index]),
            }
            if best is None or item["endpoint_clearance"] > best["endpoint_clearance"]:
                best = item
    if best is None:
        raise RuntimeError("failed to find dynamic sweep point")
    return best


def _travel_direction(outward: np.ndarray, reverse: bool) -> np.ndarray:
    axes = np.eye(3)
    axis = axes[int(np.argmin(np.abs(axes @ outward)))]
    direction = np.cross(outward, axis)
    direction /= max(float(np.linalg.norm(direction)), 1.0e-12)
    return -direction if reverse else direction


def make_forecast(config, scenario_name, model, baseline):
    scenario = config["experiment"]["scenarios"][scenario_name]
    forecast_cfg = config["forecast"]
    horizon = baseline.total_duration
    used_times = []
    forecasts = []
    details = []
    for index in range(int(scenario["obstacle_count"])):
        selected = _select_sweep_point(
            model,
            baseline,
            scenario["target_links"],
            scenario["time_range"],
            used_times,
        )
        used_times.append(selected["time"])
        direction = _travel_direction(selected["outward"], bool(index % 2))
        velocity = float(scenario["speed"]) * direction
        collision_center = (
            selected["surface_point"]
            + 0.35 * forecast_cfg["base_radius"] * selected["outward"]
        )
        center0 = collision_center - velocity * selected["time"]
        forecast = ConstantVelocitySphereForecast(
            center0,
            velocity,
            forecast_cfg["base_radius"],
            horizon,
            object_id=index + 1,
            margin=forecast_cfg["margin"],
            uncertainty=forecast_cfg["uncertainty"],
            uncertainty_growth=forecast_cfg["uncertainty_growth"],
            velocity_radius_scale=forecast_cfg["velocity_radius_scale"],
            beyond_horizon=forecast_cfg["beyond_horizon"],
        )
        predicted = forecast.occupancy_at(selected["time"]).spheres[0].center
        forecasts.append(forecast)
        details.append(
            {
                "object_id": index + 1,
                "initial_center": center0.tolist(),
                "velocity": velocity.tolist(),
                "collision_center": collision_center.tolist(),
                "collision_time": selected["time"],
                "target_link": selected["link"],
                "endpoint_clearance": selected["endpoint_clearance"],
                "time_alignment_error": float(np.linalg.norm(predicted - collision_center)),
            }
        )
    composite = forecasts[0] if len(forecasts) == 1 else CompositeForecast(forecasts)
    return composite, {"objects": details, "valid_horizon": composite.valid_horizon}


def _optimizer(config, head, tail, durations, limits, evaluator, forecast, links):
    opt, risk = config["optimizer"], config["risk"]
    return DynamicRiskNUBSOptimizer(
        head,
        tail,
        durations,
        limits,
        evaluator,
        forecast,
        lambda_risk=opt["lambda_risk"],
        risk_samples_per_segment=risk["risk_samples_per_segment"],
        risk_links=links,
        sensitivity_epsilon=opt["sensitivity_epsilon"],
        lambda_smooth=opt["lambda_smooth"],
        lambda_position=opt["lambda_position"],
        lambda_velocity=opt["lambda_velocity"],
        lambda_acceleration=opt["lambda_acceleration"],
        samples_per_segment=opt["samples_per_segment"],
        finite_difference_epsilon=opt["finite_difference_epsilon"],
        max_iterations=opt["max_iterations"],
        gradient_tolerance=opt["gradient_tolerance"],
    )


def _method_metrics(name, trajectory, solver_success, result, evaluator, truth, verifier, head, tail):
    times = np.linspace(0.0, trajectory.total_duration, 81)
    risk = evaluator.trajectory(trajectory, truth, times, with_gradient=False)
    samples = trajectory.sample(times)
    verification = verifier.verify(
        trajectory,
        truth,
        current_q=head[:, 0],
        current_qd=head[:, 1],
        current_qdd=head[:, 2],
        q_goal=tail[:, 0],
        solver_success=solver_success,
    )
    below_safe = float(np.trapezoid((risk.sample_distances < evaluator.d_safe).astype(float), times))
    below_stop = float(np.trapezoid((risk.sample_distances < verifier.d_stop).astype(float), times))
    path_length = float(np.sum(np.linalg.norm(np.diff(samples.q, axis=0), axis=1)))
    jerk = float(np.trapezoid(np.sum(samples.jerk**2, axis=1), times))
    payload = {
        "method": name,
        "solver_success": bool(solver_success),
        "true_spatiotemporal_risk": risk.cost,
        "min_distance_sampled": risk.min_distance,
        "nearest_link": risk.nearest_link,
        "nearest_object_id": risk.nearest_object_id,
        "time_below_d_safe": below_safe,
        "time_below_d_stop": below_stop,
        "path_length_joint": path_length,
        "jerk_integral": jerk,
        "verification": asdict(verification),
    }
    if result is not None:
        payload["optimization"] = {
            field.name: getattr(result, field.name)
            for field in fields(result)
            if field.name not in {"trajectory", "p_inner", "durations"}
        }
        payload["p_inner"] = result.p_inner.tolist()
    return payload


def _recorded_evidence(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"available": False, "path": str(path)}
    data = json.loads(path.read_text(encoding="utf-8"))
    metrics = data["metrics"]
    temporal = metrics["ours"]
    current = metrics["ours_wo_temporal"]
    return {
        "available": True,
        "path": str(path),
        "trials": int(data["trials"]),
        "temporal_T_lead": float(temporal["T_lead"]),
        "current_T_lead": float(current["T_lead"]),
        "lead_gain": float(temporal["T_lead"] - current["T_lead"]),
        "temporal_false_time": float(temporal["R_false_time"]),
        "current_false_time": float(current["R_false_time"]),
        "false_time_increase": float(
            temporal["R_false_time"] - current["R_false_time"]
        ),
        "position_rmse_available": False,
        "note": "Recorded results contain warning metrics but no stable 3-D center ground truth.",
    }


def _plot(name, trajectories, evaluator, truth, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for method, trajectory in trajectories.items():
        times = np.linspace(0.0, trajectory.total_duration, 161)
        risk = evaluator.trajectory(
            trajectory, truth, times, density="coarse", with_gradient=False
        )
        axes[0].plot(times, risk.sample_distances, label=method)
        axes[1].plot(times, risk.sample_costs, label=method)
    axes[0].axhline(evaluator.d_safe, color="orange", linestyle="--", label="d_safe")
    axes[0].set_ylabel("D_min / m")
    axes[0].set_title(f"Dynamic scenario {name}")
    axes[1].set_ylabel("R_body(q(t), t)")
    axes[1].set_xlabel("trajectory time / s")
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _write_table(metrics, path):
    lines = [
        "| scenario | method | solver | accepted | D_min / m | time<d_stop / s | risk | optimize / ms |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for scenario, data in metrics["scenarios"].items():
        for method, row in data["methods"].items():
            opt = row.get("optimization", {})
            lines.append(
                f"| {scenario} | {method} | {row['solver_success']} | "
                f"{row['verification']['accepted']} | {row['verification']['min_distance']:.6f} | "
                f"{row['time_below_d_stop']:.4f} | {row['true_spatiotemporal_risk']:.8g} | "
                f"{opt.get('elapsed_ms', 0.0):.3f} |"
            )
    lines.extend(["", f"Overall acceptance: **{'PASS' if metrics['accepted'] else 'FAIL'}**", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def run(config_path, output_override=None):
    config_path = Path(config_path).resolve()
    config = _load(config_path)
    output = Path(output_override or config["experiment"]["output_dir"])
    if not output.is_absolute():
        output = ROOT / output
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output / "config.yaml")
    robot, surface_cfg = config["robot"], config["surface"]
    model = RobotSurfaceModel(
        ROOT / robot["urdf_path"],
        robot["joint_names"],
        surface_cfg["density_totals"],
        seed=surface_cfg["random_seed"],
        min_points_per_link=surface_cfg["min_points_per_link"],
        cache_dir=surface_cfg["cache_dir"],
        geometry=surface_cfg["geometry"],
    )
    head, tail, durations = _states(config)
    limits = _limits(config)
    baseline_result = _baseline(config, head, tail, durations, limits)
    if not baseline_result.success:
        raise RuntimeError(f"baseline failed: {baseline_result.message}")
    baseline = baseline_result.trajectory
    risk_cfg = config["risk"]
    evaluator = SpatioTemporalRiskEvaluator(
        model,
        d_safe=risk_cfg["d_safe"],
        d_activate=risk_cfg["d_activate"],
        fd_epsilon_q=risk_cfg["fd_epsilon_q"],
        density=risk_cfg["optimizer_density"],
    )
    verifier = DynamicTrajectoryVerifier(
        evaluator,
        limits,
        d_stop=risk_cfg["d_stop"],
        time_step=config["validation"]["dense_time_step"],
        density=risk_cfg["validation_density"],
        epsilon_goal=config["validation"]["epsilon_goal"],
        epsilon_continuity_q=config["validation"]["epsilon_continuity_q"],
        epsilon_continuity_qd=config["validation"]["epsilon_continuity_qd"],
        epsilon_continuity_qdd=config["validation"]["epsilon_continuity_qdd"],
        limit_tolerance=config["validation"]["limit_tolerance"],
    )
    ee_links = set(config["experiment"]["end_effector_links"])
    metrics = {
        "accepted": False,
        "geometry": model.geometry,
        "surface_counts": model.sample_counts(),
        "scenarios": {},
        "recorded_evidence": _recorded_evidence(
            ROOT / config["experiment"]["recorded_metrics"]
        ),
    }
    gradient_check = None
    scenario_passes = []
    body_counterexample = False
    current_frame_failure = False
    for scenario_name in config["experiment"]["scenarios"]:
        truth, forecast_info = make_forecast(config, scenario_name, model, baseline)
        frozen = FrozenSphereForecast(truth)
        trajectories = {"baseline": baseline}
        rows = {
            "baseline": _method_metrics(
                "baseline", baseline, True, None, evaluator, truth, verifier, head, tail
            )
        }
        initial = baseline_result.p_inner
        method_specs = [
            ("current_full", frozen, None),
            ("temporal_ee", truth, ee_links),
            ("temporal_full", truth, None),
        ]
        for method, method_forecast, links in method_specs:
            optimizer = _optimizer(
                config, head, tail, durations, limits, evaluator, method_forecast, links
            )
            if gradient_check is None and method == "temporal_full":
                gradient_check = optimizer.check_gradient(
                    initial, config["optimizer"]["gradient_check_epsilon"]
                )
            result = optimizer.optimize(initial)
            trajectories[method] = result.trajectory
            rows[method] = _method_metrics(
                method, result.trajectory, result.success, result,
                evaluator, truth, verifier, head, tail
            )
        full = rows["temporal_full"]
        baseline_row = rows["baseline"]
        full_pass = bool(
            full["solver_success"]
            and full["verification"]["accepted"]
            and full["true_spatiotemporal_risk"] < baseline_row["true_spatiotemporal_risk"]
        )
        scenario_passes.append(full_pass)
        current_frame_failure |= not rows["current_full"]["verification"]["accepted"]
        if scenario_name == "B":
            body_counterexample = bool(
                baseline_row["nearest_link"] not in ee_links
                and not rows["temporal_ee"]["verification"]["accepted"]
                and full["verification"]["accepted"]
                and full["verification"]["min_distance"]
                > rows["temporal_ee"]["verification"]["min_distance"]
            )
        metrics["scenarios"][scenario_name] = {
            "accepted": full_pass,
            "forecast": forecast_info,
            "methods": rows,
        }
        _plot(scenario_name, trajectories, evaluator, truth, output / f"scenario_{scenario_name}.png")
    metrics["gradient_check"] = gradient_check
    metrics["body_counterexample"] = body_counterexample
    metrics["current_frame_failure_observed"] = current_frame_failure
    recorded_ok = bool(
        metrics["recorded_evidence"].get("available")
        and metrics["recorded_evidence"].get("lead_gain", 0.0) > 0.0
    )
    metrics["accepted"] = bool(
        all(scenario_passes)
        and body_counterexample
        and current_frame_failure
        and gradient_check is not None
        and gradient_check["relative_error"]
        <= config["validation"]["gradient_relative_tolerance"]
        and recorded_ok
    )
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    _write_table(metrics, output / "table_stage3.md")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config" / "ccro_stage3.yaml"))
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    metrics = run(args.config, args.output)
    print(json.dumps(metrics, indent=2, ensure_ascii=False, default=_json_default))
    if not metrics["accepted"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
