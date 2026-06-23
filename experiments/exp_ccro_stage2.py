"""CCRO-NUBS stage-two static full-body risk planning experiment."""

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

from planning.mesh_risk import MeshRiskEvaluator, StaticObstacleField
from planning.nubs_trajectory import NUBSTrajectory6D
from planning.optimizer import FixedTimeNUBSOptimizer, JointLimits
from planning.robot_surface_model import RobotSurfaceModel
from planning.static_optimizer import StaticRiskNUBSOptimizer
from planning.verifier import TrajectoryVerifier


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


def _states(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    trajectory = config["trajectory"]
    head = NUBSTrajectory6D.make_boundary_state(
        trajectory["q_start"], trajectory["qd_start"], trajectory["qdd_start"]
    )
    tail = NUBSTrajectory6D.make_boundary_state(
        trajectory["q_goal"], trajectory["qd_goal"], trajectory["qdd_goal"]
    )
    durations = np.asarray(trajectory["segment_durations"], dtype=np.float64)
    if len(durations) != int(trajectory["segment_count"]):
        raise ValueError("segment_durations does not match segment_count")
    if not np.isclose(np.sum(durations), trajectory["total_duration"]):
        raise ValueError("segment_durations does not sum to total_duration")
    return head, tail, durations


def _limits(config: dict[str, Any]) -> JointLimits:
    robot = config["robot"]
    return JointLimits.from_arrays(
        robot["q_min"], robot["q_max"], robot["qd_max"], robot["qdd_max"]
    )


def _baseline(
    config: dict[str, Any], head: np.ndarray, tail: np.ndarray, durations: np.ndarray
):
    limits = _limits(config)
    optimizer_cfg = config["optimizer"]
    optimizer = FixedTimeNUBSOptimizer(
        head,
        tail,
        durations,
        limits,
        lambda_smooth=optimizer_cfg["lambda_smooth"],
        lambda_position=optimizer_cfg["lambda_position"],
        lambda_velocity=optimizer_cfg["lambda_velocity"],
        lambda_acceleration=optimizer_cfg["lambda_acceleration"],
        samples_per_segment=optimizer_cfg["samples_per_segment"],
        finite_difference_epsilon=optimizer_cfg["finite_difference_epsilon"],
        max_iterations=optimizer_cfg["max_iterations"],
        gradient_tolerance=optimizer_cfg["gradient_tolerance"],
    )
    initial = NUBSTrajectory6D.linear_inner_points(head[:, 0], tail[:, 0], durations)
    return optimizer.optimize(initial)


def _select_sweep_center(
    surface_model: RobotSurfaceModel,
    trajectory: NUBSTrajectory6D,
    target_links: set[str],
    time_range: tuple[float, float],
    excluded_times: list[float],
) -> dict[str, Any]:
    start_surface = surface_model.surface(trajectory.evaluate(0.0), density="dense")
    goal_surface = surface_model.surface(
        trajectory.evaluate(trajectory.total_duration), density="dense"
    )
    start_tree = cKDTree(start_surface)
    goal_tree = cKDTree(goal_surface)
    best: dict[str, Any] | None = None
    fractions = np.linspace(time_range[0], time_range[1], 17)
    for fraction in fractions:
        time_value = float(fraction * trajectory.total_duration)
        if any(abs(time_value - used) < 0.15 * trajectory.total_duration for used in excluded_times):
            continue
        q = trajectory.evaluate(time_value)
        by_link = surface_model.surface_by_link(
            q, density="medium", links=target_links
        )
        for link, points in by_link.items():
            d_start, _ = start_tree.query(points, k=1)
            d_goal, _ = goal_tree.query(points, k=1)
            scores = np.minimum(d_start, d_goal)
            index = int(np.argmax(scores))
            point = points[index]
            link_center = np.mean(points, axis=0)
            outward = point - link_center
            norm = float(np.linalg.norm(outward))
            if norm < 1.0e-9:
                outward = point - np.mean(start_surface, axis=0)
                norm = float(np.linalg.norm(outward))
            direction = outward / max(norm, 1.0e-9)
            candidate = {
                "center_on_surface": point.copy(),
                "outward": direction,
                "time": time_value,
                "fraction": float(fraction),
                "link": link,
                "endpoint_clearance": float(scores[index]),
            }
            if best is None or candidate["endpoint_clearance"] > best["endpoint_clearance"]:
                best = candidate
    if best is None:
        raise RuntimeError("failed to construct a sweep obstacle candidate")
    return best


def _filled_ball(
    center: np.ndarray, radius: float, point_count: int, rng: np.random.Generator
) -> np.ndarray:
    directions = rng.normal(size=(point_count - 1, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    radii = radius * np.cbrt(rng.random(point_count - 1))
    points = center[None, :] + directions * radii[:, None]
    return np.vstack((center[None, :], points))


def make_scenario_obstacle(
    config: dict[str, Any],
    scenario_name: str,
    surface_model: RobotSurfaceModel,
    baseline: NUBSTrajectory6D,
    rng: np.random.Generator,
) -> tuple[StaticObstacleField, dict[str, Any]]:
    risk_cfg = config["risk"]
    scenario = config["experiment"]["scenarios"][scenario_name]
    target_links = set(scenario["target_links"])
    missing = target_links - set(surface_model.link_names)
    if missing:
        raise ValueError(f"scenario {scenario_name} references unknown links: {sorted(missing)}")
    used_times: list[float] = []
    chunks: list[np.ndarray] = []
    details: list[dict[str, Any]] = []
    radius = float(risk_cfg["obstacle_radius"])
    for _ in range(int(scenario["obstacle_count"])):
        selected = _select_sweep_center(
            surface_model,
            baseline,
            target_links,
            tuple(scenario["time_range"]),
            used_times,
        )
        used_times.append(selected["time"])
        center = selected["center_on_surface"] + 0.45 * radius * selected["outward"]
        chunks.append(
            _filled_ball(center, radius, int(risk_cfg["obstacle_points"]), rng)
        )
        details.append(
            {
                "center": center.tolist(),
                "surface_point": selected["center_on_surface"].tolist(),
                "time": selected["time"],
                "fraction": selected["fraction"],
                "link": selected["link"],
                "endpoint_clearance": selected["endpoint_clearance"],
                "radius": radius,
            }
        )
    points = np.vstack(chunks)
    return StaticObstacleField.from_points(points), {"obstacles": details, "point_count": len(points)}


def _risk_optimizer(
    config: dict[str, Any],
    head: np.ndarray,
    tail: np.ndarray,
    durations: np.ndarray,
    limits: JointLimits,
    evaluator: MeshRiskEvaluator,
    obstacle: StaticObstacleField,
    links: set[str] | None,
) -> StaticRiskNUBSOptimizer:
    opt = config["optimizer"]
    risk = config["risk"]
    return StaticRiskNUBSOptimizer(
        head,
        tail,
        durations,
        limits,
        evaluator,
        obstacle,
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


def _verification_dict(result) -> dict[str, Any]:
    payload = asdict(result)
    return payload


def _method_metrics(
    method: str,
    trajectory: NUBSTrajectory6D,
    solver_success: bool,
    optimizer_result,
    evaluator: MeshRiskEvaluator,
    obstacle: StaticObstacleField,
    verifier: TrajectoryVerifier,
    head: np.ndarray,
    tail: np.ndarray,
    sample_times: np.ndarray,
    links: set[str] | None,
) -> dict[str, Any]:
    risk = evaluator.trajectory(
        trajectory, obstacle, sample_times, links=links, with_gradient=False
    )
    full_risk = evaluator.trajectory(
        trajectory, obstacle, sample_times, links=None, with_gradient=False
    )
    verification = verifier.verify(
        trajectory,
        obstacle,
        current_q=head[:, 0],
        current_qd=head[:, 1],
        current_qdd=head[:, 2],
        q_goal=tail[:, 0],
        solver_success=solver_success,
    )
    metrics: dict[str, Any] = {
        "method": method,
        "solver_success": bool(solver_success),
        "optimized_links": None if links is None else sorted(links),
        "risk_cost_for_method": risk.cost,
        "full_body_risk_cost": full_risk.cost,
        "optimization_sample_min_distance": full_risk.min_distance,
        "nearest_link": full_risk.nearest_link,
        "verification": _verification_dict(verification),
    }
    if optimizer_result is not None:
        metrics["optimization"] = {
            field.name: getattr(optimizer_result, field.name)
            for field in fields(optimizer_result)
            if field.name not in {"trajectory", "p_inner", "durations"}
        }
        metrics["p_inner"] = optimizer_result.p_inner.tolist()
    return metrics


def _plot_scenario(
    scenario: str,
    methods: dict[str, NUBSTrajectory6D],
    evaluator: MeshRiskEvaluator,
    obstacle: StaticObstacleField,
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for name, trajectory in methods.items():
        times = np.linspace(0.0, trajectory.total_duration, 161)
        risk = evaluator.trajectory(
            trajectory, obstacle, times, with_gradient=False, density="coarse"
        )
        axes[0].plot(times, risk.sample_distances, label=name)
        axes[1].plot(times, risk.sample_costs, label=name)
    axes[0].axhline(evaluator.d_safe, color="orange", linestyle="--", label="d_safe")
    axes[0].set_ylabel("D_min / m")
    axes[0].set_title(f"Static scenario {scenario}")
    axes[1].set_ylabel("R_body")
    axes[1].set_xlabel("time / s")
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _write_summary(metrics: dict[str, Any], output: Path) -> None:
    lines = [
        "| scenario | method | solver | accepted | D_min dense / m | full risk | time / ms |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for scenario, payload in metrics["scenarios"].items():
        for method, row in payload["methods"].items():
            optimization = row.get("optimization", {})
            lines.append(
                f"| {scenario} | {method} | {row['solver_success']} | "
                f"{row['verification']['accepted']} | "
                f"{row['verification']['min_distance']:.6f} | "
                f"{row['full_body_risk_cost']:.8g} | "
                f"{optimization.get('elapsed_ms', 0.0):.3f} |"
            )
    lines.extend(["", f"Overall acceptance: **{'PASS' if metrics['accepted'] else 'FAIL'}**", ""])
    output.write_text("\n".join(lines), encoding="utf-8")


def run(config_path: str | Path, output_override: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    config = _load(config_path)
    output_dir = Path(output_override or config["experiment"]["output_dir"])
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output_dir / "config.yaml")
    rng = np.random.default_rng(int(config["experiment"]["random_seed"]))

    robot = config["robot"]
    surface_cfg = config["surface"]
    surface_model = RobotSurfaceModel(
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
    baseline_result = _baseline(config, head, tail, durations)
    if not baseline_result.success:
        raise RuntimeError(f"stage-one baseline failed: {baseline_result.message}")
    baseline = baseline_result.trajectory
    risk_cfg = config["risk"]
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
    all_metrics: dict[str, Any] = {
        "accepted": False,
        "surface_counts": surface_model.sample_counts(),
        "geometry": surface_model.geometry,
        "scenarios": {},
    }
    gradient_check: dict[str, float] | None = None
    full_acceptance: list[bool] = []
    body_counterexample = False
    for scenario_name in config["experiment"]["scenarios"]:
        obstacle, obstacle_info = make_scenario_obstacle(
            config, scenario_name, surface_model, baseline, rng
        )
        sample_times = np.linspace(0.0, baseline.total_duration, 41)
        methods: dict[str, NUBSTrajectory6D] = {"baseline": baseline}
        method_rows: dict[str, Any] = {
            "baseline": _method_metrics(
                "baseline", baseline, True, None, evaluator, obstacle, verifier,
                head, tail, sample_times, None
            )
        }
        initial_points = baseline_result.p_inner
        if "ee_only" in config["experiment"]["methods"]:
            ee_optimizer = _risk_optimizer(
                config, head, tail, durations, limits, evaluator, obstacle, ee_links
            )
            ee_result = ee_optimizer.optimize(initial_points)
            methods["ee_only"] = ee_result.trajectory
            method_rows["ee_only"] = _method_metrics(
                "ee_only", ee_result.trajectory, ee_result.success, ee_result,
                evaluator, obstacle, verifier, head, tail, sample_times, ee_links
            )
        full_optimizer = _risk_optimizer(
            config, head, tail, durations, limits, evaluator, obstacle, None
        )
        if gradient_check is None:
            gradient_check = full_optimizer.check_gradient(
                initial_points, config["optimizer"]["gradient_check_epsilon"]
            )
        full_result = full_optimizer.optimize(initial_points)
        methods["full_body"] = full_result.trajectory
        method_rows["full_body"] = _method_metrics(
            "full_body", full_result.trajectory, full_result.success, full_result,
            evaluator, obstacle, verifier, head, tail, sample_times, None
        )
        full_row = method_rows["full_body"]
        baseline_row = method_rows["baseline"]
        risk_decreased = full_row["full_body_risk_cost"] < baseline_row["full_body_risk_cost"]
        scenario_accepted = bool(
            full_result.success
            and full_row["verification"]["accepted"]
            and risk_decreased
        )
        full_acceptance.append(scenario_accepted)
        if scenario_name == "B" and "ee_only" in method_rows:
            baseline_link = baseline_row["nearest_link"]
            body_counterexample = bool(
                baseline_link not in ee_links
                and baseline_row["verification"]["min_distance"] < risk_cfg["d_stop"]
                and full_row["verification"]["min_distance"]
                > method_rows["ee_only"]["verification"]["min_distance"]
            )
        all_metrics["scenarios"][scenario_name] = {
            "accepted": scenario_accepted,
            "obstacle": obstacle_info,
            "methods": method_rows,
        }
        np.savez_compressed(output_dir / f"scenario_{scenario_name}_obstacle.npz", points=obstacle.points)
        _plot_scenario(
            scenario_name,
            methods,
            evaluator,
            obstacle,
            output_dir / f"scenario_{scenario_name}.png",
        )

    all_metrics["gradient_check"] = gradient_check
    all_metrics["body_counterexample"] = body_counterexample
    all_metrics["accepted"] = bool(
        all(full_acceptance)
        and body_counterexample
        and gradient_check is not None
        and gradient_check["relative_error"]
        <= config["validation"]["gradient_relative_tolerance"]
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(all_metrics, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    _write_summary(all_metrics, output_dir / "table_stage2.md")
    return all_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config" / "ccro_stage2.yaml"))
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    metrics = run(args.config, args.output)
    print(json.dumps(metrics, indent=2, ensure_ascii=False, default=_json_default))
    if not metrics["accepted"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
