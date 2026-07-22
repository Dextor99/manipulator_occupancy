"""Revised Chapter 6.3 static near-field trajectory planning benchmark."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields
import importlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SCENARIO_LABELS = {"A": "P1", "B": "P2", "C": "P3"}

MAIN_METHODS = [
    "rrt_connect_smooth",
    "minco_risk",
    "nubs_without_risk",
    "critical_point_nubs",
    "ccro_nubs",
]

METHOD_DISPLAY = {
    "rrt_connect_smooth": "RRT-Connect + smoothing",
    "minco_risk": "MINCO-risk (adapted)",
    "nubs_without_risk": "NUBS w/o risk (ablation)",
    "critical_point_nubs": "Critical-point-NUBS",
    "ccro_nubs": "CCRO-NUBS",
}

TIME_LIMIT_MS = 10_000.0


@dataclass(frozen=True)
class ObstacleSpec:
    obstacle_id: str
    center: list[float]
    radius: float
    point_count: int


@dataclass(frozen=True)
class CriticalPointGeometry:
    points: np.ndarray
    radii: np.ndarray


def make_instance_ids(scenarios: list[str], instances_per_scenario: int) -> list[str]:
    return [
        f"{scenario}_{index:02d}"
        for scenario in scenarios
        for index in range(instances_per_scenario)
    ]


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    try:
        from experiments.exp_ccro_stage2 import _json_default

        return _json_default(value)
    except TypeError:
        return str(value)


def git_commit_hash() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _sphere_points(
    rng: np.random.Generator,
    center: np.ndarray,
    radius: float,
    count: int,
    *,
    surface_noise: float = 0.0,
    dropout: float = 0.0,
) -> np.ndarray:
    directions = rng.normal(size=(count, 3))
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1.0e-12)
    points = center[None, :] + float(radius) * directions
    if surface_noise > 0.0:
        points = points + rng.normal(0.0, surface_noise, size=points.shape)
    if dropout > 0.0:
        keep = rng.random(len(points)) >= dropout
        if not np.any(keep):
            keep[int(rng.integers(0, len(points)))] = True
        points = points[keep]
    return np.ascontiguousarray(points, dtype=np.float64)


def perturb_obstacle_points(points: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    source = np.asarray(points, dtype=np.float64)
    center_shift = rng.uniform(-0.03, 0.03, size=3)
    scale = float(rng.uniform(0.90, 1.10))
    center = source.mean(axis=0, keepdims=True)
    perturbed = center + scale * (source - center) + center_shift[None, :]
    perturbed = perturbed + rng.normal(0.0, 0.005, size=perturbed.shape)
    keep = rng.random(len(perturbed)) >= 0.05
    if not np.any(keep):
        keep[int(rng.integers(0, len(perturbed)))] = True
    return np.ascontiguousarray(perturbed[keep], dtype=np.float64)


def perturb_obstacle_specs(specs: list[ObstacleSpec], seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    observed_chunks: list[np.ndarray] = []
    gt_chunks: list[np.ndarray] = []
    gt_slices: list[dict[str, int]] = []
    obstacles: list[dict[str, Any]] = []
    start = 0
    for spec in specs:
        center = np.asarray(spec.center, dtype=np.float64) + rng.uniform(-0.03, 0.03, size=3)
        radius = float(spec.radius) * float(rng.uniform(0.90, 1.10))
        gt = _sphere_points(rng, center, radius, int(spec.point_count), surface_noise=0.0, dropout=0.0)
        observed = _sphere_points(
            rng,
            center,
            radius,
            int(spec.point_count),
            surface_noise=0.005,
            dropout=0.05,
        )
        gt_chunks.append(gt)
        observed_chunks.append(observed)
        gt_slices.append({"start": start, "stop": start + len(gt)})
        start += len(gt)
        obstacles.append(
            {
                "obstacle_id": spec.obstacle_id,
                "center": center.tolist(),
                "radius": radius,
                "gt_point_count": len(gt),
                "observed_point_count": len(observed),
            }
        )
    return {
        "obstacles": obstacles,
        "gt_slices": gt_slices,
        "gt_dense_points": np.vstack(gt_chunks),
        "observed_points": np.vstack(observed_chunks),
    }


def critical_point_distance(
    critical: CriticalPointGeometry,
    *,
    obstacle_center: np.ndarray,
    obstacle_radius: float,
) -> float:
    distances = np.linalg.norm(
        np.asarray(critical.points, dtype=np.float64) - np.asarray(obstacle_center, dtype=np.float64)[None, :],
        axis=1,
    )
    return float(np.min(distances - np.asarray(critical.radii, dtype=np.float64) - float(obstacle_radius)))


class CriticalPointRiskEvaluator:
    """6.2-compatible sparse critical-point geometry risk evaluator.

    Chapter 6.2 defines critical points by body region and assigns a region
    dependent equivalent radius.  This evaluator reuses that exact definition
    for the Chapter 6.3 Critical-point-NUBS baseline, while still accepting the
    same static observed point cloud interface as the full CCRO evaluator.
    """

    def __init__(
        self,
        surface_model: RobotSurfaceModel,
        *,
        d_safe: float,
        d_activate: float,
        fd_epsilon_q: float,
        density: str = "coarse",
    ) -> None:
        self.surface_model = surface_model
        self.d_safe = float(d_safe)
        self.d_activate = float(d_activate)
        self.fd_epsilon_q = float(fd_epsilon_q)
        self.density = density
        body_coverage = importlib.import_module("experiments.new.6_2.body_coverage_62")
        self._build_critical_points = body_coverage.build_critical_points

    def _critical_points_by_link(self, q: np.ndarray, links: set[str] | None = None) -> dict[str, list[Any]]:
        selected: dict[str, list[Any]] = {}
        allowed = None if links is None else set(links)
        for point in self._build_critical_points(self.surface_model, q):
            if allowed is not None and point.link not in allowed:
                continue
            selected.setdefault(point.link, []).append(point)
        return selected

    def _evaluate_no_gradient(
        self,
        q: np.ndarray,
        obstacle: StaticObstacleField,
        links: set[str] | None,
        density: str | None,
    ) -> Any:
        from planning.mesh_risk import ConfigurationRisk

        if obstacle.tree is None:
            return ConfigurationRisk(0.0, math.inf, None, None, None, {})
        selected = self._critical_points_by_link(q, links=links)
        per_link: dict[str, float] = {}
        weighted_cost = 0.0
        min_distance = math.inf
        nearest_link = None
        nearest_robot = None
        nearest_obstacle = None
        link_count = 0
        for link, critical_points in selected.items():
            if len(critical_points) == 0:
                continue
            points = np.vstack([item.position for item in critical_points])
            radii = np.asarray([item.radius for item in critical_points], dtype=np.float64)
            distances, indices = obstacle.tree.query(points, k=1)
            signed = distances - radii
            hinge = np.maximum(self.d_safe - signed, 0.0)
            cost = float(np.mean(hinge * hinge))
            per_link[link] = cost
            weighted_cost += cost
            link_count += 1
            local = int(np.argmin(signed))
            if float(signed[local]) < min_distance:
                min_distance = float(signed[local])
                nearest_link = link
                nearest_robot = points[local].copy()
                nearest_obstacle = obstacle.points[int(indices[local])].copy()
        total = 0.0 if link_count == 0 else weighted_cost / link_count
        return ConfigurationRisk(total, min_distance, nearest_link, nearest_robot, nearest_obstacle, per_link)

    def configuration(
        self,
        q: np.ndarray,
        obstacle: StaticObstacleField,
        *,
        links: set[str] | None = None,
        density: str | None = None,
        with_gradient: bool = False,
    ) -> Any:
        values = np.asarray(q, dtype=np.float64)
        result = self._evaluate_no_gradient(values, obstacle, links, density)
        if not with_gradient:
            return result
        gradient = np.zeros(6, dtype=np.float64)
        if result.cost > 0.0 and result.min_distance < self.d_activate:
            for joint in range(6):
                plus = values.copy()
                minus = values.copy()
                plus[joint] += self.fd_epsilon_q
                minus[joint] -= self.fd_epsilon_q
                c_plus = self._evaluate_no_gradient(plus, obstacle, links, density).cost
                c_minus = self._evaluate_no_gradient(minus, obstacle, links, density).cost
                gradient[joint] = (c_plus - c_minus) / (2.0 * self.fd_epsilon_q)
        result.gradient_q = gradient
        return result

    def trajectory(
        self,
        trajectory: NUBSTrajectory6D,
        obstacle: StaticObstacleField,
        sample_times: np.ndarray,
        *,
        links: set[str] | None = None,
        density: str | None = None,
        with_gradient: bool = False,
    ) -> Any:
        from planning.mesh_risk import TrajectoryRisk, trapezoid_weights

        times = np.asarray(sample_times, dtype=np.float64)
        weights = trapezoid_weights(times)
        samples = trajectory.sample(times, max_derivative=0).q
        costs = np.zeros(len(times), dtype=np.float64)
        distances = np.full(len(times), math.inf, dtype=np.float64)
        gradients = np.zeros((len(times), 6), dtype=np.float64) if with_gradient else None
        per_link: dict[str, float] = {}
        nearest_link = None
        min_distance = math.inf
        active = 0
        for index, q in enumerate(samples):
            risk = self.configuration(q, obstacle, links=links, density=density, with_gradient=with_gradient)
            costs[index] = risk.cost
            distances[index] = risk.min_distance
            active += int(risk.cost > 0.0)
            if risk.min_distance < min_distance:
                min_distance = risk.min_distance
                nearest_link = risk.nearest_link
            if gradients is not None and risk.gradient_q is not None:
                gradients[index] = weights[index] * risk.gradient_q
            for link, value in risk.per_link_cost.items():
                per_link[link] = per_link.get(link, 0.0) + weights[index] * value
        return TrajectoryRisk(
            cost=float(np.dot(weights, costs)),
            min_distance=float(min_distance),
            nearest_link=nearest_link,
            active_sample_count=active,
            per_link_cost=per_link,
            sample_times=times.copy(),
            sample_costs=costs,
            sample_distances=distances,
            gradient_q=gradients,
        )


def compute_jerk_integral(trajectory: Any, *, time_step: float = 0.025) -> float:
    count = max(2, int(np.ceil(float(trajectory.total_duration) / time_step)) + 1)
    times = np.linspace(0.0, float(trajectory.total_duration), count)
    samples = trajectory.sample(times)
    return float(np.trapz(np.sum(samples.jerk * samples.jerk, axis=1), times))


def trajectory_plot_payload(trajectory: Any, *, time_step: float = 0.05) -> dict[str, Any]:
    times = np.linspace(0.0, float(trajectory.total_duration), max(2, int(np.ceil(float(trajectory.total_duration) / time_step)) + 1))
    samples = trajectory.sample(times)
    return {
        "times": times.tolist(),
        "q": samples.q.tolist(),
        "qd": samples.qd.tolist(),
        "qdd": samples.qdd.tolist(),
        "jerk": samples.jerk.tolist(),
    }


def mean_std(values: list[float]) -> dict[str, float | None]:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not finite:
        return {"mean": None, "std": None}
    arr = np.asarray(finite, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
    }


def aggregate_method_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    feasible = [row for row in rows if row.get("dense_feasible", row["verification"].get("accepted"))]
    budgeted = [row for row in rows if row.get("budgeted_accepted", row["verification"].get("accepted"))]
    return {
        "dense_feasible_count": len(feasible),
        "budgeted_accepted_count": len(budgeted),
        "total_count": len(rows),
        "dense_feasible_rate": len(feasible) / len(rows) if rows else 0.0,
        "budgeted_accepted_rate": len(budgeted) / len(rows) if rows else 0.0,
        "D_min": mean_std([row["verification"].get("min_distance") for row in feasible]),
        "J_smooth": mean_std([row.get("post", {}).get("J_smooth") for row in feasible]),
        "T_plan_ms": mean_std([row.get("optimization", {}).get("elapsed_ms_raw", row.get("optimization", {}).get("elapsed_ms")) for row in rows]),
        "timeout_count": sum(bool(row.get("timeout", False)) for row in rows),
    }


def validation_accept_distance(config: dict[str, Any]) -> float:
    return float(config.get("validation", {}).get("d_accept", 0.08))


def aggregate_rrt_retry_trials(trials: list[dict[str, Any]], *, max_attempts: int) -> dict[str, Any]:
    elapsed = 0.0
    attempted: list[dict[str, Any]] = []
    selected = None
    for trial in trials[:max_attempts]:
        attempted.append(trial)
        elapsed += float(trial.get("optimization", {}).get("elapsed_ms", 0.0))
        if trial.get("verification", {}).get("accepted"):
            selected = trial
            break
    if selected is None:
        selected = attempted[-1] if attempted else {
            "method": "rrt_connect_smooth",
            "solver_success": False,
            "verification": {"accepted": False, "min_distance": 0.0, "reasons": ["no_rrt_trials"]},
            "optimization": {"elapsed_ms": 0.0},
        }
    row = dict(selected)
    row["optimization"] = dict(row.get("optimization", {}))
    row["optimization"]["elapsed_ms"] = elapsed
    row["attempt_count"] = len(attempted)
    row["rrt_trials"] = attempted
    return row


def _failed_rrt_row(elapsed_ms: float, attempt_count: int) -> dict[str, Any]:
    return _annotate_budget({
        "method": "rrt_connect_smooth",
        "solver_success": False,
        "optimized_links": None,
        "planning_risk_cost": math.inf,
        "planning_min_distance": 0.0,
        "nearest_link": None,
        "verification": {
            "accepted": False,
            "reasons": ["planner_failed"],
            "min_distance": 0.0,
            "goal_error": math.inf,
        },
        "optimization": {
            "success": False,
            "elapsed_ms": float(elapsed_ms),
            "attempt_count": int(attempt_count),
        },
        "post": {"J_smooth": math.inf, "total_duration": None},
    })


def _verification_dict(result: Any) -> dict[str, Any]:
    return asdict(result)


def _optimization_dict(result: Any) -> dict[str, Any]:
    if result is None:
        return {"elapsed_ms": 0.0}
    if isinstance(result, dict):
        return {key: value for key, value in result.items() if key not in {"trajectory", "p_inner"}}
    return {
        field.name: getattr(result, field.name)
        for field in fields(result)
        if field.name not in {"trajectory", "p_inner", "durations"}
    }


def _annotate_budget(row: dict[str, Any], *, timeout_ms: float = TIME_LIMIT_MS) -> dict[str, Any]:
    optimization = dict(row.get("optimization", {}))
    elapsed = float(optimization.get("elapsed_ms_raw", optimization.get("elapsed_ms", 0.0)))
    optimization["elapsed_ms_raw"] = elapsed
    optimization["within_time_budget"] = bool(elapsed <= timeout_ms)
    row["optimization"] = optimization
    row["dense_feasible"] = bool(row.get("verification", {}).get("accepted"))
    row["within_time_budget"] = bool(optimization["within_time_budget"])
    row["budgeted_accepted"] = bool(row["dense_feasible"] and row["within_time_budget"])
    row["timeout"] = bool(not row["within_time_budget"])
    return row


def _row_for_trajectory(
    *,
    method: str,
    trajectory: Any,
    solver_success: bool,
    optimizer_result: Any,
    planning_evaluator: Any,
    planning_obstacle: StaticObstacleField,
    verifier: TrajectoryVerifier,
    validation_obstacle: StaticObstacleField,
    head: np.ndarray,
    tail: np.ndarray,
    sample_times: np.ndarray,
    links: set[str] | None = None,
    timeout_ms: float = TIME_LIMIT_MS,
) -> dict[str, Any]:
    risk = planning_evaluator.trajectory(
        trajectory, planning_obstacle, sample_times, links=links, with_gradient=False
    )
    verification = verifier.verify(
        trajectory,
        validation_obstacle,
        current_q=head[:, 0],
        current_qd=head[:, 1],
        current_qdd=head[:, 2],
        q_goal=tail[:, 0],
        solver_success=solver_success,
    )
    optimization = _optimization_dict(optimizer_result)
    elapsed = float(optimization.get("elapsed_ms", 0.0))
    verification_payload = _verification_dict(verification)
    row = {
        "method": method,
        "solver_success": bool(solver_success),
        "optimized_links": None if links is None else sorted(links),
        "planning_risk_cost": risk.cost,
        "planning_min_distance": risk.min_distance,
        "nearest_link": risk.nearest_link,
        "verification": verification_payload,
        "optimization": optimization,
        "post": {
            "J_smooth": compute_jerk_integral(trajectory, time_step=verifier.time_step),
            "total_duration": float(trajectory.total_duration),
        },
        "plot_samples": trajectory_plot_payload(trajectory),
    }
    p_inner = getattr(optimizer_result, "p_inner", None)
    if p_inner is not None:
        row["p_inner"] = np.asarray(p_inner, dtype=np.float64).tolist()
    return _annotate_budget(row, timeout_ms=timeout_ms)


def save_instance(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = dict(payload)
    serializable["observed_points"] = np.asarray(payload["observed_points"], dtype=np.float64).tolist()
    serializable["gt_dense_points"] = np.asarray(payload["gt_dense_points"], dtype=np.float64).tolist()
    path.write_text(json.dumps(serializable, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_instance(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["observed_points"] = np.asarray(payload["observed_points"], dtype=np.float64)
    payload["gt_dense_points"] = np.asarray(payload["gt_dense_points"], dtype=np.float64)
    return payload


def load_trial(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_context(config: dict[str, Any]) -> dict[str, Any]:
    from experiments.exp_ccro_stage2 import _baseline, _limits, _states
    from planning.mesh_risk import MeshRiskEvaluator
    from planning.robot_surface_model import RobotSurfaceModel
    from planning.verifier import TrajectoryVerifier

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
    risk_cfg = config["risk"]
    evaluator = MeshRiskEvaluator(
        surface_model,
        d_safe=risk_cfg["d_safe"],
        d_activate=risk_cfg["d_activate"],
        fd_epsilon_q=risk_cfg["fd_epsilon_q"],
        density=risk_cfg["optimizer_density"],
    )
    validation_evaluator = MeshRiskEvaluator(
        surface_model,
        d_safe=risk_cfg["d_safe"],
        d_activate=risk_cfg["d_activate"],
        fd_epsilon_q=risk_cfg["fd_epsilon_q"],
        density=risk_cfg["validation_density"],
    )
    verifier = TrajectoryVerifier(
        validation_evaluator,
        limits,
        d_stop=validation_accept_distance(config),
        time_step=config["validation"]["dense_time_step"],
        density=risk_cfg["validation_density"],
        epsilon_goal=config["validation"]["epsilon_goal"],
        epsilon_continuity_q=config["validation"]["epsilon_continuity_q"],
        epsilon_continuity_qd=config["validation"]["epsilon_continuity_qd"],
        epsilon_continuity_qdd=config["validation"]["epsilon_continuity_qdd"],
        limit_tolerance=config["validation"]["limit_tolerance"],
    )
    critical_evaluator = CriticalPointRiskEvaluator(
        surface_model,
        d_safe=risk_cfg["d_safe"],
        d_activate=risk_cfg["d_activate"],
        fd_epsilon_q=risk_cfg["fd_epsilon_q"],
        density="coarse",
    )
    return {
        "config": config,
        "surface_model": surface_model,
        "head": head,
        "tail": tail,
        "durations": durations,
        "limits": limits,
        "baseline_result": baseline_result,
        "baseline_trajectory": baseline_result.trajectory,
        "evaluator": evaluator,
        "validation_evaluator": validation_evaluator,
        "critical_evaluator": critical_evaluator,
        "verifier": verifier,
        "sample_times": np.linspace(0.0, float(np.sum(durations)), 41),
    }


def make_frozen_instance(context: dict[str, Any], scenario: str, index: int) -> dict[str, Any]:
    from experiments.exp_ccro_stage2 import make_scenario_obstacle

    config = context["config"]
    rng = np.random.default_rng(int(config["experiment"]["random_seed"]) + 63000 + 101 * index + ord(scenario[0]))
    obstacle, info = make_scenario_obstacle(
        config, scenario, context["surface_model"], context["baseline_trajectory"], rng
    )
    specs = [
        ObstacleSpec(
            obstacle_id=f"{scenario}_{i}",
            center=item["center"],
            radius=float(item["radius"]),
            point_count=int(config["risk"]["obstacle_points"]),
        )
        for i, item in enumerate(info["obstacles"])
    ]
    perturbed = perturb_obstacle_specs(specs, seed=int(config["experiment"]["random_seed"]) + 64000 + 113 * index + ord(scenario[0]))
    return {
        "id": f"{scenario}_{index:02d}",
        "scenario": scenario,
        "scenario_label": SCENARIO_LABELS[scenario],
        "index": index,
        "target_links": config["experiment"]["scenarios"][scenario]["target_links"],
        "source_nominal_point_count": len(obstacle.points),
        **perturbed,
    }


def run_rrt_for_instance(context: dict[str, Any], instance: dict[str, Any]) -> dict[str, Any]:
    from experiments.exp_64_external_baselines import (
        _rrt_connect,
        _rrt_to_trajectory,
        _shortcut_path,
    )
    from planning.mesh_risk import StaticObstacleField

    head = context["head"]
    tail = context["tail"]
    durations = context["durations"]
    limits = context["limits"]
    evaluator = context["evaluator"]
    verifier = context["verifier"]
    config = context["config"]
    sample_times = context["sample_times"]
    planning_obstacle = StaticObstacleField.from_points(instance["observed_points"])
    validation_obstacle = StaticObstacleField.from_points(instance["gt_dense_points"])
    rrt_clearance = validation_accept_distance(config)

    rrt_elapsed = 0.0
    rrt_attempts = 0
    selected_row = None
    for seed_index in range(3):
        rrt_attempts += 1
        rng = np.random.default_rng(
            int(config["experiment"]["random_seed"])
            + 6300
            + 97 * seed_index
            + int(instance["index"])
            + 13 * ord(instance["scenario"][0])
        )
        started = time.perf_counter()
        raw_path, _ = _rrt_connect(
            head[:, 0],
            tail[:, 0],
            limits,
            evaluator,
            planning_obstacle,
            rrt_clearance,
            rng,
        )
        trajectory = None
        if raw_path is not None:
            smooth_path = _shortcut_path(raw_path, evaluator, planning_obstacle, rrt_clearance, rng)
            trajectory = _rrt_to_trajectory(smooth_path, head, tail, float(np.sum(durations)))
        rrt_elapsed += (time.perf_counter() - started) * 1000.0
        if trajectory is None:
            continue
        row = _row_for_trajectory(
            method="rrt_connect_smooth",
            trajectory=trajectory,
            solver_success=True,
            optimizer_result={"elapsed_ms": rrt_elapsed, "success": True, "attempt_count": rrt_attempts},
            planning_evaluator=evaluator,
            planning_obstacle=planning_obstacle,
            verifier=verifier,
            validation_obstacle=validation_obstacle,
            head=head,
            tail=tail,
            sample_times=sample_times,
        )
        row["attempt_count"] = rrt_attempts
        row["planning_clearance_m"] = rrt_clearance
        selected_row = row
        if row["verification"].get("accepted"):
            break
    if selected_row is None:
        selected_row = _failed_rrt_row(rrt_elapsed, rrt_attempts)
        selected_row["planning_clearance_m"] = rrt_clearance
    return selected_row


def run_methods_for_instance(context: dict[str, Any], instance: dict[str, Any]) -> dict[str, Any]:
    from experiments.exp_ccro_stage2 import _risk_optimizer
    from experiments.exp_64_external_baselines import (
        FixedTimeMINCOOptimizer,
        _minco_metrics,
    )
    from planning.mesh_risk import StaticObstacleField

    head = context["head"]
    tail = context["tail"]
    durations = context["durations"]
    limits = context["limits"]
    evaluator = context["evaluator"]
    critical_evaluator = context["critical_evaluator"]
    verifier = context["verifier"]
    config = context["config"]
    sample_times = context["sample_times"]
    planning_obstacle = StaticObstacleField.from_points(instance["observed_points"])
    validation_obstacle = StaticObstacleField.from_points(instance["gt_dense_points"])
    baseline_result = context["baseline_result"]
    initial = baseline_result.p_inner
    opt_cfg = config["optimizer"]
    rows: dict[str, Any] = {}

    minco_result = FixedTimeMINCOOptimizer(
        head,
        tail,
        durations,
        limits,
        evaluator,
        planning_obstacle,
        lambda_smooth=opt_cfg["lambda_smooth"],
        lambda_risk=opt_cfg["lambda_risk"],
        lambda_position=opt_cfg["lambda_position"],
        lambda_velocity=opt_cfg["lambda_velocity"],
        lambda_acceleration=opt_cfg["lambda_acceleration"],
        samples_per_segment=10,
        risk_samples_per_segment=2,
        finite_difference_epsilon=2.0e-5,
        max_iterations=32,
        gradient_tolerance=opt_cfg["gradient_tolerance"],
    ).optimize(initial)
    rows["minco_risk"] = _minco_metrics(
        "minco_risk", minco_result, evaluator, planning_obstacle, verifier, head, tail, sample_times
    )
    rows["minco_risk"]["verification"] = _verification_dict(
        verifier.verify(
            minco_result["trajectory"],
            validation_obstacle,
            current_q=head[:, 0],
            current_qd=head[:, 1],
            current_qdd=head[:, 2],
            q_goal=tail[:, 0],
            solver_success=bool(minco_result["success"]),
        )
    )
    rows["minco_risk"]["post"] = {
        "J_smooth": compute_jerk_integral(minco_result["trajectory"], time_step=verifier.time_step),
        "total_duration": float(minco_result["trajectory"].total_duration),
    }
    rows["minco_risk"]["plot_samples"] = trajectory_plot_payload(minco_result["trajectory"])
    rows["minco_risk"] = _annotate_budget(rows["minco_risk"])

    rows["rrt_connect_smooth"] = run_rrt_for_instance(context, instance)

    rows["nubs_without_risk"] = _row_for_trajectory(
        method="nubs_without_risk",
        trajectory=context["baseline_trajectory"],
        solver_success=True,
        optimizer_result=baseline_result,
        planning_evaluator=evaluator,
        planning_obstacle=planning_obstacle,
        verifier=verifier,
        validation_obstacle=validation_obstacle,
        head=head,
        tail=tail,
        sample_times=sample_times,
    )

    critical_result = _risk_optimizer(
        config, head, tail, durations, limits, critical_evaluator, planning_obstacle, None
    ).optimize(initial)
    rows["critical_point_nubs"] = _row_for_trajectory(
        method="critical_point_nubs",
        trajectory=critical_result.trajectory,
        solver_success=critical_result.success,
        optimizer_result=critical_result,
        planning_evaluator=critical_evaluator,
        planning_obstacle=planning_obstacle,
        verifier=verifier,
        validation_obstacle=validation_obstacle,
        head=head,
        tail=tail,
        sample_times=sample_times,
    )

    full_result = _risk_optimizer(
        config, head, tail, durations, limits, evaluator, planning_obstacle, None
    ).optimize(initial)
    rows["ccro_nubs"] = _row_for_trajectory(
        method="ccro_nubs",
        trajectory=full_result.trajectory,
        solver_success=full_result.success,
        optimizer_result=full_result,
        planning_evaluator=evaluator,
        planning_obstacle=planning_obstacle,
        verifier=verifier,
        validation_obstacle=validation_obstacle,
        head=head,
        tail=tail,
        sample_times=sample_times,
    )
    return rows


def fmt_mean_std(stats: dict[str, float | None]) -> str:
    if stats["mean"] is None:
        return "-"
    return f"{stats['mean']:.4g} ± {stats['std']:.3g}"


def render_table(metrics: dict[str, Any]) -> str:
    lines = [
        "| 场景 | 方法 | Dense feasible | Budget accepted | $D_{\\min}$ / m | $J_{\\mathrm{smooth}}$ | $T_{\\mathrm{plan}}$ / ms | timeout |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for scenario in ["A", "B", "C"]:
        for method in MAIN_METHODS:
            row = metrics["scenarios"][scenario]["methods"][method]
            lines.append(
                "| "
                + " | ".join(
                    [
                        SCENARIO_LABELS[scenario],
                        METHOD_DISPLAY[method],
                        f"{row['dense_feasible_rate']:.3f}",
                        f"{row['budgeted_accepted_rate']:.3f}",
                        fmt_mean_std(row["D_min"]),
                        fmt_mean_std(row["J_smooth"]),
                        fmt_mean_std(row["T_plan_ms"]),
                        str(row["timeout_count"]),
                    ]
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "Note: $D_{\\min}$ and $J_{\\mathrm{smooth}}$ are computed only over dense-feasible trajectories. "
            "Budget accepted requires dense feasibility and raw planning time no greater than 10 s; the 10 s budget is an offline evaluation criterion, not a hard solver termination.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_method_list(value: str | None) -> set[str]:
    if value is None or not value.strip():
        return set()
    methods = {item.strip() for item in value.split(",") if item.strip()}
    unknown = methods - set(MAIN_METHODS)
    if unknown:
        raise ValueError(f"unknown methods for rerun: {sorted(unknown)}")
    unsupported = methods - {"rrt_connect_smooth"}
    if unsupported:
        raise ValueError(
            "selective rerun is currently supported only for rrt_connect_smooth, "
            f"got {sorted(unsupported)}"
        )
    return methods


def run(
    config_path: str | Path,
    output: str | Path = "data/results/6_3",
    instances_per_scenario: int = 10,
    force_regenerate: bool = False,
    resume: bool = True,
    rerun_methods: set[str] | None = None,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    from experiments.exp_ccro_stage2 import _load

    config = _load(config_path)
    output = Path(output)
    if not output.is_absolute():
        output = ROOT / output
    frozen = output / "frozen_instances"
    trials = output / "trials"
    paper = output / "paper"
    previous_manifest_path = output / "manifest.json"
    previous_manifest: dict[str, Any] = {}
    if previous_manifest_path.exists():
        try:
            previous_manifest = json.loads(previous_manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            previous_manifest = {}
    for path in (output, frozen, trials, paper):
        path.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output / "source_ccro_stage2.yaml")
    context = build_context(config)
    rerun_history = sorted(
        set(previous_manifest.get("rerun_history", []))
        | set(previous_manifest.get("rerun_methods", []))
        | set(rerun_methods or [])
    )
    metrics: dict[str, Any] = {
        "source": "revised Chapter 6.3 static benchmark",
        "config": str(config_path),
        "instances_per_scenario": int(instances_per_scenario),
        "time_limit_ms": TIME_LIMIT_MS,
        "validation_d_accept_m": validation_accept_distance(config),
        "rerun_methods": sorted(rerun_methods or []),
        "rerun_history": rerun_history,
        "scenarios": {},
    }
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    for scenario in ["A", "B", "C"]:
        metrics["scenarios"][scenario] = {
            "label": SCENARIO_LABELS[scenario],
            "instances": [],
            "methods": {method: [] for method in MAIN_METHODS},
        }
        for index in range(instances_per_scenario):
            instance_path = frozen / f"{scenario}_{index:02d}.json"
            if force_regenerate or not instance_path.exists():
                save_instance(instance_path, make_frozen_instance(context, scenario, index))
            instance = load_instance(instance_path)
            trial_rel = f"trials/{scenario}_{index:02d}.json"
            trial_path = output / trial_rel
            if resume and not force_regenerate and trial_path.exists():
                rows = load_trial(trial_path)
                rerun = set() if rerun_methods is None else set(rerun_methods)
                if rerun:
                    print(f"[6_3] rerun {sorted(rerun)} for {instance['id']} -> {trial_rel}", flush=True)
                    if "rrt_connect_smooth" in rerun:
                        rows["rrt_connect_smooth"] = run_rrt_for_instance(context, instance)
                    trial_path.write_text(
                        json.dumps(rows, indent=2, ensure_ascii=False, default=json_default) + "\n",
                        encoding="utf-8",
                    )
                else:
                    print(f"[6_3] reuse {instance['id']} -> {trial_rel}", flush=True)
            else:
                print(f"[6_3] run {instance['id']} ({scenario} {index + 1}/{instances_per_scenario})", flush=True)
                rows = run_methods_for_instance(context, instance)
                trial_path.write_text(
                    json.dumps(rows, indent=2, ensure_ascii=False, default=json_default) + "\n",
                    encoding="utf-8",
                )
            all_accepted = all(rows[method]["verification"].get("accepted") for method in MAIN_METHODS)
            metrics["scenarios"][scenario]["instances"].append(
                {"id": instance["id"], "all_main_accepted": bool(all_accepted), "trial_path": trial_rel}
            )
            for method in MAIN_METHODS:
                metrics["scenarios"][scenario]["methods"][method].append(rows[method])
        for method in MAIN_METHODS:
            metrics["scenarios"][scenario]["methods"][method] = aggregate_method_rows(
                metrics["scenarios"][scenario]["methods"][method]
            )
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, default=json_default) + "\n",
        encoding="utf-8",
    )
    (paper / "table_6_3_static_benchmark.md").write_text(render_table(metrics), encoding="utf-8")
    manifest = {
        "started_at": started_at,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "output": str(output),
        "config": str(config_path),
        "git_commit": git_commit_hash(),
        "benchmark_script": "experiments/new/6_3/run_static_benchmark.py",
        "plot_script": "experiments/new/6_3/plot_static_benchmark.py",
        "instances_per_scenario": int(instances_per_scenario),
        "frozen_instance_count": len(list(frozen.glob("*.json"))),
        "trial_count": len(list(trials.glob("*.json"))),
        "paper_files": sorted(path.name for path in paper.glob("*")),
        "validation_d_accept_m": validation_accept_distance(config),
        "rrt_planning_clearance_m": validation_accept_distance(config),
        "rerun_methods": sorted(rerun_methods or []),
        "rerun_history": rerun_history,
        "render_reused_existing_trials": bool(resume and not force_regenerate and not rerun_methods),
        "result_sources": {
            "frozen_instances": "reused unless --force-regenerate is supplied",
            "rrt_connect_smooth": (
                "official rows were refreshed on the unchanged frozen instances after unifying "
                "RRT planning clearance to 0.08 m; paper rerenders load the refreshed rows "
                "from existing trials"
            ),
            "minco_risk": "loaded from existing trials unless a full rerun is requested; adapted MINCO-risk baseline",
            "nubs_without_risk": "loaded from existing trials unless a full rerun is requested",
            "critical_point_nubs": "loaded from existing trials unless a full rerun is requested; critical points reuse Chapter 6.2 definition",
            "ccro_nubs": "loaded from existing trials unless a full rerun is requested",
        },
        "timeout_policy": (
            "dense_feasible and within_time_budget are reported separately; "
            "elapsed_ms_raw is preserved"
        ),
        "critical_point_definition": (
            "reuses experiments.new.6_2.body_coverage_62 build_critical_points, "
            "BODY_REGIONS, CRITICAL_POINT_RADII"
        ),
        "figure_note": (
            "Nearest CCRO pair is the representative trajectory minimum 3D "
            "robot-surface/obstacle-point distance segment."
        ),
        "result_audit_files": [
            path.name
            for path in [output / "minco_audit.json", output / "minco_audit.md"]
            if path.exists()
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config" / "ccro_stage2.yaml"))
    parser.add_argument("--output", default="data/results/6_3")
    parser.add_argument("--instances-per-scenario", type=int, default=10)
    parser.add_argument("--force-regenerate", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--rerun-methods", default="")
    args = parser.parse_args()
    rerun_methods = parse_method_list(args.rerun_methods)
    metrics = run(
        args.config,
        args.output,
        args.instances_per_scenario,
        args.force_regenerate,
        resume=not args.no_resume,
        rerun_methods=rerun_methods,
    )
    print(render_table(metrics))
    print(f"[6_3] saved results to {args.output}")


if __name__ == "__main__":
    main()
