"""Reproduce external planning baselines for Chapter 6.4.

Baselines:
- MINCO-base: joint-space minimum-control polynomial trajectory.
- MINCO-risk: same MINCO representation plus the repository's full-body risk.
- RRT-Connect + smoothing: joint-space bidirectional sampling planner followed
  by shortcutting and minimum-jerk MINCO time parameterization.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import shutil
import sys
import time
from typing import Any

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.exp_ccro_stage2 import (  # noqa: E402
    _baseline,
    _limits,
    _load,
    _method_metrics,
    _states,
    make_scenario_obstacle,
)
from planning.mesh_risk import MeshRiskEvaluator, StaticObstacleField  # noqa: E402
from planning.minco_trajectory import MinJerkMINCOTrajectory6D  # noqa: E402
from planning.nubs_trajectory import NUBSTrajectory6D  # noqa: E402
from planning.optimizer import JointLimits  # noqa: E402
from planning.robot_surface_model import RobotSurfaceModel  # noqa: E402
from planning.verifier import TrajectoryVerifier  # noqa: E402


class FixedTimeMINCOOptimizer:
    def __init__(
        self,
        head_state: np.ndarray,
        tail_state: np.ndarray,
        durations: np.ndarray,
        limits: JointLimits,
        evaluator: MeshRiskEvaluator | None = None,
        obstacle: StaticObstacleField | None = None,
        *,
        lambda_smooth: float = 0.05,
        lambda_risk: float = 5000.0,
        lambda_position: float = 20.0,
        lambda_velocity: float = 20.0,
        lambda_acceleration: float = 20.0,
        samples_per_segment: int = 10,
        risk_samples_per_segment: int = 4,
        finite_difference_epsilon: float = 1.0e-5,
        max_iterations: int = 60,
        gradient_tolerance: float = 1.0e-6,
    ) -> None:
        self.head_state = np.asarray(head_state, dtype=np.float64)
        self.tail_state = np.asarray(tail_state, dtype=np.float64)
        self.durations = np.asarray(durations, dtype=np.float64)
        self.limits = limits
        self.evaluator = evaluator
        self.obstacle = obstacle
        self.lambda_smooth = float(lambda_smooth)
        self.lambda_risk = float(lambda_risk)
        self.lambda_position = float(lambda_position)
        self.lambda_velocity = float(lambda_velocity)
        self.lambda_acceleration = float(lambda_acceleration)
        self.samples_per_segment = int(samples_per_segment)
        self.risk_samples_per_segment = int(risk_samples_per_segment)
        self.fd_epsilon = float(finite_difference_epsilon)
        self.max_iterations = int(max_iterations)
        self.gradient_tolerance = float(gradient_tolerance)
        self.sample_times = self._make_sample_times(self.samples_per_segment)
        self.risk_sample_times = self._make_sample_times(self.risk_samples_per_segment)

    @property
    def inner_shape(self) -> tuple[int, int]:
        return (len(self.durations) - 1, 6)

    def _make_sample_times(self, samples_per_segment: int) -> np.ndarray:
        chunks: list[np.ndarray] = []
        start = 0.0
        for i, duration in enumerate(self.durations):
            end = start + float(duration)
            segment = np.linspace(start, end, samples_per_segment + 1)
            chunks.append(segment if i == 0 else segment[1:])
            start = end
        return np.concatenate(chunks)

    def trajectory(self, p_inner: np.ndarray) -> MinJerkMINCOTrajectory6D:
        return MinJerkMINCOTrajectory6D.from_inner_points(
            np.asarray(p_inner, dtype=np.float64).reshape(self.inner_shape),
            self.head_state,
            self.tail_state,
            self.durations,
        )

    def violations(self, trajectory: MinJerkMINCOTrajectory6D) -> tuple[float, float, float, float]:
        samples = trajectory.sample(self.sample_times)
        q_low = np.maximum(self.limits.q_min[None, :] - samples.q, 0.0)
        q_high = np.maximum(samples.q - self.limits.q_max[None, :], 0.0)
        qd = np.maximum(np.abs(samples.qd) - self.limits.qd_max[None, :], 0.0)
        qdd = np.maximum(np.abs(samples.qdd) - self.limits.qdd_max[None, :], 0.0)
        q_cost = float(np.trapz(np.sum(q_low * q_low + q_high * q_high, axis=1), self.sample_times))
        qd_cost = float(np.trapz(np.sum(qd * qd, axis=1), self.sample_times))
        qdd_cost = float(np.trapz(np.sum(qdd * qdd, axis=1), self.sample_times))
        weighted = (
            self.lambda_position * q_cost
            + self.lambda_velocity * qd_cost
            + self.lambda_acceleration * qdd_cost
        )
        return weighted, float(np.max(np.maximum(q_low, q_high))), float(np.max(qd)), float(np.max(qdd))

    def cost_only(self, flat_points: np.ndarray) -> float:
        points = np.asarray(flat_points, dtype=np.float64).reshape(self.inner_shape)
        trajectory = self.trajectory(points)
        cost = self.lambda_smooth * trajectory.energy()
        if self.evaluator is not None and self.obstacle is not None:
            risk = self.evaluator.trajectory(
                trajectory,
                self.obstacle,
                self.risk_sample_times,
                density="coarse",
                with_gradient=False,
            )
            cost += self.lambda_risk * risk.cost
        penalty, _, _, _ = self.violations(trajectory)
        return float(cost + penalty)

    def objective(self, flat_points: np.ndarray) -> tuple[float, np.ndarray]:
        cost = self.cost_only(flat_points)
        gradient = np.zeros_like(flat_points)
        for i in range(flat_points.size):
            plus = flat_points.copy()
            minus = flat_points.copy()
            plus[i] += self.fd_epsilon
            minus[i] -= self.fd_epsilon
            gradient[i] = (self.cost_only(plus) - self.cost_only(minus)) / (
                2.0 * self.fd_epsilon
            )
        return float(cost), gradient

    def optimize(self, initial: np.ndarray | None = None) -> dict[str, Any]:
        if initial is None:
            initial = NUBSTrajectory6D.linear_inner_points(
                self.head_state[:, 0], self.tail_state[:, 0], self.durations
            )
        initial = np.asarray(initial, dtype=np.float64).reshape(self.inner_shape)
        initial_trajectory = self.trajectory(initial)
        initial_cost = self.cost_only(initial.ravel())
        if self.evaluator is None or self.obstacle is None:
            final_penalty, q_v, qd_v, qdd_v = self.violations(initial_trajectory)
            return {
                "success": True,
                "status": 0,
                "message": "analytic fixed-knot MINCO baseline",
                "trajectory": initial_trajectory,
                "p_inner": initial,
                "initial_cost": float(initial_cost),
                "final_cost": float(initial_cost),
                "initial_energy": float(initial_trajectory.energy()),
                "final_energy": float(initial_trajectory.energy()),
                "final_risk": 0.0,
                "final_min_distance": math.inf,
                "penalty_cost": float(final_penalty),
                "iterations": 0,
                "function_evaluations": 1,
                "gradient_norm": 0.0,
                "elapsed_ms": 0.0,
                "max_q_violation": q_v,
                "max_qd_violation": qd_v,
                "max_qdd_violation": qdd_v,
            }
        bounds = [
            (float(self.limits.q_min[j]), float(self.limits.q_max[j]))
            for _ in range(self.inner_shape[0])
            for j in range(6)
        ]
        started = time.perf_counter()
        rng = np.random.default_rng(20260640)
        best_flat = initial.ravel().copy()
        best_cost = self.cost_only(best_flat)
        evaluations = 1
        scales = [0.28, 0.18, 0.10, 0.05]
        for scale in scales:
            for _ in range(max(2, self.max_iterations // len(scales))):
                candidate = best_flat + rng.normal(0.0, scale, size=best_flat.shape)
                for row in range(self.inner_shape[0]):
                    for joint in range(6):
                        index = row * 6 + joint
                        candidate[index] = np.clip(
                            candidate[index],
                            self.limits.q_min[joint],
                            self.limits.q_max[joint],
                        )
                value = self.cost_only(candidate)
                evaluations += 1
                if value < best_cost:
                    best_cost = value
                    best_flat = candidate
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        final_points = np.asarray(best_flat, dtype=np.float64).reshape(self.inner_shape)
        final_trajectory = self.trajectory(final_points)
        final_penalty, q_v, qd_v, qdd_v = self.violations(final_trajectory)
        risk_cost = 0.0
        min_distance = math.inf
        if self.evaluator is not None and self.obstacle is not None:
            risk = self.evaluator.trajectory(
                final_trajectory, self.obstacle, self.risk_sample_times, with_gradient=False
            )
            risk_cost = risk.cost
            min_distance = risk.min_distance
        return {
            "success": bool(np.isfinite(best_cost)),
            "status": 0,
            "message": "fixed-budget derivative-free MINCO-risk refinement",
            "trajectory": final_trajectory,
            "p_inner": final_points,
            "initial_cost": float(initial_cost),
            "final_cost": float(best_cost),
            "initial_energy": float(initial_trajectory.energy()),
            "final_energy": float(final_trajectory.energy()),
            "final_risk": float(risk_cost),
            "final_min_distance": float(min_distance),
            "penalty_cost": float(final_penalty),
            "iterations": len(scales) * max(2, self.max_iterations // len(scales)),
            "function_evaluations": evaluations,
            "gradient_norm": 0.0,
            "elapsed_ms": float(elapsed_ms),
            "max_q_violation": q_v,
            "max_qd_violation": qd_v,
            "max_qdd_violation": qdd_v,
        }


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, MinJerkMINCOTrajectory6D):
        return "<MinJerkMINCOTrajectory6D>"
    raise TypeError(type(value).__name__)


def _minco_metrics(
    method: str,
    result: dict[str, Any],
    evaluator: MeshRiskEvaluator,
    obstacle: StaticObstacleField,
    verifier: TrajectoryVerifier,
    head: np.ndarray,
    tail: np.ndarray,
    sample_times: np.ndarray,
) -> dict[str, Any]:
    trajectory = result["trajectory"]
    risk = evaluator.trajectory(trajectory, obstacle, sample_times, with_gradient=False)
    verification = verifier.verify(
        trajectory,
        obstacle,
        current_q=head[:, 0],
        current_qd=head[:, 1],
        current_qdd=head[:, 2],
        q_goal=tail[:, 0],
        solver_success=bool(result["success"]),
    )
    return {
        "method": method,
        "solver_success": bool(result["success"]),
        "optimized_links": None,
        "risk_cost_for_method": risk.cost,
        "full_body_risk_cost": risk.cost,
        "optimization_sample_min_distance": risk.min_distance,
        "nearest_link": risk.nearest_link,
        "verification": asdict(verification),
        "optimization": {
            key: value
            for key, value in result.items()
            if key not in {"trajectory", "p_inner"}
        },
        "p_inner": result["p_inner"].tolist(),
    }


def _edge_is_safe(
    a: np.ndarray,
    b: np.ndarray,
    evaluator: MeshRiskEvaluator,
    obstacle: StaticObstacleField,
    d_stop: float,
    step: float = 0.08,
) -> bool:
    distance = float(np.linalg.norm(b - a))
    count = max(2, int(math.ceil(distance / step)) + 1)
    for ratio in np.linspace(0.0, 1.0, count):
        q = (1.0 - ratio) * a + ratio * b
        if evaluator.configuration(q, obstacle, density="coarse").min_distance < d_stop:
            return False
    return True


def _rrt_connect(
    q_start: np.ndarray,
    q_goal: np.ndarray,
    limits: JointLimits,
    evaluator: MeshRiskEvaluator,
    obstacle: StaticObstacleField,
    d_stop: float,
    rng: np.random.Generator,
    *,
    max_iterations: int = 1800,
    step_size: float = 0.22,
) -> tuple[list[np.ndarray] | None, int]:
    def nearest(tree: list[np.ndarray], q: np.ndarray) -> int:
        distances = [float(np.linalg.norm(node - q)) for node in tree]
        return int(np.argmin(distances))

    def steer(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        delta = b - a
        norm = float(np.linalg.norm(delta))
        if norm <= step_size:
            return b.copy()
        return a + step_size * delta / norm

    trees = [[q_start.copy()], [q_goal.copy()]]
    parents = [[-1], [-1]]
    for iteration in range(max_iterations):
        active = iteration % 2
        other = 1 - active
        if rng.random() < 0.15:
            q_rand = q_goal if active == 0 else q_start
        else:
            q_rand = rng.uniform(limits.q_min, limits.q_max)
        nearest_active = nearest(trees[active], q_rand)
        q_new = steer(trees[active][nearest_active], q_rand)
        if not _edge_is_safe(trees[active][nearest_active], q_new, evaluator, obstacle, d_stop):
            continue
        trees[active].append(q_new)
        parents[active].append(nearest_active)
        active_new = len(trees[active]) - 1
        while True:
            nearest_other = nearest(trees[other], trees[active][active_new])
            q_other_new = steer(trees[other][nearest_other], trees[active][active_new])
            if not _edge_is_safe(trees[other][nearest_other], q_other_new, evaluator, obstacle, d_stop):
                break
            trees[other].append(q_other_new)
            parents[other].append(nearest_other)
            other_new = len(trees[other]) - 1
            if np.linalg.norm(trees[other][other_new] - trees[active][active_new]) < 1.0e-7:
                path_a = _trace_path(trees[active], parents[active], active_new)
                path_b = _trace_path(trees[other], parents[other], other_new)
                if active == 0:
                    return path_a + list(reversed(path_b[:-1])), iteration + 1
                return path_b + list(reversed(path_a[:-1])), iteration + 1
            nearest_other = other_new
    return None, max_iterations


def _trace_path(tree: list[np.ndarray], parents: list[int], index: int) -> list[np.ndarray]:
    path = []
    while index >= 0:
        path.append(tree[index])
        index = parents[index]
    return list(reversed(path))


def _shortcut_path(
    path: list[np.ndarray],
    evaluator: MeshRiskEvaluator,
    obstacle: StaticObstacleField,
    d_stop: float,
    rng: np.random.Generator,
    attempts: int = 150,
) -> list[np.ndarray]:
    if len(path) <= 2:
        return path
    current = [p.copy() for p in path]
    for _ in range(attempts):
        if len(current) <= 2:
            break
        i, j = sorted(rng.choice(len(current), size=2, replace=False))
        if j <= i + 1:
            continue
        if _edge_is_safe(current[i], current[j], evaluator, obstacle, d_stop):
            current = current[: i + 1] + current[j:]
    return current


def _rrt_to_trajectory(
    path: list[np.ndarray],
    head: np.ndarray,
    tail: np.ndarray,
    total_duration: float,
) -> MinJerkMINCOTrajectory6D:
    positions = np.asarray(path, dtype=np.float64)
    if len(positions) == 2:
        durations = np.asarray([total_duration], dtype=np.float64)
    else:
        distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        if float(np.sum(distances)) <= 1.0e-9:
            durations = np.full(len(positions) - 1, total_duration / (len(positions) - 1))
        else:
            durations = total_duration * distances / float(np.sum(distances))
            durations = np.maximum(durations, 0.25)
            durations *= total_duration / float(np.sum(durations))
    return MinJerkMINCOTrajectory6D(positions, head, tail, durations)


def _rrt_metrics(
    trajectory: MinJerkMINCOTrajectory6D | None,
    success: bool,
    planning_ms: float,
    iterations: int,
    waypoint_count: int,
    evaluator: MeshRiskEvaluator,
    obstacle: StaticObstacleField,
    verifier: TrajectoryVerifier,
    head: np.ndarray,
    tail: np.ndarray,
    sample_times: np.ndarray,
) -> dict[str, Any]:
    if trajectory is None:
        return {
            "method": "rrt_connect_smooth",
            "solver_success": False,
            "optimized_links": None,
            "risk_cost_for_method": math.inf,
            "full_body_risk_cost": math.inf,
            "optimization_sample_min_distance": 0.0,
            "nearest_link": None,
            "verification": {
                "accepted": False,
                "reasons": ["planner_failed"],
                "min_distance": 0.0,
                "goal_error": math.inf,
            },
            "optimization": {
                "success": False,
                "elapsed_ms": planning_ms,
                "iterations": iterations,
                "waypoint_count": waypoint_count,
            },
        }
    risk = evaluator.trajectory(trajectory, obstacle, sample_times, with_gradient=False)
    verification = verifier.verify(
        trajectory,
        obstacle,
        current_q=head[:, 0],
        current_qd=head[:, 1],
        current_qdd=head[:, 2],
        q_goal=tail[:, 0],
        solver_success=success,
    )
    return {
        "method": "rrt_connect_smooth",
        "solver_success": success,
        "optimized_links": None,
        "risk_cost_for_method": risk.cost,
        "full_body_risk_cost": risk.cost,
        "optimization_sample_min_distance": risk.min_distance,
        "nearest_link": risk.nearest_link,
        "verification": asdict(verification),
        "optimization": {
            "success": success,
            "elapsed_ms": planning_ms,
            "iterations": iterations,
            "waypoint_count": waypoint_count,
            "duration": trajectory.total_duration,
            "final_energy": trajectory.energy(),
        },
    }


def _run_rrt_once(
    seed: int,
    head: np.ndarray,
    tail: np.ndarray,
    durations: np.ndarray,
    limits: JointLimits,
    evaluator: MeshRiskEvaluator,
    obstacle: StaticObstacleField,
    verifier: TrajectoryVerifier,
    d_stop: float,
    sample_times: np.ndarray,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    started = time.perf_counter()
    raw_path, iterations = _rrt_connect(
        head[:, 0],
        tail[:, 0],
        limits,
        evaluator,
        obstacle,
        d_stop,
        rng,
    )
    trajectory = None
    waypoint_count = 0
    if raw_path is not None:
        smooth_path = _shortcut_path(raw_path, evaluator, obstacle, d_stop, rng)
        waypoint_count = len(smooth_path)
        trajectory = _rrt_to_trajectory(smooth_path, head, tail, float(np.sum(durations)))
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    row = _rrt_metrics(
        trajectory,
        trajectory is not None,
        elapsed_ms,
        iterations,
        waypoint_count,
        evaluator,
        obstacle,
        verifier,
        head,
        tail,
        sample_times,
    )
    row["seed"] = int(seed)
    return row


def _mean_std_ci(values: list[float]) -> dict[str, float | None]:
    finite = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not finite:
        return {"mean": None, "std": None, "ci95": None}
    arr = np.asarray(finite, dtype=np.float64)
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    ci95 = float(1.96 * std / math.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return {"mean": float(np.mean(arr)), "std": std, "ci95": ci95}


def _aggregate_rrt_trials(trials: list[dict[str, Any]]) -> dict[str, Any]:
    accepted = [bool(row["verification"].get("accepted")) for row in trials]
    solver = [bool(row["solver_success"]) for row in trials]
    dmins = [row["verification"].get("min_distance") for row in trials]
    risks = [row.get("full_body_risk_cost") for row in trials]
    energies = [row.get("optimization", {}).get("final_energy") for row in trials]
    times = [row.get("optimization", {}).get("elapsed_ms") for row in trials]
    waypoints = [row.get("optimization", {}).get("waypoint_count") for row in trials]
    d_stats = _mean_std_ci(dmins)
    risk_stats = _mean_std_ci(risks)
    energy_stats = _mean_std_ci(energies)
    time_stats = _mean_std_ci(times)
    waypoint_stats = _mean_std_ci(waypoints)
    best_index = int(np.argmax([v if v is not None and np.isfinite(v) else -1.0 for v in dmins]))
    best = trials[best_index]
    return {
        "method": "rrt_connect_smooth",
        "solver_success": float(np.mean(solver)) >= 0.5,
        "optimized_links": None,
        "risk_cost_for_method": risk_stats["mean"],
        "full_body_risk_cost": risk_stats["mean"],
        "optimization_sample_min_distance": d_stats["mean"],
        "nearest_link": best.get("nearest_link"),
        "verification": {
            "accepted": bool(all(accepted)),
            "success_rate": float(np.mean(accepted)),
            "min_distance": d_stats["mean"],
            "min_distance_std": d_stats["std"],
            "min_distance_ci95": d_stats["ci95"],
            "goal_error": best["verification"].get("goal_error"),
            "reasons": [] if all(accepted) else ["some_rrt_seeds_failed_dense_verification"],
        },
        "optimization": {
            "success": bool(all(solver)),
            "elapsed_ms": time_stats["mean"],
            "elapsed_ms_std": time_stats["std"],
            "elapsed_ms_ci95": time_stats["ci95"],
            "waypoint_count": waypoint_stats["mean"],
            "waypoint_count_std": waypoint_stats["std"],
            "final_energy": energy_stats["mean"],
            "final_energy_std": energy_stats["std"],
            "final_energy_ci95": energy_stats["ci95"],
        },
        "rrt_trials": trials,
        "statistics": {
            "n": len(trials),
            "success_rate": float(np.mean(solver)),
            "accepted_rate": float(np.mean(accepted)),
            "D_min": d_stats,
            "J_smooth": energy_stats,
            "elapsed_ms": time_stats,
        },
    }


def markdown(headers: list[str], rows: list[list[str]]) -> str:
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
            *["| " + " | ".join(row) + " |" for row in rows],
        ]
    )


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if math.isinf(value):
            return "inf"
        return f"{value:.6g}"
    return str(value)


def run(config_path: str | Path, output_override: str | Path | None = None, *, rrt_seeds: int = 5) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    config = _load(config_path)
    output = Path(output_override or "data/results/ch6_4_external")
    if not output.is_absolute():
        output = ROOT / output
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output / "source_ccro_stage2.yaml")
    rng = np.random.default_rng(int(config["experiment"]["random_seed"]) + 640)
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
    nubs_baseline_result = _baseline(config, head, tail, durations)
    nubs_baseline = nubs_baseline_result.trajectory
    risk_cfg = config["risk"]
    opt_cfg = config["optimizer"]
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
    metrics: dict[str, Any] = {
        "source": "External baselines reproduced for Chapter 6.4",
        "references": {
            "minco": {
                "paper": "Wang et al., Geometrically Constrained Trajectory Optimization for Multicopters, IEEE T-RO 2022",
                "repo": "https://github.com/ZJU-FAST-Lab/GCOPTER",
                "adaptation": "joint-space s=3 minimum-jerk MINCO with shared full-body risk verifier",
            },
            "rrt_connect": {
                "adaptation": "joint-space bidirectional RRT-Connect, shortcut smoothing, MINCO time parameterization",
            },
        },
        "scenarios": {},
    }
    sample_times = np.linspace(0.0, float(np.sum(durations)), 41)
    for scenario_name in config["experiment"]["scenarios"]:
        obstacle, obstacle_info = make_scenario_obstacle(
            config, scenario_name, surface_model, nubs_baseline, rng
        )
        initial = NUBSTrajectory6D.linear_inner_points(head[:, 0], tail[:, 0], durations)
        minco_base = FixedTimeMINCOOptimizer(
            head,
            tail,
            durations,
            limits,
            lambda_smooth=opt_cfg["lambda_smooth"],
            lambda_position=opt_cfg["lambda_position"],
            lambda_velocity=opt_cfg["lambda_velocity"],
            lambda_acceleration=opt_cfg["lambda_acceleration"],
            samples_per_segment=10,
            finite_difference_epsilon=2.0e-5,
            max_iterations=45,
            gradient_tolerance=opt_cfg["gradient_tolerance"],
        ).optimize(initial)
        minco_risk = FixedTimeMINCOOptimizer(
            head,
            tail,
            durations,
            limits,
            evaluator,
            obstacle,
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
        ).optimize(minco_base["p_inner"])
        rrt_trials = [
            _run_rrt_once(
                int(config["experiment"]["random_seed"]) + 6400 + 97 * seed_index + 13 * ord(scenario_name[0]),
                head,
                tail,
                durations,
                limits,
                evaluator,
                obstacle,
                verifier,
                risk_cfg["d_stop"],
                sample_times,
            )
            for seed_index in range(int(rrt_seeds))
        ]
        rows = {
            "minco_base": _minco_metrics(
                "minco_base", minco_base, evaluator, obstacle, verifier, head, tail, sample_times
            ),
            "minco_risk": _minco_metrics(
                "minco_risk", minco_risk, evaluator, obstacle, verifier, head, tail, sample_times
            ),
            "rrt_connect_smooth": _aggregate_rrt_trials(rrt_trials),
        }
        metrics["scenarios"][scenario_name] = {
            "obstacle": obstacle_info,
            "methods": rows,
        }
    metrics["accepted"] = True
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )
    table_rows = []
    for scenario, payload in metrics["scenarios"].items():
        for method, row in payload["methods"].items():
            opt = row.get("optimization", {})
            ver = row["verification"]
            table_rows.append(
                [
                    scenario,
                    method,
                    str(row["solver_success"]),
                    str(ver.get("accepted")),
                    fmt(ver.get("min_distance")),
                    fmt(row.get("full_body_risk_cost")),
                    fmt(opt.get("final_energy")),
                    fmt(ver.get("goal_error")),
                    row.get("nearest_link") or "-",
                    fmt(opt.get("elapsed_ms")),
                ]
            )
    table = markdown(
        ["scenario", "method", "solver", "accepted", "D_min dense/m", "J_risk", "J_smooth", "goal error", "nearest link", "time/ms"],
        table_rows,
    )
    (output / "table_6_4_external_baselines.md").write_text(table + "\n", encoding="utf-8")
    print(table)
    print(f"\n[exp_64_external] saved results to {output}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config" / "ccro_stage2.yaml"))
    parser.add_argument("--output", default="data/results/ch6_4_external")
    parser.add_argument("--rrt-seeds", type=int, default=5)
    args = parser.parse_args()
    run(args.config, args.output, rrt_seeds=args.rrt_seeds)


if __name__ == "__main__":
    main()
