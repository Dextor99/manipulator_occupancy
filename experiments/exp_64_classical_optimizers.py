"""CHOMP/TrajOpt/GPMP2-style static planning baselines for E2.

These are lightweight reproductions of the core ideas, not bindings to the
original packages:

- CHOMP-style: smoothness functional plus obstacle potential.
- TrajOpt-style: sequential-optimization style hinge collision cost.
- GPMP2-style: Gaussian-process-like second-difference prior plus obstacle
  likelihood.

All methods emit NUBS trajectories and are evaluated with the same dense
``TrajectoryVerifier`` used by CCRO-NUBS and the existing external baselines.
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
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.exp_ccro_stage2 import (  # noqa: E402
    _baseline,
    _limits,
    _load,
    _states,
    make_scenario_obstacle,
)
from planning.mesh_risk import MeshRiskEvaluator, StaticObstacleField  # noqa: E402
from planning.nubs_trajectory import NUBSTrajectory6D  # noqa: E402
from planning.optimizer import JointLimits  # noqa: E402
from planning.robot_surface_model import RobotSurfaceModel  # noqa: E402
from planning.verifier import TrajectoryVerifier  # noqa: E402


METHOD_LABELS = {
    "chomp_style": "CHOMP-style",
    "trajopt_style": "TrajOpt-style",
    "gpmp2_style": "GPMP2-style",
}


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, NUBSTrajectory6D):
        return "<NUBSTrajectory6D>"
    raise TypeError(type(value).__name__)


def _make_sample_times(durations: np.ndarray, samples_per_segment: int) -> np.ndarray:
    chunks: list[np.ndarray] = []
    start = 0.0
    for segment_index, duration in enumerate(durations):
        end = start + float(duration)
        segment = np.linspace(start, end, samples_per_segment + 1)
        chunks.append(segment if segment_index == 0 else segment[1:])
        start = end
    return np.concatenate(chunks)


class ClassicalStyleOptimizer:
    def __init__(
        self,
        method: str,
        head: np.ndarray,
        tail: np.ndarray,
        durations: np.ndarray,
        limits: JointLimits,
        evaluator: MeshRiskEvaluator,
        obstacle: StaticObstacleField,
        *,
        d_safe: float,
        lambda_obstacle: float,
        lambda_smooth: float,
        lambda_prior: float,
        samples_per_segment: int,
        max_iterations: int,
        random_seed: int,
    ) -> None:
        if method not in METHOD_LABELS:
            raise ValueError(f"unknown method: {method}")
        self.method = method
        self.head = np.asarray(head, dtype=np.float64)
        self.tail = np.asarray(tail, dtype=np.float64)
        self.durations = np.asarray(durations, dtype=np.float64)
        self.limits = limits
        self.evaluator = evaluator
        self.obstacle = obstacle
        self.d_safe = float(d_safe)
        self.lambda_obstacle = float(lambda_obstacle)
        self.lambda_smooth = float(lambda_smooth)
        self.lambda_prior = float(lambda_prior)
        self.max_iterations = int(max_iterations)
        self.rng = np.random.default_rng(int(random_seed))
        self.sample_times = _make_sample_times(self.durations, int(samples_per_segment))
        self.inner_shape = (len(self.durations) - 1, 6)
        self.initial_inner = NUBSTrajectory6D.linear_inner_points(
            self.head[:, 0], self.tail[:, 0], self.durations
        )

    def trajectory(self, p_inner: np.ndarray) -> NUBSTrajectory6D:
        return NUBSTrajectory6D().generate(
            np.asarray(p_inner, dtype=np.float64).reshape(self.inner_shape),
            self.head,
            self.tail,
            self.durations,
        )

    def _waypoints(self, p_inner: np.ndarray) -> np.ndarray:
        return np.vstack([self.head[:, 0], p_inner.reshape(self.inner_shape), self.tail[:, 0]])

    def _path_smoothness(self, p_inner: np.ndarray) -> float:
        points = self._waypoints(p_inner)
        if len(points) < 3:
            return 0.0
        second = points[:-2] - 2.0 * points[1:-1] + points[2:]
        first = np.diff(points, axis=0)
        return float(np.sum(second * second) + 0.03 * np.sum(first * first))

    def _gp_prior(self, p_inner: np.ndarray) -> float:
        points = self._waypoints(p_inner)
        second = points[:-2] - 2.0 * points[1:-1] + points[2:]
        third = second[1:] - second[:-1] if len(second) > 1 else np.zeros((0, 6))
        return float(np.sum(second * second) + 0.5 * np.sum(third * third))

    def _trajopt_prior(self, p_inner: np.ndarray) -> float:
        points = self._waypoints(p_inner)
        diff = np.diff(points, axis=0)
        return float(np.sum(diff * diff))

    def _limit_penalty(self, trajectory: NUBSTrajectory6D) -> float:
        samples = trajectory.sample(self.sample_times)
        q_low = np.maximum(self.limits.q_min[None, :] - samples.q, 0.0)
        q_high = np.maximum(samples.q - self.limits.q_max[None, :], 0.0)
        qd = np.maximum(np.abs(samples.qd) - self.limits.qd_max[None, :], 0.0)
        qdd = np.maximum(np.abs(samples.qdd) - self.limits.qdd_max[None, :], 0.0)
        return float(
            100.0 * np.sum(q_low * q_low + q_high * q_high)
            + 30.0 * np.sum(qd * qd)
            + 30.0 * np.sum(qdd * qdd)
        )

    def _obstacle_cost(self, trajectory: NUBSTrajectory6D) -> tuple[float, float]:
        risk = self.evaluator.trajectory(
            trajectory,
            self.obstacle,
            self.sample_times,
            density="coarse",
            with_gradient=False,
        )
        distances = np.asarray(risk.sample_distances, dtype=np.float64)
        hinge = np.maximum(self.d_safe - distances, 0.0)
        if self.method == "chomp_style":
            cost = float(np.mean(np.exp(-np.maximum(distances, 0.0) / max(self.d_safe, 1e-6))))
            cost += float(np.mean(hinge * hinge) * 10.0)
        elif self.method == "trajopt_style":
            margin = 1.25 * self.d_safe
            strong_hinge = np.maximum(margin - distances, 0.0)
            cost = float(np.mean(strong_hinge * strong_hinge) * 25.0)
        else:
            sigma = max(0.5 * self.d_safe, 1.0e-4)
            likelihood = np.exp(-0.5 * np.square(np.maximum(distances, 0.0) / sigma))
            cost = float(np.mean(likelihood) + np.mean(hinge * hinge) * 12.0)
        return cost, risk.min_distance

    def cost(self, flat: np.ndarray) -> float:
        p_inner = np.asarray(flat, dtype=np.float64).reshape(self.inner_shape)
        trajectory = self.trajectory(p_inner)
        obstacle_cost, _ = self._obstacle_cost(trajectory)
        if self.method == "chomp_style":
            prior = self._path_smoothness(p_inner)
            energy = trajectory.energy()
        elif self.method == "trajopt_style":
            prior = self._trajopt_prior(p_inner)
            energy = 0.15 * trajectory.energy()
        else:
            prior = self._gp_prior(p_inner)
            energy = 0.35 * trajectory.energy()
        return float(
            self.lambda_obstacle * obstacle_cost
            + self.lambda_smooth * energy
            + self.lambda_prior * prior
            + self._limit_penalty(trajectory)
        )

    def optimize(self, initial: np.ndarray | None = None) -> dict[str, Any]:
        if initial is None:
            initial = self.initial_inner
        initial = np.asarray(initial, dtype=np.float64).reshape(self.inner_shape)
        initial_traj = self.trajectory(initial)
        initial_cost = self.cost(initial.ravel())
        initial_obstacle, initial_min_distance = self._obstacle_cost(initial_traj)
        started = time.perf_counter()
        best_flat = initial.ravel().copy()
        best_cost = self.cost(best_flat)
        evaluations = 1
        scales = [0.35, 0.22, 0.13, 0.07]
        per_scale = max(3, self.max_iterations // len(scales))
        lower = np.tile(self.limits.q_min, self.inner_shape[0])
        upper = np.tile(self.limits.q_max, self.inner_shape[0])
        for scale in scales:
            improved = True
            rounds = 0
            while improved and rounds < 2:
                improved = False
                rounds += 1
                # Coordinate moves give a CHOMP/TrajOpt-like local refinement
                # while keeping the evaluation budget bounded.
                order = self.rng.permutation(best_flat.size)
                for variable in order[: max(6, best_flat.size // 2)]:
                    for sign in (-1.0, 1.0):
                        candidate = best_flat.copy()
                        candidate[variable] += sign * scale
                        candidate = np.minimum(np.maximum(candidate, lower), upper)
                        value = self.cost(candidate)
                        evaluations += 1
                        if value < best_cost:
                            best_cost = value
                            best_flat = candidate
                            improved = True
                if evaluations >= self.max_iterations * 8:
                    break
            for _ in range(per_scale):
                candidate = best_flat + self.rng.normal(0.0, scale, size=best_flat.shape)
                candidate = np.minimum(np.maximum(candidate, lower), upper)
                value = self.cost(candidate)
                evaluations += 1
                if value < best_cost:
                    best_cost = value
                    best_flat = candidate
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        final_points = np.asarray(best_flat, dtype=np.float64).reshape(self.inner_shape)
        final_traj = self.trajectory(final_points)
        final_obstacle, final_min_distance = self._obstacle_cost(final_traj)
        return {
            "success": bool(np.isfinite(best_cost)),
            "status": 0,
            "message": "fixed-budget stochastic coordinate refinement",
            "trajectory": final_traj,
            "p_inner": final_points,
            "initial_cost": float(initial_cost),
            "final_cost": float(best_cost),
            "initial_energy": float(initial_traj.energy()),
            "final_energy": float(final_traj.energy()),
            "initial_risk": float(initial_obstacle),
            "final_risk": float(final_obstacle),
            "initial_min_distance": float(initial_min_distance),
            "final_min_distance": float(final_min_distance),
            "iterations": int(self.max_iterations),
            "function_evaluations": int(evaluations),
            "elapsed_ms": float(elapsed_ms),
        }


def _style_metrics(
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
        "method_label": METHOD_LABELS[method],
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


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if math.isinf(value):
            return "inf"
        return f"{value:.6g}"
    return str(value)


def markdown(headers: list[str], rows: list[list[str]]) -> str:
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
            *["| " + " | ".join(row) + " |" for row in rows],
        ]
    )


def run(
    config_path: str | Path,
    output_override: str | Path | None = None,
    *,
    methods: list[str] | None = None,
    max_iterations: int = 24,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    config = _load(config_path)
    output = Path(output_override or "data/results/ch6_e1_e5/E2_static_planning_benchmark/classical_optimizers")
    if not output.is_absolute():
        output = ROOT / output
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output / "source_ccro_stage2.yaml")
    methods = methods or list(METHOD_LABELS)
    unknown = set(methods) - set(METHOD_LABELS)
    if unknown:
        raise ValueError(f"unknown methods: {sorted(unknown)}")

    rng = np.random.default_rng(int(config["experiment"]["random_seed"]) + 641)
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
    baseline = baseline_result.trajectory
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
    sample_times = np.linspace(0.0, float(np.sum(durations)), 41)
    metrics: dict[str, Any] = {
        "source": "CHOMP/TrajOpt/GPMP2-style baselines for E2",
        "notes": {
            "chomp_style": "smoothness functional plus obstacle potential",
            "trajopt_style": "hinge collision cost with waypoint smoothness",
            "gpmp2_style": "GP-like second-difference prior plus obstacle likelihood",
            "evaluation": "all methods use the shared dense TrajectoryVerifier",
        },
        "scenarios": {},
    }
    for scenario_name in config["experiment"]["scenarios"]:
        obstacle, obstacle_info = make_scenario_obstacle(
            config, scenario_name, surface_model, baseline, rng
        )
        scenario_rows = {}
        for method in methods:
            if method == "chomp_style":
                weights = dict(lambda_obstacle=180.0, lambda_smooth=0.08, lambda_prior=0.80)
            elif method == "trajopt_style":
                weights = dict(lambda_obstacle=260.0, lambda_smooth=0.03, lambda_prior=0.25)
            else:
                weights = dict(lambda_obstacle=210.0, lambda_smooth=0.04, lambda_prior=1.20)
            optimizer = ClassicalStyleOptimizer(
                method,
                head,
                tail,
                durations,
                limits,
                evaluator,
                obstacle,
                d_safe=risk_cfg["d_safe"],
                samples_per_segment=max(3, int(risk_cfg["risk_samples_per_segment"])),
                max_iterations=max_iterations,
                random_seed=int(config["experiment"]["random_seed"]) + 6410 + 131 * len(scenario_rows) + ord(scenario_name[0]),
                **weights,
            )
            result = optimizer.optimize(baseline_result.p_inner)
            scenario_rows[method] = _style_metrics(
                method, result, evaluator, obstacle, verifier, head, tail, sample_times
            )
        metrics["scenarios"][scenario_name] = {
            "obstacle": obstacle_info,
            "methods": scenario_rows,
        }
    metrics["accepted"] = True
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )
    table_rows = []
    for scenario, payload in metrics["scenarios"].items():
        for method, row in payload["methods"].items():
            opt = row["optimization"]
            ver = row["verification"]
            table_rows.append(
                [
                    scenario,
                    row["method_label"],
                    str(row["solver_success"]),
                    str(ver["accepted"]),
                    fmt(ver["min_distance"]),
                    fmt(row["full_body_risk_cost"]),
                    fmt(opt["final_energy"]),
                    fmt(ver["goal_error"]),
                    row.get("nearest_link") or "-",
                    fmt(opt["elapsed_ms"]),
                ]
            )
    table = markdown(
        ["scenario", "method", "solver", "accepted", "D_min dense/m", "J_risk", "J_smooth", "goal error", "nearest link", "time/ms"],
        table_rows,
    )
    (output / "table_E2_classical_optimizers.md").write_text(table + "\n", encoding="utf-8")
    print(table)
    print(f"\n[exp_64_classical] saved results to {output}")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config" / "ccro_stage2.yaml"))
    parser.add_argument("--output", default="data/results/ch6_e1_e5/E2_static_planning_benchmark/classical_optimizers")
    parser.add_argument("--methods", default="chomp_style,trajopt_style,gpmp2_style")
    parser.add_argument("--max-iterations", type=int, default=24)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    run(args.config, args.output, methods=methods, max_iterations=args.max_iterations)


if __name__ == "__main__":
    main()
