"""Fixed-time NUBS optimization with full-body spatio-temporal risk."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from scipy.optimize import minimize

from .nubs_trajectory import DIMENSION, NUBSTrajectory6D
from .obstacle_forecast import ObstacleForecast
from .optimizer import FixedTimeNUBSOptimizer, JointLimits
from .spatiotemporal_risk import (
    SpatioTemporalRiskEvaluator,
    SpatioTemporalTrajectoryRisk,
)


@dataclass
class DynamicRiskOptimizationResult:
    success: bool
    status: int
    message: str
    trajectory: NUBSTrajectory6D
    p_inner: np.ndarray
    durations: np.ndarray
    initial_cost: float
    final_cost: float
    initial_energy: float
    final_energy: float
    initial_risk: float
    final_risk: float
    initial_min_distance: float
    final_min_distance: float
    penalty_cost: float
    iterations: int
    function_evaluations: int
    gradient_norm: float
    elapsed_ms: float


class _OptimizationTimeout(RuntimeError):
    def __init__(self, x: np.ndarray) -> None:
        super().__init__("optimization time budget exceeded")
        self.x = np.asarray(x, dtype=np.float64).copy()


class DynamicRiskNUBSOptimizer(FixedTimeNUBSOptimizer):
    def __init__(
        self,
        head_state: np.ndarray,
        tail_state: np.ndarray,
        durations: np.ndarray,
        joint_limits: JointLimits,
        risk_evaluator: SpatioTemporalRiskEvaluator,
        forecast: ObstacleForecast,
        *,
        lambda_risk: float = 5000.0,
        risk_samples_per_segment: int = 5,
        risk_links: set[str] | None = None,
        sensitivity_epsilon: float = 1.0e-6,
        **kwargs,
    ) -> None:
        super().__init__(head_state, tail_state, durations, joint_limits, **kwargs)
        if lambda_risk <= 0.0 or risk_samples_per_segment < 2:
            raise ValueError("lambda_risk must be positive and risk_samples_per_segment >= 2")
        self.risk_evaluator = risk_evaluator
        self.forecast = forecast
        self.lambda_risk = float(lambda_risk)
        self.risk_samples_per_segment = int(risk_samples_per_segment)
        self.risk_links = None if risk_links is None else set(risk_links)
        self.sensitivity_epsilon = float(sensitivity_epsilon)
        self.risk_sample_times = self._make_risk_sample_times()
        if self.risk_sample_times[-1] > forecast.valid_horizon + 1.0e-12:
            forecast.occupancy_at(float(self.risk_sample_times[-1]))
        reference = NUBSTrajectory6D.linear_inner_points(
            self.head_state[:, 0], self.tail_state[:, 0], self.durations
        )
        self._sample_sensitivity = self._build_sample_sensitivity(reference)

    def _make_risk_sample_times(self) -> np.ndarray:
        chunks: list[np.ndarray] = []
        start = 0.0
        for segment_index, duration in enumerate(self.durations):
            end = start + float(duration)
            segment = np.linspace(start, end, self.risk_samples_per_segment + 1)
            chunks.append(segment if segment_index == 0 else segment[1:])
            start = end
        return np.concatenate(chunks)

    def _build_sample_sensitivity(self, reference: np.ndarray) -> np.ndarray:
        sensitivity = np.zeros(
            (len(self.risk_sample_times), DIMENSION, reference.size), dtype=np.float64
        )
        for variable in range(reference.size):
            row, col = np.unravel_index(variable, self.inner_shape)
            plus = reference.copy()
            minus = reference.copy()
            plus[row, col] += self.sensitivity_epsilon
            minus[row, col] -= self.sensitivity_epsilon
            q_plus = self._trajectory(plus).sample(
                self.risk_sample_times, max_derivative=0
            ).q
            q_minus = self._trajectory(minus).sample(
                self.risk_sample_times, max_derivative=0
            ).q
            sensitivity[:, :, variable] = (q_plus - q_minus) / (
                2.0 * self.sensitivity_epsilon
            )
        return sensitivity

    def evaluate_risk(
        self, trajectory: NUBSTrajectory6D, *, with_gradient: bool
    ) -> SpatioTemporalTrajectoryRisk:
        return self.risk_evaluator.trajectory(
            trajectory,
            self.forecast,
            self.risk_sample_times,
            links=self.risk_links,
            with_gradient=with_gradient,
        )

    def cost_only(self, flat_points: np.ndarray) -> float:
        points = np.asarray(flat_points, dtype=np.float64).reshape(self.inner_shape)
        trajectory = self._trajectory(points)
        risk = self.evaluate_risk(trajectory, with_gradient=False)
        penalty, _, _, _ = self._violations(trajectory)
        return float(
            self.lambda_smooth * trajectory.energy()
            + self.lambda_risk * risk.cost
            + penalty
        )

    def objective(self, flat_points: np.ndarray) -> tuple[float, np.ndarray]:
        points = np.asarray(flat_points, dtype=np.float64).reshape(self.inner_shape)
        trajectory = self._trajectory(points)
        energy, energy_gradient, _ = trajectory.energy_and_gradient()
        risk = self.evaluate_risk(trajectory, with_gradient=True)
        if risk.gradient_q is None:
            raise RuntimeError("spatio-temporal risk did not return gradients")
        risk_gradient = np.einsum(
            "ni,niv->v", risk.gradient_q, self._sample_sensitivity, optimize=True
        )
        penalty, _, _, _ = self._violations(trajectory)
        penalty_gradient = self._penalty_gradient(points, penalty).ravel()
        cost = self.lambda_smooth * energy + self.lambda_risk * risk.cost + penalty
        gradient = (
            self.lambda_smooth * energy_gradient.ravel()
            + self.lambda_risk * risk_gradient
            + penalty_gradient
        )
        if not np.isfinite(cost) or not np.all(np.isfinite(gradient)):
            raise FloatingPointError("dynamic-risk objective produced NaN or Inf")
        return float(cost), gradient

    def check_gradient(
        self, p_inner: np.ndarray, epsilon: float = 1.0e-5
    ) -> dict[str, float]:
        points = np.asarray(p_inner, dtype=np.float64)
        cost, analytic = self.objective(points.ravel())
        numeric = np.zeros(points.size)
        for variable in range(points.size):
            plus = points.ravel().copy()
            minus = points.ravel().copy()
            plus[variable] += epsilon
            minus[variable] -= epsilon
            numeric[variable] = (self.cost_only(plus) - self.cost_only(minus)) / (
                2.0 * epsilon
            )
        difference = analytic - numeric
        denominator = max(float(np.linalg.norm(numeric)), 1.0e-12)
        product = float(np.linalg.norm(analytic) * np.linalg.norm(numeric))
        cosine = 1.0 if product < 1.0e-16 else float(np.dot(analytic, numeric) / product)
        return {
            "cost": float(cost),
            "relative_error": float(np.linalg.norm(difference) / denominator),
            "cosine_similarity": cosine,
            "max_absolute_error": float(np.max(np.abs(difference), initial=0.0)),
            "analytic_norm": float(np.linalg.norm(analytic)),
            "numeric_norm": float(np.linalg.norm(numeric)),
        }

    def optimize(
        self,
        p_inner_initial: np.ndarray | None = None,
        *,
        time_limit_s: float | None = None,
    ) -> DynamicRiskOptimizationResult:
        if p_inner_initial is None:
            p_inner_initial = NUBSTrajectory6D.linear_inner_points(
                self.head_state[:, 0], self.tail_state[:, 0], self.durations
            )
        initial = np.asarray(p_inner_initial, dtype=np.float64)
        if initial.shape != self.inner_shape:
            raise ValueError(f"p_inner_initial must have shape {self.inner_shape}")
        initial_trajectory = self._trajectory(initial)
        initial_risk = self.evaluate_risk(initial_trajectory, with_gradient=False)
        initial_energy = initial_trajectory.energy()
        initial_penalty, _, _, _ = self._violations(initial_trajectory)
        initial_cost = (
            self.lambda_smooth * initial_energy
            + self.lambda_risk * initial_risk.cost
            + initial_penalty
        )
        bounds = [
            (float(self.limits.q_min[joint]), float(self.limits.q_max[joint]))
            for _ in range(self.inner_shape[0])
            for joint in range(DIMENSION)
        ]
        started = time.perf_counter()
        timed_out = False
        best_x = initial.ravel().copy()

        def _callback(xk: np.ndarray) -> None:
            nonlocal best_x
            best_x = np.asarray(xk, dtype=np.float64).copy()
            if time_limit_s is not None and time.perf_counter() - started > float(time_limit_s):
                raise _OptimizationTimeout(best_x)

        try:
            result = minimize(
                self.objective,
                initial.ravel(),
                method="L-BFGS-B",
                jac=True,
                bounds=bounds,
                callback=_callback,
                options={
                    "maxiter": self.max_iterations,
                    "gtol": self.gradient_tolerance,
                    "ftol": 1.0e-10,
                    "maxls": 40,
                },
            )
            final_flat = np.asarray(result.x, dtype=np.float64)
            status = int(result.status)
            message = str(result.message)
            iterations = int(result.nit)
            function_evaluations = int(result.nfev)
            jac = np.asarray(result.jac, dtype=np.float64)
            optimizer_success = bool(result.success) and np.isfinite(result.fun)
            final_fun = float(result.fun)
        except _OptimizationTimeout as exc:
            timed_out = True
            final_flat = exc.x
            final_fun = self.cost_only(final_flat)
            _, jac = self.objective(final_flat)
            status = 9
            message = "STOP: OPTIMIZATION TIME BUDGET EXCEEDED"
            iterations = -1
            function_evaluations = -1
            optimizer_success = False
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        final_points = final_flat.reshape(self.inner_shape)
        final_trajectory = self._trajectory(final_points)
        final_risk = self.evaluate_risk(final_trajectory, with_gradient=False)
        final_energy = final_trajectory.energy()
        penalty, _, _, _ = self._violations(final_trajectory)
        return DynamicRiskOptimizationResult(
            success=optimizer_success and not timed_out,
            status=status,
            message=message,
            trajectory=final_trajectory,
            p_inner=final_points,
            durations=self.durations.copy(),
            initial_cost=float(initial_cost),
            final_cost=float(final_fun),
            initial_energy=float(initial_energy),
            final_energy=float(final_energy),
            initial_risk=float(initial_risk.cost),
            final_risk=float(final_risk.cost),
            initial_min_distance=float(initial_risk.min_distance),
            final_min_distance=float(final_risk.min_distance),
            penalty_cost=float(penalty),
            iterations=iterations,
            function_evaluations=function_evaluations,
            gradient_norm=float(np.linalg.norm(jac)),
            elapsed_ms=float(elapsed_ms),
        )
