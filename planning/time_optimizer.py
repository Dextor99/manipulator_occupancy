"""Joint waypoint and physical-time optimization for CCRO-NUBS P2."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from scipy.optimize import minimize

from .nubs_trajectory import DIMENSION, NUBSTrajectory6D
from .optimizer import JointLimits


@dataclass
class TimeOptimizationResult:
    success: bool
    message: str
    mode: str
    trajectory: NUBSTrajectory6D
    p_inner: np.ndarray
    durations: np.ndarray
    initial_cost: float
    final_cost: float
    final_energy: float
    time_cost: float
    penalty_cost: float
    iterations: int
    function_evaluations: int
    gradient_norm: float
    elapsed_ms: float
    max_q_violation: float
    max_qd_violation: float
    max_qdd_violation: float


class VariableTimeNUBSOptimizer:
    """Optimize NUBS interpolation points and either total or segment times.

    ``total`` keeps the initial duration ratios fixed and optimizes one log total
    duration. ``segment`` optimizes one log duration per segment. Log variables
    guarantee strictly positive physical durations; L-BFGS-B bounds provide the
    configured hard time interval.
    """

    def __init__(
        self,
        head_state: np.ndarray,
        tail_state: np.ndarray,
        durations: np.ndarray,
        joint_limits: JointLimits,
        *,
        mode: str = "total",
        lambda_smooth: float = 1.0,
        lambda_time: float = 0.05,
        lambda_position: float = 10.0,
        lambda_velocity: float = 10.0,
        lambda_acceleration: float = 10.0,
        samples_per_segment: int = 12,
        finite_difference_epsilon: float = 1.0e-6,
        min_total_duration: float = 3.0,
        max_total_duration: float = 10.0,
        min_segment_duration: float = 0.25,
        max_segment_duration: float = 4.0,
        max_iterations: int = 200,
        gradient_tolerance: float = 1.0e-7,
    ) -> None:
        self.head_state = np.asarray(head_state, dtype=np.float64)
        self.tail_state = np.asarray(tail_state, dtype=np.float64)
        self.initial_durations = np.asarray(durations, dtype=np.float64)
        if self.head_state.shape != (DIMENSION, 3) or self.tail_state.shape != (DIMENSION, 3):
            raise ValueError("head_state and tail_state must have shape (6, 3)")
        if self.initial_durations.ndim != 1 or np.any(self.initial_durations <= 0.0):
            raise ValueError("durations must be a positive vector")
        if mode not in {"total", "segment"}:
            raise ValueError("mode must be 'total' or 'segment'")
        self.mode = mode
        self.limits = joint_limits
        self.lambda_smooth = float(lambda_smooth)
        self.lambda_time = float(lambda_time)
        self.lambda_position = float(lambda_position)
        self.lambda_velocity = float(lambda_velocity)
        self.lambda_acceleration = float(lambda_acceleration)
        self.samples_per_segment = max(2, int(samples_per_segment))
        self.fd_epsilon = float(finite_difference_epsilon)
        self.min_total = float(min_total_duration)
        self.max_total = float(max_total_duration)
        self.min_segment = float(min_segment_duration)
        self.max_segment = float(max_segment_duration)
        self.max_iterations = int(max_iterations)
        self.gradient_tolerance = float(gradient_tolerance)
        if not 0.0 < self.min_total < self.max_total:
            raise ValueError("invalid total-duration bounds")
        if not 0.0 < self.min_segment < self.max_segment:
            raise ValueError("invalid segment-duration bounds")
        if self.lambda_smooth <= 0.0 or self.lambda_time <= 0.0 or self.fd_epsilon <= 0.0:
            raise ValueError("weights and finite-difference epsilon must be positive")
        self._ratios = self.initial_durations / np.sum(self.initial_durations)

    @property
    def inner_shape(self) -> tuple[int, int]:
        return (len(self.initial_durations) - 1, DIMENSION)

    @property
    def time_variable_count(self) -> int:
        return 1 if self.mode == "total" else len(self.initial_durations)

    def encode(self, points: np.ndarray, durations: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64).reshape(self.inner_shape)
        durations = np.asarray(durations, dtype=np.float64)
        if durations.shape != self.initial_durations.shape or np.any(durations <= 0.0):
            raise ValueError("durations have an invalid shape or value")
        time_vars = [np.log(np.sum(durations))] if self.mode == "total" else np.log(durations)
        return np.concatenate((points.ravel(), np.asarray(time_vars)))

    def decode(self, variables: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        variables = np.asarray(variables, dtype=np.float64)
        point_count = int(np.prod(self.inner_shape))
        if variables.shape != (point_count + self.time_variable_count,):
            raise ValueError("optimization variable vector has an invalid shape")
        points = variables[:point_count].reshape(self.inner_shape)
        if self.mode == "total":
            durations = self._ratios * np.exp(variables[-1])
        else:
            durations = np.exp(variables[point_count:])
        return points, durations

    def _trajectory(self, points: np.ndarray, durations: np.ndarray) -> NUBSTrajectory6D:
        return NUBSTrajectory6D().generate(points, self.head_state, self.tail_state, durations)

    def _sample_times(self, durations: np.ndarray) -> np.ndarray:
        chunks, start = [], 0.0
        for index, duration in enumerate(durations):
            stop = start + float(duration)
            values = np.linspace(start, stop, self.samples_per_segment + 1)
            chunks.append(values if index == 0 else values[1:])
            start = stop
        times = np.concatenate(chunks)
        # Avoid a one-ulp cumulative-sum overshoot at the physical endpoint.
        times[-1] = float(np.sum(durations))
        return times

    def _penalty(self, points: np.ndarray, durations: np.ndarray) -> tuple[float, float, float, float]:
        trajectory = self._trajectory(points, durations)
        sample_times = self._sample_times(durations)
        sample_times[-1] = trajectory.total_duration
        samples = trajectory.sample(sample_times)
        q_low = np.maximum(self.limits.q_min - samples.q, 0.0)
        q_high = np.maximum(samples.q - self.limits.q_max, 0.0)
        qd = np.maximum(np.abs(samples.qd) - self.limits.qd_max, 0.0)
        qdd = np.maximum(np.abs(samples.qdd) - self.limits.qdd_max, 0.0)
        q_cost = np.trapz(np.sum(q_low**2 + q_high**2, axis=1), sample_times)
        qd_cost = np.trapz(np.sum(qd**2, axis=1), sample_times)
        qdd_cost = np.trapz(np.sum(qdd**2, axis=1), sample_times)
        cost = self.lambda_position*q_cost + self.lambda_velocity*qd_cost + self.lambda_acceleration*qdd_cost
        return float(cost), float(np.max(np.maximum(q_low, q_high))), float(np.max(qd)), float(np.max(qdd))

    def _penalty_gradient(self, variables: np.ndarray, penalty: float) -> np.ndarray:
        if penalty == 0.0:
            return np.zeros_like(variables)
        gradient = np.zeros_like(variables)
        for index in range(len(variables)):
            plus, minus = variables.copy(), variables.copy()
            plus[index] += self.fd_epsilon
            minus[index] -= self.fd_epsilon
            pp, pd = self.decode(plus)
            mp, md = self.decode(minus)
            gradient[index] = (self._penalty(pp, pd)[0] - self._penalty(mp, md)[0]) / (2*self.fd_epsilon)
        return gradient

    def objective(self, variables: np.ndarray) -> tuple[float, np.ndarray]:
        points, durations = self.decode(variables)
        trajectory = self._trajectory(points, durations)
        energy, grad_points, grad_durations = trajectory.energy_and_gradient_full()
        penalty = self._penalty(points, durations)[0]
        gradient = np.zeros_like(variables)
        point_count = points.size
        gradient[:point_count] = (self.lambda_smooth * grad_points).ravel()
        if self.mode == "total":
            gradient[-1] = self.lambda_smooth * float(np.dot(grad_durations, durations)) + self.lambda_time * float(np.sum(durations))
        else:
            gradient[point_count:] = self.lambda_smooth * grad_durations * durations + self.lambda_time * durations
        gradient += self._penalty_gradient(variables, penalty)
        cost = self.lambda_smooth*energy + self.lambda_time*float(np.sum(durations)) + penalty
        if not np.isfinite(cost) or not np.all(np.isfinite(gradient)):
            raise FloatingPointError("variable-time objective produced NaN or Inf")
        return float(cost), gradient

    def bounds(self) -> list[tuple[float, float]]:
        point_bounds = [(float(self.limits.q_min[j]), float(self.limits.q_max[j])) for _ in range(self.inner_shape[0]) for j in range(DIMENSION)]
        if self.mode == "total":
            time_bounds = [(np.log(self.min_total), np.log(self.max_total))]
        else:
            time_bounds = [(np.log(self.min_segment), np.log(self.max_segment))] * len(self.initial_durations)
        return point_bounds + time_bounds

    def optimize(self, p_inner_initial: np.ndarray | None = None) -> TimeOptimizationResult:
        if p_inner_initial is None:
            p_inner_initial = NUBSTrajectory6D.linear_inner_points(self.head_state[:, 0], self.tail_state[:, 0], self.initial_durations)
        initial = self.encode(p_inner_initial, self.initial_durations)
        initial_cost = self.objective(initial)[0]
        start = time.perf_counter()
        result = minimize(self.objective, initial, method="L-BFGS-B", jac=True, bounds=self.bounds(), options={"maxiter": self.max_iterations, "gtol": self.gradient_tolerance, "ftol": 1e-12, "maxls": 40})
        elapsed_ms = 1000.0 * (time.perf_counter() - start)
        points, durations = self.decode(result.x)
        trajectory = self._trajectory(points, durations)
        energy = trajectory.energy()
        penalty, qv, qdv, qddv = self._penalty(points, durations)
        return TimeOptimizationResult(bool(result.success) and np.isfinite(result.fun), str(result.message), self.mode, trajectory, points, durations, float(initial_cost), float(result.fun), float(energy), self.lambda_time*float(np.sum(durations)), penalty, int(result.nit), int(result.nfev), float(np.linalg.norm(result.jac)), float(elapsed_ms), qv, qdv, qddv)
