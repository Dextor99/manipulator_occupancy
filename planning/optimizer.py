"""Fixed-time waypoint optimization for the first CCRO-NUBS stage."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from scipy.optimize import minimize

from .nubs_trajectory import DIMENSION, NUBSTrajectory6D


@dataclass(frozen=True)
class JointLimits:
    q_min: np.ndarray
    q_max: np.ndarray
    qd_max: np.ndarray
    qdd_max: np.ndarray

    @classmethod
    def from_arrays(cls, q_min, q_max, qd_max, qdd_max) -> "JointLimits":
        arrays = [np.asarray(v, dtype=np.float64) for v in (q_min, q_max, qd_max, qdd_max)]
        if any(v.shape != (DIMENSION,) for v in arrays):
            raise ValueError("all joint-limit arrays must have shape (6,)")
        if not all(np.all(np.isfinite(v)) for v in arrays):
            raise ValueError("joint limits must be finite")
        if np.any(arrays[0] >= arrays[1]):
            raise ValueError("q_min must be smaller than q_max")
        if np.any(arrays[2] <= 0.0) or np.any(arrays[3] <= 0.0):
            raise ValueError("velocity and acceleration limits must be positive")
        return cls(*(v.copy() for v in arrays))


@dataclass
class NUBSOptimizationResult:
    success: bool
    status: int
    message: str
    trajectory: NUBSTrajectory6D
    p_inner: np.ndarray
    durations: np.ndarray
    initial_energy: float
    final_energy: float
    final_cost: float
    penalty_cost: float
    iterations: int
    function_evaluations: int
    gradient_norm: float
    elapsed_ms: float
    max_q_violation: float
    max_qd_violation: float
    max_qdd_violation: float


class FixedTimeNUBSOptimizer:
    """Optimize interpolation configurations while keeping durations fixed.

    The NUBS endpoint states are hard constraints.  There is deliberately no
    goal penalty in this stage-one objective.
    """

    def __init__(
        self,
        head_state: np.ndarray,
        tail_state: np.ndarray,
        durations: np.ndarray,
        joint_limits: JointLimits,
        *,
        lambda_smooth: float = 1.0,
        lambda_position: float = 10.0,
        lambda_velocity: float = 10.0,
        lambda_acceleration: float = 10.0,
        samples_per_segment: int = 12,
        finite_difference_epsilon: float = 1.0e-6,
        max_iterations: int = 200,
        gradient_tolerance: float = 1.0e-7,
    ) -> None:
        self.head_state = np.asarray(head_state, dtype=np.float64)
        self.tail_state = np.asarray(tail_state, dtype=np.float64)
        self.durations = np.asarray(durations, dtype=np.float64)
        if self.head_state.shape != (DIMENSION, 3) or self.tail_state.shape != (DIMENSION, 3):
            raise ValueError("head_state and tail_state must have shape (6, 3)")
        if self.durations.ndim != 1 or len(self.durations) == 0 or np.any(self.durations <= 1.0e-8):
            raise ValueError("durations must be a positive one-dimensional array")
        self.limits = joint_limits
        self.lambda_smooth = float(lambda_smooth)
        self.lambda_position = float(lambda_position)
        self.lambda_velocity = float(lambda_velocity)
        self.lambda_acceleration = float(lambda_acceleration)
        self.samples_per_segment = max(int(samples_per_segment), 2)
        self.fd_epsilon = float(finite_difference_epsilon)
        self.max_iterations = int(max_iterations)
        self.gradient_tolerance = float(gradient_tolerance)
        if self.lambda_smooth <= 0.0:
            raise ValueError("lambda_smooth must be positive")
        if self.fd_epsilon <= 0.0:
            raise ValueError("finite_difference_epsilon must be positive")
        self._sample_times = self._make_sample_times()

    @property
    def inner_shape(self) -> tuple[int, int]:
        return (len(self.durations) - 1, DIMENSION)

    def _make_sample_times(self) -> np.ndarray:
        chunks: list[np.ndarray] = []
        start = 0.0
        for index, duration in enumerate(self.durations):
            endpoint = start + float(duration)
            segment = np.linspace(start, endpoint, self.samples_per_segment + 1)
            chunks.append(segment if index == 0 else segment[1:])
            start = endpoint
        return np.concatenate(chunks)

    def _trajectory(self, p_inner: np.ndarray) -> NUBSTrajectory6D:
        return NUBSTrajectory6D().generate(
            p_inner, self.head_state, self.tail_state, self.durations
        )

    def _violations(self, trajectory: NUBSTrajectory6D) -> tuple[float, float, float, float]:
        samples = trajectory.sample(self._sample_times)
        q_low = np.maximum(self.limits.q_min[None, :] - samples.q, 0.0)
        q_high = np.maximum(samples.q - self.limits.q_max[None, :], 0.0)
        qd = np.maximum(np.abs(samples.qd) - self.limits.qd_max[None, :], 0.0)
        qdd = np.maximum(np.abs(samples.qdd) - self.limits.qdd_max[None, :], 0.0)

        q_integrand = np.sum(q_low * q_low + q_high * q_high, axis=1)
        qd_integrand = np.sum(qd * qd, axis=1)
        qdd_integrand = np.sum(qdd * qdd, axis=1)
        q_cost = float(np.trapz(q_integrand, self._sample_times))
        qd_cost = float(np.trapz(qd_integrand, self._sample_times))
        qdd_cost = float(np.trapz(qdd_integrand, self._sample_times))
        weighted = (
            self.lambda_position * q_cost
            + self.lambda_velocity * qd_cost
            + self.lambda_acceleration * qdd_cost
        )
        return weighted, float(np.max(np.maximum(q_low, q_high))), float(np.max(qd)), float(np.max(qdd))

    def _penalty_only(self, p_inner: np.ndarray) -> float:
        return self._violations(self._trajectory(p_inner))[0]

    def _penalty_gradient(self, p_inner: np.ndarray, penalty: float) -> np.ndarray:
        if penalty == 0.0 or p_inner.size == 0:
            return np.zeros_like(p_inner)
        gradient = np.zeros_like(p_inner)
        for row in range(p_inner.shape[0]):
            for col in range(p_inner.shape[1]):
                plus = p_inner.copy()
                minus = p_inner.copy()
                plus[row, col] += self.fd_epsilon
                minus[row, col] -= self.fd_epsilon
                gradient[row, col] = (
                    self._penalty_only(plus) - self._penalty_only(minus)
                ) / (2.0 * self.fd_epsilon)
        return gradient

    def objective(self, flat_points: np.ndarray) -> tuple[float, np.ndarray]:
        points = np.asarray(flat_points, dtype=np.float64).reshape(self.inner_shape)
        trajectory = self._trajectory(points)
        energy, energy_gradient, _ = trajectory.energy_and_gradient()
        penalty, _, _, _ = self._violations(trajectory)
        penalty_gradient = self._penalty_gradient(points, penalty)
        cost = self.lambda_smooth * energy + penalty
        gradient = self.lambda_smooth * energy_gradient + penalty_gradient
        if not np.isfinite(cost) or not np.all(np.isfinite(gradient)):
            raise FloatingPointError("stage-one objective produced NaN or Inf")
        return float(cost), gradient.ravel()

    def optimize(self, p_inner_initial: np.ndarray | None = None) -> NUBSOptimizationResult:
        if p_inner_initial is None:
            p_inner_initial = NUBSTrajectory6D.linear_inner_points(
                self.head_state[:, 0], self.tail_state[:, 0], self.durations
            )
        initial = np.asarray(p_inner_initial, dtype=np.float64)
        if initial.shape != self.inner_shape:
            raise ValueError(f"p_inner_initial must have shape {self.inner_shape}")
        initial_trajectory = self._trajectory(initial)
        initial_energy = initial_trajectory.energy()
        bounds = [
            (float(self.limits.q_min[j]), float(self.limits.q_max[j]))
            for _ in range(self.inner_shape[0])
            for j in range(DIMENSION)
        ]

        start = time.perf_counter()
        result = minimize(
            self.objective,
            initial.ravel(),
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={
                "maxiter": self.max_iterations,
                "gtol": self.gradient_tolerance,
                "ftol": 1.0e-12,
                "maxls": 30,
            },
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        final_points = np.asarray(result.x, dtype=np.float64).reshape(self.inner_shape)
        final_trajectory = self._trajectory(final_points)
        final_energy = final_trajectory.energy()
        penalty, q_v, qd_v, qdd_v = self._violations(final_trajectory)
        gradient_norm = float(np.linalg.norm(np.asarray(result.jac, dtype=np.float64)))
        success = bool(result.success) and np.isfinite(result.fun)
        return NUBSOptimizationResult(
            success=success,
            status=int(result.status),
            message=str(result.message),
            trajectory=final_trajectory,
            p_inner=final_points,
            durations=self.durations.copy(),
            initial_energy=float(initial_energy),
            final_energy=float(final_energy),
            final_cost=float(result.fun),
            penalty_cost=float(penalty),
            iterations=int(result.nit),
            function_evaluations=int(result.nfev),
            gradient_norm=gradient_norm,
            elapsed_ms=float(elapsed_ms),
            max_q_violation=q_v,
            max_qd_violation=qd_v,
            max_qdd_violation=qdd_v,
        )
