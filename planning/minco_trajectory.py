"""Joint-space minimum-control polynomial trajectory baselines.

This module implements the MINCO idea used in GCOPTER-style trajectory
optimization in a form that matches this repository's 6-DOF joint-space
experiments: segment durations and intermediate positions parameterize a
piecewise quintic trajectory, while intermediate velocities and accelerations
are eliminated by minimizing the integrated squared jerk.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .nubs_trajectory import DIMENSION, TrajectorySamples


@dataclass(frozen=True)
class MincoBoundaryState:
    q: np.ndarray
    qd: np.ndarray
    qdd: np.ndarray


def _as_array(value, shape: tuple[int | None, ...], name: str) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=np.float64)
    if array.ndim != len(shape):
        raise ValueError(f"{name} must have {len(shape)} dimensions, got {array.shape}")
    for axis, expected in enumerate(shape):
        if expected is not None and array.shape[axis] != expected:
            raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or Inf")
    return array


def _quintic_coefficients(
    q0: np.ndarray,
    v0: np.ndarray,
    acc0: np.ndarray,
    q1: np.ndarray,
    v1: np.ndarray,
    acc1: np.ndarray,
    duration: float,
) -> np.ndarray:
    t = float(duration)
    if t <= 1.0e-9:
        raise ValueError("segment duration must be positive")
    coeff = np.empty((DIMENSION, 6), dtype=np.float64)
    coeff[:, 0] = q0
    coeff[:, 1] = v0
    coeff[:, 2] = 0.5 * acc0
    rhs0 = q1 - (coeff[:, 0] + coeff[:, 1] * t + coeff[:, 2] * t**2)
    rhs1 = v1 - (coeff[:, 1] + 2.0 * coeff[:, 2] * t)
    rhs2 = acc1 - (2.0 * coeff[:, 2])
    matrix = np.array(
        [
            [t**3, t**4, t**5],
            [3.0 * t**2, 4.0 * t**3, 5.0 * t**4],
            [6.0 * t, 12.0 * t**2, 20.0 * t**3],
        ],
        dtype=np.float64,
    )
    coeff[:, 3:6] = np.linalg.solve(matrix, np.column_stack((rhs0, rhs1, rhs2)).T).T
    return coeff


def _jerk_energy_from_coefficients(coeff: np.ndarray, duration: float) -> float:
    c3 = coeff[:, 3]
    c4 = coeff[:, 4]
    c5 = coeff[:, 5]
    t = float(duration)
    value = (
        36.0 * np.sum(c3 * c3) * t
        + 144.0 * np.sum(c3 * c4) * t**2
        + (240.0 * np.sum(c3 * c5) + 192.0 * np.sum(c4 * c4)) * t**3
        + 720.0 * np.sum(c4 * c5) * t**4
        + 720.0 * np.sum(c5 * c5) * t**5
    )
    return float(value)


class MinJerkMINCOTrajectory6D:
    """Piecewise quintic minimum-jerk trajectory with sparse knot positions."""

    def __init__(
        self,
        positions: np.ndarray,
        head_state: np.ndarray,
        tail_state: np.ndarray,
        durations: np.ndarray,
        *,
        derivative_tolerance: float = 1.0e-9,
    ) -> None:
        self.positions = _as_array(positions, (None, DIMENSION), "positions")
        self.durations = _as_array(durations, (None,), "durations")
        if len(self.positions) != len(self.durations) + 1:
            raise ValueError("positions must contain one more row than durations")
        if np.any(self.durations <= 1.0e-8):
            raise ValueError("durations must be positive")
        head = _as_array(head_state, (DIMENSION, 3), "head_state")
        tail = _as_array(tail_state, (DIMENSION, 3), "tail_state")
        if np.linalg.norm(self.positions[0] - head[:, 0]) > 1.0e-9:
            raise ValueError("first position must match head state")
        if np.linalg.norm(self.positions[-1] - tail[:, 0]) > 1.0e-9:
            raise ValueError("last position must match tail state")
        self.head_state = head.copy()
        self.tail_state = tail.copy()
        self._knot_times = np.concatenate(([0.0], np.cumsum(self.durations)))
        self._velocities, self._accelerations = self._solve_internal_derivatives(
            derivative_tolerance
        )
        self._coefficients = self._build_coefficients()

    @property
    def total_duration(self) -> float:
        return float(self._knot_times[-1])

    @property
    def piece_count(self) -> int:
        return len(self.durations)

    @property
    def inner_points(self) -> np.ndarray:
        return self.positions[1:-1].copy()

    @staticmethod
    def from_inner_points(
        p_inner: np.ndarray,
        head_state: np.ndarray,
        tail_state: np.ndarray,
        durations: np.ndarray,
    ) -> "MinJerkMINCOTrajectory6D":
        inner = _as_array(p_inner, (len(durations) - 1, DIMENSION), "p_inner")
        positions = np.vstack((head_state[:, 0], inner, tail_state[:, 0]))
        return MinJerkMINCOTrajectory6D(positions, head_state, tail_state, durations)

    def _derivative_vector_to_states(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        velocities = np.zeros_like(self.positions)
        accelerations = np.zeros_like(self.positions)
        velocities[0] = self.head_state[:, 1]
        accelerations[0] = self.head_state[:, 2]
        velocities[-1] = self.tail_state[:, 1]
        accelerations[-1] = self.tail_state[:, 2]
        if len(self.positions) > 2:
            split = (len(self.positions) - 2) * DIMENSION
            velocities[1:-1] = values[:split].reshape((-1, DIMENSION))
            accelerations[1:-1] = values[split:].reshape((-1, DIMENSION))
        return velocities, accelerations

    def _energy_for_derivatives(self, values: np.ndarray) -> float:
        velocities, accelerations = self._derivative_vector_to_states(values)
        total = 0.0
        for i, duration in enumerate(self.durations):
            coeff = _quintic_coefficients(
                self.positions[i],
                velocities[i],
                accelerations[i],
                self.positions[i + 1],
                velocities[i + 1],
                accelerations[i + 1],
                float(duration),
            )
            total += _jerk_energy_from_coefficients(coeff, float(duration))
        return float(total)

    def _solve_internal_derivatives(self, tolerance: float) -> tuple[np.ndarray, np.ndarray]:
        variable_count = max(0, len(self.positions) - 2) * DIMENSION * 2
        if variable_count == 0:
            return self._derivative_vector_to_states(np.empty(0, dtype=np.float64))
        durations_key = tuple(float(f"{value:.12g}") for value in self.durations)
        hessian = _cached_internal_hessian(durations_key)
        gradient_at_zero = self._derivative_gradient(
            np.zeros(variable_count, dtype=np.float64)
        )
        regularized = hessian + max(tolerance, 1.0e-10) * np.eye(variable_count)
        solution = np.linalg.solve(regularized, -gradient_at_zero)
        return self._derivative_vector_to_states(solution)

    def _derivative_gradient(self, values: np.ndarray, epsilon: float = 1.0e-6) -> np.ndarray:
        gradient = np.zeros_like(values)
        for i in range(values.size):
            plus = values.copy()
            minus = values.copy()
            plus[i] += epsilon
            minus[i] -= epsilon
            gradient[i] = (self._energy_for_derivatives(plus) - self._energy_for_derivatives(minus)) / (
                2.0 * epsilon
            )
        return gradient

    def _build_coefficients(self) -> np.ndarray:
        coefficients = []
        for i, duration in enumerate(self.durations):
            coefficients.append(
                _quintic_coefficients(
                    self.positions[i],
                    self._velocities[i],
                    self._accelerations[i],
                    self.positions[i + 1],
                    self._velocities[i + 1],
                    self._accelerations[i + 1],
                    float(duration),
                )
            )
        return np.asarray(coefficients, dtype=np.float64)

    def energy(self) -> float:
        return float(
            sum(
                _jerk_energy_from_coefficients(coeff, float(duration))
                for coeff, duration in zip(self._coefficients, self.durations)
            )
        )

    def _segment_index(self, time: float) -> tuple[int, float]:
        t = float(np.clip(time, 0.0, self.total_duration))
        if t >= self.total_duration:
            index = len(self.durations) - 1
        else:
            index = int(np.searchsorted(self._knot_times, t, side="right") - 1)
            index = min(max(index, 0), len(self.durations) - 1)
        return index, t - float(self._knot_times[index])

    def evaluate(self, time: float, derivative_order: int = 0) -> np.ndarray:
        if not 0 <= derivative_order <= 3:
            raise ValueError("derivative_order must be in [0, 3]")
        index, local = self._segment_index(time)
        coeff = self._coefficients[index]
        if derivative_order == 0:
            powers = np.array([1.0, local, local**2, local**3, local**4, local**5])
            return coeff @ powers
        if derivative_order == 1:
            powers = np.array([1.0, local, local**2, local**3, local**4])
            return coeff[:, 1:6] @ (np.array([1.0, 2.0, 3.0, 4.0, 5.0]) * powers)
        if derivative_order == 2:
            powers = np.array([1.0, local, local**2, local**3])
            return coeff[:, 2:6] @ (np.array([2.0, 6.0, 12.0, 20.0]) * powers)
        powers = np.array([1.0, local, local**2])
        return coeff[:, 3:6] @ (np.array([6.0, 24.0, 60.0]) * powers)

    def sample(self, times: np.ndarray, max_derivative: int = 3) -> TrajectorySamples:
        sample_times = _as_array(times, (None,), "times")
        if np.any(sample_times < -1.0e-10) or np.any(sample_times > self.total_duration + 1.0e-10):
            raise ValueError("sample times must lie in [0, total_duration]")
        q = np.vstack([self.evaluate(t, 0) for t in sample_times])
        qd = np.vstack([self.evaluate(t, 1) for t in sample_times])
        qdd = np.vstack([self.evaluate(t, 2) for t in sample_times])
        jerk = np.vstack([self.evaluate(t, 3) for t in sample_times])
        return TrajectorySamples(sample_times.copy(), q, qd, qdd, jerk)

    def dense_sample(self, time_step: float = 0.01) -> TrajectorySamples:
        count = max(2, int(np.ceil(self.total_duration / time_step)) + 1)
        return self.sample(np.linspace(0.0, self.total_duration, count))


@lru_cache(maxsize=32)
def _cached_internal_hessian(durations_key: tuple[float, ...]) -> np.ndarray:
    durations = np.asarray(durations_key, dtype=np.float64)
    knot_count = len(durations) + 1
    variable_count = max(0, knot_count - 2) * DIMENSION * 2
    if variable_count == 0:
        return np.empty((0, 0), dtype=np.float64)
    positions = np.zeros((knot_count, DIMENSION), dtype=np.float64)
    boundary = np.zeros((DIMENSION, 3), dtype=np.float64)
    reference = MinJerkMINCOTrajectory6D.__new__(MinJerkMINCOTrajectory6D)
    reference.positions = positions
    reference.durations = durations
    reference.head_state = boundary
    reference.tail_state = boundary

    def gradient(values: np.ndarray, epsilon: float = 1.0e-6) -> np.ndarray:
        grad = np.zeros_like(values)
        for i in range(values.size):
            plus = values.copy()
            minus = values.copy()
            plus[i] += epsilon
            minus[i] -= epsilon
            grad[i] = (
                reference._energy_for_derivatives(plus)
                - reference._energy_for_derivatives(minus)
            ) / (2.0 * epsilon)
        return grad

    hessian = np.zeros((variable_count, variable_count), dtype=np.float64)
    step = 1.0e-4
    zeros = np.zeros(variable_count, dtype=np.float64)
    for j in range(variable_count):
        plus = zeros.copy()
        minus = zeros.copy()
        plus[j] += step
        minus[j] -= step
        hessian[:, j] = (gradient(plus) - gradient(minus)) / (2.0 * step)
    return 0.5 * (hessian + hessian.T)
