"""Python-facing 6-DOF minimum-jerk NUBS trajectory API."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from . import _nubs_cpp
except ImportError as exc:  # pragma: no cover - exercised before the extension is built
    _EXTENSION_ERROR = exc
    _nubs_cpp = None
else:
    _EXTENSION_ERROR = None


DIMENSION = 6
SYSTEM_ORDER = 3
MAX_DERIVATIVE = 3


@dataclass(frozen=True)
class TrajectorySamples:
    """Batch samples of one physical-time trajectory."""

    times: np.ndarray
    q: np.ndarray
    qd: np.ndarray
    qdd: np.ndarray
    jerk: np.ndarray


def _as_finite_array(value, shape: tuple[int | None, ...], name: str) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=np.float64)
    if array.ndim != len(shape):
        raise ValueError(f"{name} must have {len(shape)} dimensions, got {array.shape}")
    for axis, expected in enumerate(shape):
        if expected is not None and array.shape[axis] != expected:
            raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or Inf")
    return array


class NUBSTrajectory6D:
    """Validated wrapper around ``nubs::QuinticNUBS<6>``.

    ``p_inner`` contains interpolation configurations at internal segment
    boundaries.  It does not contain B-spline control points.
    """

    def __init__(self) -> None:
        if _nubs_cpp is None:
            root = Path(__file__).resolve().parents[1]
            raise ImportError(
                "CCRO-NUBS extension is not built. Run "
                f"`bash {root / 'scripts' / 'build_ccro_stage1.sh'}` first."
            ) from _EXTENSION_ERROR
        self._impl = _nubs_cpp.NUBSTrajectory6D()
        self._generated = False
        self._p_inner = np.empty((0, DIMENSION), dtype=np.float64)
        self._head_state = np.empty((DIMENSION, SYSTEM_ORDER), dtype=np.float64)
        self._tail_state = np.empty((DIMENSION, SYSTEM_ORDER), dtype=np.float64)
        self._durations = np.empty(0, dtype=np.float64)

    @staticmethod
    def make_boundary_state(
        q: np.ndarray,
        qd: np.ndarray | None = None,
        qdd: np.ndarray | None = None,
    ) -> np.ndarray:
        q_array = _as_finite_array(q, (DIMENSION,), "q")
        qd_array = np.zeros(DIMENSION) if qd is None else _as_finite_array(
            qd, (DIMENSION,), "qd"
        )
        qdd_array = np.zeros(DIMENSION) if qdd is None else _as_finite_array(
            qdd, (DIMENSION,), "qdd"
        )
        return np.column_stack((q_array, qd_array, qdd_array))

    @staticmethod
    def linear_inner_points(
        q_start: np.ndarray,
        q_goal: np.ndarray,
        durations: np.ndarray,
    ) -> np.ndarray:
        q0 = _as_finite_array(q_start, (DIMENSION,), "q_start")
        q1 = _as_finite_array(q_goal, (DIMENSION,), "q_goal")
        times = _as_finite_array(durations, (None,), "durations")
        if len(times) == 0 or np.any(times <= 1.0e-8):
            raise ValueError("durations must be non-empty and greater than 1e-8")
        if len(times) == 1:
            return np.empty((0, DIMENSION), dtype=np.float64)
        ratios = np.cumsum(times)[:-1] / np.sum(times)
        return q0[None, :] + ratios[:, None] * (q1 - q0)[None, :]

    def generate(
        self,
        p_inner: np.ndarray,
        head_state: np.ndarray,
        tail_state: np.ndarray,
        durations: np.ndarray,
    ) -> "NUBSTrajectory6D":
        times = _as_finite_array(durations, (None,), "durations")
        if len(times) == 0 or np.any(times <= 1.0e-8):
            raise ValueError("durations must be non-empty and greater than 1e-8")
        inner = _as_finite_array(p_inner, (len(times) - 1, DIMENSION), "p_inner")
        head = _as_finite_array(
            head_state, (DIMENSION, SYSTEM_ORDER), "head_state"
        )
        tail = _as_finite_array(
            tail_state, (DIMENSION, SYSTEM_ORDER), "tail_state"
        )
        self._impl.generate(inner, head, tail, times)
        self._generated = True
        self._p_inner = inner.copy()
        self._head_state = head.copy()
        self._tail_state = tail.copy()
        self._durations = times.copy()
        return self

    def _require_generated(self) -> None:
        if not self._generated:
            raise RuntimeError("trajectory has not been generated")

    @property
    def piece_count(self) -> int:
        self._require_generated()
        return len(self._durations)

    @property
    def total_duration(self) -> float:
        self._require_generated()
        return float(self._impl.total_duration)

    @property
    def durations(self) -> np.ndarray:
        self._require_generated()
        return self._durations.copy()

    @property
    def inner_points(self) -> np.ndarray:
        self._require_generated()
        return self._p_inner.copy()

    @property
    def control_points(self) -> np.ndarray:
        self._require_generated()
        return np.asarray(self._impl.control_points, dtype=np.float64).copy()

    @property
    def head_state(self) -> np.ndarray:
        self._require_generated()
        return self._head_state.copy()

    @property
    def tail_state(self) -> np.ndarray:
        self._require_generated()
        return self._tail_state.copy()

    def evaluate(self, time: float, derivative_order: int = 0) -> np.ndarray:
        self._require_generated()
        if not np.isfinite(time):
            raise ValueError("time must be finite")
        if not 0 <= derivative_order <= 5:
            raise ValueError("derivative_order must be in [0, 5]")
        return np.asarray(
            self._impl.evaluate(float(time), int(derivative_order)), dtype=np.float64
        )

    def sample(
        self,
        times: np.ndarray,
        max_derivative: int = MAX_DERIVATIVE,
    ) -> TrajectorySamples:
        self._require_generated()
        sample_times = _as_finite_array(times, (None,), "times")
        if np.any(sample_times < 0.0) or np.any(sample_times > self.total_duration):
            raise ValueError("sample times must lie in [0, total_duration]")
        if not 0 <= max_derivative <= MAX_DERIVATIVE:
            raise ValueError(f"max_derivative must be in [0, {MAX_DERIVATIVE}]")
        q, qd, qdd, jerk = self._impl.sample(sample_times, max_derivative)
        return TrajectorySamples(
            times=sample_times.copy(),
            q=np.asarray(q, dtype=np.float64),
            qd=np.asarray(qd, dtype=np.float64),
            qdd=np.asarray(qdd, dtype=np.float64),
            jerk=np.asarray(jerk, dtype=np.float64),
        )

    def dense_sample(self, time_step: float = 0.01) -> TrajectorySamples:
        self._require_generated()
        if not np.isfinite(time_step) or time_step <= 0.0:
            raise ValueError("time_step must be positive and finite")
        count = max(2, int(np.ceil(self.total_duration / time_step)) + 1)
        return self.sample(np.linspace(0.0, self.total_duration, count))

    def energy(self) -> float:
        self._require_generated()
        return float(self._impl.energy())

    def energy_and_gradient(self) -> tuple[float, np.ndarray, np.ndarray]:
        self._require_generated()
        energy, grad_points, grad_times = self._impl.energy_and_gradient()
        return (
            float(energy),
            np.asarray(grad_points, dtype=np.float64),
            np.asarray(grad_times, dtype=np.float64),
        )

    def energy_and_gradient_full(self) -> tuple[float, np.ndarray, np.ndarray]:
        self._require_generated()
        energy, grad_points, grad_times = self._impl.energy_and_gradient_full()
        return (
            float(energy),
            np.asarray(grad_points, dtype=np.float64),
            np.asarray(grad_times, dtype=np.float64),
        )

    def boundary_errors(self) -> dict[str, float]:
        self._require_generated()
        end_time = self.total_duration
        return {
            "q_start": float(np.linalg.norm(self.evaluate(0.0, 0) - self._head_state[:, 0])),
            "qd_start": float(np.linalg.norm(self.evaluate(0.0, 1) - self._head_state[:, 1])),
            "qdd_start": float(np.linalg.norm(self.evaluate(0.0, 2) - self._head_state[:, 2])),
            "q_goal": float(np.linalg.norm(self.evaluate(end_time, 0) - self._tail_state[:, 0])),
            "qd_goal": float(np.linalg.norm(self.evaluate(end_time, 1) - self._tail_state[:, 1])),
            "qdd_goal": float(np.linalg.norm(self.evaluate(end_time, 2) - self._tail_state[:, 2])),
        }

    def waypoint_error(self) -> float:
        self._require_generated()
        if self.piece_count == 1:
            return 0.0
        knot_times = np.cumsum(self._durations)[:-1]
        values = self.sample(knot_times, max_derivative=0).q
        return float(np.max(np.linalg.norm(values - self._p_inner, axis=1)))
