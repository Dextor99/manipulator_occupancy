"""Independent dense candidate-trajectory verification."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np

from .mesh_risk import MeshRiskEvaluator, StaticObstacleField
from .nubs_trajectory import NUBSTrajectory6D
from .optimizer import JointLimits
from .obstacle_forecast import ObstacleForecast
from .spatiotemporal_risk import SpatioTemporalRiskEvaluator


@dataclass
class VerificationResult:
    accepted: bool
    reasons: list[str]
    checks: dict[str, bool]
    goal_error: float
    min_distance: float
    nearest_link: str | None
    max_q_violation: float
    max_qd_violation: float
    max_qdd_violation: float
    continuity_q: float
    continuity_qd: float
    continuity_qdd: float
    validation_ms: float
    self_collision_checked: bool


class TrajectoryVerifier:
    def __init__(
        self,
        risk_evaluator: MeshRiskEvaluator,
        joint_limits: JointLimits,
        *,
        d_stop: float,
        time_step: float = 0.02,
        density: str = "dense",
        epsilon_goal: float = 1.0e-6,
        epsilon_continuity_q: float = 1.0e-6,
        epsilon_continuity_qd: float = 1.0e-5,
        epsilon_continuity_qdd: float = 1.0e-4,
        limit_tolerance: float = 1.0e-8,
    ) -> None:
        if d_stop <= 0.0 or time_step <= 0.0:
            raise ValueError("d_stop and time_step must be positive")
        self.risk_evaluator = risk_evaluator
        self.limits = joint_limits
        self.d_stop = float(d_stop)
        self.time_step = float(time_step)
        self.density = density
        self.epsilon_goal = float(epsilon_goal)
        self.epsilon_continuity_q = float(epsilon_continuity_q)
        self.epsilon_continuity_qd = float(epsilon_continuity_qd)
        self.epsilon_continuity_qdd = float(epsilon_continuity_qdd)
        self.limit_tolerance = float(limit_tolerance)

    def verify(
        self,
        trajectory: NUBSTrajectory6D,
        obstacle: StaticObstacleField,
        *,
        current_q: np.ndarray,
        current_qd: np.ndarray,
        current_qdd: np.ndarray,
        q_goal: np.ndarray,
        solver_success: bool,
    ) -> VerificationResult:
        started = time.perf_counter()
        count = max(2, int(np.ceil(trajectory.total_duration / self.time_step)) + 1)
        times = np.linspace(0.0, trajectory.total_duration, count)
        samples = trajectory.sample(times)
        finite_ok = bool(
            np.all(np.isfinite(samples.q))
            and np.all(np.isfinite(samples.qd))
            and np.all(np.isfinite(samples.qdd))
        )
        goal_error = float(np.linalg.norm(samples.q[-1] - np.asarray(q_goal)))
        continuity_q = float(np.linalg.norm(samples.q[0] - np.asarray(current_q)))
        continuity_qd = float(np.linalg.norm(samples.qd[0] - np.asarray(current_qd)))
        continuity_qdd = float(np.linalg.norm(samples.qdd[0] - np.asarray(current_qdd)))
        q_low = np.maximum(self.limits.q_min[None, :] - samples.q, 0.0)
        q_high = np.maximum(samples.q - self.limits.q_max[None, :], 0.0)
        qd_v = np.maximum(np.abs(samples.qd) - self.limits.qd_max[None, :], 0.0)
        qdd_v = np.maximum(np.abs(samples.qdd) - self.limits.qdd_max[None, :], 0.0)
        max_q = float(np.max(np.maximum(q_low, q_high)))
        max_qd = float(np.max(qd_v))
        max_qdd = float(np.max(qdd_v))

        min_distance = math.inf
        nearest_link: str | None = None
        for q in samples.q:
            result = self.risk_evaluator.configuration(
                q, obstacle, density=self.density, with_gradient=False
            )
            if result.min_distance < min_distance:
                min_distance = result.min_distance
                nearest_link = result.nearest_link

        checks = {
            "solver_ok": bool(solver_success),
            "finite_ok": finite_ok,
            "goal_ok": goal_error <= self.epsilon_goal,
            "distance_ok": min_distance >= self.d_stop,
            "position_ok": max_q <= self.limit_tolerance,
            "velocity_ok": max_qd <= self.limit_tolerance,
            "acceleration_ok": max_qdd <= self.limit_tolerance,
            "continuity_q_ok": continuity_q <= self.epsilon_continuity_q,
            "continuity_qd_ok": continuity_qd <= self.epsilon_continuity_qd,
            "continuity_qdd_ok": continuity_qdd <= self.epsilon_continuity_qdd,
        }
        reasons = [name for name, passed in checks.items() if not passed]
        return VerificationResult(
            accepted=bool(all(checks.values())),
            reasons=reasons,
            checks=checks,
            goal_error=goal_error,
            min_distance=float(min_distance),
            nearest_link=nearest_link,
            max_q_violation=max_q,
            max_qd_violation=max_qd,
            max_qdd_violation=max_qdd,
            continuity_q=continuity_q,
            continuity_qd=continuity_qd,
            continuity_qdd=continuity_qdd,
            validation_ms=(time.perf_counter() - started) * 1000.0,
            self_collision_checked=False,
        )


@dataclass
class DynamicVerificationResult(VerificationResult):
    extrapolated_sample_count: int = 0


class DynamicTrajectoryVerifier:
    def __init__(
        self,
        risk_evaluator: SpatioTemporalRiskEvaluator,
        joint_limits: JointLimits,
        *,
        d_stop: float,
        time_step: float = 0.025,
        density: str = "dense",
        epsilon_goal: float = 1.0e-6,
        epsilon_continuity_q: float = 1.0e-6,
        epsilon_continuity_qd: float = 1.0e-5,
        epsilon_continuity_qdd: float = 1.0e-4,
        limit_tolerance: float = 1.0e-8,
    ) -> None:
        if d_stop <= 0.0 or time_step <= 0.0:
            raise ValueError("d_stop and time_step must be positive")
        self.risk_evaluator = risk_evaluator
        self.limits = joint_limits
        self.d_stop = float(d_stop)
        self.time_step = float(time_step)
        self.density = density
        self.epsilon_goal = float(epsilon_goal)
        self.epsilon_continuity_q = float(epsilon_continuity_q)
        self.epsilon_continuity_qd = float(epsilon_continuity_qd)
        self.epsilon_continuity_qdd = float(epsilon_continuity_qdd)
        self.limit_tolerance = float(limit_tolerance)

    def verify(
        self,
        trajectory: NUBSTrajectory6D,
        forecast: ObstacleForecast,
        *,
        current_q: np.ndarray,
        current_qd: np.ndarray,
        current_qdd: np.ndarray,
        q_goal: np.ndarray,
        solver_success: bool,
    ) -> DynamicVerificationResult:
        started = time.perf_counter()
        count = max(2, int(np.ceil(trajectory.total_duration / self.time_step)) + 1)
        times = np.linspace(0.0, trajectory.total_duration, count)
        samples = trajectory.sample(times)
        finite_ok = bool(
            np.all(np.isfinite(samples.q))
            and np.all(np.isfinite(samples.qd))
            and np.all(np.isfinite(samples.qdd))
        )
        goal_error = float(np.linalg.norm(samples.q[-1] - np.asarray(q_goal)))
        continuity_q = float(np.linalg.norm(samples.q[0] - np.asarray(current_q)))
        continuity_qd = float(np.linalg.norm(samples.qd[0] - np.asarray(current_qd)))
        continuity_qdd = float(np.linalg.norm(samples.qdd[0] - np.asarray(current_qdd)))
        q_low = np.maximum(self.limits.q_min[None, :] - samples.q, 0.0)
        q_high = np.maximum(samples.q - self.limits.q_max[None, :], 0.0)
        qd_v = np.maximum(np.abs(samples.qd) - self.limits.qd_max[None, :], 0.0)
        qdd_v = np.maximum(np.abs(samples.qdd) - self.limits.qdd_max[None, :], 0.0)
        max_q = float(np.max(np.maximum(q_low, q_high)))
        max_qd = float(np.max(qd_v))
        max_qdd = float(np.max(qdd_v))
        min_distance = math.inf
        nearest_link = None
        extrapolated = 0
        for tau, q in zip(times, samples.q):
            result = self.risk_evaluator.configuration_clearance(
                q, forecast, float(tau), density=self.density
            )
            extrapolated += int(result.extrapolated)
            if result.min_distance < min_distance:
                min_distance = result.min_distance
                nearest_link = result.nearest_link
        checks = {
            "solver_ok": bool(solver_success),
            "finite_ok": finite_ok,
            "goal_ok": goal_error <= self.epsilon_goal,
            "distance_ok": min_distance >= self.d_stop,
            "position_ok": max_q <= self.limit_tolerance,
            "velocity_ok": max_qd <= self.limit_tolerance,
            "acceleration_ok": max_qdd <= self.limit_tolerance,
            "continuity_q_ok": continuity_q <= self.epsilon_continuity_q,
            "continuity_qd_ok": continuity_qd <= self.epsilon_continuity_qd,
            "continuity_qdd_ok": continuity_qdd <= self.epsilon_continuity_qdd,
            "forecast_horizon_ok": extrapolated == 0,
        }
        reasons = [name for name, passed in checks.items() if not passed]
        return DynamicVerificationResult(
            accepted=bool(all(checks.values())),
            reasons=reasons,
            checks=checks,
            goal_error=goal_error,
            min_distance=float(min_distance),
            nearest_link=nearest_link,
            max_q_violation=max_q,
            max_qd_violation=max_qd,
            max_qdd_violation=max_qdd,
            continuity_q=continuity_q,
            continuity_qd=continuity_qd,
            continuity_qdd=continuity_qdd,
            validation_ms=(time.perf_counter() - started) * 1000.0,
            self_collision_checked=False,
            extrapolated_sample_count=extrapolated,
        )
