from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
import pytest

from planning.dynamic_optimizer import DynamicRiskOptimizationResult
from planning.nubs_trajectory import NUBSTrajectory6D
from planning.obstacle_forecast import ConstantVelocitySphereForecast, ShiftedForecast
from planning.optimizer import JointLimits
from planning.replanner import FutureRiskReport, ReplanManager, RiskLevel
from planning.trajectory_buffer import TrajectoryBuffer
from planning.verifier import DynamicVerificationResult


def make_trajectory(duration=4.0):
    q0 = np.zeros(6)
    q1 = np.full(6, 0.4)
    durations = np.full(4, duration / 4.0)
    head = NUBSTrajectory6D.make_boundary_state(q0)
    tail = NUBSTrajectory6D.make_boundary_state(q1)
    inner = NUBSTrajectory6D.linear_inner_points(q0, q1, durations)
    trajectory = NUBSTrajectory6D().generate(inner, head, tail, durations)
    return q0, q1, durations, trajectory


class DummyOptimizer:
    def __init__(self, head, tail, durations):
        self.head = head
        self.tail = tail
        self.durations = durations

    def optimize(self, warm):
        trajectory = NUBSTrajectory6D().generate(
            np.asarray(warm), self.head, self.tail, self.durations
        )
        return DynamicRiskOptimizationResult(
            True, 0, "ok", trajectory, np.asarray(warm), self.durations,
            1.0, 0.5, 1.0, 0.5, 0.01, 0.001, 0.05, 0.10,
            0.0, 2, 3, 1.0e-6, 5.0,
        )


class DummyVerifier:
    def verify(self, *args, **kwargs):
        return DynamicVerificationResult(
            True, [], {"solver_ok": True, "distance_ok": True}, 0.0, 0.20,
            "foreArm_Link", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, False, 0,
        )


def make_manager(forecast, *, budget=0.12, delay=0.0):
    def factory(head, tail, durations, local_forecast):
        return DummyOptimizer(head, tail, durations)

    limits = JointLimits.from_arrays([-6] * 6, [6] * 6, [2] * 6, [4] * 6)
    return ReplanManager(
        factory, None, forecast, DummyVerifier(), limits,
        d_replan=0.10, d_safe=0.08, d_accept=0.06, d_stop=0.035,
        planning_budget=budget, artificial_worker_delay=delay,
    )


def test_shifted_forecast_preserves_global_time():
    source = ConstantVelocitySphereForecast(
        np.zeros(3), np.array([0.2, 0.0, 0.0]), 0.03, 5.0
    )
    shifted = ShiftedForecast(source, 1.5, 2.0)
    np.testing.assert_allclose(shifted.occupancy_at(0.0).spheres[0].center, [0.3, 0, 0])
    np.testing.assert_allclose(shifted.occupancy_at(1.0).spheres[0].center, [0.5, 0, 0])
    assert shifted.valid_horizon == pytest.approx(2.0)
    with pytest.raises(ValueError, match="shifted horizon"):
        shifted.occupancy_at(2.1)


def test_buffer_warm_start_shape_and_pause_continuity():
    _, q1, durations, trajectory = make_trajectory()
    buffer = TrajectoryBuffer()
    buffer.set_active(trajectory, 0.0, q1)
    warm = buffer.remaining_waypoints(0.5, durations)
    assert warm.shape == (3, 6)
    state_before = buffer.sample_state(1.0)
    buffer.pause(1.0)
    state_while_paused = buffer.sample_state(2.0)
    for before, paused in zip(state_before, state_while_paused):
        np.testing.assert_allclose(before, paused)
    buffer.resume(3.0)
    np.testing.assert_allclose(buffer.sample_now(3.0), state_before[0])


def test_high_risk_transition_is_immediate_from_low():
    forecast = ConstantVelocitySphereForecast(np.zeros(3), np.zeros(3), 0.03, 5.0)
    manager = make_manager(forecast)
    assert manager.current_level == RiskLevel.LOW
    assert manager._resolve_level(0.02) == RiskLevel.HIGH
    manager.shutdown()


def test_async_candidate_accepts_without_blocking_control_loop():
    _, q_goal, durations, trajectory = make_trajectory()
    forecast = ConstantVelocitySphereForecast(np.zeros(3), np.zeros(3), 0.03, 5.0)
    manager = make_manager(forecast, budget=0.15)
    buffer = TrajectoryBuffer()
    buffer.set_active(trajectory, 0.0, q_goal)
    report = FutureRiskReport(RiskLevel.MEDIUM, 0.05, 0.1, 1.0, "foreArm_Link", 1)
    manager._current_level = RiskLevel.MEDIUM
    started = time.perf_counter()
    assert manager.submit_replan(0.0, buffer, q_goal, durations, report)
    assert time.perf_counter() - started < 0.10
    assert not buffer.is_paused and manager.replan_in_flight
    outcome = "running"
    deadline = time.perf_counter() + 1.0
    while outcome in {"running", "ready"} and time.perf_counter() < deadline:
        simulated_time = time.perf_counter() - started
        outcome = manager.poll_candidate(simulated_time, buffer, q_goal).outcome
        time.sleep(0.01)
    assert outcome == "accepted"
    assert not buffer.is_paused
    assert manager.replan_events[-1].candidate_accepted
    manager.shutdown()


def test_async_timeout_holds_position_and_rejects_switch():
    _, q_goal, durations, trajectory = make_trajectory()
    forecast = ConstantVelocitySphereForecast(np.zeros(3), np.zeros(3), 0.03, 5.0)
    manager = make_manager(forecast, budget=0.08, delay=0.30)
    buffer = TrajectoryBuffer()
    buffer.set_active(trajectory, 0.0, q_goal)
    report = FutureRiskReport(RiskLevel.MEDIUM, 0.05, 0.1, 1.0, "foreArm_Link", 1)
    manager._current_level = RiskLevel.MEDIUM
    assert manager.submit_replan(0.0, buffer, q_goal, durations, report)
    outcome = manager.poll_candidate(0.09, buffer, q_goal).outcome
    assert outcome == "timeout"
    assert manager.safety_hold_required
    assert manager.safety_takeover_count == 1
    q_at_timeout = buffer.sample_now(0.09)
    np.testing.assert_allclose(buffer.sample_now(1.0), q_at_timeout)
    manager.shutdown()
