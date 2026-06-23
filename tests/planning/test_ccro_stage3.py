from __future__ import annotations

import numpy as np
import pytest

from planning.dynamic_optimizer import DynamicRiskNUBSOptimizer
from planning.nubs_trajectory import NUBSTrajectory6D
from planning.obstacle_forecast import ConstantVelocitySphereForecast
from planning.optimizer import JointLimits
from planning.robot_surface_model import RobotSurfaceModel
from planning.spatiotemporal_risk import SpatioTemporalRiskEvaluator
from planning.verifier import DynamicTrajectoryVerifier


JOINT_NAMES = [
    "shoulder_joint",
    "upperArm_joint",
    "foreArm_joint",
    "wrist1_joint",
    "wrist2_joint",
    "wrist3_joint",
]


def make_model(tmp_path, totals=None):
    return RobotSurfaceModel(
        "urdf/aubo_i16_gripper.urdf",
        JOINT_NAMES,
        totals or {"coarse": 120, "medium": 240, "dense": 600},
        seed=20260623,
        min_points_per_link=12,
        cache_dir=tmp_path,
        geometry="collision",
    )


def make_trajectory():
    q0 = np.array([0.0, -0.6, 1.3, 0.0, 1.0, 0.0])
    q1 = np.array([0.7, -0.1, 0.7, 0.5, 0.5, -0.4])
    head = NUBSTrajectory6D.make_boundary_state(q0)
    tail = NUBSTrajectory6D.make_boundary_state(q1)
    durations = np.array([2.0, 2.0])
    inner = NUBSTrajectory6D.linear_inner_points(q0, q1, durations)
    trajectory = NUBSTrajectory6D().generate(inner, head, tail, durations)
    return q0, q1, head, tail, durations, inner, trajectory


def test_constant_velocity_forecast_has_tau_zero_and_explicit_horizon():
    forecast = ConstantVelocitySphereForecast(
        np.array([1.0, 2.0, 3.0]),
        np.array([0.2, -0.1, 0.0]),
        0.04,
        2.0,
        margin=0.01,
        uncertainty=0.02,
        uncertainty_growth=0.003,
    )
    initial = forecast.occupancy_at(0.0)
    future = forecast.occupancy_at(1.5)
    np.testing.assert_allclose(initial.spheres[0].center, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(future.spheres[0].center, [1.3, 1.85, 3.0])
    assert future.spheres[0].radius > initial.spheres[0].radius
    assert not future.extrapolated
    with pytest.raises(ValueError, match="valid_horizon"):
        forecast.occupancy_at(2.01)


def test_dynamic_risk_uses_matching_physical_time(tmp_path):
    model = make_model(tmp_path)
    q = np.array([0.0, -0.6, 1.3, 0.0, 1.0, 0.0])
    point = model.surface_by_link(q, "medium", {"foreArm_Link"})["foreArm_Link"][0]
    velocity = np.array([0.08, 0.0, 0.0])
    collision_time = 1.25
    forecast = ConstantVelocitySphereForecast(
        point - velocity * collision_time, velocity, 0.03, 2.0
    )
    evaluator = SpatioTemporalRiskEvaluator(
        model, d_safe=0.10, d_activate=0.15, density="medium"
    )
    near = evaluator.configuration(q, forecast, collision_time, with_gradient=True)
    assert near.cost > 0.0
    assert near.min_distance == pytest.approx(0.0, abs=1.0e-10)
    assert near.nearest_link == "foreArm_Link"
    assert near.gradient_q is not None


def test_dynamic_verifier_rejects_a_timed_body_collision(tmp_path):
    model = make_model(tmp_path)
    q0, q1, head, _, _, _, trajectory = make_trajectory()
    collision_time = 0.5 * trajectory.total_duration
    q_collision = trajectory.evaluate(collision_time)
    point = model.surface_by_link(
        q_collision, "dense", {"foreArm_Link"}
    )["foreArm_Link"][0]
    velocity = np.array([0.10, 0.0, 0.0])
    forecast = ConstantVelocitySphereForecast(
        point - velocity * collision_time,
        velocity,
        0.03,
        trajectory.total_duration,
    )
    evaluator = SpatioTemporalRiskEvaluator(
        model, d_safe=0.10, d_activate=0.15, density="medium"
    )
    limits = JointLimits.from_arrays([-6] * 6, [6] * 6, [2] * 6, [4] * 6)
    verifier = DynamicTrajectoryVerifier(
        evaluator, limits, d_stop=0.04, time_step=0.05, density="dense"
    )
    result = verifier.verify(
        trajectory,
        forecast,
        current_q=q0,
        current_qd=head[:, 1],
        current_qdd=head[:, 2],
        q_goal=q1,
        solver_success=True,
    )
    assert not result.accepted
    assert not result.checks["distance_ok"]
    assert result.checks["forecast_horizon_ok"]
    assert result.extrapolated_sample_count == 0


def test_dynamic_optimizer_gradient_and_risk_reduction(tmp_path):
    model = make_model(tmp_path)
    q0, _, head, tail, durations, inner, trajectory = make_trajectory()
    collision_time = 0.5 * trajectory.total_duration
    point = model.surface_by_link(
        trajectory.evaluate(collision_time), "medium", {"foreArm_Link"}
    )["foreArm_Link"][0]
    velocity = np.array([0.08, 0.0, 0.0])
    forecast = ConstantVelocitySphereForecast(
        point - velocity * collision_time,
        velocity,
        0.035,
        trajectory.total_duration,
        margin=0.005,
    )
    evaluator = SpatioTemporalRiskEvaluator(
        model,
        d_safe=0.10,
        d_activate=0.15,
        fd_epsilon_q=2.0e-4,
        density="medium",
    )
    limits = JointLimits.from_arrays([-6] * 6, [6] * 6, [2] * 6, [4] * 6)
    optimizer = DynamicRiskNUBSOptimizer(
        head,
        tail,
        durations,
        limits,
        evaluator,
        forecast,
        lambda_smooth=0.05,
        lambda_risk=1500.0,
        risk_links={"foreArm_Link"},
        risk_samples_per_segment=4,
        samples_per_segment=6,
        max_iterations=35,
    )
    check = optimizer.check_gradient(inner, epsilon=1.0e-5)
    assert check["relative_error"] < 0.05
    assert check["cosine_similarity"] > 0.99
    result = optimizer.optimize(inner)
    assert result.success, result.message
    assert result.final_risk < result.initial_risk
    np.testing.assert_allclose(result.trajectory.evaluate(0.0), q0, atol=1.0e-12)
