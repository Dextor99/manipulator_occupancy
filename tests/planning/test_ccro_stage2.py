from __future__ import annotations

import numpy as np

from planning.mesh_risk import MeshRiskEvaluator, StaticObstacleField
from planning.nubs_trajectory import NUBSTrajectory6D
from planning.optimizer import JointLimits
from planning.robot_surface_model import RobotSurfaceModel
from planning.static_optimizer import StaticRiskNUBSOptimizer
from planning.verifier import TrajectoryVerifier


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
        totals or {"coarse": 160, "medium": 320, "dense": 800},
        seed=1234,
        min_points_per_link=16,
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


def test_surface_model_is_deterministic_and_multiresolution(tmp_path):
    first = make_model(tmp_path)
    second = make_model(tmp_path)
    q = np.array([0.1, -0.5, 1.2, 0.2, 0.8, -0.1])
    for density in ("coarse", "medium", "dense"):
        np.testing.assert_allclose(first.surface(q, density), second.surface(q, density))
    assert len(first.surface(q, "coarse")) < len(first.surface(q, "medium"))
    assert len(first.surface(q, "medium")) < len(first.surface(q, "dense"))
    assert "foreArm_Link" in first.surface_by_link(q, "medium")


def test_mesh_risk_is_positive_near_robot_and_zero_far_away(tmp_path):
    model = make_model(tmp_path)
    q = np.array([0.0, -0.6, 1.3, 0.0, 1.0, 0.0])
    robot_point = model.surface_by_link(q, "medium", {"foreArm_Link"})[
        "foreArm_Link"
    ][0]
    evaluator = MeshRiskEvaluator(model, d_safe=0.12, d_activate=0.18, density="medium")
    near = evaluator.configuration(
        q, StaticObstacleField.from_points(robot_point[None, :]), with_gradient=True
    )
    far = evaluator.configuration(
        q, StaticObstacleField.from_points(np.array([[10.0, 10.0, 10.0]]))
    )
    assert near.cost > 0.0
    assert near.min_distance < 1.0e-9
    assert near.gradient_q is not None
    assert far.cost == 0.0
    assert far.min_distance > 1.0


def test_verifier_rejects_colliding_trajectory(tmp_path):
    model = make_model(tmp_path)
    q0, q1, head, _, _, _, trajectory = make_trajectory()
    midpoint = trajectory.evaluate(0.5 * trajectory.total_duration)
    obstacle_point = model.surface_by_link(
        midpoint, "dense", {"foreArm_Link"}
    )["foreArm_Link"][0]
    obstacle = StaticObstacleField.from_points(obstacle_point[None, :])
    evaluator = MeshRiskEvaluator(model, d_safe=0.12, d_activate=0.18)
    limits = JointLimits.from_arrays([-6] * 6, [6] * 6, [2] * 6, [4] * 6)
    verifier = TrajectoryVerifier(
        evaluator, limits, d_stop=0.04, time_step=0.05, density="dense"
    )
    result = verifier.verify(
        trajectory,
        obstacle,
        current_q=q0,
        current_qd=np.zeros(6),
        current_qdd=np.zeros(6),
        q_goal=q1,
        solver_success=True,
    )
    assert not result.accepted
    assert not result.checks["distance_ok"]
    assert "distance_ok" in result.reasons


def test_static_optimizer_gradient_and_risk_reduction(tmp_path):
    model = make_model(tmp_path, {"coarse": 120, "medium": 240, "dense": 600})
    q0, q1, head, tail, durations, inner, trajectory = make_trajectory()
    midpoint = trajectory.evaluate(0.5 * trajectory.total_duration)
    points = model.surface_by_link(midpoint, "medium", {"foreArm_Link"})[
        "foreArm_Link"
    ]
    obstacle = StaticObstacleField.from_points(points[:8])
    evaluator = MeshRiskEvaluator(
        model,
        d_safe=0.10,
        d_activate=0.15,
        fd_epsilon_q=2.0e-4,
        density="medium",
    )
    limits = JointLimits.from_arrays([-6] * 6, [6] * 6, [2] * 6, [4] * 6)
    optimizer = StaticRiskNUBSOptimizer(
        head,
        tail,
        durations,
        limits,
        evaluator,
        obstacle,
        lambda_smooth=0.05,
        lambda_risk=1000.0,
        risk_samples_per_segment=3,
        samples_per_segment=6,
        max_iterations=30,
    )
    check = optimizer.check_gradient(inner, epsilon=1.0e-5)
    assert check["relative_error"] < 0.05
    assert check["cosine_similarity"] > 0.99
    result = optimizer.optimize(inner)
    assert result.success, result.message
    assert result.final_risk < result.initial_risk
