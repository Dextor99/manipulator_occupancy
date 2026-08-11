from __future__ import annotations

import unittest

import numpy as np

from experiments.exp_ccro_stage1 import check_energy_gradient
from planning.nubs_trajectory import CompositeTrajectory6D, NUBSTrajectory6D
from planning.optimizer import FixedTimeNUBSOptimizer, JointLimits


class NUBSStageOneTests(unittest.TestCase):
    def setUp(self):
        self.q0 = np.array([0.0, -0.4, 1.2, 0.0, 1.0, 0.0])
        self.q1 = np.array([0.5, -0.1, 0.9, 0.3, 0.8, -0.2])
        self.head = NUBSTrajectory6D.make_boundary_state(self.q0)
        self.tail = NUBSTrajectory6D.make_boundary_state(self.q1)
        self.durations = np.array([1.2, 1.4, 1.3, 1.1])
        self.inner = NUBSTrajectory6D.linear_inner_points(
            self.q0, self.q1, self.durations
        )

    def test_boundary_and_waypoint_constraints(self):
        trajectory = NUBSTrajectory6D().generate(
            self.inner, self.head, self.tail, self.durations
        )
        errors = trajectory.boundary_errors()
        self.assertLess(errors["q_start"], 1.0e-9)
        self.assertLess(errors["q_goal"], 1.0e-8)
        self.assertLess(errors["qd_start"], 1.0e-8)
        self.assertLess(errors["qd_goal"], 1.0e-7)
        self.assertLess(errors["qdd_start"], 1.0e-7)
        self.assertLess(errors["qdd_goal"], 1.0e-6)
        self.assertLess(trajectory.waypoint_error(), 1.0e-8)

    def test_energy_gradient_matches_external_finite_difference(self):
        result = check_energy_gradient(
            self.inner, self.head, self.tail, self.durations, 1.0e-6
        )
        self.assertLess(result["relative_error"], 1.0e-3)
        self.assertGreater(result["cosine_similarity"], 0.999)

    def test_local_and_full_energy_gradients_match(self):
        trajectory = NUBSTrajectory6D().generate(
            self.inner, self.head, self.tail, self.durations
        )
        energy_local, points_local, times_local = trajectory.energy_and_gradient()
        energy_full, points_full, times_full = trajectory.energy_and_gradient_full()
        self.assertAlmostEqual(energy_local, energy_full, places=12)
        np.testing.assert_allclose(points_local, points_full, rtol=1.0e-8, atol=1.0e-9)
        np.testing.assert_allclose(times_local, times_full, rtol=1.0e-6, atol=1.0e-7)

    def test_single_segment_trajectory(self):
        durations = np.array([3.0])
        trajectory = NUBSTrajectory6D().generate(
            np.empty((0, 6)), self.head, self.tail, durations
        )
        self.assertEqual(trajectory.piece_count, 1)
        self.assertEqual(trajectory.inner_points.shape, (0, 6))
        self.assertLess(trajectory.boundary_errors()["q_goal"], 1.0e-8)
        samples = trajectory.sample(np.linspace(0.0, 3.0, 11))
        self.assertEqual(samples.q.shape, (11, 6))
        self.assertTrue(np.all(np.isfinite(samples.jerk)))

    def test_fixed_time_optimizer_does_not_increase_energy(self):
        limits = JointLimits.from_arrays(
            [-6.0] * 6, [6.0] * 6, [2.0] * 6, [4.0] * 6
        )
        optimizer = FixedTimeNUBSOptimizer(
            self.head,
            self.tail,
            self.durations,
            limits,
            max_iterations=100,
        )
        result = optimizer.optimize(self.inner)
        self.assertTrue(result.success, result.message)
        self.assertLessEqual(result.final_energy, result.initial_energy + 1.0e-9)
        self.assertEqual(result.p_inner.shape, self.inner.shape)

    def test_invalid_duration_is_rejected(self):
        with self.assertRaises(ValueError):
            NUBSTrajectory6D().generate(
                self.inner, self.head, self.tail, np.array([1.0, 0.0, 1.0, 1.0])
            )

    def test_composite_trajectory_preserves_c2_join(self):
        middle = NUBSTrajectory6D.make_boundary_state(self.q1, np.full(6, 0.02), np.full(6, -0.01))
        first = NUBSTrajectory6D().generate(
            NUBSTrajectory6D.linear_inner_points(self.q0, self.q1, np.array([0.5, 0.5])),
            self.head,
            middle,
            np.array([0.5, 0.5]),
        )
        q2 = self.q1 + 0.1
        second = NUBSTrajectory6D().generate(
            NUBSTrajectory6D.linear_inner_points(self.q1, q2, np.array([0.4, 0.6])),
            middle,
            NUBSTrajectory6D.make_boundary_state(q2),
            np.array([0.4, 0.6]),
        )
        composite = CompositeTrajectory6D([first, second])
        for derivative in range(3):
            np.testing.assert_allclose(
                composite.evaluate(1.0 - 1.0e-8, derivative),
                composite.evaluate(1.0 + 1.0e-8, derivative),
                atol=1.0e-5,
            )
        self.assertAlmostEqual(composite.total_duration, 2.0)


if __name__ == "__main__":
    unittest.main()
