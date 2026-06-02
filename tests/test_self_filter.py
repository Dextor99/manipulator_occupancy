import numpy as np

from perception.self_filter import filter_robot_self_points
from robot.capsule_model import Capsule


def test_filter_robot_self_points_keeps_external_points():
    points = np.array([[0.5, 0.05, 0.0], [0.5, 0.5, 0.0], [2.0, 2.0, 2.0]])
    capsule = Capsule("link", np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), 0.1)

    external, robot = filter_robot_self_points(points, [capsule], margin=0.03)

    assert robot.shape == (1, 3)
    assert external.shape == (2, 3)
    np.testing.assert_allclose(external[0], [0.5, 0.5, 0.0])
