import numpy as np

from perception.self_filter import point_to_capsule_signed_distance
from robot.capsule_model import Capsule


def test_point_to_capsule_distance_near_segment_middle():
    capsule = Capsule("link", np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), 0.1)
    distance = point_to_capsule_signed_distance(np.array([0.5, 0.2, 0.0]), capsule)
    assert abs(distance - 0.1) < 1e-9


def test_point_to_capsule_distance_clamps_to_endpoint():
    capsule = Capsule("link", np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), 0.1)
    distance = point_to_capsule_signed_distance(np.array([1.3, 0.0, 0.0]), capsule)
    assert abs(distance - 0.2) < 1e-9


def test_point_inside_capsule_has_negative_signed_distance():
    capsule = Capsule("link", np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), 0.1)
    distance = point_to_capsule_signed_distance(np.array([0.5, 0.05, 0.0]), capsule)
    assert distance < 0.0
