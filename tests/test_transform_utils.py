import numpy as np

from calibration.transform_utils import make_transform, transform_points


def test_identity_transform_keeps_points_unchanged():
    points = np.array([[0.0, 0.0, 1.0], [1.0, 2.0, 3.0]])
    transformed = transform_points(points, np.eye(4))
    np.testing.assert_allclose(transformed, points)


def test_translation_transform_moves_points():
    points = np.array([[1.0, 2.0, 3.0]])
    transform = make_transform(translation=[0.5, -1.0, 2.0])
    transformed = transform_points(points, transform)
    np.testing.assert_allclose(transformed, [[1.5, 1.0, 5.0]])
