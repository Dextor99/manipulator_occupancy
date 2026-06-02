import numpy as np

from camera.depth_to_pointcloud import depth_to_points


def test_depth_to_points_uses_pinhole_intrinsics():
    depth = np.array([[1000, 2000]], dtype=np.uint16)
    intrinsic = {"fx": 100.0, "fy": 100.0, "cx": 0.0, "cy": 0.0}

    points = depth_to_points(depth, intrinsic, depth_scale=0.001)

    np.testing.assert_allclose(points, [[0.0, 0.0, 1.0], [0.02, 0.0, 2.0]])


def test_depth_to_points_drops_zero_depth():
    depth = np.array([[0, 1000]], dtype=np.uint16)
    intrinsic = {"fx": 100.0, "fy": 100.0, "cx": 0.0, "cy": 0.0}

    points = depth_to_points(depth, intrinsic, depth_scale=0.001)

    assert points.shape == (1, 3)
    np.testing.assert_allclose(points[0], [0.01, 0.0, 1.0])
