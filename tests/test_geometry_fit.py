import numpy as np

from perception.geometry_fit import fit_aabb, fit_sphere


def test_fit_sphere_covers_all_points_with_margin():
    points = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    sphere = fit_sphere(points, margin=0.1)

    assert sphere.kind == "sphere"
    np.testing.assert_allclose(sphere.center, [1.0, 0.0, 0.0])
    assert sphere.radius >= 1.1


def test_fit_aabb_reports_min_and_max_bounds():
    points = np.array([[-1.0, 2.0, 0.5], [3.0, -2.0, 1.5]])
    aabb = fit_aabb(points)

    np.testing.assert_allclose(aabb.extents["min"], [-1.0, -2.0, 0.5])
    np.testing.assert_allclose(aabb.extents["max"], [3.0, 2.0, 1.5])
