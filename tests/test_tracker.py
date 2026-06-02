import numpy as np

from perception.geometry_fit import fit_sphere
from perception.occupancy_object import OccupancyObject
from perception.occupancy_tracker import OccupancyTracker


def make_object(center, timestamp):
    shape = fit_sphere(np.array([center]), margin=0.05)
    return OccupancyObject(
        id=-1,
        center=np.array(center, dtype=float),
        velocity=np.zeros(3),
        radius=shape.radius,
        shape=shape,
        confidence=0.0,
        risk="UNKNOWN",
        point_count=20,
        age=0,
        timestamp=timestamp,
    )


def test_tracker_reuses_id_and_estimates_smoothed_velocity():
    tracker = OccupancyTracker(association_distance=0.2, alpha=0.5)
    first = tracker.update([make_object([0.0, 0.0, 0.0], 0.0)], timestamp=0.0)
    second = tracker.update([make_object([0.1, 0.0, 0.0], 1.0)], timestamp=1.0)

    assert second[0].id == first[0].id
    np.testing.assert_allclose(second[0].velocity, [0.05, 0.0, 0.0])
    assert second[0].age == 2
