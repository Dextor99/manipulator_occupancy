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
    tracker = OccupancyTracker(association_distance=0.2, alpha=0.5, pos_alpha=1.0)
    first = tracker.update([make_object([0.0, 0.0, 0.0], 0.0)], timestamp=0.0)
    second = tracker.update([make_object([0.1, 0.0, 0.0], 1.0)], timestamp=1.0)

    assert second[0].id == first[0].id
    np.testing.assert_allclose(second[0].velocity, [0.05, 0.0, 0.0])
    assert second[0].age == 2


def test_speed_motion_gate_is_independent_of_frame_period():
    tracker = OccupancyTracker(
        association_distance=0.2, alpha=1.0, pos_alpha=1.0,
        motion_gate_speed=0.03,
    )
    tracker.update([make_object([0.0, 0.0, 0.0], 0.0)], timestamp=0.0)
    moving = tracker.update([make_object([0.004, 0.0, 0.0], 0.04)], timestamp=0.04)
    np.testing.assert_allclose(moving[0].velocity, [0.1, 0.0, 0.0])


def test_track_identity_survives_two_misses_without_outputting_predictions():
    tracker = OccupancyTracker(association_distance=0.2, max_miss=2)
    first = tracker.update([make_object([0.0, 0.0, 0.0], 0.0)], timestamp=0.0)
    assert tracker.update([], timestamp=0.04) == []
    assert tracker.update([], timestamp=0.08) == []
    resumed = tracker.update([make_object([0.02, 0.0, 0.0], 0.12)], timestamp=0.12)
    assert resumed[0].id == first[0].id
