import numpy as np

from perception.geometry_fit import ShapeModel
from perception.occupancy_object import OccupancyObject
from risk.prediction import predict_risk_spheres
from risk.safety_policy import RiskLevel, SafetyPolicy


def test_prediction_radius_grows_with_velocity_and_time():
    obj = OccupancyObject(
        id=7,
        center=np.zeros(3),
        velocity=np.array([1.0, 0.0, 0.0]),
        radius=0.2,
        shape=ShapeModel(kind="sphere", center=np.zeros(3), radius=0.2),
        confidence=1.0,
        risk="UNKNOWN",
        point_count=100,
        age=3,
        timestamp=0.0,
    )

    spheres = predict_risk_spheres([obj], horizon=0.2, step=0.1, margin=0.05, uncertainty=0.02)

    assert len(spheres) == 2
    assert spheres[1].radius > spheres[0].radius
    np.testing.assert_allclose(spheres[1].center, [0.2, 0.0, 0.0])


def test_safety_policy_thresholds_and_speed_scale():
    policy = SafetyPolicy(d_safe=0.15, d_slow=0.10, d_stop=0.05)

    assert policy.evaluate(0.2).level == RiskLevel.SAFE
    assert policy.evaluate(0.12).level == RiskLevel.WARNING
    assert policy.evaluate(0.07).level == RiskLevel.SLOW
    stop = policy.evaluate(0.04)
    assert stop.level == RiskLevel.STOP
    assert stop.speed_scale == 0.0
