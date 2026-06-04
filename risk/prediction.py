from dataclasses import dataclass

import numpy as np

from perception.occupancy_object import OccupancyObject


@dataclass
class RiskSphere:
    object_id: int
    center: np.ndarray
    radius: float
    tau: float


def predict_risk_spheres(
    objects: list[OccupancyObject],
    horizon: float = 0.5,
    step: float = 0.1,
    margin: float = 0.05,
    uncertainty: float = 0.02,
) -> list[RiskSphere]:
    if step <= 0.0:
        raise ValueError("step must be positive")
    taus = np.arange(step, horizon + 1e-9, step)
    predictions = []
    for obj in objects:
        speed = float(np.linalg.norm(obj.velocity))
        if speed < 0.01:
            # 静态障碍：至少生成一个当前位置 RiskSphere，否则距离检查完全找不到它
            predictions.append(RiskSphere(
                obj.id, obj.center.copy(),
                float(obj.radius + margin + uncertainty),
                0.0,
            ))
        else:
            for tau in taus:
                center = obj.center + obj.velocity * tau
                radius = obj.radius + margin + speed * tau + uncertainty
                predictions.append(RiskSphere(obj.id, center, float(radius), float(tau)))
    return predictions
