import math

import numpy as np

from perception.self_filter import point_to_capsule_signed_distance
from risk.prediction import RiskSphere
from robot.capsule_model import Capsule


def capsule_sphere_distance(capsule: Capsule, center: np.ndarray, radius: float) -> float:
    return point_to_capsule_signed_distance(center, capsule) - float(radius)


def min_capsule_sphere_distance(capsules: list[Capsule], spheres: list[RiskSphere]):
    best_distance = math.inf
    best_object_id = None
    if not capsules or not spheres:
        return best_distance, best_object_id
    for capsule in capsules:
        for sphere in spheres:
            distance = capsule_sphere_distance(capsule, sphere.center, sphere.radius)
            if distance < best_distance:
                best_distance = float(distance)
                best_object_id = sphere.object_id
    return best_distance, best_object_id
