import numpy as np

from robot.capsule_model import Capsule


def point_to_capsule_signed_distance(point: np.ndarray, capsule: Capsule) -> float:
    point = np.asarray(point, dtype=float)
    a = np.asarray(capsule.a, dtype=float)
    b = np.asarray(capsule.b, dtype=float)
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1e-12:
        segment_distance = np.linalg.norm(point - a)
    else:
        t = float(np.dot(point - a, ab) / denom)
        t = min(1.0, max(0.0, t))
        projection = a + t * ab
        segment_distance = np.linalg.norm(point - projection)
    return float(segment_distance - capsule.radius)


def filter_robot_self_points(points: np.ndarray, capsules: list[Capsule], margin: float = 0.03):
    points = np.asarray(points, dtype=float)
    if points.size == 0 or not capsules:
        return points.reshape(-1, 3), np.empty((0, 3), dtype=float)

    robot_mask = np.zeros(points.shape[0], dtype=bool)
    for capsule in capsules:
        a = np.asarray(capsule.a, dtype=float)
        b = np.asarray(capsule.b, dtype=float)
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom <= 1e-12:
            distances = np.linalg.norm(points - a, axis=1) - capsule.radius
        else:
            t = ((points - a) @ ab) / denom
            t = np.clip(t, 0.0, 1.0)
            projections = a + t[:, None] * ab
            distances = np.linalg.norm(points - projections, axis=1) - capsule.radius
        robot_mask |= distances < margin
    return points[~robot_mask], points[robot_mask]
