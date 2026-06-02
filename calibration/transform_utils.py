import json
from pathlib import Path

import numpy as np


def make_transform(translation=None, rotation=None) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    if rotation is not None:
        rotation = np.asarray(rotation, dtype=float)
        if rotation.shape != (3, 3):
            raise ValueError("rotation must be a 3x3 matrix")
        transform[:3, :3] = rotation
    if translation is not None:
        translation = np.asarray(translation, dtype=float)
        if translation.shape != (3,):
            raise ValueError("translation must have 3 values")
        transform[:3, 3] = translation
    return transform


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.size == 0:
        return points.reshape(0, 3)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    ones = np.ones((points.shape[0], 1), dtype=float)
    homogeneous = np.hstack([points, ones])
    return (np.asarray(transform, dtype=float) @ homogeneous.T).T[:, :3]


def load_transform_json(path: str | Path, key: str = "base_T_cam") -> np.ndarray:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    matrix = data.get(key, data)
    transform = np.asarray(matrix, dtype=float)
    if transform.shape != (4, 4):
        raise ValueError(f"{path} must contain a 4x4 transform")
    return transform
