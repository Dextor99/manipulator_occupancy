import numpy as np


def depth_to_points(depth: np.ndarray, intrinsic: dict, depth_scale: float = 0.001) -> np.ndarray:
    depth = np.asarray(depth)
    if depth.ndim != 2:
        raise ValueError("depth must be a 2D array")

    fx = float(intrinsic["fx"])
    fy = float(intrinsic["fy"])
    cx = float(intrinsic["cx"])
    cy = float(intrinsic["cy"])

    v, u = np.indices(depth.shape)
    z = depth.astype(float) * float(depth_scale)
    valid = z > 0.0
    if not np.any(valid):
        return np.empty((0, 3), dtype=float)

    x = (u[valid] - cx) * z[valid] / fx
    y = (v[valid] - cy) * z[valid] / fy
    return np.column_stack([x, y, z[valid]])
