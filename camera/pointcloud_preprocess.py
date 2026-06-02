import numpy as np


def remove_invalid_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.size == 0:
        return points.reshape(0, 3)
    mask = np.isfinite(points).all(axis=1)
    return points[mask]


def crop_workspace(points: np.ndarray, workspace: dict) -> np.ndarray:
    points = remove_invalid_points(points)
    if points.size == 0:
        return points
    x_min, x_max = workspace.get("x", [-1.2, 1.2])
    y_min, y_max = workspace.get("y", [-1.2, 1.2])
    z_min, z_max = workspace.get("z", [0.0, 1.5])
    mask = (
        (points[:, 0] >= x_min)
        & (points[:, 0] <= x_max)
        & (points[:, 1] >= y_min)
        & (points[:, 1] <= y_max)
        & (points[:, 2] >= z_min)
        & (points[:, 2] <= z_max)
    )
    points = points[mask]

    sphere_cfg = workspace.get("sphere")
    if sphere_cfg is not None:
        center = np.asarray(sphere_cfg["center"], dtype=float)
        radius = float(sphere_cfg["radius"])
        points = crop_sphere(points, center, radius)

    return points


def crop_sphere(points: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    """保留以 center 为球心、radius 为半径的球体内的点。"""
    points = remove_invalid_points(points)
    if points.size == 0:
        return points
    distances = np.linalg.norm(points - np.asarray(center), axis=1)
    return points[distances <= radius]


def voxel_downsample(points: np.ndarray, voxel_size: float = 0.02) -> np.ndarray:
    points = remove_invalid_points(points)
    if points.size == 0 or voxel_size <= 0.0:
        return points
    try:
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        down = pcd.voxel_down_sample(voxel_size=float(voxel_size))
        return np.asarray(down.points)
    except Exception:
        keys = np.floor(points / float(voxel_size)).astype(np.int64)
        _, unique_indices = np.unique(keys, axis=0, return_index=True)
        return points[np.sort(unique_indices)]
