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


# ── OBB 有符号距离 ──────────────────────────────────────────────


def capsule_obb_signed_distance(
    capsule: Capsule,
    obb_center: np.ndarray,
    rotation: np.ndarray,
    half_lengths: np.ndarray,
) -> float:
    """胶囊体骨架到 OBB 表面的有符号距离。

    将胶囊体骨架线段的两个端点变换到 OBB 局部坐标系 → OBB 退化为 AABB，
    计算线段到 AABB 的有符号距离再减去胶囊体半径。

    Parameters
    ----------
    capsule : Capsule
        机器人胶囊体（线段 + 半径）
    obb_center : (3,)
        OBB 在世界系中的中心
    rotation : (3, 3)
        OBB 的旋转矩阵（列为主）
    half_lengths : (3,)
        OBB 的三个半轴长度

    Returns
    -------
    float
        有符号距离：正=有间隙，负=穿透深度
    """
    a = rotation.T @ (np.asarray(capsule.a, dtype=float) - obb_center)
    b = rotation.T @ (np.asarray(capsule.b, dtype=float) - obb_center)
    h = np.asarray(half_lengths, dtype=float)

    seg_dist = _segment_aabb_signed_distance(a, b, h)
    return float(seg_dist - capsule.radius)


def _segment_aabb_signed_distance(
    a: np.ndarray, b: np.ndarray, h: np.ndarray
) -> float:
    """线段 [a, b] 到原点 AABB [-h, +h] 的有符号距离。

    距离为正 = 线段在 AABB 外部；距离为负 = 穿透深度。
    """
    d = b - a
    eps = 1e-10

    # ── 裁剪法检测相交 ──
    t_in, t_out = 0.0, 1.0
    intersects = True
    for i in range(3):
        if abs(d[i]) < eps:
            if abs(a[i]) > h[i]:
                intersects = False
                break
        else:
            t1 = (-h[i] - a[i]) / d[i]
            t2 = (h[i] - a[i]) / d[i]
            if t1 > t2:
                t1, t2 = t2, t1
            t_in = max(t_in, t1)
            t_out = min(t_out, t2)
            if t_in > t_out:
                intersects = False
                break

    if intersects:
        # 线段穿过 AABB → 返回穿透深度（到最近面的距离）
        p = a + 0.5 * (t_in + t_out) * d
        pen = min(h[0] - abs(p[0]), h[1] - abs(p[1]), h[2] - abs(p[2]))
        return -max(pen, 0.0)

    # ── 不相交 → 采样线段上的最近点 ──
    best = math.inf
    # 端点 + 均匀采样 9 个内点，共 11 个点
    # 对于最长 1m 的胶囊体骨架，采样间距 ≈ 10cm，精度足够
    for t in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        p = a + t * d
        dx = max(0.0, abs(p[0]) - h[0])
        dy = max(0.0, abs(p[1]) - h[1])
        dz = max(0.0, abs(p[2]) - h[2])
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        if dist < best:
            best = dist

    return best


# ── 双层距离检查 ──────────────────────────────────────────────


def min_capsule_obb_distance(
    capsules: list[Capsule],
    objects: list,
    horizon: float = 0.5,
    step: float = 0.1,
    margin: float = 0.05,
    uncertainty: float = 0.02,
    obb_threshold: float = 0.15,
):
    """双层距离检查：球体快速筛 → OBB 精确算（近距离时）。

    Parameters
    ----------
    capsules : list[Capsule]
        机器人胶囊体模型
    objects : list[OccupancyObject]
        当前帧的占据物体（含 shape.rotation + shape.extents['half_lengths']）
    horizon, step : float
        预测时间范围
    margin, uncertainty : float
        球体边距和不确定性（与 predict_risk_spheres 一致）
    obb_threshold : float
        球体距离低于此值时触发 OBB 精确计算（默认 0.15m = d_safe）

    Returns
    -------
    min_distance : float
    nearest_object_id : int | None
    obb_used : bool
        最终距离是否来自 OBB 精确计算
    """
    best = float("inf")
    best_id = None
    obb_used = False

    if not capsules or not objects:
        return best, best_id, obb_used

    taus = np.arange(step, horizon + 1e-9, step)

    for obj in objects:
        speed = float(np.linalg.norm(obj.velocity))
        rotation = getattr(obj.shape, "rotation", None)
        half_lengths = obj.shape.extents.get("half_lengths") if hasattr(obj.shape, "extents") else None
        has_obb = rotation is not None and half_lengths is not None

        for tau in taus:
            center = obj.center + obj.velocity * tau
            radius = obj.radius + margin + speed * tau + uncertainty

            for capsule in capsules:
                # Stage 1: 球体距离（快速）
                sphere_d = point_to_capsule_signed_distance(center, capsule) - radius
                if sphere_d < best:
                    best = sphere_d
                    best_id = obj.id

                # Stage 2: OBB 精确距离（仅近距离时触发）
                if sphere_d < obb_threshold and has_obb:
                    obb_d = capsule_obb_signed_distance(
                        capsule, center, rotation, half_lengths
                    )
                    if obb_d < best:
                        best = obb_d
                        best_id = obj.id
                        obb_used = True

    return best, best_id, obb_used
