import numpy as np

from perception.occupancy_object import ShapeModel


def fit_sphere(points: np.ndarray, margin: float = 0.02) -> ShapeModel:
    points = np.asarray(points, dtype=float)
    if points.size == 0:
        return ShapeModel(kind="sphere", center=np.zeros(3), radius=0.0)
    center = points.mean(axis=0)
    radius = float(np.max(np.linalg.norm(points - center, axis=1)) + margin)
    return ShapeModel(kind="sphere", center=center, radius=radius)


def fit_aabb(points: np.ndarray) -> ShapeModel:
    points = np.asarray(points, dtype=float)
    if points.size == 0:
        center = np.zeros(3)
        return ShapeModel(kind="aabb", center=center, radius=0.0, extents={"min": center, "max": center})
    min_bound = points.min(axis=0)
    max_bound = points.max(axis=0)
    center = (min_bound + max_bound) * 0.5
    radius = float(np.linalg.norm(max_bound - center))
    return ShapeModel(kind="aabb", center=center, radius=radius, extents={"min": min_bound, "max": max_bound})


def fit_obb(points: np.ndarray) -> ShapeModel:
    points = np.asarray(points, dtype=float)
    if points.shape[0] < 3:
        sphere = fit_sphere(points)
        return ShapeModel(kind="obb", center=sphere.center, radius=sphere.radius, rotation=np.eye(3))
    center = points.mean(axis=0)
    centered = points - center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    rotation = vh.T
    local = centered @ rotation
    min_local = local.min(axis=0)
    max_local = local.max(axis=0)
    half_lengths = (max_local - min_local) * 0.5
    obb_center = center + rotation @ ((min_local + max_local) * 0.5)
    radius = float(np.linalg.norm(half_lengths))
    return ShapeModel(
        kind="obb",
        center=obb_center,
        radius=radius,
        extents={"half_lengths": half_lengths},
        rotation=rotation,
    )


def make_occupancy_object(cluster: np.ndarray, timestamp: float, margin: float = 0.02):
    from perception.occupancy_object import OccupancyObject

    shape = fit_sphere(cluster, margin=margin)
    return OccupancyObject(
        id=-1,
        center=shape.center,
        velocity=np.zeros(3),
        radius=float(shape.radius or 0.0),
        shape=shape,
        confidence=0.0,
        risk="UNKNOWN",
        point_count=int(cluster.shape[0]),
        age=0,
        timestamp=timestamp,
    )


# ── 可视化辅助：OBB / 球体线框 + 3D 文字标签 ──────────────────

def create_obb_wireframe(points: np.ndarray,
                         color=(1.0, 0.5, 0.0),
                         line_width: int = 1) -> object:
    """从点云创建 OBB 包围盒线框 (open3d.geometry.LineSet)。"""
    import open3d as o3d

    obb_model = fit_obb(points)
    R = obb_model.rotation                   # (3, 3)
    half = obb_model.extents.get("half_lengths", np.ones(3) * 0.01)
    center = obb_model.center

    # 8 个角点 (局部 → 世界)
    local = np.array([
        [-1, -1, -1], [ 1, -1, -1], [ 1, -1,  1], [-1, -1,  1],
        [-1,  1, -1], [ 1,  1, -1], [ 1,  1,  1], [-1,  1,  1],
    ]) * half
    world = local @ R.T + center

    edges = np.array([
        [0, 1], [1, 2], [2, 3], [3, 0],   # 底面
        [4, 5], [5, 6], [6, 7], [7, 4],   # 顶面
        [0, 4], [1, 5], [2, 6], [3, 7],   # 立柱
    ])

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(world)
    ls.lines = o3d.utility.Vector2iVector(edges)
    ls.paint_uniform_color(color)
    return ls


def create_sphere_wireframe(center: np.ndarray,
                            radius: float,
                            color=(0.0, 1.0, 0.5),
                            n_meridians: int = 14,
                            n_parallels: int = 7) -> object:
    """创建球体的经纬线框 (open3d.geometry.LineSet)。"""
    import open3d as o3d

    pts = []
    lines = []

    # ── 经线 (Meridians) ──
    for mi in range(n_meridians):
        theta = 2 * np.pi * mi / n_meridians
        base = len(pts)
        for pi in range(n_parallels + 1):
            phi = np.pi * pi / n_parallels
            pts.append(center + np.array([
                radius * np.sin(phi) * np.cos(theta),
                radius * np.sin(phi) * np.sin(theta),
                radius * np.cos(phi),
            ]))
        for pi in range(n_parallels):
            lines.append([base + pi, base + pi + 1])

    # ── 纬线 (Parallels) ──
    for pi in range(n_parallels + 1):
        phi = np.pi * pi / n_parallels
        base = len(pts)
        for mi in range(n_meridians):
            theta = 2 * np.pi * mi / n_meridians
            pts.append(center + np.array([
                radius * np.sin(phi) * np.cos(theta),
                radius * np.sin(phi) * np.sin(theta),
                radius * np.cos(phi),
            ]))
        for mi in range(n_meridians):
            lines.append([base + mi, base + (mi + 1) % n_meridians])

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(np.array(pts))
    ls.lines = o3d.utility.Vector2iVector(np.array(lines))
    ls.paint_uniform_color(color)
    return ls


def create_text_label(center: np.ndarray,
                      text: str,
                      color=(1.0, 1.0, 1.0),
                      size: float = 0.025) -> object:
    """创建 3D 文字标签 (open3d.geometry.TriangleMesh)。

    基于 Open3D 0.15+ 的 create_text，不支持时降级为彩色小球标记。
    """
    import open3d as o3d

    try:
        mesh = o3d.geometry.TriangleMesh.create_text(text, depth=0)
        # 缩放到目标尺寸
        bbox = mesh.get_axis_aligned_bounding_box()
        extent = bbox.get_extent()
        if np.linalg.norm(extent) > 0:
            mesh.scale(size / max(extent), center=(0, 0, 0))
        # 居中并移动到目标位置
        bb = mesh.get_axis_aligned_bounding_box()
        mesh.translate(center - bb.get_center())
        if color is not None:
            mesh.paint_uniform_color(color)
        mesh.compute_vertex_normals()
        return mesh
    except Exception:
        # 降级：彩色小球 + 小杆标记
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=size * 0.4)
        sphere.translate(center)
        if color is not None:
            sphere.paint_uniform_color(color)
        return sphere
