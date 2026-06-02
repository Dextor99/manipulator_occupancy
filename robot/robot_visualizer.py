import numpy as np


def capsule_line_sets(capsules):
    try:
        import open3d as o3d
    except Exception:
        return []
    geometries = []
    for capsule in capsules:
        line = o3d.geometry.LineSet()
        line.points = o3d.utility.Vector3dVector(np.vstack([capsule.a, capsule.b]))
        line.lines = o3d.utility.Vector2iVector([[0, 1]])
        line.colors = o3d.utility.Vector3dVector([[0.0, 0.2, 1.0]])
        geometries.append(line)
    return geometries
