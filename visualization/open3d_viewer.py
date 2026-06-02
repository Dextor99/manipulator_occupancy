import numpy as np


class Open3DViewer:
    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.o3d = None
        self.vis = None
        if enabled:
            try:
                import open3d as o3d

                self.o3d = o3d
                self.vis = o3d.visualization.Visualizer()
                self.vis.create_window(window_name="Manipulator Occupancy")
            except Exception:
                self.enabled = False

    def update(self, points, capsules, objects, risk_spheres):
        if not self.enabled or self.o3d is None or self.vis is None:
            return
        self.vis.clear_geometries()
        if points.size:
            pcd = self.o3d.geometry.PointCloud()
            pcd.points = self.o3d.utility.Vector3dVector(points)
            pcd.paint_uniform_color([0.5, 0.5, 0.5])
            self.vis.add_geometry(pcd)
        for capsule in capsules:
            line = self.o3d.geometry.LineSet()
            line.points = self.o3d.utility.Vector3dVector(np.vstack([capsule.a, capsule.b]))
            line.lines = self.o3d.utility.Vector2iVector([[0, 1]])
            line.colors = self.o3d.utility.Vector3dVector([[0.0, 0.1, 1.0]])
            self.vis.add_geometry(line)
        for obj in objects:
            mesh = self.o3d.geometry.TriangleMesh.create_sphere(radius=max(obj.radius, 0.01), resolution=12)
            mesh.translate(obj.center)
            mesh.paint_uniform_color([1.0, 0.6, 0.0])
            self.vis.add_geometry(mesh)
        for sphere in risk_spheres:
            mesh = self.o3d.geometry.TriangleMesh.create_sphere(radius=max(sphere.radius, 0.01), resolution=8)
            mesh.translate(sphere.center)
            mesh.paint_uniform_color([1.0, 0.0, 0.0])
            self.vis.add_geometry(mesh)
        self.vis.poll_events()
        self.vis.update_renderer()

    def close(self):
        if self.vis is not None:
            self.vis.destroy_window()
