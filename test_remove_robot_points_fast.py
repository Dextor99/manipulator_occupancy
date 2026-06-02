"""实时点云处理 — 自动去除机械臂本体点云（高性能优化版）。

优化策略：
  1. 预处理采样：网格表面在 link 本地坐标系下采样一次，每帧只做矩阵变换
  2. KD-tree 距离查询：scipy.spatial.cKDTree 替代 Open3D compute_point_cloud_distance
  3. 跳帧处理：机械臂去除每 N 帧执行一次，其余帧复用上一帧的过滤掩码
  4. 场景点云限幅：防止极端帧拖垮性能
  5. 快速坐标变换：pts @ R.T + t 替代齐次坐标 4x4 乘法
  6. 原始点云预下采样：921K → 100K，避免体素降采样成为瓶颈
  7. 机器人表面点体素压缩：KD-tree 前压缩 50K → ~5K，大幅加速查询

输出：
  - scene_points : (N, 3)  环境点云（机械臂去除后）
  - robot_points : (M, 3)  机械臂表面点云

对每一帧：
  1. 读取 RealSense 点云 → 变换到基坐标系 → 裁剪 → 体素降采样
  2. 读取关节角 → FK → 矩阵变换预采样点 → KD-tree 最近邻查询
  3. 移除距离 < threshold 的点（机械臂本体）
  4. 输出 scene_points 和 robot_points 两套点云

外部调用：
  from test_remove_robot_points_fast import SceneProcessor
  processor = SceneProcessor(...)
  result = processor.process_frame()
  env_pts = result.scene_points    # 环境点云
  robot_pts = result.robot_points  # 机械臂点云

命令行用法：
  # 带可视化（调试）
  python test_remove_robot_points_fast.py --real-robot --visualize
  # 无可视化，后台计算
  python test_remove_robot_points_fast.py --real-robot
"""
from __future__ import annotations

import argparse
import dataclasses
import math
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from calibration.transform_utils import load_transform_json
from camera.pointcloud_preprocess import crop_workspace, voxel_downsample
from camera.realsense_pipeline_reader import RealSensePipelineReader
from robot.urdf_model import URDFModel
from robot.robot_state_reader import RealRobotStateReader, MockRobotStateReader
from utils.config import load_config_dir

# ── 默认参数 ────────────────────────────────────────────────────
ROBOT_REMOVAL_THRESHOLD = 0.05     # 5 cm
MESH_SAMPLE_POINTS = 50000          # 网格预采样点数（预采样后会被体素压缩）
MAX_RAW_POINTS = 100000             # 每帧原始点云上限（超过则随机下采样，避免体素降采样瓶颈）
MAX_SCENE_POINTS = 30000            # 场景点云上限
ROBOT_VOXEL_SIZE = 0.01            # 机器人表面点 KD-tree 前体素压缩分辨率（= threshold）
PROCESS_INTERVAL = 1               # 每 N 帧执行一次去除（其余帧复用掩码）
TIMING_BUCKETS = 30                 # 每多少帧打印一次详细耗时


# ── data class for processed frame ──────────────────────────────

@dataclasses.dataclass
class ProcessedFrame:
    """单帧处理结果，供外部程序调用。"""
    scene_points: np.ndarray          # (N, 3)  环境点云
    robot_points: np.ndarray          # (M, 3)  机械臂表面点云
    timestamp: float = 0.0            # 帧时间戳


# ── mock camera for testing ────────────────────────────────────

@dataclasses.dataclass
class _MockFrame:
    points_cam: np.ndarray
    timestamp: float = 0.0


class _MockReader:
    """生成工作空间内的随机点云用于测试。"""
    def __init__(self, n_points: int = 5000):
        self._n = n_points
        self._t = 0.0

    def read(self):
        self._t += 0.033
        n_bg = self._n - 200
        bg = np.random.uniform(-0.8, 0.8, (n_bg, 3))
        cx, cy, cz = 0.3 * math.cos(self._t * 0.5), 0.0, 0.3 + 0.2 * math.sin(self._t * 0.3)
        cluster = np.random.normal(0, 0.03, (200, 3)) + [cx, cy, cz]
        return _MockFrame(points_cam=np.vstack([bg, cluster]))

    def stop(self):
        pass


# ── scene helpers ──────────────────────────────────────────────

def _static_geometries():
    """坐标系指示：原点和线框球（辅助观察）。"""
    import open3d as o3d

    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
    sphere.compute_vertex_normals()
    sphere.paint_uniform_color([0.25, 0.25, 0.70])
    wire = o3d.geometry.LineSet.create_from_triangle_mesh(sphere)
    wire.paint_uniform_color([0.25, 0.25, 0.70])

    origin = o3d.geometry.TriangleMesh.create_sphere(radius=0.03)
    origin.paint_uniform_color([1.0, 0.0, 0.0])

    return [origin, wire]


# ── 机械臂点云去除器（核心优化类）────────────────────────────────

class RobotPointRemover:
    """高性能机械臂点云去除器。

    原理
    ----
    - 初始化时对每个 link 网格在本地坐标系下采样一次，存为 (N, 3) 数组
    - 每帧只需将预采样点通过 FK 矩阵变换到世界坐标系（纯 numpy 乘法）
    - 使用 scipy KD-tree 进行最近邻距离查询
    - 配合跳帧策略进一步降低计算量
    """

    def __init__(self, urdf: URDFModel,
                 n_samples: int = MESH_SAMPLE_POINTS,
                 threshold: float = ROBOT_REMOVAL_THRESHOLD,
                 process_interval: int = PROCESS_INTERVAL):
        self.urdf = urdf
        self.n_samples = n_samples
        self.threshold = threshold
        self.process_interval = process_interval

        # 本地坐标系预采样点 {link_name: (N, 3)}
        self._local_samples: dict[str, np.ndarray] = {}
        self._load_and_presample()

        # 跳帧状态
        self._frame_counter = 0
        # 上一帧的过滤掩码（跳帧时复用）
        self._last_keep_mask: np.ndarray | None = None
        # 上一帧使用的场景点数量（用于检测掩码是否可复用）
        self._last_n_pts = 0
        # 上一帧的机械臂表面点云（跳帧时复用）
        self._last_robot_pts: np.ndarray = np.empty((0, 3))

        # 耗时统计（外部读取）
        self.last_timings: dict[str, float] = {}

    # ── 初始化 ──────────────────────────────────────────────────

    def _load_and_presample(self):
        """加载网格并预采样（仅执行一次）。"""
        import open3d as o3d

        # 统计实际有 mesh 的 link 数量，避免被无 mesh 的 link 稀释采样点
        meshed_links = [
            ln for ln in self.urdf.links
            if self.urdf.resolve_mesh(ln) is not None
        ]
        n_links = 0
        for link_name in meshed_links:
            mesh_path = self.urdf.resolve_mesh(link_name)
            if mesh_path is None:
                continue
            mesh = o3d.io.read_triangle_mesh(mesh_path)
            if not mesh.has_triangles():
                continue
            self._scale_to_meters(mesh)
            mesh.compute_vertex_normals()
            n_links += 1

            # 在本地坐标系下采样，并合并 visual_origin
            vo = self.urdf.links[link_name]['visual_origin']
            sampled = mesh.sample_points_uniformly(
                number_of_points=max(self.n_samples // max(len(meshed_links), 1), 50))
            pts_local = np.asarray(sampled.points)  # (N, 3)

            # mesh 坐标系 → link 坐标系
            ones = np.ones((pts_local.shape[0], 1))
            pts_h = np.hstack([pts_local, ones])
            pts_link = (vo @ pts_h.T).T[:, :3]
            self._local_samples[link_name] = pts_link

        n_total = sum(v.shape[0] for v in self._local_samples.values())
        print(f'[Remover] Pre-sampled {n_total} points from {n_links} links')

    @staticmethod
    def _scale_to_meters(mesh) -> None:
        extent = mesh.get_axis_aligned_bounding_box().get_extent()
        if np.linalg.norm(extent) > 10:
            mesh.scale(0.001, center=(0, 0, 0))

    # ── 核心方法 ────────────────────────────────────────────────

    def _transform_to_world(self, fk: dict[str, np.ndarray]) -> np.ndarray:
        """将预采样点从 link 坐标系变换到世界坐标系。"""
        all_pts = []
        for link_name, pts_link in self._local_samples.items():
            T = fk.get(link_name, np.eye(4))
            # 避免齐次坐标分配：pts_world = R @ pts^T + t
            pts_world = pts_link @ T[:3, :3].T + T[:3, 3]
            all_pts.append(pts_world)
        return np.vstack(all_pts) if all_pts else np.empty((0, 3))

    def remove(self, scene_pts: np.ndarray,
               fk: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """去除场景点云中的机械臂本体点。

        Returns
        -------
        scene_points : (N, 3)  环境点云（机械臂去除后）
        robot_points : (M, 3)  机械臂表面点云（预采样网格点变换到世界坐标）
        """
        t = {}

        if len(scene_pts) == 0:
            self.last_timings = t
            return scene_pts, np.empty((0, 3))

        # ── 跳帧判断 ──
        run_remove = (self._frame_counter % self.process_interval == 0)
        self._frame_counter += 1

        if not run_remove and self._last_keep_mask is not None:
            if len(scene_pts) == self._last_n_pts:
                t['skip_reuse'] = 0.0
                self.last_timings = t
                return scene_pts[self._last_keep_mask], self._last_robot_pts.copy()

        # ── 场景点云限幅 ──
        t0 = time.perf_counter()
        if len(scene_pts) > MAX_SCENE_POINTS:
            idx = np.random.default_rng().choice(
                len(scene_pts), MAX_SCENE_POINTS, replace=False)
            scene_pts = scene_pts[idx]
        t['cap'] = (time.perf_counter() - t0) * 1000

        # ── 变换预采样点到世界坐标 ──
        t0 = time.perf_counter()
        robot_pts = self._transform_to_world(fk)
        t['transform'] = (time.perf_counter() - t0) * 1000

        if len(robot_pts) == 0:
            self.last_timings = t
            return scene_pts, np.empty((0, 3))

        # ── 体素压缩机器人表面点（减少 KD-tree 规模） ──
        t0 = time.perf_counter()
        robot_pts_full = robot_pts  # 保留完整点云供外部返回
        if len(robot_pts) > 5000:
            robot_pts = voxel_downsample(robot_pts, self.threshold)
        t['robot_voxel'] = (time.perf_counter() - t0) * 1000

        # ── KD-tree 构建 + 查询 ──
        t0 = time.perf_counter()
        tree = cKDTree(robot_pts)
        t['kdtree_build'] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        distances, _ = tree.query(scene_pts, k=1)
        t['kdtree_query'] = (time.perf_counter() - t0) * 1000

        # ── 阈值过滤 ──
        t0 = time.perf_counter()
        keep = distances > self.threshold
        filtered = scene_pts[keep]
        t['filter'] = (time.perf_counter() - t0) * 1000

        # ── 缓存掩码供跳帧复用 ──
        self._last_keep_mask = keep
        self._last_n_pts = len(scene_pts)
        self._last_robot_pts = robot_pts_full

        self.last_timings = t
        return filtered, robot_pts_full


# ── 场景处理器（供外部调用）───────────────────────────────────────

@dataclasses.dataclass
class _CameraTransform:
    """预分解的相机→基座标系变换参数。"""
    R: np.ndarray  # (3, 3)
    t: np.ndarray  # (3,)


class SceneProcessor:
    """一站式场景处理器：采集 → 预处理 → 机械臂去除 → 返回双点云集。

    外部调用示例
    ------------
    processor = SceneProcessor(config_dir='config', urdf_path='urdf/aubo_i16_gripper.urdf',
                               use_real_robot=True)
    for result in processor.run():
        env_pts = result.scene_points    # 处理环境点云
        robot_pts = result.robot_points  # 机械臂表面点云
        break  # 取一帧后退出
    processor.stop()
    """

    def __init__(self,
                 config_dir: str = 'config',
                 urdf_path: str = 'urdf/aubo_i16_gripper.urdf',
                 width: int = 1280,
                 height: int = 720,
                 threshold: float = ROBOT_REMOVAL_THRESHOLD,
                 mesh_samples: int = MESH_SAMPLE_POINTS,
                 voxel_size: float = 0.02,
                 process_interval: int = PROCESS_INTERVAL,
                 use_real_robot: bool = False,
                 use_mock_camera: bool = False):
        self.voxel_size = voxel_size

        # ── 配置 ──
        config = load_config_dir(config_dir)
        extrinsic = load_transform_json(Path(config_dir) / 'camera_extrinsic.json')
        self._workspace = config['workspace']

        # 预分解变换矩阵
        self._cam_xf = _CameraTransform(
            R=extrinsic[:3, :3],
            t=extrinsic[:3, 3],
        )

        # ── robot state reader ──
        if use_real_robot:
            reader = RealRobotStateReader()
            if not reader.connect():
                print('[SceneProcessor] Failed to connect to real robot, falling back to mock')
                reader = MockRobotStateReader()
        else:
            reader = MockRobotStateReader()
        self._state_reader = reader

        # ── URDF ──
        self._urdf = URDFModel(urdf_path)
        movable = self._urdf.movable_joints()
        print(f'[SceneProcessor] URDF  joints={len(self._urdf.joints)}  movable={movable}')

        # ── camera ──
        if use_mock_camera:
            reader = _MockReader()
            print('[SceneProcessor] Using mock camera (random point cloud)')
        else:
            reader = RealSensePipelineReader(width=width, height=height)
        self._reader = reader

        # ── 机械臂点云去除器 ──
        self._remover = RobotPointRemover(
            self._urdf, mesh_samples, threshold, process_interval)

        self._running = False

    def process_frame(self) -> ProcessedFrame:
        """采集并处理一帧，返回环境点云和机械臂点云。"""
        # 1. 读取相机帧
        frame_data = self._reader.read()

        # 2. 点云预处理（坐标变换 + 裁剪 + 降采样）
        raw = frame_data.points_cam
        if len(raw):
            if len(raw) > MAX_RAW_POINTS:
                idx = np.random.default_rng().choice(
                    len(raw), MAX_RAW_POINTS, replace=False)
                raw = raw[idx]
            pts = raw @ self._cam_xf.R.T + self._cam_xf.t
            pts = crop_workspace(pts, self._workspace)
            pts = voxel_downsample(pts, self.voxel_size)
        else:
            pts = np.empty((0, 3))

        # 3. 获取关节角 + FK
        robot_pts = np.empty((0, 3))
        if len(pts) > 0:
            angles = self._state_reader.get_joint_positions()
            fk = self._urdf.link_transforms(angles)
            # 4. 去除机械臂本体点云 → 环境 + 机械臂两套点云
            pts, robot_pts = self._remover.remove(pts, fk)

        return ProcessedFrame(
            scene_points=pts,
            robot_points=robot_pts,
            timestamp=frame_data.timestamp if hasattr(frame_data, 'timestamp') else 0.0,
        )

    def run(self):
        """Generator：持续 yield ProcessedFrame。"""
        self._running = True
        while self._running:
            yield self.process_frame()

    def stop(self):
        """释放资源。"""
        self._running = False
        self._reader.stop()
        if hasattr(self._state_reader, 'disconnect'):
            self._state_reader.disconnect()

    @property
    def remover(self) -> RobotPointRemover:
        """暴露去除器供外部读取耗时统计等。"""
        return self._remover


# ── main ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='场景点云处理 — 自动去除机械臂本体点云（优化版）')
    parser.add_argument('--config', default='config')
    parser.add_argument('--urdf', default='urdf/aubo_i16_gripper.urdf')
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--threshold', type=float, default=ROBOT_REMOVAL_THRESHOLD,
                        help=f'去除阈值 (米，默认 {ROBOT_REMOVAL_THRESHOLD})')
    parser.add_argument('--mesh-samples', type=int, default=MESH_SAMPLE_POINTS,
                        help=f'网格采样点数 (默认 {MESH_SAMPLE_POINTS})')
    parser.add_argument('--process-interval', type=int, default=PROCESS_INTERVAL,
                        help=f'跳帧间隔 (默认 {PROCESS_INTERVAL})')
    parser.add_argument('--voxel-size', type=float, default=0.02,
                        help='体素降采样尺寸 (默认 0.02m)')
    parser.add_argument('--mock-camera', action='store_true',
                        help='使用随机点云代替 RealSense（无需相机）')
    parser.add_argument('--real-robot', action='store_true',
                        help='从真实 AUBO 机器人读取关节角')
    parser.add_argument('--visualize', action='store_true', default=False,
                        help='启用 Open3D 可视化（默认关闭，关闭时仅后台计算）')
    args = parser.parse_args()

    threshold = args.threshold
    n_mesh_samples = args.mesh_samples
    process_interval = args.process_interval
    voxel_size = args.voxel_size
    visualize = args.visualize

    # ── 配置 ──
    config = load_config_dir(args.config)
    extrinsic = load_transform_json(Path(args.config) / 'camera_extrinsic.json')
    workspace = config['workspace']

    # 预分解变换矩阵（避免每帧齐次坐标分配）
    R_cam2base = extrinsic[:3, :3]
    t_cam2base = extrinsic[:3, 3]

    # ── robot state reader ──
    if args.real_robot:
        state_reader = RealRobotStateReader()
        if not state_reader.connect():
            print('Failed to connect to real robot, falling back to mock')
            state_reader = MockRobotStateReader()
    else:
        state_reader = MockRobotStateReader()
    print(f'Using robot state reader: {type(state_reader).__name__}')

    # ── URDF ──
    urdf = URDFModel(args.urdf)
    movable = urdf.movable_joints()
    print(f'URDF  joints={len(urdf.joints)}  movable={movable}')

    # ── camera ──
    if args.mock_camera:
        reader = _MockReader()
        print('Using mock camera (random point cloud)')
    else:
        reader = RealSensePipelineReader(width=args.width, height=args.height)

    # ── 高性能机械臂点云去除器（预采样 + KD-tree + 跳帧） ──
    remover = RobotPointRemover(urdf, n_mesh_samples, threshold, process_interval)

    # ── Open3D 可视化（仅 --visualize 时开启） ──
    if visualize:
        import open3d as o3d
        vis = o3d.visualization.Visualizer()
        vis.create_window(
            window_name='Scene Point Cloud (Robot Points Removed) [Optimized]',
            width=1024, height=768)

        for geom in _static_geometries():
            vis.add_geometry(geom)

        # 环境点云（绿色）
        scene_pcd = o3d.geometry.PointCloud()
        scene_pcd.points = o3d.utility.Vector3dVector(np.empty((0, 3)))
        scene_pcd.paint_uniform_color([0.0, 0.5, 0.0])
        vis.add_geometry(scene_pcd)

        # 机械臂表面点云（红色）
        robot_pcd = o3d.geometry.PointCloud()
        robot_pcd.points = o3d.utility.Vector3dVector(np.empty((0, 3)))
        robot_pcd.paint_uniform_color([0.8, 0.2, 0.2])
        vis.add_geometry(robot_pcd)
    else:
        vis = None
        scene_pcd = None
        robot_pcd = None

    # ── main loop ──
    print('=== Running (Ctrl+C to exit) ===')
    frame = 0
    running = True

    try:
        while running:
            t0 = time.perf_counter()

            # 1. 读取相机帧
            frame_data = reader.read()
            t_read = (time.perf_counter() - t0) * 1000

            # 2. 点云预处理（快速坐标变换 + 裁剪 + 降采样）
            t1 = time.perf_counter()
            raw = frame_data.points_cam
            if len(raw):
                # 原始点云随机下采样 → 避免体素降采样成为瓶颈
                if len(raw) > MAX_RAW_POINTS:
                    idx = np.random.default_rng().choice(
                        len(raw), MAX_RAW_POINTS, replace=False)
                    raw = raw[idx]
                # 快速坐标变换（避免齐次坐标构造）
                pts = raw @ R_cam2base.T + t_cam2base
                pts = crop_workspace(pts, workspace)
                pts = voxel_downsample(pts, voxel_size)
            else:
                pts = np.empty((0, 3))
            t_preproc = (time.perf_counter() - t1) * 1000

            # 3. 获取关节角 + FK
            t2 = time.perf_counter()
            angles = None
            fk = None
            if len(pts) > 0:
                angles = state_reader.get_joint_positions()
                fk = urdf.link_transforms(angles)
            t_fk = (time.perf_counter() - t2) * 1000

            # 4. 去除机械臂本体点云 → 同时得到环境点云 + 机械臂点云
            t3 = time.perf_counter()
            robot_pts = np.empty((0, 3))
            if len(pts) > 0 and fk is not None:
                pts, robot_pts = remover.remove(pts, fk)
            t_remove = (time.perf_counter() - t3) * 1000

            # 5. 可视化更新（仅 --visualize 时）
            if visualize and vis is not None:
                t4 = time.perf_counter()
                scene_pcd.points = o3d.utility.Vector3dVector(pts)
                vis.update_geometry(scene_pcd)

                robot_pcd.points = o3d.utility.Vector3dVector(robot_pts)
                vis.update_geometry(robot_pcd)

                vis.update_renderer()
                t_vis = (time.perf_counter() - t4) * 1000
                running = vis.poll_events()
            else:
                t_vis = 0.0

            elapsed = (time.perf_counter() - t0) * 1000
            frame += 1
            if frame % TIMING_BUCKETS == 0:
                rt = remover.last_timings
                skip_info = ''
                if 'skip_reuse' in rt:
                    skip_info = '  [跳帧复用掩码]'
                print(
                    f'[{frame:4d}]  {elapsed:5.1f} ms  '
                    f'scene={len(pts):5d}  robot={len(robot_pts):5d}  '
                    f'vox={voxel_size:.2f} thr={threshold:.3f}  '
                    f'|  read={t_read:.1f}  '
                    f'proc={t_preproc:.1f}  '
                    f'FK={t_fk:.1f}  '
                    f'remove={t_remove:.1f}  '
                    f'vis={t_vis:.1f}  '
                    f'|  xfrm={rt.get("transform",0):.1f}  '
                    f'rvox={rt.get("robot_voxel",0):.1f}  '
                    f'kdtree={rt.get("kdtree_build",0):.1f}+{rt.get("kdtree_query",0):.1f}  '
                    f'cap={rt.get("cap",0):.1f}'
                    f'{skip_info}')

    except KeyboardInterrupt:
        print('\nInterrupted by user')
    finally:
        if visualize and vis is not None:
            vis.destroy_window()
        reader.stop()
        if args.real_robot and hasattr(state_reader, 'disconnect'):
            state_reader.disconnect()


if __name__ == '__main__':
    main()
