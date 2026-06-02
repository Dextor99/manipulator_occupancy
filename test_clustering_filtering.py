"""
场景外部占据点云聚类与筛选。

流水线：
  1. [可选] RANSAC 平面分离 → 将桌面/地面从点云移除（蓝色）
  2. DBSCAN 聚类（仅对非平面点）
  3. 对每个聚类施加过滤条件：
     - 点数太少 → 噪声簇，删除
     - 体积太小 → 传感器反光/离群点，删除
     - 距离工作空间边缘太远 → 深度图边缘伪影，删除
     - 深度异常 → 深度值不连续/超范围，删除
  4. 可视化：
     - 机械臂表面点云（红色）
     - 有效外部占据簇（不同颜色，含 OBB 线框 + 球体线框 + 信息标签）
     - 被过滤掉的簇（灰色）
     - 分离出的平面点（蓝色）

外部调用：
  from test_clustering_filtering import FastClusteringFilter
  filter = FastClusteringFilter(scene_pts, robot_pts)
  clusters = filter.clusters          # 有效占据簇列表
  filtered_out = filter.filtered_out  # 被过滤掉的点
  plane_points = filter.plane_points  # 分离出的平面点

命令行用法：

  # ── 基本用法 ──
  python test_clustering_filtering.py --mock-camera --visualize          # 模拟数据
  python test_clustering_filtering.py --real-robot --visualize           # 真实机械臂

  # ── 实时模式（默认） ──
  python test_clustering_filtering.py --mock-camera --visualize

  # ── 单帧调试（采集一帧后退出，适合调参） ──
  python test_clustering_filtering.py --mock-camera --single --visualize

  # ── 显示被过滤的簇和噪声点 ──
  python test_clustering_filtering.py --mock-camera --visualize --show-filtered --show-noise

  # ── 启用平面分离（聚类前先移除桌面/地面） ──
  python test_clustering_filtering.py --real-robot --visualize --remove-planes

  # ── 调整平面分离参数 ──
  python test_clustering_filtering.py --mock-camera --visualize --remove-planes \
      --plane-dist 0.03       # 平面拟合距离阈值（默认 0.02m）
      --max-planes 2          # 最多分离几个平面（默认 1）

  # ── 调整 DBSCAN 和过滤参数 ──
  python test_clustering_filtering.py --mock-camera --visualize \
      --eps 0.08              # DBSCAN 邻域半径
      --min-samples 20        # DBSCAN 核心点最少邻居
      --min-points 50         # 簇最少点数
      --min-volume 0.001      # 簇最小体积（m³）
      --edge-margin 0.08      # 工作空间边界裕度
      --max-z-var 0.10        # Z 方差上限

  # ── 时域去噪（滤除帧间闪烁噪点） ──
  python test_clustering_filtering.py --real-robot --visualize --remove-planes \
      --temporal-denoise                  # 启用时域体素置信度去噪
      --denoise-voxel 0.04               # 体素大小（米，默认 0.04）
      --denoise-conf 3                   # 置信度阈值（默认 3，越大去噪越强）
      --denoise-decay 0.4               # 衰减系数（默认 0.4，越小忘得越快）
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN

from test_remove_robot_points_fast import SceneProcessor, ProcessedFrame, ROBOT_REMOVAL_THRESHOLD
from perception.geometry_fit import create_obb_wireframe, create_sphere_wireframe, create_text_label  # 仅用于可视化，无逻辑依赖

# ── 默认聚类参数 ──────────────────────────────────────────────
DBSCAN_EPS = 0.06           # DBSCAN 邻域半径（米）
DBSCAN_MIN_SAMPLES = 15     # DBSCAN 核心点最少邻居数
CLUSTER_MIN_POINTS = 15     # 聚类最少点数
CLUSTER_MIN_VOLUME = 0.0005 # 聚类最小体积（立方米，≈8cm 立方体）
EDGE_MARGIN = 0.05          # 工作空间边界裕度（米），聚类中心离边界 < 此值则可能是伪影
DEPTH_Z_RANGE = (0.0, 1.5)  # 深度 Z 合理范围
MAX_Z_VARIANCE = 0.15       # 聚类内 Z 方差上限（超过则可能是跨越边缘的伪影）
FRAME_INTERVAL_MS = 50      # 可视化帧间隔控制（毫秒）


# ── 数据结构 ──────────────────────────────────────────────────

@dataclass
class OccupancyCluster:
    """单个外部占据聚类的结果。"""
    points: np.ndarray               # (N, 3) 该簇的点云
    label: int                       # DBSCAN 标签
    center: np.ndarray               # (3,) 几何中心
    bbox_min: np.ndarray             # (3,) AABB 最小值
    bbox_max: np.ndarray             # (3,) AABB 最大值
    volume: float                    # AABB 体积（立方米）
    n_points: int                    # 点数
    passed: bool                     # 是否通过所有过滤
    filter_reasons: list[str]        # 被过滤的原因（passed=True 时为空）


# ── 时域去噪 ──────────────────────────────────────────────────

class TemporalDenoiser:
    """基于体素置信度的时域去噪器，滤除帧间闪烁噪点。

    原理
    ----
    将空间划分为体素网格，每个体素维护一个置信度分数：
      - 当前帧有点落在体素中 → 置信度 +1
      - 当前帧没有点 → 置信度 × decay（衰减）
      - 置信度 < threshold → 该体素中的点作为闪烁噪点剔除

    这样短暂出现一两帧的噪点置信度积累不足，会被滤除；
    而稳定出现的物体（连续多帧都在）置信度逐渐升高，稳定保留。

    用法
    ----
    denoiser = TemporalDenoiser(voxel_size=0.04)
    clean_pts = denoiser.filter(scene_pts)   # 每帧调用一次
    temporal_noise = denoiser.last_noise     # 本帧被滤除的点（可选可视化）
    """

    def __init__(self,
                 voxel_size: float = 0.04,
                 confidence_threshold: int = 3,
                 decay: float = 0.4):
        """
        Parameters
        ----------
        voxel_size : float
            体素大小（米）。越小对细节保留越多，但去噪效果越弱。
            建议 0.03~0.05。
        confidence_threshold : int
            置信度阈值。低于此值的体素中的点被视为噪点剔除。
            越大去噪越强（物体需要更多帧确认）。
            - 2：去掉仅出现1-2帧的点
            - 3：去掉仅出现1-3帧的点（推荐）
            - 4：更强去噪，但运动物体会被削弱
        decay : float
            每帧未命中时的衰减系数 [0, 1]。
            - 0.0：一帧消失立刻清零（只考虑连续出现）
            - 0.4：逐渐衰减（推荐）
            - 1.0：永不衰减（只增不减，不推荐）
        """
        self.voxel_size = max(voxel_size, 0.001)
        self.threshold = confidence_threshold
        self.decay = decay
        self._confidence: dict[int, float] = {}
        self.last_noise: np.ndarray = np.empty((0, 3))

    def reset(self):
        """清空历史置信度（重新开始积累）。"""
        self._confidence.clear()

    def filter(self, points: np.ndarray) -> np.ndarray:
        """输入当前帧点云，返回时域去噪后的点云。

        Parameters
        ----------
        points : (N, 3)  当前帧场景点云

        Returns
        -------
        clean : (M, 3)  置信度 >= threshold 的稳定点
        """
        points = np.asarray(points, dtype=np.float64)
        if len(points) == 0:
            self.last_noise = np.empty((0, 3))
            return points

        # 1. 向量化计算体素键（int64，一步算完）
        idx = np.floor(points / self.voxel_size).astype(np.int64)
        keys = (idx[:, 0] << 40) | (idx[:, 1] << 20) | idx[:, 2]

        # 唯一体素 + 逆映射（一次 np.unique 替代多次 Python 循环）
        unique_keys, inverse = np.unique(keys, return_inverse=True)
        uniq_list = unique_keys.tolist()
        hit_set = set(uniq_list)

        # 2. 置信度更新（只在唯一体素上循环，~100 次）
        conf = self._confidence
        for k, v in list(conf.items()):
            if k in hit_set:
                conf[k] = v + 1.0
            else:
                v *= self.decay
                if v < 0.1:
                    del conf[k]
                else:
                    conf[k] = v
        for k in uniq_list:
            if k not in conf:
                conf[k] = 1.0

        # 3. 向量化过滤（逆映射：纯 numpy 索引，零 Python 循环）
        conf_vals = np.array([conf[k] for k in uniq_list], dtype=np.float64)
        keep = conf_vals[inverse] >= self.threshold

        self.last_noise = points[~keep] if (~keep).any() else np.empty((0, 3))
        return points[keep]


# ── 半径时域平滑 ────────────────────────────────────────────

@dataclass
class _TrackedSphere:
    """单个被跟踪的球体，供时域平滑半径。"""
    center: np.ndarray        # (3,) 当前平滑后的中心
    radius: float             # 当前平滑后的半径
    age: int = 0              # 连续跟踪帧数
    miss: int = 0             # 连续丢失帧数
    _seen_ids: set = field(default_factory=set)


class SphereSmoother:
    """对每个簇做时域跟踪：半径平滑 + 簇稳定度滤波（滤除闪烁簇）。

    用法
    ----
    smoother = SphereSmoother(alpha=0.25, min_age=3, max_miss=5)
    tracks = smoother.update(clusters)   # clusters = [(center, radius), ...]
    # 只有 track.age >= min_age 的簇才是稳定的
    # track.radius 是平滑后的半径
    """

    def __init__(self, alpha: float = 0.25, min_age: int = 3, max_miss: int = 5):
        """
        Parameters
        ----------
        alpha : float
            EMA 半径平滑系数 (0.1~0.3)。
        min_age : int
            簇需连续出现至少 min_age 帧才视为稳定（默认 3 帧）。
        max_miss : int
            丢失后保留的最大帧数，超时自动删除。
        """
        self.alpha = alpha
        self.min_age = min_age
        self.max_miss = max_miss
        self._tracks: dict[int, _TrackedSphere] = {}
        self._next_id = 0
        # 上一次匹配结果（供可视化读取）
        self.tracks: list[_TrackedSphere] = []

    def update(self, clusters: list[tuple[np.ndarray, float]],
               ) -> list[_TrackedSphere]:
        """输入当前帧的簇（中心, 半径），返回平滑后的跟踪记录。

        Parameters
        ----------
        clusters : list of (center, radius)
            当前帧每个有效簇的中心和原始球体半径

        Returns
        -------
        tracks : list of _TrackedSphere
            smoothed_tracks[i] 与 clusters[i] 一一对应
        """
        if not clusters:
            # 没有簇 → 所有 track 计数 miss
            to_del = []
            for tid, t in self._tracks.items():
                t.miss += 1
                if t.miss > self.max_miss:
                    to_del.append(tid)
            for tid in to_del:
                del self._tracks[tid]
            self.tracks = []
            return []

        centers = np.array([c[0] for c in clusters])
        radii = np.array([c[1] for c in clusters])

        # 最近邻匹配：用 scipy KD-tree 做关联
        prev_ids = list(self._tracks.keys())
        if prev_ids:
            prev_centers = np.array([self._tracks[tid].center for tid in prev_ids])
            from scipy.spatial import cKDTree
            tree = cKDTree(prev_centers)
            dists, idxs = tree.query(centers, k=1)

            matched = set()
            for i, (dist, prev_idx) in enumerate(zip(dists, idxs)):
                tid = prev_ids[prev_idx]
                # 匹配距离阈值 25cm
                if dist < 0.25:
                    t = self._tracks[tid]
                    # EMA 平滑
                    t.radius = self.alpha * radii[i] + (1 - self.alpha) * t.radius
                    t.center = self.alpha * centers[i] + (1 - self.alpha) * t.center
                    t.age += 1
                    t.miss = 0
                    matched.add(tid)
                    radii[i] = t.radius
                else:
                    # 新物体
                    new_id = self._next_id
                    self._next_id += 1
                    self._tracks[new_id] = _TrackedSphere(
                        center=centers[i].copy(), radius=float(radii[i]), age=1)
                    matched.add(new_id)

            # 未匹配的旧 track 计数 miss
            to_del = []
            for tid in prev_ids:
                if tid not in matched:
                    self._tracks[tid].miss += 1
                    if self._tracks[tid].miss > self.max_miss:
                        to_del.append(tid)
            for tid in to_del:
                del self._tracks[tid]
        else:
            for i in range(len(clusters)):
                new_id = self._next_id
                self._next_id += 1
                self._tracks[new_id] = _TrackedSphere(
                    center=centers[i].copy(), radius=float(radii[i]), age=1)

        # 按当前帧聚类顺序返回平滑结果
        self.tracks = []
        for i in range(len(clusters)):
            # 找到匹配的 track
            c = clusters[i][0]
            best_id = min(self._tracks.keys(),
                          key=lambda tid: np.linalg.norm(self._tracks[tid].center - c))
            self.tracks.append(self._tracks[best_id])

        return self.tracks


# ── 快速聚类器 ────────────────────────────────────────────────

class FastClusteringFilter:
    """对场景点云进行 DBSCAN 聚类并过滤。

    用法
    ----
    filter = FastClusteringFilter(scene_pts, robot_pts)
    for cluster in filter.clusters:
        print(cluster.center, cluster.volume)
    """

    def __init__(self,
                 scene_pts: np.ndarray,
                 robot_pts: np.ndarray,
                 eps: float = DBSCAN_EPS,
                 min_samples: int = DBSCAN_MIN_SAMPLES,
                 min_points: int = CLUSTER_MIN_POINTS,
                 min_volume: float = CLUSTER_MIN_VOLUME,
                 edge_margin: float = EDGE_MARGIN,
                 workspace: dict | None = None,
                 depth_z_range: tuple[float, float] = DEPTH_Z_RANGE,
                 max_z_variance: float = MAX_Z_VARIANCE,
                 plane_removal: dict | None = None):
        """
        Parameters
        ----------
        scene_pts : (N, 3)  去除机械臂后的环境点云
        robot_pts : (M, 3)  机械臂表面点云
        eps, min_samples : DBSCAN 参数
        min_points : 聚类最少点数（低于此值视为噪声）
        min_volume : 聚类最小 AABB 体积（低于此值视为噪声）
        edge_margin : 到工作空间边界的最小距离，低于此值且靠近边界的簇可能被标记
        workspace : 工作空间配置字典（含 x/y/z 范围或 sphere），用于边缘判断
        depth_z_range : Z 轴合理范围
        max_z_variance : Z 方差上限
        plane_removal : dict or None
            平面分离配置。None 或 {} = 不启用。示例：
            {'enabled': True, 'distance_threshold': 0.02, 'max_planes': 1}
        """
        self.scene_pts = np.asarray(scene_pts, dtype=np.float64)
        self.robot_pts = np.asarray(robot_pts, dtype=np.float64)
        self.eps = eps
        self.min_samples = min_samples
        self.min_points = min_points
        self.min_volume = min_volume
        self.edge_margin = edge_margin
        self.depth_z_range = depth_z_range
        self.max_z_variance = max_z_variance

        # 平面分离配置
        if plane_removal and plane_removal.get('enabled', False):
            self._plane_cfg = {
                'distance_threshold': plane_removal.get('distance_threshold', 0.02),
                'max_planes': plane_removal.get('max_planes', 1),
                'min_plane_points': plane_removal.get('min_plane_points', 50),
            }
        else:
            self._plane_cfg = None

        # 工作空间边界（AABB + 球体）
        self._ws: dict[str, Any] = workspace or {}

        # 聚类结果
        self._raw_labels: np.ndarray | None = None     # DBSCAN 原始标签
        self._clusters: list[OccupancyCluster] = []      # 所有被识别的簇
        self._noise_points: np.ndarray = np.empty((0, 3))  # DBSCAN 标记为噪声的点
        self._filtered_out_pts: np.ndarray = np.empty((0, 3))  # 被过滤掉的簇点
        self._plane_points: np.ndarray = np.empty((0, 3))  # 分离出的平面点
        self._plane_normals: list[np.ndarray] = []  # 平面法向量

        self._run()

    # ── 公共属性 ──────────────────────────────────────────────

    @property
    def clusters(self) -> list[OccupancyCluster]:
        """通过所有过滤的有效占据簇。"""
        return [c for c in self._clusters if c.passed]

    @property
    def filtered_out(self) -> list[OccupancyCluster]:
        """被过滤掉的簇。"""
        return [c for c in self._clusters if not c.passed]

    @property
    def noise_points(self) -> np.ndarray:
        """DBSCAN 标为噪声的离散点。"""
        return self._noise_points

    @property
    def filtered_out_points(self) -> np.ndarray:
        """被过滤掉的所有点（被淘汰簇的点云）。"""
        return self._filtered_out_pts

    @property
    def all_points(self) -> np.ndarray:
        """场景点云中属于某个簇（无论是否通过过滤）的所有点。"""
        pts_list = [c.points for c in self._clusters]
        return np.vstack(pts_list) if pts_list else np.empty((0, 3))

    @property
    def plane_points(self) -> np.ndarray:
        """被 RANSAC 分离出的平面/地面点（桌、地板等）。"""
        return self._plane_points

    @property
    def plane_normals(self) -> list[np.ndarray]:
        """每个分离平面的法向量（单位向量）。"""
        return self._plane_normals

    # ── 核心处理 ──────────────────────────────────────────────

    def _remove_planes(self) -> np.ndarray:
        """RANSAC 迭代拟合平面，从 scene_pts 中移除平面点。"""
        if self._plane_cfg is None:
            return self.scene_pts

        pts = self.scene_pts
        plane_pts_list: list[np.ndarray] = []
        self._plane_normals = []

        import open3d as o3d

        min_pts = self._plane_cfg['min_plane_points']
        dist_thr = self._plane_cfg['distance_threshold']
        max_iter = self._plane_cfg['max_planes']

        for _ in range(max_iter):
            if len(pts) < min_pts:
                break

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)

            plane_model, inliers = pcd.segment_plane(
                distance_threshold=dist_thr,
                ransac_n=3,
                num_iterations=50,
            )

            if len(inliers) < min_pts:
                break

            a, b, c, _ = plane_model
            normal = np.array([a, b, c], dtype=np.float64)
            norm_len = np.linalg.norm(normal)
            if norm_len > 0:
                normal /= norm_len
            self._plane_normals.append(normal)

            plane_pts_list.append(pts[inliers])
            mask = np.ones(len(pts), dtype=bool)
            mask[inliers] = False
            pts = pts[mask]

        self._plane_points = np.vstack(plane_pts_list) if plane_pts_list else np.empty((0, 3))
        return pts

    def _run(self):
        """执行平面分离 → DBSCAN 聚类 → 分析 → 过滤。"""
        if len(self.scene_pts) < self.min_samples:
            self._noise_points = self.scene_pts.copy()
            return

        # 1. 平面分离（聚类前先移除桌面/地面）
        cluster_pts = self._remove_planes()
        if len(cluster_pts) < self.min_samples:
            self._noise_points = cluster_pts.copy()
            return

        # 2. DBSCAN 聚类
        clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples,
                            metric='euclidean', n_jobs=-1)
        self._raw_labels = clustering.fit_predict(cluster_pts)

        # 3. 按标签分组
        unique_labels = set(self._raw_labels)
        for label in sorted(unique_labels):
            mask = self._raw_labels == label
            pts = cluster_pts[mask]
            if label == -1:
                self._noise_points = pts
                continue

            self._clusters.append(self._analyze_cluster(pts, label))

        # 4. 汇总被过滤掉的点云
        fo_pts = [c.points for c in self._clusters if not c.passed]
        self._filtered_out_pts = np.vstack(fo_pts) if fo_pts else np.empty((0, 3))

    def _analyze_cluster(self, pts: np.ndarray, label: int) -> OccupancyCluster:
        """分析单个簇的各项指标并判断是否通过过滤。"""
        center = np.mean(pts, axis=0)
        bmin = pts.min(axis=0)
        bmax = pts.max(axis=0)
        dims = bmax - bmin
        volume = float(dims[0] * dims[1] * dims[2])
        n_pts = len(pts)
        z_var = float(np.var(pts[:, 2]))

        reasons: list[str] = []

        # ── 过滤条件 1：点数太少 ──
        if n_pts < self.min_points:
            reasons.append(f'too_few_points({n_pts}<{self.min_points})')

        # ── 过滤条件 2：体积太小 ──
        if volume < self.min_volume:
            reasons.append(f'too_small_volume({volume:.6f}<{self.min_volume})')

        # ── 过滤条件 3：深度 Z 异常 ──
        z_min, z_max = self.depth_z_range
        if center[2] < z_min or center[2] > z_max:
            reasons.append(f'depth_out_of_range(z={center[2]:.3f})')
        if z_var > self.max_z_variance:
            reasons.append(f'depth_high_variance(var_z={z_var:.5f})')

        # ── 过滤条件 4：靠近工作空间边缘（深度伪影多发区） ──
        if self._is_near_workspace_boundary(center):
            reasons.append('near_workspace_boundary')

        passed = len(reasons) == 0

        return OccupancyCluster(
            points=pts,
            label=label,
            center=center,
            bbox_min=bmin,
            bbox_max=bmax,
            volume=volume,
            n_points=n_pts,
            passed=passed,
            filter_reasons=reasons,
        )

    def _is_near_workspace_boundary(self, center: np.ndarray) -> bool:
        """判断簇中心是否靠近工作空间边界。

        深度相机在视野边缘的深度值往往不可靠，靠近工作空间裁剪边界的簇
        很可能是物体被裁剪后的不完整点云或传感器伪影。若簇中心到任意
        AABB 边界的距离 < edge_margin，或到球体表面距离 < edge_margin，
        则视为边缘伪影。
        """
        ws = self._ws
        x_range = ws.get('x', [-1.5, 1.5])
        y_range = ws.get('y', [-1.5, 1.5])
        z_range = ws.get('z', [-0.5, 1.8])

        # 计算到 AABB 六个面的最近距离
        dist_to_edges = [
            abs(center[0] - x_range[0]), abs(center[0] - x_range[1]),
            abs(center[1] - y_range[0]), abs(center[1] - y_range[1]),
            abs(center[2] - z_range[0]), abs(center[2] - z_range[1]),
        ]
        min_aabb_edge_dist = min(dist_to_edges)

        # 如果到某个 AABB 边界太近 → 可能是边缘伪影
        if min_aabb_edge_dist < self.edge_margin:
            return True

        # 球体边界判断（如果配置了）
        sphere_cfg = ws.get('sphere')
        if sphere_cfg is not None:
            sp_center = np.asarray(sphere_cfg['center'], dtype=float)
            sp_radius = float(sphere_cfg['radius'])
            dist_from_sphere_center = np.linalg.norm(center - sp_center)
            # 离球体表面太近（边缘区域）
            if abs(sp_radius - dist_from_sphere_center) < self.edge_margin:
                return True

        return False


# ── 可视化 ─────────────────────────────────────────────────────

def _random_colors(n: int, seed: int = 42) -> np.ndarray:
    """生成 n 个不同的随机颜色。"""
    rng = np.random.default_rng(seed)
    colors = rng.uniform(0.3, 0.9, (n, 3))
    # 避免与红色（机械臂，0.8,0.2,0.2）过于接近
    for c in colors:
        while np.linalg.norm(c - [0.8, 0.2, 0.2]) < 0.5:
            c[:] = rng.uniform(0.3, 0.9, 3)
    return colors


def visualize_clusters(scene_pts: np.ndarray,
                       robot_pts: np.ndarray,
                       cluster_result: FastClusteringFilter,
                       window_title: str = 'External Occupancy Clusters',
                       block: bool = True):
    """Open3D 可视化：机械臂点云 + 场景点云聚类结果。

    颜色方案
    --------
    - 红色   : 机械臂表面点云
    - 彩色   : 通过过滤的有效占据簇（每个簇一个颜色）
    - 灰色   : DBSCAN 噪声点
    - 淡灰色 : 被过滤掉的簇点
    """
    import open3d as o3d

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_title, width=1280, height=800)

    # 坐标系
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
    vis.add_geometry(coord)

    # ── 机械臂表面点云（红色） ──
    if len(robot_pts) > 0:
        robot_pcd = o3d.geometry.PointCloud()
        robot_pcd.points = o3d.utility.Vector3dVector(robot_pts)
        robot_pcd.paint_uniform_color([0.8, 0.2, 0.2])
        vis.add_geometry(robot_pcd)

    # ── 通过过滤的有效簇（彩色） ──
    valid_clusters = cluster_result.clusters
    colors = _random_colors(max(len(valid_clusters), 1))
    for i, cluster in enumerate(valid_clusters):
        ci = colors[i % len(colors)]
        # 簇点云
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cluster.points)
        pcd.paint_uniform_color(ci)
        vis.add_geometry(pcd)

        # ── 球体线框（绿色调） ──
        sphere_radius = float(np.max(np.linalg.norm(
            cluster.points - cluster.center, axis=1))) + 0.02
        sphere_ls = create_sphere_wireframe(
            cluster.center, sphere_radius, color=(0.0, 1.0, 0.5))
        vis.add_geometry(sphere_ls)

        # ── OBB 线框（橙色） ──
        if len(cluster.points) >= 4:
            obb_ls = create_obb_wireframe(
                cluster.points, color=ci, line_width=2)
            vis.add_geometry(obb_ls)

        # ── 中心标记 ──
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.015)
        sphere.translate(cluster.center)
        sphere.paint_uniform_color(ci)
        vis.add_geometry(sphere)

        # ── 信息标签 ──
        text_str = f'#{i} c=({cluster.center[0]:.2f},{cluster.center[1]:.2f},{cluster.center[2]:.2f})\n{cluster.n_points}pts {cluster.volume:.3f}m³'
        label_pos = cluster.center + np.array([0, 0, 0.08])
        label = create_text_label(label_pos, text_str, color=ci, size=0.025)
        vis.add_geometry(label)

    # ── 被过滤掉的簇（淡灰色） ──
    if len(cluster_result.filtered_out_points) > 0:
        fo_pcd = o3d.geometry.PointCloud()
        fo_pcd.points = o3d.utility.Vector3dVector(cluster_result.filtered_out_points)
        fo_pcd.paint_uniform_color([0.6, 0.6, 0.6])
        vis.add_geometry(fo_pcd)

    # ── 分离出的平面/地面点（蓝色半透明） ──
    if len(cluster_result.plane_points) > 0:
        plane_pcd = o3d.geometry.PointCloud()
        plane_pcd.points = o3d.utility.Vector3dVector(cluster_result.plane_points)
        plane_pcd.paint_uniform_color([0.2, 0.4, 0.9])
        vis.add_geometry(plane_pcd)

    # ── DBSCAN 噪声点（暗灰色） ──
    if len(cluster_result.noise_points) > 0:
        noise_pcd = o3d.geometry.PointCloud()
        noise_pcd.points = o3d.utility.Vector3dVector(cluster_result.noise_points)
        noise_pcd.paint_uniform_color([0.3, 0.3, 0.3])
        vis.add_geometry(noise_pcd)

    # ── 信息叠加（中心坐标标签用文本——Open3D 不支持直接文本，改用小球） ──
    print(f'\n=== Cluster Summary ===')
    print(f'  Valid clusters     : {len(valid_clusters)}')
    for i, cl in enumerate(valid_clusters):
        print(f'    [{i}] center=({cl.center[0]:.3f}, {cl.center[1]:.3f}, {cl.center[2]:.3f})  '
              f'pts={cl.n_points}  vol={cl.volume:.5f}')
    print(f'  Filtered-out clus  : {len(cluster_result.filtered_out)}')
    for cl in cluster_result.filtered_out:
        print(f'    label={cl.label}  reasons={cl.filter_reasons}  '
              f'pts={cl.n_points}  vol={cl.volume:.5f}')
    print(f'  Noise points       : {len(cluster_result.noise_points)}')
    print(f'  Plane points       : {len(cluster_result.plane_points)}  '
          f'n_planes={len(cluster_result.plane_normals)}')
    print(f'  Scene points total : {len(scene_pts)}')
    print(f'  Robot points total : {len(robot_pts)}')
    print('=' * 40)

    # ── 调整视角：自动适配所有几何体，缩小点尺寸 ──
    vis.reset_view_point(True)
    opt = vis.get_render_option()
    opt.point_size = 2.0

    # ── 渲染循环 ──
    try:
        while vis.poll_events():
            vis.update_renderer()
            if block:
                # 等待用户按 Q 退出
                pass
            else:
                # 非阻塞，仅渲染一帧即可返回
                break
    finally:
        if block:
            vis.destroy_window()


# ── 实时可视化循环（持续处理帧） ─────────────────────────────

def run_live(config_dir: str = 'config',
             urdf_path: str = 'urdf/aubo_i16_gripper.urdf',
             use_real_robot: bool = False,
             use_mock_camera: bool = False,
             visualize: bool = True,
             show_noise: bool = False,
             show_filtered: bool = False,
             plane_removal: dict | None = None,
             temporal_denoise: dict | None = None,
             **kwargs):
    """启动实时场景处理器 + 时域去噪 + 聚类过滤 + 可视化。

    Parameters
    ----------
    show_noise : bool
        是否显示 DBSCAN 噪声点（灰色，默认不显示）
    show_filtered : bool
        是否显示被过滤掉的簇点（灰色，默认不显示）
    plane_removal : dict or None
        平面分离配置（见 FastClusteringFilter）
    temporal_denoise : dict or None
        时域去噪配置。示例：
        {'enabled': True, 'voxel_size': 0.04, 'confidence_threshold': 2, 'decay': 0.5}
        启用后在聚类前先做时域体素置信度滤波，滤除帧间闪烁噪点。
    """
    import open3d as o3d

    processor = SceneProcessor(
        config_dir=config_dir,
        urdf_path=urdf_path,
        use_real_robot=use_real_robot,
        use_mock_camera=use_mock_camera,
    )

    # ── 时域去噪（可选） ──
    denoiser = None
    denoise_noise_pcd = None
    if temporal_denoise and temporal_denoise.get('enabled', False):
        denoiser = TemporalDenoiser(
            voxel_size=temporal_denoise.get('voxel_size', 0.04),
            confidence_threshold=temporal_denoise.get('confidence_threshold', 2),
            decay=temporal_denoise.get('decay', 0.5),
        )
        print(f'[Denoiser] enabled  voxel={temporal_denoise.get("voxel_size",0.04):.3f}m  '
              f'conf_thr={temporal_denoise.get("confidence_threshold",2)}  '
              f'decay={temporal_denoise.get("decay",0.5):.1f}')

    # ── 球体半径时域平滑 ──
    sphere_smoother = SphereSmoother(alpha=0.25, max_miss=5)

    # ── Open3D 可视化初始化 ──
    if visualize:
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name='Occupancy Clustering — Live',
                          width=1280, height=800)
        coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
        vis.add_geometry(coord)

        # 预创建几何体（方便每帧更新）
        robot_pcd = o3d.geometry.PointCloud()
        robot_pcd.paint_uniform_color([0.8, 0.2, 0.2])
        vis.add_geometry(robot_pcd)

        scene_pcd = o3d.geometry.PointCloud()
        scene_pcd.paint_uniform_color([0.0, 0.6, 0.0])
        vis.add_geometry(scene_pcd)

        # 被过滤掉的簇点（灰色，按 show_filtered 决定显示）
        filtered_pcd = o3d.geometry.PointCloud()
        filtered_pcd.paint_uniform_color([0.6, 0.6, 0.6])
        vis.add_geometry(filtered_pcd)

        # 噪声点（暗灰色，按 show_noise 决定显示）
        noise_pcd = o3d.geometry.PointCloud()
        noise_pcd.paint_uniform_color([0.3, 0.3, 0.3])
        vis.add_geometry(noise_pcd)

        # 时域去噪滤掉的点（紫色，仅 denoiser 启用时更新）
        denoise_noise_pcd = o3d.geometry.PointCloud()
        denoise_noise_pcd.paint_uniform_color([0.6, 0.2, 0.8])
        vis.add_geometry(denoise_noise_pcd)

        # 分离出的平面点（蓝色）
        plane_pcd = o3d.geometry.PointCloud()
        plane_pcd.paint_uniform_color([0.2, 0.4, 0.9])
        vis.add_geometry(plane_pcd)

        # 预分配几何线框和标记（最多支持 MAX_GEO_CLUSTERS 个簇）
        MAX_GEO_CLUSTERS = 10
        obb_lines = []     # 每个簇的 OBB 线框
        sphere_lines = []  # 每个簇的球体线框
        center_pts = o3d.geometry.PointCloud()  # 中心标记
        vis.add_geometry(center_pts)
        for _ in range(MAX_GEO_CLUSTERS):
            ls = o3d.geometry.LineSet()
            ls.points = o3d.utility.Vector3dVector(np.empty((0, 3)))
            vis.add_geometry(ls)
            obb_lines.append(ls)
            ls2 = o3d.geometry.LineSet()
            ls2.points = o3d.utility.Vector3dVector(np.empty((0, 3)))
            vis.add_geometry(ls2)
            sphere_lines.append(ls2)

        _toggle = {'show_noise': show_noise, 'show_filtered': show_filtered}
        flags_str = []
        if show_noise:
            flags_str.append('noise')
        if show_filtered:
            flags_str.append('filtered')
        flag_info = f' (showing: {",".join(flags_str)})' if flags_str else ' (all hidden)'
        print(f'  Noise:     {"shown" if show_noise else "hidden"}{flag_info}')
        print(f'  Filtered:  {"shown" if show_filtered else "hidden"}')
        print('  Use --show-noise / --show-filtered to toggle on startup')
    else:
        vis = None
        _toggle = None

    print('=== Live Clustering & Filtering (Ctrl+C to exit) ===')
    frame_idx = 0
    _view_fitted = False

    try:
        for frame_data in processor.run():
            t0 = time.perf_counter()
            scene_pts = frame_data.scene_points
            robot_pts = frame_data.robot_points

            # ── [可选] 时域去噪 ──
            n_denoised = 0
            if denoiser is not None:
                scene_pts = denoiser.filter(scene_pts)
                n_denoised = len(denoiser.last_noise)

            # ── 聚类 + 过滤 ──
            cluster_result = FastClusteringFilter(
                scene_pts, robot_pts,
                workspace=processor._workspace if hasattr(processor, '_workspace') else None,
                plane_removal=plane_removal,
            )

            elapsed = (time.perf_counter() - t0) * 1000
            frame_idx += 1

            # ── 控制台摘要 ──
            n_valid = len(cluster_result.clusters)
            n_filtered = len(cluster_result.filtered_out)
            n_noise = len(cluster_result.noise_points)
            n_plane = len(cluster_result.plane_points)
            d_tag = f'  denoised={n_denoised}' if n_denoised > 0 else ''
            print(f'[{frame_idx:4d}] {elapsed:5.1f}ms  '
                  f'valid={n_valid}  filtered={n_filtered}  noise={n_noise}  '
                  f'plane={n_plane}  scene={len(scene_pts)}  robot={len(robot_pts)}'
                  f'{d_tag}')

            if n_valid > 0:
                for i, cl in enumerate(cluster_result.clusters[:3]):
                    c = cl.center
                    print(f'         cls[{i}] center=({c[0]:.3f},{c[1]:.3f},{c[2]:.3f}) '
                          f'pts={cl.n_points} vol={cl.volume:.5f}')
                if len(cluster_result.clusters) > 3:
                    print(f'         ... and {len(cluster_result.clusters)-3} more')

            # ── 簇时域稳定度滤波 ──
            # 用 SphereSmoother 跟踪每帧的簇，只保留连续出现 >= min_age 帧的簇
            valid_all = cluster_result.clusters
            if valid_all:
                raw_centers = [c.center for c in valid_all]
                raw_radii = [float(np.max(np.linalg.norm(
                    c.points - c.center, axis=1))) + 0.02 for c in valid_all]
                sphere_smoother.update(list(zip(raw_centers, raw_radii)))

                # 过滤出稳定的簇（age >= min_age）
                stable = []
                for cl, track in zip(valid_all, sphere_smoother.tracks):
                    if track.age >= sphere_smoother.min_age:
                        stable.append(cl)
                n_flicker = len(valid_all) - len(stable)
                valid = stable
                if n_flicker > 0:
                    print(f'         (filtered {n_flicker} flickering cluster(s), '
                          f'{len(valid)} stable)')
            else:
                valid = []
                sphere_smoother.update([])

            # ── 可视化更新 ──
            if visualize and vis is not None:
                # 机械臂点云
                robot_pcd.points = o3d.utility.Vector3dVector(robot_pts)

                # 有效聚类：不同颜色
                if valid:
                    colors = _random_colors(len(valid))
                    all_valid_pts = np.vstack([c.points for c in valid])
                    all_colors = np.repeat(colors, [len(c.points) for c in valid], axis=0)
                    scene_pcd.points = o3d.utility.Vector3dVector(all_valid_pts)
                    scene_pcd.colors = o3d.utility.Vector3dVector(all_colors)
                else:
                    scene_pcd.points = o3d.utility.Vector3dVector(np.empty((0, 3)))

                # 被过滤掉的簇点 → 按开关决定
                fo_pts = cluster_result.filtered_out_points if _toggle['show_filtered'] else np.empty((0, 3))
                filtered_pcd.points = o3d.utility.Vector3dVector(fo_pts)

                # 噪声点 → 按开关决定
                nz_pts = cluster_result.noise_points if _toggle['show_noise'] else np.empty((0, 3))
                noise_pcd.points = o3d.utility.Vector3dVector(nz_pts)

                # 分离出的平面点（蓝色）
                plane_pcd.points = o3d.utility.Vector3dVector(cluster_result.plane_points)

                # 时域去噪滤除的点（紫色，仅 denoiser 启用时有数据）
                if denoiser is not None:
                    dn_pts = denoiser.last_noise if _toggle['show_noise'] else np.empty((0, 3))
                    denoise_noise_pcd.points = o3d.utility.Vector3dVector(dn_pts)
                    vis.update_geometry(denoise_noise_pcd)

                # ── 几何线框和标记更新 ──
                n_show = min(len(valid), MAX_GEO_CLUSTERS)

                # 构建稳定簇 → track 的映射（平滑后的球体半径）
                stable_tracks = {}
                if n_show > 0 and sphere_smoother.tracks:
                    from scipy.spatial import cKDTree
                    tc = np.array([t.center for t in sphere_smoother.tracks])
                    if len(tc) > 0:
                        tree = cKDTree(tc)
                        for vi in range(n_show):
                            dist, idx = tree.query(valid[vi].center, k=1)
                            if dist < 0.25:
                                stable_tracks[vi] = sphere_smoother.tracks[idx]

                # 中心标记
                if n_show > 0:
                    centers = np.array([valid[i].center for i in range(n_show)])
                    center_colors = np.array([colors[i] for i in range(n_show)])
                    center_pts.points = o3d.utility.Vector3dVector(centers)
                    center_pts.colors = o3d.utility.Vector3dVector(center_colors)
                else:
                    center_pts.points = o3d.utility.Vector3dVector(np.empty((0, 3)))

                for i in range(MAX_GEO_CLUSTERS):
                    if i < n_show:
                        cl = valid[i]
                        ci = colors[i]
                        obb = create_obb_wireframe(cl.points, color=ci)
                        obb_lines[i].points = obb.points
                        obb_lines[i].lines = obb.lines
                        obb_lines[i].paint_uniform_color(ci)
                        vis.update_geometry(obb_lines[i])
                        # 球体线框（平滑后的半径）
                        if i in stable_tracks:
                            sphere_radius = stable_tracks[i].radius
                            sphere_center = stable_tracks[i].center
                        else:
                            sphere_radius = float(np.max(np.linalg.norm(
                                cl.points - cl.center, axis=1))) + 0.02
                            sphere_center = cl.center
                        sphere = create_sphere_wireframe(
                            sphere_center, sphere_radius, color=(0.0, 1.0, 0.5))
                        sphere_lines[i].points = sphere.points
                        sphere_lines[i].lines = sphere.lines
                        sphere_lines[i].paint_uniform_color((0.0, 1.0, 0.5))
                        vis.update_geometry(sphere_lines[i])
                    else:
                        # 清空未使用的线框
                        obb_lines[i].points = o3d.utility.Vector3dVector(np.empty((0, 3)))
                        vis.update_geometry(obb_lines[i])
                        sphere_lines[i].points = o3d.utility.Vector3dVector(np.empty((0, 3)))
                        vis.update_geometry(sphere_lines[i])

                vis.update_geometry(center_pts)
                vis.update_geometry(robot_pcd)
                vis.update_geometry(scene_pcd)
                vis.update_geometry(filtered_pcd)
                vis.update_geometry(noise_pcd)
                vis.update_geometry(plane_pcd)
                vis.update_renderer()

                # 首帧有数据后自动适配视角 + 缩小点尺寸
                if not _view_fitted and (len(robot_pts) > 0 or len(cluster_result.all_points) > 0):
                    vis.reset_view_point(True)
                    opt = vis.get_render_option()
                    opt.point_size = 2.0
                    _view_fitted = True

                if not vis.poll_events():
                    break

            # 控制帧率
            sleep_ms = FRAME_INTERVAL_MS - elapsed
            if sleep_ms > 0:
                time.sleep(sleep_ms / 1000)

    except KeyboardInterrupt:
        print('\nInterrupted by user')
    finally:
        if visualize and vis is not None:
            vis.destroy_window()
        processor.stop()


# ── 单帧测试（用于调试/验证过滤参数） ─────────────────────

def run_single_frame(config_dir: str = 'config',
                     urdf_path: str = 'urdf/aubo_i16_gripper.urdf',
                     use_real_robot: bool = False,
                     use_mock_camera: bool = False,
                     visualize: bool = True,
                     plane_removal: dict | None = None):
    """采集一帧 → 聚类过滤 → 可视化（适合调试过滤参数）。"""
    processor = SceneProcessor(
        config_dir=config_dir,
        urdf_path=urdf_path,
        use_real_robot=use_real_robot,
        use_mock_camera=use_mock_camera,
    )

    print('Capturing one frame...')
    frame_data = processor.process_frame()
    print(f'  Scene points: {len(frame_data.scene_points)}')
    print(f'  Robot points: {len(frame_data.robot_points)}')

    # 获取 workspace 配置
    ws = None
    try:
        from utils.config import load_config_dir
        cfg = load_config_dir(config_dir)
        ws = cfg.get('workspace')
    except Exception:
        pass

    cluster_result = FastClusteringFilter(
        frame_data.scene_points,
        frame_data.robot_points,
        workspace=ws,
        plane_removal=plane_removal,
    )

    print(f'\nClusters valid  : {len(cluster_result.clusters)}')
    print(f'Filtered out    : {len(cluster_result.filtered_out)}')
    print(f'Noise points    : {len(cluster_result.noise_points)}')

    for i, cl in enumerate(cluster_result.clusters):
        c = cl.center
        print(f'  Valid[{i}] center=({c[0]:.3f},{c[1]:.3f},{c[2]:.3f})  '
              f'pts={cl.n_points}  vol={cl.volume:.5f}  '
              f'dims=({cl.bbox_max[0]-cl.bbox_min[0]:.3f},'
              f'{cl.bbox_max[1]-cl.bbox_min[1]:.3f},'
              f'{cl.bbox_max[2]-cl.bbox_min[2]:.3f})')

    for cl in cluster_result.filtered_out:
        c = cl.center
        print(f'  Filtered[{cl.label}] center=({c[0]:.3f},{c[1]:.3f},{c[2]:.3f})  '
              f'reasons={cl.filter_reasons}')

    if visualize:
        visualize_clusters(
            frame_data.scene_points,
            frame_data.robot_points,
            cluster_result,
            window_title='Single Frame — Occupancy Clusters',
        )

    processor.stop()
    return frame_data, cluster_result


# ── main ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='外部占据点云聚类与筛选 — DBSCAN + 多条件过滤 + 可视化')
    parser.add_argument('--config', default='config')
    parser.add_argument('--urdf', default='urdf/aubo_i16_gripper.urdf')
    parser.add_argument('--mock-camera', action='store_true',
                        help='使用随机点云代替 RealSense')
    parser.add_argument('--real-robot', action='store_true',
                        help='从真实 AUBO 机器人读取关节角')
    parser.add_argument('--visualize', action='store_true', default=False,
                        help='启用 Open3D 可视化')
    parser.add_argument('--single', action='store_true', default=False,
                        help='单帧模式（采集一帧后退出，适合调试参数）')
    parser.add_argument('--eps', type=float, default=DBSCAN_EPS,
                        help=f'DBSCAN 邻域半径 (默认 {DBSCAN_EPS})')
    parser.add_argument('--min-samples', type=int, default=DBSCAN_MIN_SAMPLES,
                        help=f'DBSCAN 核心点最少邻居 (默认 {DBSCAN_MIN_SAMPLES})')
    parser.add_argument('--min-points', type=int, default=CLUSTER_MIN_POINTS,
                        help=f'聚类最少点数 (默认 {CLUSTER_MIN_POINTS})')
    parser.add_argument('--min-volume', type=float, default=CLUSTER_MIN_VOLUME,
                        help=f'聚类最小体积 m³ (默认 {CLUSTER_MIN_VOLUME})')
    parser.add_argument('--edge-margin', type=float, default=EDGE_MARGIN,
                        help=f'工作空间边界裕度 (默认 {EDGE_MARGIN})')
    parser.add_argument('--max-z-var', type=float, default=MAX_Z_VARIANCE,
                        help=f'Z 方差上限 (默认 {MAX_Z_VARIANCE})')
    parser.add_argument('--show-noise', action='store_true', default=False,
                        help='显示 DBSCAN 噪声点（灰色，默认隐藏）')
    parser.add_argument('--show-filtered', action='store_true', default=False,
                        help='显示被过滤掉的簇点（灰色，默认隐藏）')
    parser.add_argument('--remove-planes', action='store_true', default=False,
                        help='聚类前先用 RANSAC 分离平面（桌面/地面）')
    parser.add_argument('--plane-dist', type=float, default=0.02,
                        help='RANSAC 平面拟合的距离阈值 (默认 0.02m)')
    parser.add_argument('--max-planes', type=int, default=1,
                        help='最多分离几个平面 (默认 1)')
    parser.add_argument('--temporal-denoise', action='store_true', default=False,
                        help='启用时域体素置信度去噪（滤除帧间闪烁噪点）')
    parser.add_argument('--denoise-voxel', type=float, default=0.04,
                        help='时域去噪体素大小，米 (默认 0.04)')
    parser.add_argument('--denoise-conf', type=int, default=3,
                        help='时域去噪置信度阈值 (默认 3，越大去噪越强)')
    parser.add_argument('--denoise-decay', type=float, default=0.4,
                        help='时域去噪衰减系数 (默认 0.4，越小忘得越快)')
    args = parser.parse_args()

    # 平面分离配置
    plane_cfg = None
    if args.remove_planes:
        plane_cfg = {
            'enabled': True,
            'distance_threshold': args.plane_dist,
            'max_planes': args.max_planes,
        }

    # 时域去噪配置
    denoise_cfg = None
    if args.temporal_denoise:
        denoise_cfg = {
            'enabled': True,
            'voxel_size': args.denoise_voxel,
            'confidence_threshold': args.denoise_conf,
            'decay': args.denoise_decay,
        }

    if args.single:
        run_single_frame(
            config_dir=args.config,
            urdf_path=args.urdf,
            use_real_robot=args.real_robot,
            use_mock_camera=args.mock_camera,
            visualize=args.visualize,
            plane_removal=plane_cfg,
        )
    else:
        run_live(
            config_dir=args.config,
            urdf_path=args.urdf,
            use_real_robot=args.real_robot,
            use_mock_camera=args.mock_camera,
            visualize=args.visualize,
            show_noise=args.show_noise,
            show_filtered=args.show_filtered,
            plane_removal=plane_cfg,
            temporal_denoise=denoise_cfg,
        )


if __name__ == '__main__':
    main()
