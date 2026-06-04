import numpy as np

from perception.occupancy_object import OccupancyObject


class OccupancyTracker:
    def __init__(self, association_distance: float = 0.2, alpha: float = 0.3,
                 pos_alpha: float = 0.3, motion_gate: float = 0.005,
                 velocity_dead_zone: float = 0.01,
                 shape_alpha: float = 0.4):
        """
        Parameters
        ----------
        association_distance : float
            相邻帧同目标关联最大距离 (m)
        alpha : float
            速度 EMA 平滑系数（越大越灵敏）
        pos_alpha : float
            位置 EMA 平滑系数（抑制点云中心帧间抖动）
        motion_gate : float
            位移门限 (m)。平滑后中心位移低于此值 → 视为静止 → 速度置零
        velocity_dead_zone : float
            速度死区 (m/s)。最终速度模长低于此值 → 强制归零（二次保险）
        shape_alpha : float
            形状 EMA 平滑系数。对 radius / OBB half_lengths 做 EMA，
            抑制点云噪声引起的包围盒/包围球尺寸抖动。
            输出取 max(smoothed, raw) 以确保不低估障碍物尺寸。
        """
        self.association_distance = association_distance
        self.alpha = alpha
        self.pos_alpha = pos_alpha
        self.motion_gate = motion_gate
        self.velocity_dead_zone = velocity_dead_zone
        self.shape_alpha = shape_alpha
        self.next_id = 1
        self.tracks: dict[int, OccupancyObject] = {}
        # 每个 track 的 EMA 平滑后中心（与 raw center 独立）
        self._smoothed_centers: dict[int, np.ndarray] = {}
        # 形状 EMA 缓存
        self._smoothed_radius: dict[int, float] = {}
        self._smoothed_half_lengths: dict[int, np.ndarray] = {}

    def update(self, detections: list[OccupancyObject], timestamp: float) -> list[OccupancyObject]:
        assigned: set[int] = set()
        output = []
        for detection in detections:
            track_id = self._nearest_track_id(detection.center, assigned)
            if track_id is None:
                detection.id = self.next_id
                self.next_id += 1
                detection.age = 1
                detection.velocity = np.zeros(3)
                self._smoothed_centers[detection.id] = detection.center.copy()
                # 新 track：用原始形状初始化平滑缓存
                self._smoothed_radius[detection.id] = detection.radius
                hl = detection.shape.extents.get("half_lengths", None)
                if hl is not None:
                    self._smoothed_half_lengths[detection.id] = np.asarray(hl, dtype=float).copy()
            else:
                track_id = track_id  # type: ignore
                previous = self.tracks[track_id]
                dt = max(float(timestamp - previous.timestamp), 1e-6)

                # ── 1. 位置 EMA 平滑 ──
                prev_smoothed = self._smoothed_centers.get(track_id, previous.center)
                smoothed_center = self.pos_alpha * detection.center + \
                                  (1.0 - self.pos_alpha) * prev_smoothed
                self._smoothed_centers[track_id] = smoothed_center

                # ── 2. 位移门限：位移太小 → 视为静止噪声 ──
                displacement = np.linalg.norm(smoothed_center - prev_smoothed)
                if displacement < self.motion_gate:
                    detection.velocity = np.zeros(3)
                else:
                    raw_velocity = (smoothed_center - prev_smoothed) / dt
                    detection.velocity = self.alpha * raw_velocity + \
                                         (1.0 - self.alpha) * previous.velocity

                # ── 3. 速度死区：最终速度太小 → 强制归零 ──
                speed = np.linalg.norm(detection.velocity)
                if speed < self.velocity_dead_zone:
                    detection.velocity = np.zeros(3)

                # ── 4. 形状 EMA 平滑（包围球半径 + OBB half_lengths） ──
                prev_r = self._smoothed_radius.get(track_id, previous.radius)
                smoothed_r = self.shape_alpha * detection.radius + (1.0 - self.shape_alpha) * prev_r
                self._smoothed_radius[track_id] = smoothed_r
                detection.radius = max(smoothed_r, detection.radius)  # 安全上界

                raw_hl = detection.shape.extents.get("half_lengths", None)
                if raw_hl is not None:
                    prev_hl = self._smoothed_half_lengths.get(
                        track_id,
                        previous.shape.extents.get("half_lengths", raw_hl),
                    )
                    prev_hl = np.asarray(prev_hl, dtype=float)
                    raw_hl_arr = np.asarray(raw_hl, dtype=float)
                    smoothed_hl = self.shape_alpha * raw_hl_arr + (1.0 - self.shape_alpha) * prev_hl
                    self._smoothed_half_lengths[track_id] = smoothed_hl
                    # 安全上界：逐元素取 max
                    safe_hl = np.maximum(smoothed_hl, raw_hl_arr)
                    detection.shape.extents["half_lengths"] = safe_hl
                    # 同步更新 shape.radius（与 OBB 半长一致）
                    detection.shape.radius = float(np.linalg.norm(safe_hl))

                detection.id = track_id
                detection.age = previous.age + 1
                assigned.add(track_id)
            detection.timestamp = timestamp
            detection.confidence = self._confidence(detection)
            output.append(detection)
        # 清理已消失 track 的缓存
        self.tracks = {obj.id: obj for obj in output}
        active = set(self.tracks)
        self._smoothed_centers = {tid: c for tid, c in self._smoothed_centers.items()
                                  if tid in active}
        self._smoothed_radius = {tid: r for tid, r in self._smoothed_radius.items()
                                 if tid in active}
        self._smoothed_half_lengths = {tid: h for tid, h in self._smoothed_half_lengths.items()
                                       if tid in active}
        return output

    def _nearest_track_id(self, center: np.ndarray, assigned: set[int]) -> int | None:
        best_id = None
        best_distance = self.association_distance
        for track_id, track in self.tracks.items():
            if track_id in assigned:
                continue
            distance = float(np.linalg.norm(center - track.center))
            if distance < best_distance:
                best_id = track_id
                best_distance = distance
        return best_id

    @staticmethod
    def _confidence(obj: OccupancyObject) -> float:
        count_score = min(1.0, obj.point_count / 100.0)
        age_score = min(1.0, obj.age / 5.0)
        return float(0.6 * count_score + 0.4 * age_score)
