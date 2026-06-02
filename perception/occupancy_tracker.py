import numpy as np

from perception.occupancy_object import OccupancyObject


class OccupancyTracker:
    def __init__(self, association_distance: float = 0.2, alpha: float = 0.3):
        self.association_distance = association_distance
        self.alpha = alpha
        self.next_id = 1
        self.tracks: dict[int, OccupancyObject] = {}

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
            else:
                previous = self.tracks[track_id]
                dt = max(float(timestamp - previous.timestamp), 1e-6)
                raw_velocity = (detection.center - previous.center) / dt
                detection.velocity = self.alpha * raw_velocity + (1.0 - self.alpha) * previous.velocity
                detection.id = track_id
                detection.age = previous.age + 1
                assigned.add(track_id)
            detection.timestamp = timestamp
            detection.confidence = self._confidence(detection)
            output.append(detection)
        self.tracks = {obj.id: obj for obj in output}
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
