import csv
from pathlib import Path


class CSVLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.handle)
        self.writer.writerow(["frame", "timestamp", "object_count", "risk_level", "min_distance", "speed_scale", "elapsed_ms"])

    def write_row(self, frame_idx, timestamp, object_count, decision, elapsed_ms):
        self.writer.writerow(
            [
                frame_idx,
                f"{timestamp:.3f}",
                object_count,
                decision.level.value,
                f"{decision.min_distance:.4f}",
                f"{decision.speed_scale:.3f}",
                f"{elapsed_ms:.3f}",
            ]
        )

    def close(self):
        self.handle.close()
