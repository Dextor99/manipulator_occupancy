from dataclasses import dataclass
import time

import numpy as np

from camera.depth_to_pointcloud import depth_to_points


@dataclass
class Frame:
    color: np.ndarray
    depth: np.ndarray
    points_cam: np.ndarray
    timestamp: float


class RealSensePipelineReader:
    """RealSense RGB-D 读取器，输出 pipeline 兼容的 Frame（含 points_cam）。"""

    def __init__(self, width: int = 1280, height: int = 720, fps: int = 30):
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RuntimeError("pyrealsense2 is required") from exc

        self.rs = rs
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.align = rs.align(rs.stream.color)
        self.profile = self.pipeline.start(config)

        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        color_profile = self.profile.get_stream(rs.stream.color)
        intr = color_profile.as_video_stream_profile().get_intrinsics()
        self.intrinsic = {
            "fx": intr.fx,
            "fy": intr.fy,
            "cx": intr.ppx,
            "cy": intr.ppy,
        }

    def read(self) -> Frame:
        frames = self.align.process(self.pipeline.wait_for_frames())
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            raise RuntimeError("RealSense frame is incomplete")

        color = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data())

        points_cam = depth_to_points(depth, self.intrinsic, self.depth_scale)

        return Frame(
            color=color,
            depth=depth,
            points_cam=points_cam,
            timestamp=time.time(),
        )

    def stop(self):
        self.pipeline.stop()
