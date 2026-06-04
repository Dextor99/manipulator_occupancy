import argparse
from dataclasses import dataclass
from pathlib import Path
import time

from calibration.transform_utils import load_transform_json, transform_points
from camera.mock_reader import MockRGBDReader
from camera.pointcloud_preprocess import crop_workspace, voxel_downsample
from camera.realsense_pipeline_reader import RealSensePipelineReader
from perception.clustering import cluster_points
from perception.geometry_fit import make_occupancy_object
from perception.occupancy_object import OccupancyObject
from perception.occupancy_tracker import OccupancyTracker
from perception.self_filter import filter_robot_self_points
from risk.distance_check import min_capsule_sphere_distance, min_capsule_obb_distance
from risk.prediction import predict_risk_spheres
from risk.safety_policy import SafetyDecision, SafetyPolicy
from robot.capsule_model import capsules_from_config, mock_capsules
from robot.robot_state_reader import MockRobotStateReader
from utils.config import load_config_dir
from visualization.open3d_viewer import Open3DViewer
from visualization.plot_logger import CSVLogger


@dataclass
class PipelineResult:
    frames: int
    objects: list[OccupancyObject]
    safety_decision: SafetyDecision


def run_pipeline(source: str, config_dir: str, max_frames: int, visualize: bool = False) -> PipelineResult:
    config = load_config_dir(config_dir)
    extrinsic = load_transform_json(Path(config_dir) / "camera_extrinsic.json")
    workspace = config["workspace"]
    safety_cfg = config["safety"]

    reader = _make_reader(source)
    robot_state = MockRobotStateReader()
    tracker = OccupancyTracker(
        association_distance=safety_cfg.get("association_distance", 0.2),
        alpha=safety_cfg.get("velocity_alpha", 0.3),
        pos_alpha=safety_cfg.get("pos_alpha", 0.3),
        motion_gate=safety_cfg.get("motion_gate", 0.005),
        velocity_dead_zone=safety_cfg.get("velocity_dead_zone", 0.01),
        shape_alpha=safety_cfg.get("shape_alpha", 0.4),
    )
    policy = SafetyPolicy(
        d_safe=safety_cfg.get("d_safe", 0.15),
        d_slow=safety_cfg.get("d_slow", 0.10),
        d_stop=safety_cfg.get("d_stop", 0.05),
    )
    viewer = Open3DViewer(enabled=visualize)
    logger = CSVLogger(Path("data") / "logs" / "pipeline_log.csv")
    objects: list[OccupancyObject] = []
    decision = policy.evaluate(float("inf"))

    for frame_idx in range(max_frames):
        start = time.perf_counter()
        frame = reader.read()
        _ = robot_state.get_joint_positions()
        points_base = transform_points(frame.points_cam, extrinsic)
        cropped = crop_workspace(points_base, workspace)
        downsampled = voxel_downsample(cropped, workspace.get("voxel_size", 0.02))
        capsules = capsules_from_config(config["capsules"]) or mock_capsules()
        external, _robot_points = filter_robot_self_points(
            downsampled,
            capsules,
            margin=safety_cfg.get("self_filter_margin", 0.03),
        )
        clusters = cluster_points(
            external,
            eps=safety_cfg.get("cluster_eps", 0.05),
            min_points=safety_cfg.get("cluster_min_points", 30),
        )
        detections = []
        enable_split = safety_cfg.get("enable_cluster_split", True)
        split_threshold = safety_cfg.get("cluster_split_threshold", 500)
        sub_eps = safety_cfg.get("cluster_sub_eps", 0.12)
        sub_min_pts = safety_cfg.get("cluster_sub_min_points", 10)
        for cluster in clusters:
            if cluster.shape[0] < safety_cfg.get("cluster_min_points", 30):
                continue
            # 子聚类拆分：异常大的簇可能是两个物体被 DBSCAN 合并，尝试拆开
            if enable_split and cluster.shape[0] >= split_threshold:
                sub_clusters = cluster_points(cluster, eps=sub_eps, min_points=sub_min_pts)
                if len(sub_clusters) > 1:
                    for sub in sub_clusters:
                        if sub.shape[0] < safety_cfg.get("cluster_min_points", 30):
                            continue
                        detections.append(
                            make_occupancy_object(sub, timestamp=frame.timestamp, margin=safety_cfg.get("shape_margin", 0.02))
                        )
                    continue
            detections.append(
                make_occupancy_object(cluster, timestamp=frame.timestamp, margin=safety_cfg.get("shape_margin", 0.02))
            )
        objects = tracker.update(detections, timestamp=frame.timestamp)
        # 滤除闪烁簇：只对连续跟踪 ≥ N 帧的物体做安全判断
        # tracker 仍会跟踪所有物体（包括新出现的），积累 age
        # 但安全管道只看见稳物体，避免单帧闪现噪声引发急停
        min_track_age = safety_cfg.get("min_track_age", 3)
        stable_objects = [obj for obj in objects if obj.age >= min_track_age]
        risk_spheres = predict_risk_spheres(
            stable_objects,
            horizon=safety_cfg.get("prediction_horizon", 0.5),
            step=safety_cfg.get("prediction_step", 0.1),
            margin=safety_cfg.get("risk_margin", 0.05),
            uncertainty=safety_cfg.get("prediction_uncertainty", 0.02),
        )
        # 双层距离检查：球体快速筛 → 近距离时 OBB 精确算
        min_distance, nearest_id = min_capsule_sphere_distance(capsules, risk_spheres)
        safe_thr = safety_cfg.get("d_safe", 0.15)
        if min_distance < safe_thr + 0.05:
            obb_dist, obb_id, _ = min_capsule_obb_distance(
                capsules, stable_objects,
                horizon=safety_cfg.get("prediction_horizon", 0.5),
                step=safety_cfg.get("prediction_step", 0.1),
                margin=safety_cfg.get("risk_margin", 0.05),
                uncertainty=safety_cfg.get("prediction_uncertainty", 0.02),
                obb_threshold=safe_thr,
            )
            if obb_dist < min_distance:
                min_distance = obb_dist
                nearest_id = obb_id
        decision = policy.evaluate(min_distance, nearest_id)
        for obj in objects:
            obj.risk = decision.level.value if obj.id == nearest_id else "OBSERVED"
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        logger.write_row(frame_idx, frame.timestamp, len(objects), decision, elapsed_ms)
        viewer.update(external, capsules, objects, risk_spheres)

    viewer.close()
    logger.close()
    return PipelineResult(frames=max_frames, objects=objects, safety_decision=decision)


def _make_reader(source: str):
    if source == "mock":
        return MockRGBDReader()
    if source == "realsense":
        return RealSensePipelineReader(width=1280, height=720, fps=30)
    raise ValueError(f"unknown source: {source}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["mock", "realsense"], default="mock")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--config", default="config")
    parser.add_argument("--max-frames", type=int, default=100)
    args = parser.parse_args()

    result = run_pipeline(args.source, args.config, args.max_frames, args.visualize)
    decision = result.safety_decision
    print(
        f"frames={result.frames} objects={len(result.objects)} "
        f"risk={decision.level.value} min_distance={decision.min_distance:.3f} "
        f"speed_scale={decision.speed_scale:.2f}"
    )


if __name__ == "__main__":
    main()
