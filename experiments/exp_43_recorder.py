"""Chapter 4.3 recorder with optional synchronized Y-axis robot motion."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from experiments.recorder import Recorder, _make_camera
from robot.robot_commander import RobotCommander
from robot.robot_state_reader import MockRobotStateReader, RealRobotStateReader


JOINT_NAMES = [
    "shoulder_joint",
    "upperArm_joint",
    "foreArm_joint",
    "wrist1_joint",
    "wrist2_joint",
    "wrist3_joint",
    "left_joint",
    "right_joint",
]


def _wait_for_space() -> None:
    """Block until the user presses Space; Enter is accepted for SSH terminals."""
    print("\n[exp_43_recorder] 准备就绪。按 Space 开始录制并启动运动；按 q 退出。", flush=True)
    try:
        import termios
        import tty

        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while True:
                ch = sys.stdin.read(1)
                if ch in (" ", "\n", "\r"):
                    return
                if ch.lower() == "q":
                    raise KeyboardInterrupt
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
    except Exception:
        input("[exp_43_recorder] 当前终端不支持单键读取，按 Enter 开始...")


def _joint_dict_from_commander(commander: RobotCommander) -> dict[str, float]:
    joints = commander.get_joints()
    if joints is None:
        raise RuntimeError("failed to read joints from RobotCommander")
    values = list(joints[:6]) + [-0.02, -0.02]
    return {name: float(values[i]) for i, name in enumerate(JOINT_NAMES)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Record Chapter 4.3 data with synchronized robot Y-axis motion.")
    parser.add_argument("--output", required=True, help="Recording directory, e.g. data/recordings/ch4_3_dynamic_01")
    parser.add_argument("--config", default="config")
    parser.add_argument("--scene", choices=["A", "B", "C"], required=True)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--source", choices=["realsense", "mock"], default="realsense")
    parser.add_argument("--real-robot", action="store_true", help="Read joints from the real robot.")
    parser.add_argument("--auto-y-motion", action="store_true", help="Start real robot Y-axis oscillation after Space.")
    parser.add_argument("--ip", default="192.168.123.96", help="Robot IP for --auto-y-motion.")
    parser.add_argument("--motion-range", type=float, default=0.20, help="Y-axis half range in meters.")
    parser.add_argument("--motion-omega", type=float, default=0.8, help="Y-axis oscillation angular speed.")
    parser.add_argument(
        "--motion-x-offset",
        type=float,
        default=0.0,
        help="Cartesian X offset applied before Y oscillation; negative values reduce X.",
    )
    parser.add_argument("--home-joints-deg", default="0,0,90,0,90,0", help="Comma-separated home joints in degrees.")
    parser.add_argument("--no-wait", action="store_true", help="Start immediately instead of waiting for Space.")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--record-fps",
        type=float,
        default=0.0,
        help="Limit saved frames per second; <=0 saves every camera frame.",
    )
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--obstacle-desc", default="")
    args = parser.parse_args()

    if args.auto_y_motion and not args.real_robot:
        parser.error("--auto-y-motion requires --real-robot")

    camera = _make_camera(args.source, args.width, args.height, args.fps)
    recorder = Recorder(
        args.output,
        config_dir=args.config,
        scene=args.scene,
        obstacle_desc=args.obstacle_desc,
        save_images=args.save_images,
        clean_existing=True,
    )
    recorder.manifest["requested_camera"] = {
        "source": args.source,
        "width": args.width,
        "height": args.height,
        "fps": args.fps,
        "record_fps": args.record_fps,
    }
    recorder.manifest["exp_43_recorder"] = {
        "auto_y_motion": bool(args.auto_y_motion),
        "motion_range": args.motion_range,
        "motion_omega": args.motion_omega,
        "motion_x_offset": args.motion_x_offset,
        "waited_for_space": not args.no_wait,
    }
    if hasattr(camera, "intrinsic"):
        recorder.manifest["intrinsic"] = dict(camera.intrinsic)
        if hasattr(camera, "depth_scale"):
            recorder.manifest["intrinsic"]["depth_scale"] = float(camera.depth_scale)

    commander: RobotCommander | None = None
    state_reader = None
    if args.auto_y_motion:
        home = [float(item.strip()) for item in args.home_joints_deg.split(",") if item.strip()]
        if len(home) != 6:
            raise SystemExit("--home-joints-deg must contain 6 values")
        state_reader = RealRobotStateReader()
        if not state_reader.connect():
            raise SystemExit("[exp_43_recorder] failed to connect robot state reader")
        commander = RobotCommander(ip=args.ip, base_speed=0.05, robot_mod=state_reader.sdk_module)
        if not commander.connect(home_joints_deg=home):
            raise SystemExit("[exp_43_recorder] failed to connect robot for auto motion")
        print(
            f"[exp_43_recorder] robot connected; "
            f"Y motion range=±{args.motion_range:.3f}m, X offset={args.motion_x_offset:+.3f}m"
        )
    elif args.real_robot:
        state_reader = RealRobotStateReader()
        if not state_reader.connect():
            raise SystemExit("[exp_43_recorder] failed to connect robot state reader")
    else:
        state_reader = MockRobotStateReader()

    print(f"[exp_43_recorder] writing to {Path(args.output).resolve()}")
    print("[exp_43_recorder] Ctrl+C 可提前停止，finally 中会停止机器人运动并保存 manifest。")
    if not args.no_wait:
        _wait_for_space()

    if commander is not None:
        commander.start_y_oscillate(
            range_m=args.motion_range,
            base_omega=args.motion_omega,
            x_offset=args.motion_x_offset,
        )

    deadline = time.time() + args.duration
    record_interval = 0.0 if args.record_fps <= 0.0 else 1.0 / args.record_fps
    next_record_time = 0.0
    try:
        while time.time() < deadline:
            frame = camera.read()
            now = time.time()
            if record_interval > 0.0 and now < next_record_time:
                continue
            joints = _joint_dict_from_commander(commander) if commander is not None else state_reader.get_joint_positions()
            recorder.record_frame(frame, joints)
            if record_interval > 0.0:
                next_record_time = now + record_interval
            if recorder.frame_count % max(args.fps, 1) == 0:
                print(f"[exp_43_recorder] frames={recorder.frame_count}")
    except KeyboardInterrupt:
        print("\n[exp_43_recorder] interrupted")
    finally:
        if commander is not None:
            commander.stop()
        if state_reader is not None and hasattr(state_reader, "disconnect"):
            state_reader.disconnect()
        recorder.close()
        if hasattr(camera, "stop"):
            camera.stop()
        print(f"[exp_43_recorder] saved {recorder.frame_count} frames")


if __name__ == "__main__":
    main()
