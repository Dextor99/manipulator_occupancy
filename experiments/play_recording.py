"""Play recorded Chapter 4 RGB-D/point-cloud sequences."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

from calibration.transform_utils import load_transform_json
from camera.pointcloud_preprocess import crop_workspace, remove_invalid_points
from utils.config import load_config_dir


def _load_manifest(record_dir: Path) -> dict:
    path = record_dir / "manifest.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _format_time(value: float | None, t0: float | None) -> str:
    if value is None or t0 is None:
        return "t=?"
    return f"t={value - t0:6.2f}s"


def _depth_to_vis(depth: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth)
    valid = depth[np.isfinite(depth) & (depth > 0)]
    if valid.size == 0:
        return np.zeros((*depth.shape[:2], 3), dtype=np.uint8)
    lo, hi = np.percentile(valid, [2, 98])
    if hi <= lo:
        hi = lo + 1.0
    norm = np.clip((depth.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
    img = (norm * 255).astype(np.uint8)
    return cv2.applyColorMap(img, cv2.COLORMAP_TURBO)


def _thin(points: np.ndarray, limit: int) -> np.ndarray:
    if limit <= 0 or len(points) <= limit:
        return points
    idx = np.linspace(0, len(points) - 1, limit).astype(np.int64)
    return points[idx]


def _project_panel(
    points: np.ndarray,
    x_idx: int,
    y_idx: int,
    bounds: tuple[float, float, float, float],
    size: tuple[int, int],
    color: tuple[int, int, int],
    title: str,
) -> np.ndarray:
    width, height = size
    img = np.full((height, width, 3), 248, dtype=np.uint8)
    x0, x1, y0, y1 = bounds
    cv2.rectangle(img, (0, 0), (width - 1, height - 1), (220, 220, 220), 1)
    cv2.putText(img, title, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 40, 40), 2, cv2.LINE_AA)

    if len(points) == 0 or x1 <= x0 or y1 <= y0:
        return img
    xs = points[:, x_idx]
    ys = points[:, y_idx]
    mask = np.isfinite(xs) & np.isfinite(ys) & (xs >= x0) & (xs <= x1) & (ys >= y0) & (ys <= y1)
    xs = xs[mask]
    ys = ys[mask]
    if len(xs) == 0:
        return img

    px = ((xs - x0) / (x1 - x0) * (width - 1)).astype(np.int32)
    py = ((1.0 - (ys - y0) / (y1 - y0)) * (height - 1)).astype(np.int32)
    img[py, px] = color
    return img


def _cloud_view(
    points: np.ndarray,
    *,
    max_points: int,
    base_frame: bool,
    config_dir: Path,
    size: tuple[int, int] = (1280, 720),
) -> np.ndarray:
    points = remove_invalid_points(points)
    if base_frame:
        T = load_transform_json(config_dir / "camera_extrinsic.json")
        points = points @ T[:3, :3].T + T[:3, 3]
        cfg = load_config_dir(config_dir)
        points = crop_workspace(points, cfg["workspace"])
        bounds_left = (-1.0, 1.0, -1.0, 1.0)
        bounds_right = (-1.0, 1.0, -0.1, 1.2)
        title_left = "Base top view: X-Y"
        title_right = "Base front view: X-Z"
        left = _project_panel(_thin(points, max_points), 0, 1, bounds_left, (size[0] // 2, size[1]), (40, 90, 230), title_left)
        right = _project_panel(_thin(points, max_points), 0, 2, bounds_right, (size[0] // 2, size[1]), (40, 150, 70), title_right)
    else:
        points = points[(points[:, 2] > 0.05) & (points[:, 2] < 2.5)]
        bounds_left = (-0.9, 0.9, 0.0, 1.6)
        bounds_right = (-0.9, 0.9, -0.8, 0.8)
        title_left = "Camera top view: X-Z"
        title_right = "Camera image-like view: X-Y"
        pts = _thin(points, max_points)
        left = _project_panel(pts, 0, 2, bounds_left, (size[0] // 2, size[1]), (40, 90, 230), title_left)
        right = _project_panel(pts, 0, 1, bounds_right, (size[0] // 2, size[1]), (40, 150, 70), title_right)
    return np.hstack([left, right])


def _put_overlay(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 42), (0, 0, 0), -1)
    cv2.putText(out, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def _make_frame(
    frame_path: Path,
    *,
    mode: str,
    t0: float | None,
    index: int,
    total: int,
    max_points: int,
    base_frame: bool,
    config_dir: Path,
) -> np.ndarray:
    data = np.load(frame_path)
    timestamp = float(data["timestamp"]) if "timestamp" in data.files else None
    resolved_mode = mode
    if resolved_mode == "auto":
        resolved_mode = "rgb" if "color_image" in data.files else "cloud"

    if resolved_mode == "rgb" and "color_image" in data.files:
        img = data["color_image"]
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = np.asarray(img, dtype=np.uint8)
    elif resolved_mode == "depth" and "depth_image" in data.files:
        img = _depth_to_vis(data["depth_image"])
    else:
        if "points_cam" not in data.files:
            raise RuntimeError(f"{frame_path} has no points_cam/color/depth data")
        img = _cloud_view(
            data["points_cam"].astype(np.float64),
            max_points=max_points,
            base_frame=base_frame,
            config_dir=config_dir,
        )

    overlay = (
        f"{frame_path.parent.parent.name}  frame {index + 1}/{total}  "
        f"{_format_time(timestamp, t0)}  mode={resolved_mode}  "
        "Space=pause  A/D=step  Q=quit"
    )
    return _put_overlay(img, overlay)


def main() -> None:
    parser = argparse.ArgumentParser(description="Play a recorded RGB-D/point-cloud sequence.")
    parser.add_argument("--record-dir", required=True, help="Recording directory containing manifest.json and frames/")
    parser.add_argument("--mode", choices=["auto", "rgb", "depth", "cloud"], default="auto")
    parser.add_argument("--fps", type=float, default=8.0, help="Playback FPS.")
    parser.add_argument("--stride", type=int, default=1, help="Play every Nth frame.")
    parser.add_argument("--start", type=int, default=0, help="0-based frame index to start from.")
    parser.add_argument("--max-points", type=int, default=120000, help="Max points rendered per cloud frame.")
    parser.add_argument("--base-frame", action="store_true", help="Transform cloud to robot base frame before projection.")
    parser.add_argument("--config", default="config")
    parser.add_argument("--output-video", default=None, help="Optional MP4 path to save playback.")
    parser.add_argument("--no-window", action="store_true", help="Do not open OpenCV window; useful with --output-video.")
    args = parser.parse_args()

    record_dir = Path(args.record_dir)
    frame_paths = sorted((record_dir / "frames").glob("frame_*.npz"))
    if not frame_paths:
        raise SystemExit(f"no frame_*.npz found under {record_dir / 'frames'}")
    frame_paths = frame_paths[max(args.start, 0)::max(args.stride, 1)]
    manifest = _load_manifest(record_dir)

    t0 = None
    if frame_paths:
        first = np.load(frame_paths[0])
        if "timestamp" in first.files:
            t0 = float(first["timestamp"])

    print(f"[play_recording] record_dir={record_dir}")
    print(f"[play_recording] frames={len(frame_paths)} mode={args.mode} stride={args.stride}")
    if manifest:
        print(f"[play_recording] scene={manifest.get('scene')} desc={manifest.get('obstacle_desc')}")
    if args.mode != "cloud":
        sample = np.load(frame_paths[0])
        if "color_image" not in sample.files and "depth_image" not in sample.files:
            print("[play_recording] no saved color/depth images; falling back to point-cloud projection")

    writer = None
    paused = False
    i = 0
    delay = max(1, int(1000.0 / max(args.fps, 0.1)))
    try:
        while 0 <= i < len(frame_paths):
            img = _make_frame(
                frame_paths[i],
                mode=args.mode,
                t0=t0,
                index=i,
                total=len(frame_paths),
                max_points=args.max_points,
                base_frame=args.base_frame,
                config_dir=Path(args.config),
            )
            if writer is None and args.output_video:
                out_path = Path(args.output_video)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(out_path), fourcc, max(args.fps, 0.1), (img.shape[1], img.shape[0]))
            if writer is not None and not paused:
                writer.write(img)

            if not args.no_window:
                cv2.imshow("recording playback", img)
                key = cv2.waitKey(0 if paused else delay) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord(" "):
                    paused = not paused
                elif key in (ord("a"), 81):
                    i = max(0, i - 1)
                    paused = True
                    continue
                elif key in (ord("d"), 83):
                    i = min(len(frame_paths) - 1, i + 1)
                    paused = True
                    continue
            if not paused:
                i += 1
        if args.output_video:
            print(f"[play_recording] saved video: {args.output_video}")
    finally:
        if writer is not None:
            writer.release()
        if not args.no_window:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
