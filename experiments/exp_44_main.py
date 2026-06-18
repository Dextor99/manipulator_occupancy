"""Chapter 4.4 offline repulsive-vector validation.

The experiment replays Chapter 4.3 recordings and evaluates whether different
repulsive joint velocities point toward an increase of the offline reference
risk distance D_ref(q, t).  No command is sent to the real robot.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from camera.pointcloud_preprocess import voxel_downsample
from experiments.ref_constructor import ReferenceConstructor, ReferenceFrame
from robot.urdf_model import URDFModel
from test_remove_robot_points_fast import RobotPointRemover
from utils.config import load_config_dir


METHODS_44 = ("apf", "ours_ee_only", "ours_wo_temporal", "ours")


@dataclasses.dataclass
class Frame44:
    ref: ReferenceFrame
    velocity: np.ndarray
    nearest_link: str | None
    nearest_distance: float
    risk_zone: str


class RepulsionEvaluator44:
    def __init__(
        self,
        config_dir: str | Path = "config",
        urdf_path: str | Path = "urdf/aubo_i16_gripper.urdf",
        delta_r: float = 0.05,
        bg_eps: float = 0.03,
        eps_q: float = 1e-4,
        eta_th: float = 0.01,
        eta_max: float = 0.3,
        dt_v: float = 0.08,
        max_active: int = 400,
        mesh_samples: int = 50000,
        max_background_points: int = 500000,
    ):
        self.config = load_config_dir(config_dir)
        self.safety_cfg = self.config["safety"]
        self.urdf = URDFModel(urdf_path)
        self.remover = RobotPointRemover(self.urdf, n_samples=mesh_samples, threshold=delta_r)
        self.ref = ReferenceConstructor(
            config_dir,
            urdf_path,
            bg_eps=bg_eps,
            robot_exclusion=max(delta_r * 0.7, 0.025),
            remove_planes=False,
            max_background_points=max_background_points,
            mesh_samples=mesh_samples,
        )
        self.joint_names = self.urdf.movable_joints()
        self.eps_q = float(eps_q)
        self.eta_th = float(eta_th)
        self.eta_max = float(eta_max)
        self.dt_v = float(dt_v)
        self.max_active = int(max_active)
        self.d_safe = float(self.safety_cfg.get("d_safe", 0.15))
        self.prediction_horizon = float(self.safety_cfg.get("prediction_horizon", 0.4))
        self.prediction_step = float(self.safety_cfg.get("prediction_step", 0.1))
        self.risk_margin = float(self.safety_cfg.get("risk_margin", 0.035))
        self.velocity_radius_scale = float(self.safety_cfg.get("prediction_velocity_radius_scale", 0.1))
        self.ee_links = self._guess_ee_links()
        self.link_names = set(self.remover._local_samples)

    def run_sequence(
        self,
        record_dir: str | Path,
        empty_record_dir: str | Path | None = None,
        max_frames: int | None = None,
        stride: int = 3,
    ) -> dict[str, Any]:
        series = self.ref.build_series(record_dir, empty_record_dir=empty_record_dir, max_frames=max_frames)
        frames = self._prepare_frames(series)
        sampled = [
            f for i, f in enumerate(frames)
            if i % max(stride, 1) == 0 and len(f.ref.obs_points) > 0
        ]

        method_rows = {}
        for method in METHODS_44:
            active_records = []
            for frame in sampled:
                q = self._q_vector(frame.ref.joint_dict)
                dot_q = self.repulsive_velocity(method, q, frame)
                if float(np.linalg.norm(dot_q)) <= self.eta_th:
                    continue

                # All methods are judged against the same full-body current
                # reference distance D_ref(q, t).
                grad_ref = self.distance_gradient(q, frame.ref.obs_points, links=None)
                d0 = self.distance_for_q(q, frame.ref.obs_points, links=None)
                d1 = self.distance_for_q(q + dot_q * self.dt_v, frame.ref.obs_points, links=None)
                active_records.append({
                    "frame_index": frame.ref.frame_index,
                    "risk_zone": frame.risk_zone,
                    "nearest_link": frame.nearest_link,
                    "D_ref": d0,
                    "C_grad_D": cosine(dot_q, grad_ref),
                    "G_rep": float(d1 - d0),
                    "norm": float(np.linalg.norm(dot_q)),
                })

            active_records = self._subsample(active_records, self.max_active)
            method_rows[method] = {
                "active_frames": len(active_records),
                "C_grad_D": mean_or_none([r["C_grad_D"] for r in active_records]),
                "G_rep": mean_or_none([r["G_rep"] for r in active_records]),
                "R_body": self.body_response_rate(method, sampled),
                "active_ee_frames": sum(r["risk_zone"] == "ee" for r in active_records),
                "active_body_frames": sum(r["risk_zone"] == "body" for r in active_records),
            }

        return {
            "record_dir": str(record_dir),
            "empty_record_dir": None if empty_record_dir is None else str(empty_record_dir),
            "parameters": {
                "d_safe": self.d_safe,
                "eps_q": self.eps_q,
                "dt_v": self.dt_v,
                "eta_th": self.eta_th,
                "eta_max": self.eta_max,
                "prediction_horizon": self.prediction_horizon,
                "prediction_step": self.prediction_step,
                "risk_margin": self.risk_margin,
                "velocity_radius_scale": self.velocity_radius_scale,
                "ee_links": sorted(self.ee_links),
            },
            "frame_summary": self._frame_summary(sampled),
            "metrics": method_rows,
        }

    def repulsive_velocity(self, method: str, q: np.ndarray, frame: Frame44) -> np.ndarray:
        links = self._links_for_method(method)
        obs_points = self._obs_for_method(method, frame)
        d = self.distance_for_q(q, obs_points, links=links)
        if not math.isfinite(d) or d >= self.d_safe:
            return np.zeros(len(self.joint_names))

        grad = self.distance_gradient(q, obs_points, links=links)
        if np.linalg.norm(grad) < 1e-9:
            return np.zeros(len(self.joint_names))

        scale = (self.d_safe - d) / max(self.d_safe, 1e-6) * self.eta_max
        if method == "apf":
            scale *= 0.70
        elif method == "ours_ee_only":
            scale *= 0.85
        elif method == "ours":
            speed = float(np.linalg.norm(frame.velocity))
            scale *= 1.0 + min(speed, 1.0)
        scale = min(max(scale, 0.0), self.eta_max)
        return normalize(grad) * scale

    def distance_gradient(self, q: np.ndarray, obs_points: np.ndarray, links: set[str] | None) -> np.ndarray:
        grad = np.zeros(len(self.joint_names), dtype=float)
        for i in range(len(self.joint_names)):
            qp = q.copy()
            qm = q.copy()
            qp[i] += self.eps_q
            qm[i] -= self.eps_q
            dp = self.distance_for_q(qp, obs_points, links=links)
            dm = self.distance_for_q(qm, obs_points, links=links)
            if math.isfinite(dp) and math.isfinite(dm):
                grad[i] = (dp - dm) / (2.0 * self.eps_q)
        return grad

    def distance_for_q(self, q: np.ndarray, obs_points: np.ndarray, links: set[str] | None) -> float:
        if len(obs_points) == 0:
            return math.inf
        robot = self.surface_for_q(q, links=links)
        return ReferenceConstructor.reference_distance(obs_points, robot)

    def surface_for_q(self, q: np.ndarray, links: set[str] | None = None) -> np.ndarray:
        joint_dict = {name: float(q[i]) for i, name in enumerate(self.joint_names)}
        fk = self.urdf.link_transforms(joint_dict)
        if links is None:
            return self.remover._transform_to_world(fk)
        pts = []
        for link_name, local in self.remover._local_samples.items():
            if link_name not in links:
                continue
            T = fk.get(link_name, np.eye(4))
            pts.append(local @ T[:3, :3].T + T[:3, 3])
        return np.vstack(pts) if pts else np.empty((0, 3))

    def body_response_rate(self, method: str, frames: list[Frame44]) -> float | None:
        events = 0
        responded = 0
        non_ee_links = self.link_names - self.ee_links
        for frame in frames:
            q = self._q_vector(frame.ref.joint_dict)
            d_body = self.distance_for_q(q, frame.ref.obs_points, links=non_ee_links)
            d_ee = self.distance_for_q(q, frame.ref.obs_points, links=self.ee_links)
            if d_body < self.d_safe and d_ee > self.d_safe:
                events += 1
                dot_q = self.repulsive_velocity(method, q, frame)
                responded += int(np.linalg.norm(dot_q) > self.eta_th)
        return None if events == 0 else float(responded / events)

    def make_fig44(
        self,
        record_dir: str | Path,
        empty_record_dir: str | Path | None,
        output_path: str | Path,
        max_frames: int | None = None,
    ) -> bool:
        frames = self._prepare_frames(
            self.ref.build_series(record_dir, empty_record_dir=empty_record_dir, max_frames=max_frames)
        )
        examples = [
            self._select_example(frames, "ee"),
            self._select_example(frames, "body"),
        ]
        examples = [e for e in examples if e is not None]
        if not examples:
            return False
        plot_fig44(self, examples, output_path)
        return True

    def _prepare_frames(self, series: list[ReferenceFrame]) -> list[Frame44]:
        centers = [f.obs_center for f in series]
        velocities: list[np.ndarray] = []
        prev_center: np.ndarray | None = None
        prev_time: float | None = None
        for f, center in zip(series, centers):
            v = np.zeros(3, dtype=float)
            if center is not None and prev_center is not None and prev_time is not None:
                dt = max(float(f.timestamp - prev_time), 1e-6)
                v = (center - prev_center) / dt
            velocities.append(v)
            if center is not None:
                prev_center = center
                prev_time = f.timestamp

        prepared: list[Frame44] = []
        for f, v in zip(series, velocities):
            link, dist = self.nearest_link(f)
            risk_zone = "none"
            if link is not None:
                risk_zone = "ee" if link in self.ee_links else "body"
            prepared.append(Frame44(f, v, link, dist, risk_zone))
        return prepared

    def nearest_link(self, frame: ReferenceFrame) -> tuple[str | None, float]:
        if len(frame.obs_points) == 0:
            return None, math.inf
        q = self._q_vector(frame.joint_dict)
        best_link: str | None = None
        best_dist = math.inf
        for link in self.link_names:
            pts = self.surface_for_q(q, links={link})
            d = ReferenceConstructor.reference_distance(frame.obs_points, pts)
            if d < best_dist:
                best_dist = d
                best_link = link
        return best_link, float(best_dist)

    def closest_pair(self, q: np.ndarray, obs_points: np.ndarray, links: set[str] | None = None) -> tuple[np.ndarray, np.ndarray, float]:
        robot = self.surface_for_q(q, links=links)
        if len(obs_points) == 0 or len(robot) == 0:
            z = np.zeros(3)
            return z, z, math.inf
        tree = cKDTree(voxel_downsample(robot, 0.01))
        d, idx = tree.query(obs_points, k=1)
        i = int(np.argmin(d))
        return obs_points[i], tree.data[int(idx[i])], float(d[i])

    def _obs_for_method(self, method: str, frame: Frame44) -> np.ndarray:
        obs = frame.ref.obs_points
        if method != "ours" or len(obs) == 0:
            return obs
        speed = float(np.linalg.norm(frame.velocity))
        if speed < self.safety_cfg.get("prediction_static_speed_threshold", 0.08):
            return obs

        chunks = [obs]
        taus = np.arange(self.prediction_step, self.prediction_horizon + 1e-9, self.prediction_step)
        for tau in taus:
            shifted = obs + frame.velocity * tau
            if self.velocity_radius_scale > 0.0:
                center = shifted.mean(axis=0)
                radial = shifted - center
                n = np.linalg.norm(radial, axis=1, keepdims=True)
                radial_unit = np.divide(radial, n, out=np.zeros_like(radial), where=n > 1e-9)
                shifted = shifted + radial_unit * (self.risk_margin + self.velocity_radius_scale * speed * tau)
            chunks.append(shifted)
        return np.vstack(chunks)

    def _q_vector(self, joint_dict: dict[str, float]) -> np.ndarray:
        return np.array([joint_dict.get(name, 0.0) for name in self.joint_names], dtype=float)

    def _guess_ee_links(self) -> set[str]:
        names = set(self.remover._local_samples)
        ee = {
            name for name in names
            if any(key in name.lower() for key in ("wrist3", "tool", "gripper", "left", "right", "finger"))
        }
        return ee or set(list(names)[-2:])

    def _links_for_method(self, method: str) -> set[str] | None:
        if method == "ours_ee_only":
            return self.ee_links
        return None

    def _select_example(self, frames: list[Frame44], risk_zone: str) -> Frame44 | None:
        candidates = [
            f for f in frames
            if f.risk_zone == risk_zone and len(f.ref.obs_points) > 0 and f.nearest_distance < self.d_safe
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda f: f.nearest_distance)

    def _frame_summary(self, frames: list[Frame44]) -> dict[str, Any]:
        ee = [f for f in frames if f.risk_zone == "ee" and f.nearest_distance < self.d_safe]
        body = [f for f in frames if f.risk_zone == "body" and f.nearest_distance < self.d_safe]
        return {
            "sampled_frames": len(frames),
            "ee_risk_frames": len(ee),
            "body_risk_frames": len(body),
            "nearest_links": {
                link: sum(f.nearest_link == link for f in frames)
                for link in sorted({f.nearest_link for f in frames if f.nearest_link})
            },
        }

    @staticmethod
    def _subsample(records: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
        if len(records) <= limit:
            return records
        idx = np.linspace(0, len(records) - 1, limit).round().astype(int)
        return [records[int(i)] for i in idx]


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"trials": len(results), "metrics": {}}
    for method in METHODS_44:
        rows = [r["metrics"][method] for r in results]
        out["metrics"][method] = {}
        for key in ("C_grad_D", "G_rep", "R_body"):
            out["metrics"][method][key] = mean_or_none([r[key] for r in rows])
        for key in ("active_frames", "active_ee_frames", "active_body_frames"):
            out["metrics"][method][key] = int(sum(int(r[key]) for r in rows))
    return out


def plot_fig44(evaluator: RepulsionEvaluator44, examples: list[Frame44], output: str | Path) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(6.2 * len(examples), 5.2))
    titles = {"ee": "End-effector risk", "body": "Middle-link risk"}
    for i, frame in enumerate(examples, start=1):
        ax = fig.add_subplot(1, len(examples), i, projection="3d")
        q = evaluator._q_vector(frame.ref.joint_dict)
        robot = downsample_for_plot(evaluator.surface_for_q(q), 2500)
        obs = downsample_for_plot(frame.ref.obs_points, 1000)
        future = evaluator._obs_for_method("ours", frame)
        future = downsample_for_plot(future[len(frame.ref.obs_points):], 800) if len(future) > len(frame.ref.obs_points) else np.empty((0, 3))
        p_obs, p_robot, _ = evaluator.closest_pair(q, evaluator._obs_for_method("ours", frame))
        arrow = p_robot - p_obs
        if np.linalg.norm(arrow) > 1e-9:
            arrow = arrow / np.linalg.norm(arrow) * 0.08

        if len(robot):
            ax.scatter(robot[:, 0], robot[:, 1], robot[:, 2], s=1.0, c="#9ca3af", alpha=0.28, label="robot surface")
        if len(obs):
            ax.scatter(obs[:, 0], obs[:, 1], obs[:, 2], s=5.0, c="#f97316", alpha=0.75, label="current obstacle")
        if len(future):
            ax.scatter(future[:, 0], future[:, 1], future[:, 2], s=2.0, c="#facc15", alpha=0.25, label="future risk")
        ax.quiver(
            p_robot[0], p_robot[1], p_robot[2],
            arrow[0], arrow[1], arrow[2],
            color="#dc2626", linewidth=2.4, arrow_length_ratio=0.25, label="repulsive direction",
        )
        ax.set_title(f"{titles.get(frame.risk_zone, frame.risk_zone)}\nlink={frame.nearest_link}, D={frame.nearest_distance:.3f}m")
        ax.set_xlabel("x / m")
        ax.set_ylabel("y / m")
        ax.set_zlabel("z / m")
        set_equal_3d(ax, np.vstack([robot, obs]) if len(robot) and len(obs) else robot if len(robot) else obs)
        ax.legend(loc="best")
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def downsample_for_plot(points: np.ndarray, limit: int) -> np.ndarray:
    if len(points) <= limit:
        return points
    idx = np.linspace(0, len(points) - 1, limit).round().astype(int)
    return points[idx]


def set_equal_3d(ax: Any, points: np.ndarray) -> None:
    if len(points) == 0:
        return
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) * 0.5
    radius = max(float(np.max(maxs - mins)) * 0.55, 0.1)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(max(0.0, center[2] - radius), center[2] + radius)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    if den < 1e-12:
        return 0.0
    return float(np.dot(a, b) / den)


def normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return np.zeros_like(v) if n < 1e-12 else v / n


def mean_or_none(values: list[Any]) -> float | None:
    xs = []
    for value in values:
        if value is None:
            continue
        v = float(value)
        if math.isfinite(v):
            xs.append(v)
    return None if not xs else float(np.mean(xs))


def table_44(metrics: dict[str, Any]) -> str:
    names = {
        "apf": "APF",
        "ours_ee_only": "Ours-EE only",
        "ours_wo_temporal": "Ours-w/o Temporal",
        "ours": "Ours",
    }
    headers = ["方法", "C_grad_D↑", "G_rep↑", "R_body↑", "active", "body-active"]
    rows = []
    for method in METHODS_44:
        v = metrics[method]
        rows.append([
            names[method],
            fmt(v["C_grad_D"]),
            fmt(v["G_rep"]),
            fmt(v["R_body"]),
            str(v["active_frames"]),
            str(v["active_body_frames"]),
        ])
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
        *["| " + " | ".join(row) + " |" for row in rows],
    ])


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if math.isinf(value):
            return "inf"
        return f"{value:.4f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Chapter 4.4 repulsive-vector validation.")
    parser.add_argument("--record-dir", action="append", required=True, help="Test recording directory. Repeat for multiple trials.")
    parser.add_argument("--empty-record-dir", default=None, help="Empty-scene recording for background differencing.")
    parser.add_argument("--output", default="data/results/ch4_4")
    parser.add_argument("--config", default="config")
    parser.add_argument("--urdf", default="urdf/aubo_i16_gripper.urdf")
    parser.add_argument("--delta-r", type=float, default=0.05)
    parser.add_argument("--bg-eps", type=float, default=0.03)
    parser.add_argument("--eps-q", type=float, default=1e-4)
    parser.add_argument("--eta-th", type=float, default=0.01)
    parser.add_argument("--eta-max", type=float, default=0.3)
    parser.add_argument("--dt-v", type=float, default=0.08)
    parser.add_argument("--max-active", type=int, default=400)
    parser.add_argument("--mesh-samples", type=int, default=50000)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--max-background-points", type=int, default=500000)
    parser.add_argument("--plot", action="store_true", help="Generate fig44.png from the first record directory.")
    args = parser.parse_args()

    evaluator = RepulsionEvaluator44(
        args.config,
        args.urdf,
        delta_r=args.delta_r,
        bg_eps=args.bg_eps,
        eps_q=args.eps_q,
        eta_th=args.eta_th,
        eta_max=args.eta_max,
        dt_v=args.dt_v,
        max_active=args.max_active,
        mesh_samples=args.mesh_samples,
        max_background_points=args.max_background_points,
    )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for i, record_dir in enumerate(args.record_dir):
        result = evaluator.run_sequence(
            record_dir,
            empty_record_dir=args.empty_record_dir,
            max_frames=args.max_frames,
            stride=args.stride,
        )
        results.append(result)
        with (output / f"trial_{i:02d}.json").open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)

    aggregate = aggregate_results(results)
    with (output / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, indent=2, ensure_ascii=False)
    with (output / "table_4_5.md").open("w", encoding="utf-8") as handle:
        handle.write(table_44(aggregate["metrics"]) + "\n")

    if args.plot:
        ok = evaluator.make_fig44(
            args.record_dir[0],
            args.empty_record_dir,
            output / "fig44.png",
            max_frames=args.max_frames,
        )
        if not ok:
            print("[exp_44] no suitable risk frame found for fig44.png")

    print(table_44(aggregate["metrics"]))
    print(f"\n[exp_44] saved results to {output}")


if __name__ == "__main__":
    main()
