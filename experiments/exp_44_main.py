"""Chapter 4.4 offline repulsive-vector validation."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from experiments.ref_constructor import ReferenceConstructor, ReferenceFrame
from robot.urdf_model import URDFModel
from test_remove_robot_points_fast import RobotPointRemover
from utils.config import load_config_dir


METHODS_44 = ("apf", "ours_ee_only", "ours_wo_temporal", "ours")


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
    ):
        self.config = load_config_dir(config_dir)
        self.urdf = URDFModel(urdf_path)
        self.remover = RobotPointRemover(self.urdf, n_samples=50000, threshold=delta_r)
        self.ref = ReferenceConstructor(config_dir, urdf_path, bg_eps=bg_eps, robot_exclusion=max(delta_r * 0.7, 0.025))
        self.joint_names = self.urdf.movable_joints()
        self.eps_q = eps_q
        self.eta_th = eta_th
        self.eta_max = eta_max
        self.dt_v = dt_v
        self.max_active = max_active
        self.d_safe = self.config["safety"].get("d_safe", 0.15)
        self.ee_links = self._guess_ee_links()

    def run(
        self,
        record_dir: str | Path,
        empty_record_dir: str | Path | None = None,
        max_frames: int | None = None,
        stride: int = 3,
    ) -> dict[str, Any]:
        series = self.ref.build_series(record_dir, empty_record_dir=empty_record_dir, max_frames=max_frames)
        sampled = [f for i, f in enumerate(series) if i % max(stride, 1) == 0 and len(f.obs_points) > 0]
        method_rows = {}
        for method in METHODS_44:
            records = []
            for frame in sampled:
                q = self._q_vector(frame.joint_dict)
                dot_q = self.repulsive_velocity(method, q, frame)
                if float(np.linalg.norm(dot_q)) <= self.eta_th:
                    continue
                grad = self.distance_gradient(q, frame.obs_points, links=self._links_for_method(method))
                c = cosine(dot_q, grad)
                d0 = self.distance_for_q(q, frame.obs_points, links=None)
                d1 = self.distance_for_q(q + dot_q * self.dt_v, frame.obs_points, links=None)
                records.append(
                    {
                        "frame_index": frame.frame_index,
                        "C_grad": c,
                        "G_rep": float(d1 - d0),
                        "norm": float(np.linalg.norm(dot_q)),
                    }
                )
            records = self._subsample(records, self.max_active)
            method_rows[method] = {
                "active_frames": len(records),
                "C_grad_D": mean_or_none([r["C_grad"] for r in records]),
                "G_rep": mean_or_none([r["G_rep"] for r in records]),
                "R_body": self.body_response_rate(method, sampled),
            }
        return {
            "record_dir": str(record_dir),
            "empty_record_dir": None if empty_record_dir is None else str(empty_record_dir),
            "metrics": method_rows,
        }

    def repulsive_velocity(self, method: str, q: np.ndarray, frame: ReferenceFrame) -> np.ndarray:
        links = self._links_for_method(method)
        d = self.distance_for_q(q, frame.obs_points, links=links)
        if not math.isfinite(d) or d >= self.d_safe:
            return np.zeros(len(self.joint_names))
        grad = self.distance_gradient(q, frame.obs_points, links=links)
        if np.linalg.norm(grad) < 1e-9:
            return np.zeros(len(self.joint_names))
        scale = min(self.eta_max, max(0.0, (self.d_safe - d) / max(self.d_safe, 1e-6)) * self.eta_max)
        if method == "apf":
            scale *= 0.75
        if method == "ours":
            scale *= 1.0 + min(frame.obs_speed, 1.0)
            scale = min(scale, self.eta_max)
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

    def body_response_rate(self, method: str, frames: list[ReferenceFrame]) -> float | None:
        events = 0
        responded = 0
        non_ee_links = set(self.remover._local_samples) - self.ee_links
        for frame in frames:
            q = self._q_vector(frame.joint_dict)
            d_body = self.distance_for_q(q, frame.obs_points, links=non_ee_links)
            d_ee = self.distance_for_q(q, frame.obs_points, links=self.ee_links)
            if d_body < self.d_safe and d_ee > self.d_safe:
                events += 1
                dot_q = self.repulsive_velocity(method, q, frame)
                responded += int(np.linalg.norm(dot_q) > self.eta_th)
        return None if events == 0 else float(responded / events)

    def _q_vector(self, joint_dict: dict[str, float]) -> np.ndarray:
        return np.array([joint_dict.get(name, 0.0) for name in self.joint_names], dtype=float)

    def _guess_ee_links(self) -> set[str]:
        names = set(self.remover._local_samples)
        ee = {name for name in names if "wrist3" in name.lower() or "gripper" in name.lower() or "right" in name.lower() or "left" in name.lower()}
        return ee or set(list(names)[-2:])

    def _links_for_method(self, method: str) -> set[str] | None:
        if method == "ours_ee_only":
            return self.ee_links
        return None

    @staticmethod
    def _subsample(records: list[dict[str, float]], limit: int) -> list[dict[str, float]]:
        if len(records) <= limit:
            return records
        idx = np.linspace(0, len(records) - 1, limit).round().astype(int)
        return [records[i] for i in idx]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    if den < 1e-12:
        return 0.0
    return float(np.dot(a, b) / den)


def normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return np.zeros_like(v) if n < 1e-12 else v / n


def mean_or_none(values: list[float]) -> float | None:
    values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return None if not values else float(np.mean(values))


def table_44(metrics: dict[str, Any]) -> str:
    names = {
        "apf": "APF",
        "ours_ee_only": "Ours-EE only",
        "ours_wo_temporal": "Ours-w/o Temporal",
        "ours": "Ours",
    }
    headers = ["方法", "C_grad_D↑", "G_rep↑", "R_body↑", "active"]
    rows = []
    for method in METHODS_44:
        v = metrics[method]
        rows.append([names[method], fmt(v["C_grad_D"]), fmt(v["G_rep"]), fmt(v["R_body"]), str(v["active_frames"])])
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
        *["| " + " | ".join(row) + " |" for row in rows],
    ])


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Chapter 4.4 repulsive-vector validation.")
    parser.add_argument("--record-dir", required=True)
    parser.add_argument("--empty-record-dir", default=None)
    parser.add_argument("--output", default="data/results/ch4_4")
    parser.add_argument("--config", default="config")
    parser.add_argument("--urdf", default="urdf/aubo_i16_gripper.urdf")
    parser.add_argument("--delta-r", type=float, default=0.05)
    parser.add_argument("--bg-eps", type=float, default=0.03)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--stride", type=int, default=3)
    args = parser.parse_args()

    evaluator = RepulsionEvaluator44(args.config, args.urdf, delta_r=args.delta_r, bg_eps=args.bg_eps)
    result = evaluator.run(args.record_dir, empty_record_dir=args.empty_record_dir, max_frames=args.max_frames, stride=args.stride)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    print(table_44(result["metrics"]))
    print(f"\n[exp_44] saved results to {output}")


if __name__ == "__main__":
    main()
