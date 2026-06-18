"""Chapter 4.3 object-level spatiotemporal risk warning evaluation."""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from experiments.decoupling_eval import DecouplingEvaluator
from experiments.recorder import load_sequence
from experiments.ref_constructor import ReferenceConstructor, ReferenceFrame, save_reference_series
from perception.geometry_fit import make_occupancy_object
from perception.occupancy_tracker import OccupancyTracker
from risk.prediction import predict_risk_spheres
from risk.safety_policy import RiskLevel, SafetyPolicy
from test_clustering_filtering import FastClusteringFilter
from utils.config import load_config_dir


METHODS_43 = ("dsa", "ssm", "ours_wo_temporal", "ours")
STATE_LEVEL = {RiskLevel.SAFE.value: 0, RiskLevel.WARNING.value: 1, RiskLevel.SLOW.value: 2, RiskLevel.STOP.value: 3}


@dataclasses.dataclass
class MethodSeries:
    method: str
    states: list[str]
    distances: list[float]
    trigger_index: int | None
    trigger_d_ref: float | None
    n_switch: int


def _min_points_to_robot(points: np.ndarray, robot_points: np.ndarray) -> float:
    if len(points) == 0 or len(robot_points) == 0:
        return math.inf
    tree = cKDTree(robot_points)
    d, _ = tree.query(points, k=1)
    return float(np.min(d))


def _min_clusters_to_robot(clusters: list[Any], robot_points: np.ndarray) -> float:
    if not clusters or len(robot_points) == 0:
        return math.inf
    tree = cKDTree(robot_points)
    best = math.inf
    for cluster in clusters:
        if len(cluster.points) == 0:
            continue
        d, _ = tree.query(cluster.points, k=1)
        best = min(best, float(np.min(d)))
    return best


def _min_spheres_to_robot(spheres: list[Any], robot_points: np.ndarray) -> float:
    if not spheres or len(robot_points) == 0:
        return math.inf
    tree = cKDTree(robot_points)
    best = math.inf
    for sphere in spheres:
        d, _ = tree.query(np.asarray(sphere.center), k=1)
        best = min(best, float(d - sphere.radius))
    return best


class WarningEvaluator43:
    def __init__(
        self,
        config_dir: str | Path = "config",
        urdf_path: str | Path = "urdf/aubo_i16_gripper.urdf",
        delta_r: float = 0.05,
        bg_eps: float = 0.03,
        t_req: float = 0.25,
        remove_planes: bool = True,
        max_background_points: int = 500000,
        danger_threshold: float | None = None,
        prediction_velocity_radius_scale: float | None = None,
    ):
        self.config = load_config_dir(config_dir)
        self.safety_cfg = self.config["safety"]
        self.prediction_velocity_radius_scale = (
            self.safety_cfg.get("prediction_velocity_radius_scale", 0.2)
            if prediction_velocity_radius_scale is None
            else float(prediction_velocity_radius_scale)
        )
        self.prediction_horizon = self.safety_cfg.get("prediction_horizon", 0.5)
        self.risk_margin = self.safety_cfg.get("risk_margin", 0.05)
        self.policy = SafetyPolicy(
            d_safe=self.safety_cfg.get("d_safe", 0.15),
            d_slow=self.safety_cfg.get("d_slow", 0.10),
            d_stop=self.safety_cfg.get("d_stop", 0.05),
        )
        self.t_req = float(t_req)
        self.danger_threshold = danger_threshold
        self.ref = ReferenceConstructor(
            config_dir=config_dir,
            urdf_path=urdf_path,
            bg_eps=bg_eps,
            robot_exclusion=max(delta_r * 0.7, 0.025),
            # D_ref is an offline weak truth for external obstacles.  Hand-held
            # boxes/boards are often planar, so RANSAC plane removal can delete
            # the obstacle itself and make D_ref = inf for the whole trial.
            remove_planes=False,
            max_background_points=max_background_points,
        )
        dec = DecouplingEvaluator(
            config_dir=config_dir,
            urdf_path=urdf_path,
            delta_r=delta_r,
            remove_planes=remove_planes,
        )
        self.ours_filter = dec.build_method("ours")
        self.cluster_kwargs = dec.cluster_kwargs
        self.tracker_cfg = {
            "association_distance": self.safety_cfg.get("association_distance", 0.2),
            "alpha": self.safety_cfg.get("velocity_alpha", 0.3),
            "pos_alpha": self.safety_cfg.get("pos_alpha", 0.3),
            "motion_gate": self.safety_cfg.get("motion_gate", 0.005),
            "velocity_dead_zone": self.safety_cfg.get("velocity_dead_zone", 0.01),
            "shape_alpha": self.safety_cfg.get("shape_alpha", 0.4),
        }

    def run_sequence(
        self,
        test_record_dir: str | Path,
        empty_record_dir: str | Path | None = None,
        max_frames: int | None = None,
    ) -> dict[str, Any]:
        empty_tree = self.ref.build_empty_tree(empty_record_dir, max_frames=max_frames)
        trackers = {method: OccupancyTracker(**self.tracker_cfg) for method in METHODS_43}
        raw: dict[str, dict[str, list[Any]]] = {
            method: {"states": [], "distances": []} for method in METHODS_43
        }
        reference: list[dict[str, Any]] = []
        prev_center: np.ndarray | None = None
        prev_time: float | None = None

        for idx, frame in enumerate(load_sequence(test_record_dir)):
            if max_frames is not None and idx >= max_frames:
                break
            ref_frame, prev_center, prev_time = self.ref.frame_reference(
                frame,
                idx,
                empty_tree,
                prev_center=prev_center,
                prev_time=prev_time,
            )
            reference.append(
                {
                    "frame_index": ref_frame.frame_index,
                    "timestamp": ref_frame.timestamp,
                    "d_ref": ref_frame.d_ref,
                    "obs_speed": ref_frame.obs_speed,
                    "obs_count": len(ref_frame.obs_points),
                }
            )
            output = self.ours_filter.filter(frame["points_cam"], frame["joint_dict"])
            clusters = FastClusteringFilter(output.external_points, ref_frame.robot_points, **self.cluster_kwargs).clusters
            detections = [
                make_occupancy_object(cluster.points, timestamp=frame["timestamp"], margin=self.safety_cfg.get("shape_margin", 0.02))
                for cluster in clusters
            ]

            for method in METHODS_43:
                tracked = trackers[method].update(detections, timestamp=frame["timestamp"])
                stable = [obj for obj in tracked if obj.age >= self.safety_cfg.get("min_track_age", 3)]

                if method == "dsa":
                    distance = _min_points_to_robot(output.external_points, ref_frame.robot_points)
                elif method in ("ssm", "ours_wo_temporal"):
                    # Current-frame object-level baselines should use the
                    # observed obstacle surface.  A sphere enclosing a flat box
                    # or board is much larger than its nearest visible surface
                    # and makes SSM trigger unrealistically early.
                    distance = _min_clusters_to_robot(clusters, ref_frame.robot_points)
                else:
                    risk_spheres = predict_risk_spheres(
                        stable,
                        horizon=self.prediction_horizon,
                        step=self.safety_cfg.get("prediction_step", 0.1),
                        margin=self.risk_margin,
                        uncertainty=self.safety_cfg.get("prediction_uncertainty", 0.02),
                        static_speed_threshold=self.safety_cfg.get("prediction_static_speed_threshold", 0.08),
                        static_margin=self.safety_cfg.get("prediction_static_margin", 0.0),
                        velocity_radius_scale=self.prediction_velocity_radius_scale,
                    )
                    distance = _min_spheres_to_robot(risk_spheres, ref_frame.robot_points)

                decision = self.policy.evaluate(distance)
                raw[method]["states"].append(decision.level.value)
                raw[method]["distances"].append(float(distance))

        d_stop = self.safety_cfg.get("d_stop", 0.05)
        d_safe = self.safety_cfg.get("d_safe", 0.15)
        d_danger = float(self.danger_threshold if self.danger_threshold is not None else d_stop)
        danger = danger_index(reference, d_stop=d_danger)
        leave = None if danger is None else leave_index(reference, d_safe=d_safe, start=danger)
        method_series = {
            method: self._build_method_series(method, raw[method], reference)
            for method in METHODS_43
        }
        metrics = {
            method: self._metrics(series, reference, danger, leave)
            for method, series in method_series.items()
        }
        speed = self._median_speed(reference)
        return {
            "record_dir": str(test_record_dir),
            "empty_record_dir": None if empty_record_dir is None else str(empty_record_dir),
            "parameters": {
                "d_safe": d_safe,
                "d_stop": d_stop,
                "d_danger": d_danger,
                "t_req": self.t_req,
                "prediction_horizon": self.prediction_horizon,
                "risk_margin": self.risk_margin,
                "prediction_velocity_radius_scale": self.prediction_velocity_radius_scale,
            },
            "speed_median": speed,
            "danger_index": danger,
            "leave_index": leave,
            "reference": reference,
            "series": {m: dataclasses.asdict(s) for m, s in method_series.items()},
            "metrics": metrics,
        }

    def _build_method_series(self, method: str, raw: dict[str, list[Any]], reference: list[dict[str, Any]]) -> MethodSeries:
        trigger = None
        for i, state in enumerate(raw["states"]):
            if state != RiskLevel.SAFE.value:
                trigger = i
                break
        return MethodSeries(
            method=method,
            states=list(raw["states"]),
            distances=list(raw["distances"]),
            trigger_index=trigger,
            trigger_d_ref=None if trigger is None else reference[trigger]["d_ref"],
            n_switch=count_switches(raw["states"]),
        )

    def _metrics(
        self,
        series: MethodSeries,
        reference: list[dict[str, Any]],
        danger: int | None,
        leave: int | None,
    ) -> dict[str, float | int | None]:
        times = [f["timestamp"] for f in reference]
        t_warn = None if series.trigger_index is None else times[series.trigger_index]
        t_danger = None if danger is None else times[danger]
        if t_danger is None:
            t_lead = None
            miss = None
        elif t_warn is None:
            t_lead = None
            miss = 1.0
        else:
            t_lead = float(t_danger - t_warn)
            miss = float(t_warn > t_danger - self.t_req)

        d_safe = self.safety_cfg.get("d_safe", 0.15)
        false_den = 0
        false_num = 0
        for state, ref in zip(series.states, reference):
            if ref["d_ref"] > d_safe:
                false_den += 1
                false_num += int(state != RiskLevel.SAFE.value)

        recover = None
        if leave is not None:
            for i in range(leave, len(series.states)):
                if series.states[i] == RiskLevel.SAFE.value:
                    recover = float(times[i] - times[leave])
                    break

        return {
            "T_lead": t_lead,
            "R_miss": miss,
            "R_false_time": None if false_den == 0 else float(false_num / false_den),
            "D_trigger_ref": series.trigger_d_ref,
            "T_recover": recover,
            "N_switch": series.n_switch,
        }

    @staticmethod
    def _median_speed(reference: list[dict[str, Any]]) -> float:
        speeds = [f["obs_speed"] for f in reference if f["obs_speed"] > 1e-6 and f["obs_count"] > 0]
        return float(np.median(speeds)) if speeds else 0.0


def danger_index(reference: list[dict[str, Any]], d_stop: float) -> int | None:
    for i, frame in enumerate(reference):
        if frame["d_ref"] <= d_stop:
            return i
    return None


def leave_index(reference: list[dict[str, Any]], d_safe: float, start: int = 0) -> int | None:
    for i in range(max(start, 0), len(reference)):
        if reference[i]["d_ref"] > d_safe:
            return i
    return None


def count_switches(states: list[str]) -> int:
    if not states:
        return 0
    return int(sum(a != b for a, b in zip(states[:-1], states[1:])))


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"trials": len(results), "metrics": {}}
    for method in METHODS_43:
        vals = [r["metrics"][method] for r in results]
        out["metrics"][method] = {}
        for key in ("T_lead", "R_miss", "R_false_time", "D_trigger_ref", "T_recover", "N_switch"):
            xs = [v[key] for v in vals if v[key] is not None]
            out["metrics"][method][key] = None if not xs else float(np.mean(xs))
    return out


def table_43(metrics: dict[str, Any]) -> str:
    headers = ["方法", "T_lead↑", "R_miss↓", "R_false-time↓", "D_trigger_ref", "N_switch↓"]
    names = {
        "dsa": "DSA",
        "ssm": "SSM",
        "ours_wo_temporal": "Ours-w/o Temporal",
        "ours": "Ours",
    }
    rows = []
    for method in METHODS_43:
        v = metrics[method]
        rows.append([
            names[method],
            _fmt(v.get("T_lead")),
            _fmt(v.get("R_miss")),
            _fmt(v.get("R_false_time")),
            _fmt(v.get("D_trigger_ref")),
            _fmt(v.get("N_switch")),
        ])
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
        *["| " + " | ".join(row) + " |" for row in rows],
    ])


def plot_fig43(result: dict[str, Any], output: str | Path) -> None:
    import matplotlib.pyplot as plt

    ref = result["reference"]
    t0 = ref[0]["timestamp"] if ref else 0.0
    ts = np.array([r["timestamp"] - t0 for r in ref])
    d_ref = np.array([r["d_ref"] for r in ref], dtype=float)
    d_ref_plot = np.where(np.isfinite(d_ref), d_ref, np.nan)
    d_safe = result.get("parameters", {}).get("d_safe", 0.15)
    d_stop = result.get("parameters", {}).get("d_stop", 0.05)
    d_danger = result.get("parameters", {}).get("d_danger", d_stop)
    fig, axes = plt.subplots(1 + len(METHODS_43), 1, figsize=(9, 7), sharex=True)
    axes[0].plot(ts, d_ref_plot, label="D_ref")
    axes[0].axhline(d_safe, color="tab:orange", linestyle="--", label="d_safe")
    axes[0].axhline(d_stop, color="tab:red", linestyle="--", label="d_stop")
    if abs(d_danger - d_stop) > 1e-9:
        axes[0].axhline(d_danger, color="tab:purple", linestyle=":", label="d_danger")
    if result.get("danger_index") is not None:
        axes[0].axvline(ts[result["danger_index"]], color="k", linestyle=":", label="t_danger")
    axes[0].set_ylabel("distance (m)")
    axes[0].legend(loc="best")

    for ax, method in zip(axes[1:], METHODS_43):
        states = result["series"][method]["states"]
        ys = [STATE_LEVEL.get(s, 0) for s in states]
        ax.step(ts[: len(ys)], ys, where="post")
        ax.set_yticks([0, 1, 2, 3], ["SAFE", "WARN", "SLOW", "STOP"])
        ax.set_ylabel(method)
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if math.isinf(value):
            return "inf"
        return f"{value:.4f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Chapter 4.3 warning evaluation.")
    parser.add_argument("--record-dir", action="append", required=True, help="Test recording directory. Repeat for multiple trials.")
    parser.add_argument("--empty-record-dir", default=None, help="Empty-scene recording for background differencing.")
    parser.add_argument("--output", default="data/results/ch4_3")
    parser.add_argument("--config", default="config")
    parser.add_argument("--urdf", default="urdf/aubo_i16_gripper.urdf")
    parser.add_argument("--delta-r", type=float, default=0.05)
    parser.add_argument("--bg-eps", type=float, default=0.03)
    parser.add_argument("--t-req", type=float, default=0.25)
    parser.add_argument(
        "--danger-threshold",
        type=float,
        default=None,
        help="Reference distance threshold for t_danger. Defaults to config d_stop; use 0.10 for warning-zone dynamic trials.",
    )
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--max-background-points",
        type=int,
        default=500000,
        help="Maximum downsampled points retained for the empty-scene background KD-tree.",
    )
    parser.add_argument(
        "--prediction-velocity-radius-scale",
        type=float,
        default=None,
        help="Override config prediction_velocity_radius_scale for Ours dynamic risk-sphere expansion.",
    )
    parser.add_argument("--no-remove-planes", action="store_true")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    evaluator = WarningEvaluator43(
        config_dir=args.config,
        urdf_path=args.urdf,
        delta_r=args.delta_r,
        bg_eps=args.bg_eps,
        t_req=args.t_req,
        remove_planes=not args.no_remove_planes,
        max_background_points=args.max_background_points,
        danger_threshold=args.danger_threshold,
        prediction_velocity_radius_scale=args.prediction_velocity_radius_scale,
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for i, record_dir in enumerate(args.record_dir):
        result = evaluator.run_sequence(record_dir, empty_record_dir=args.empty_record_dir, max_frames=args.max_frames)
        results.append(result)
        with (output / f"trial_{i:02d}.json").open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)
        save_reference_series(
            [
                ReferenceFrame(
                    frame_index=r["frame_index"],
                    timestamp=r["timestamp"],
                    joint_dict={},
                    common_points=np.empty((0, 3)),
                    obs_points=np.empty((r["obs_count"], 3)),
                    robot_points=np.empty((0, 3)),
                    d_ref=r["d_ref"],
                    obs_center=None,
                    obs_speed=r["obs_speed"],
                )
                for r in result["reference"]
            ],
            output / f"trial_{i:02d}_reference.json",
        )
        if args.plot and i == 0:
            plot_fig43(result, output / "fig43.png")

    aggregate = aggregate_results(results)
    with (output / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, indent=2, ensure_ascii=False)
    print(table_43(aggregate["metrics"]))
    print(f"\n[exp_43] saved results to {output}")


if __name__ == "__main__":
    main()
