"""Replay runner for Chapter 4.5 safe motion generation experiments."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from experiments.exp_44_main import RepulsionEvaluator44
from experiments.exp_45_controller import CONTROLLERS_45, make_controller
from experiments.exp_45_eval import aggregate_trials, table_45
from experiments.ref_constructor import ReferenceConstructor
from risk.safety_policy import SafetyPolicy
from utils.config import load_config_dir


def _q_series(ref_series: list, joint_names: list[str]) -> np.ndarray:
    return np.array([[f.joint_dict.get(name, 0.0) for name in joint_names] for f in ref_series], dtype=float)


def _ref_velocities(qs: np.ndarray, ts: np.ndarray) -> np.ndarray:
    if len(qs) == 0:
        return qs
    out = np.zeros_like(qs)
    for i in range(1, len(qs)):
        dt = max(float(ts[i] - ts[i - 1]), 1e-6)
        out[i] = (qs[i] - qs[i - 1]) / dt
    if len(qs) > 1:
        out[0] = out[1]
    return np.clip(out, -0.25, 0.25)


class ExperimentRunner45:
    """Generate controller logs from recorded RGB-D/joint sequences."""

    def __init__(
        self,
        config_dir: str = "config",
        urdf_path: str = "urdf/aubo_i16_gripper.urdf",
        delta_r: float = 0.05,
        bg_eps: float = 0.03,
        output_dir: str | Path = "data/results/ch4_5",
        mesh_samples: int = 10000,
    ):
        self.config = load_config_dir(config_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ref = ReferenceConstructor(
            config_dir,
            urdf_path,
            bg_eps=bg_eps,
            robot_exclusion=max(delta_r * 0.7, 0.025),
            mesh_samples=mesh_samples,
            remove_planes=False,
        )
        self.rep = RepulsionEvaluator44(
            config_dir,
            urdf_path,
            delta_r=delta_r,
            bg_eps=bg_eps,
            mesh_samples=mesh_samples,
        )
        self.policy = SafetyPolicy(
            d_safe=self.config["safety"].get("d_safe", 0.15),
            d_slow=self.config["safety"].get("d_slow", 0.10),
            d_stop=self.config["safety"].get("d_stop", 0.05),
        )

    def run_replay(
        self,
        record_dir: str | Path,
        empty_record_dir: str | Path | None,
        scenario: str,
        controller_name: str,
        trial_id: int = 0,
        max_frames: int | None = None,
    ) -> Path:
        ref_series = self.ref.build_series(record_dir, empty_record_dir=empty_record_dir, max_frames=max_frames)
        eval_frames = self.rep._prepare_frames(ref_series)
        joint_names = self.rep.joint_names
        qs = _q_series(ref_series, joint_names)
        ts_abs = np.array([f.timestamp for f in ref_series], dtype=float)
        ref_qd = _ref_velocities(qs, ts_abs)
        return self._write_controller_log(
            record_dir,
            empty_record_dir,
            scenario,
            controller_name,
            trial_id,
            eval_frames,
            qs,
            ref_qd,
            joint_names,
        )

    def run_replay_all(
        self,
        record_dir: str | Path,
        empty_record_dir: str | Path | None,
        scenario: str,
        controller_names: list[str],
        trial_id: int = 0,
        max_frames: int | None = None,
    ) -> dict[str, Path]:
        ref_series = self.ref.build_series(record_dir, empty_record_dir=empty_record_dir, max_frames=max_frames)
        eval_frames = self.rep._prepare_frames(ref_series)
        joint_names = self.rep.joint_names
        qs = _q_series(ref_series, joint_names)
        ts_abs = np.array([f.timestamp for f in ref_series], dtype=float)
        ref_qd = _ref_velocities(qs, ts_abs)
        return {
            name: self._write_controller_log(
                record_dir,
                empty_record_dir,
                scenario,
                name,
                trial_id,
                eval_frames,
                qs,
                ref_qd,
                joint_names,
            )
            for name in controller_names
        }

    def _write_controller_log(
        self,
        record_dir: str | Path,
        empty_record_dir: str | Path | None,
        scenario: str,
        controller_name: str,
        trial_id: int,
        eval_frames: list,
        qs: np.ndarray,
        ref_qd: np.ndarray,
        joint_names: list[str],
    ) -> Path:
        controller = make_controller(controller_name, self.policy)

        frames = []
        for i, frame in enumerate(eval_frames):
            ref = frame.ref
            t0 = time.perf_counter()
            out = controller.step(ref_qd[i], qs[i], frame, self.rep)
            t_cmd = (time.perf_counter() - t0) * 1000.0
            frames.append(
                {
                    "frame_index": i,
                    "timestamp": ref.timestamp,
                    "d_ref": ref.d_ref,
                    "obs_count": int(len(ref.obs_points)),
                    "obs_speed": float(ref.obs_speed),
                    "estimated_velocity": frame.velocity.tolist(),
                    "nearest_link": frame.nearest_link,
                    "nearest_distance": frame.nearest_distance,
                    "risk_zone": frame.risk_zone,
                    "cmd_velocity": out.cmd_velocity.tolist(),
                    "ref_velocity": ref_qd[i].tolist(),
                    "speed_scale": out.speed_scale,
                    "rep_velocity": out.rep_velocity.tolist(),
                    "safety_state": out.safety_state,
                    "risk_distance": out.risk_distance,
                    "timing": {
                        "T_pre_ms": 0.0,
                        "T_dec_ms": 0.0,
                        "T_obj_ms": 0.0,
                        "T_trk_ms": 0.0,
                        "T_risk_ms": 0.0,
                        "T_rep_ms": t_cmd,
                        "T_cmd_ms": t_cmd,
                        "T_frame_ms": t_cmd,
                    },
                }
            )

        payload = {
            "scenario": scenario,
            "controller": controller_name,
            "trial_id": trial_id,
            "record_dir": str(record_dir),
            "empty_record_dir": None if empty_record_dir is None else str(empty_record_dir),
            "joint_names": joint_names,
            "parameters": {
                "d_safe": self.policy.d_safe,
                "d_slow": self.policy.d_slow,
                "d_stop": self.policy.d_stop,
                "delta_r": self.rep.remover.threshold,
                "bg_eps": self.ref.bg_eps,
            },
            "frames": frames,
        }
        out_path = self.output_dir / f"trial_{scenario}_{controller_name}_{trial_id:02d}.json"
        with out_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Chapter 4.5 replay safe-motion experiments.")
    parser.add_argument("--record-dir", required=True)
    parser.add_argument("--empty-record-dir", default=None)
    parser.add_argument("--scenario", choices=["A", "B", "C"], required=True)
    parser.add_argument("--controller", choices=[*CONTROLLERS_45, "all"], default="all")
    parser.add_argument("--trial-id", type=int, default=0)
    parser.add_argument("--output", default="data/results/ch4_5")
    parser.add_argument("--config", default="config")
    parser.add_argument("--urdf", default="urdf/aubo_i16_gripper.urdf")
    parser.add_argument("--delta-r", type=float, default=0.05)
    parser.add_argument("--bg-eps", type=float, default=0.03)
    parser.add_argument("--mesh-samples", type=int, default=10000)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    runner = ExperimentRunner45(args.config, args.urdf, args.delta_r, args.bg_eps, args.output, args.mesh_samples)
    controllers = list(CONTROLLERS_45) if args.controller == "all" else [args.controller]
    paths = runner.run_replay_all(args.record_dir, args.empty_record_dir, args.scenario, controllers, args.trial_id, args.max_frames)
    for name, path in paths.items():
        print(f"[exp_45] saved {path}")
    rows = {name: aggregate_trials([path]) for name, path in paths.items()}
    print(table_45(rows))


if __name__ == "__main__":
    main()
