"""Virtual closed-loop replay for Chapter 4.5 safe motion generation.

This experiment keeps the recorded obstacle observations fixed, but integrates
each controller's joint-velocity command into a simulated robot configuration.
The reference distance is recomputed as D_ref(q_sim, t) at every frame, so
distance metrics can differ across methods without commanding the real robot.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np

from experiments.exp_44_main import Frame44
from experiments.exp_45_controller import CONTROLLERS_45, make_controller
from experiments.exp_45_eval import aggregate_trials, table_45
from experiments.exp_45_runner import ExperimentRunner45, _q_series, _ref_velocities


def _sim_frame(frame: Frame44, d_sim: float, q_sim: np.ndarray, runner: ExperimentRunner45) -> Frame44:
    robot = runner.rep.surface_for_q(q_sim, links=None)
    ref = dataclasses.replace(frame.ref, d_ref=float(d_sim), robot_points=robot)
    return dataclasses.replace(frame, ref=ref, nearest_distance=float(d_sim))


class VirtualClosedLoopRunner45(ExperimentRunner45):
    """Integrate controller outputs and evaluate distances on simulated q."""

    def run_virtual_all(
        self,
        record_dir: str | Path,
        empty_record_dir: str | Path | None,
        scenario: str,
        controller_names: list[str],
        trial_id: int = 0,
        max_frames: int | None = None,
        dt_scale: float = 1.0,
    ) -> dict[str, Path]:
        ref_series = self.ref.build_series(record_dir, empty_record_dir=empty_record_dir, max_frames=max_frames)
        eval_frames = self.rep._prepare_frames(ref_series)
        joint_names = self.rep.joint_names
        qs_recorded = _q_series(ref_series, joint_names)
        ts_abs = np.array([f.timestamp for f in ref_series], dtype=float)
        ref_qd = _ref_velocities(qs_recorded, ts_abs)
        return {
            name: self._write_virtual_log(
                record_dir,
                empty_record_dir,
                scenario,
                name,
                trial_id,
                eval_frames,
                qs_recorded,
                ref_qd,
                ts_abs,
                joint_names,
                dt_scale,
            )
            for name in controller_names
        }

    def _write_virtual_log(
        self,
        record_dir: str | Path,
        empty_record_dir: str | Path | None,
        scenario: str,
        controller_name: str,
        trial_id: int,
        eval_frames: list[Frame44],
        qs_recorded: np.ndarray,
        ref_qd: np.ndarray,
        ts_abs: np.ndarray,
        joint_names: list[str],
        dt_scale: float,
    ) -> Path:
        controller = make_controller(controller_name, self.policy)
        q_sim = qs_recorded[0].copy() if len(qs_recorded) else np.zeros(len(joint_names))
        frames = []

        for i, frame in enumerate(eval_frames):
            ref = frame.ref
            if len(ref.obs_points):
                d_sim = self.rep.distance_for_q(q_sim, ref.obs_points, links=None)
            else:
                d_sim = float("inf")
            frame_sim = _sim_frame(frame, d_sim, q_sim, self)

            t0 = time.perf_counter()
            out = controller.step(ref_qd[i], q_sim, frame_sim, self.rep)
            t_cmd = (time.perf_counter() - t0) * 1000.0

            if i + 1 < len(ts_abs):
                dt = max(float(ts_abs[i + 1] - ts_abs[i]), 1e-6) * dt_scale
            elif i > 0:
                dt = max(float(ts_abs[i] - ts_abs[i - 1]), 1e-6) * dt_scale
            else:
                dt = 0.05 * dt_scale

            frames.append(
                {
                    "frame_index": i,
                    "timestamp": ref.timestamp,
                    "d_ref": float(d_sim),
                    "recorded_d_ref": float(ref.d_ref),
                    "obs_count": int(len(ref.obs_points)),
                    "obs_speed": float(ref.obs_speed),
                    "estimated_velocity": frame.velocity.tolist(),
                    "nearest_link": frame.nearest_link,
                    "nearest_distance": float(d_sim),
                    "risk_zone": frame.risk_zone,
                    "q_sim": q_sim.tolist(),
                    "q_recorded": qs_recorded[i].tolist(),
                    "cmd_velocity": out.cmd_velocity.tolist(),
                    "ref_velocity": ref_qd[i].tolist(),
                    "speed_scale": out.speed_scale,
                    "rep_velocity": out.rep_velocity.tolist(),
                    "safety_state": out.safety_state,
                    "risk_distance": out.risk_distance,
                    "dt": dt,
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
            q_sim = q_sim + out.cmd_velocity * dt

        payload = {
            "mode": "virtual_closed_loop",
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
                "dt_scale": dt_scale,
            },
            "frames": frames,
        }
        out_path = self.output_dir / f"trial_{scenario}_{controller_name}_{trial_id:02d}.json"
        with out_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Chapter 4.5 virtual closed-loop experiments.")
    parser.add_argument("--record-dir", required=True)
    parser.add_argument("--empty-record-dir", default=None)
    parser.add_argument("--scenario", choices=["A", "B", "C"], required=True)
    parser.add_argument("--controller", choices=[*CONTROLLERS_45, "all"], default="all")
    parser.add_argument("--trial-id", type=int, default=0)
    parser.add_argument("--output", default="data/results/ch4_5_virtual")
    parser.add_argument("--config", default="config")
    parser.add_argument("--urdf", default="urdf/aubo_i16_gripper.urdf")
    parser.add_argument("--delta-r", type=float, default=0.05)
    parser.add_argument("--bg-eps", type=float, default=0.03)
    parser.add_argument("--mesh-samples", type=int, default=50000)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--dt-scale", type=float, default=1.0)
    args = parser.parse_args()

    runner = VirtualClosedLoopRunner45(
        args.config,
        args.urdf,
        args.delta_r,
        args.bg_eps,
        args.output,
        args.mesh_samples,
    )
    controllers = list(CONTROLLERS_45) if args.controller == "all" else [args.controller]
    paths = runner.run_virtual_all(
        args.record_dir,
        args.empty_record_dir,
        args.scenario,
        controllers,
        args.trial_id,
        args.max_frames,
        args.dt_scale,
    )
    for path in paths.values():
        print(f"[exp_45_virtual] saved {path}")
    rows = {name: aggregate_trials([path]) for name, path in paths.items()}
    print(table_45(rows))


if __name__ == "__main__":
    main()
