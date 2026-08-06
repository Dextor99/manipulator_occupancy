#!/usr/bin/env python3
"""Audit objective terms for 6.5.2 static-obstacle candidate plans."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from experiments.exp_ccro_stage2 import _load  # noqa: E402
from planning.mesh_risk import StaticObstacleField  # noqa: E402
from planning.nubs_trajectory import NUBSTrajectory6D  # noqa: E402
from plan_652_static_ccro_nubs_from_trial import trajectory_preference_metrics  # noqa: E402
from run_652_static_avoidance import make_evaluator_and_verifier, make_surface_model  # noqa: E402


DEFAULT_TRIAL_DIR = (
    ROOT
    / "results"
    / "new"
    / "6_5"
    / "6_5_2"
    / "planar_static_live"
    / "rs1_lateral_table_obstacle"
    / "trials"
    / "rs1_lateral_table_obstacle_r08"
)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_trajectory(data: Any, key: str) -> NUBSTrajectory6D:
    head = NUBSTrajectory6D.make_boundary_state(data["q_start"])
    tail = NUBSTrajectory6D.make_boundary_state(data["q_goal"])
    return NUBSTrajectory6D().generate(data[key], head, tail, data["durations"])


def optimizer_sample_times(durations: np.ndarray, samples_per_segment: int) -> np.ndarray:
    chunks: list[np.ndarray] = []
    start = 0.0
    for idx, duration in enumerate(durations):
        end = start + float(duration)
        segment = np.linspace(start, end, samples_per_segment + 1)
        chunks.append(segment if idx == 0 else segment[1:])
        start = end
    return np.concatenate(chunks)


def summarize_plan(plan_dir: Path, config: dict[str, Any], surface_model: Any, evaluator: Any, args: argparse.Namespace) -> dict[str, Any]:
    summary = json.loads((plan_dir / "summary.json").read_text(encoding="utf-8"))
    data = np.load(plan_dir / "ccro_nubs_trajectories.npz")
    obstacle = StaticObstacleField.from_points(np.asarray(data["obstacle_points"], dtype=np.float64))
    reference = load_trajectory(data, "reference_inner")
    candidate = load_trajectory(data, "candidate_inner")

    opt_times = optimizer_sample_times(data["durations"], int(config["risk"]["risk_samples_per_segment"]))
    dense_times = np.linspace(
        0.0,
        candidate.total_duration,
        max(2, int(np.ceil(candidate.total_duration / float(config["validation"]["dense_time_step"]))) + 1),
    )
    opt_risk = evaluator.trajectory(
        candidate,
        obstacle,
        opt_times,
        density=config["risk"]["optimizer_density"],
        with_gradient=False,
    )
    dense_risk = evaluator.trajectory(
        candidate,
        obstacle,
        dense_times,
        density=config["risk"]["validation_density"],
        with_gradient=False,
    )
    energy = candidate.energy()
    metrics = trajectory_preference_metrics(
        surface_model,
        reference,
        candidate,
        args.tcp_link,
        samples=args.samples,
    )
    lambda_smooth = float(config["optimizer"]["lambda_smooth"])
    lambda_risk = float(config["optimizer"]["lambda_risk"])
    return {
        "name": plan_dir.name,
        "status": summary.get("status"),
        "accepted_for_real_execution": bool(summary.get("accepted_for_real_execution", False)),
        "optimizer_type": summary.get("optimizer_type"),
        "optimizer_objective_terms": {
            "energy": float(energy),
            "lambda_smooth_energy": float(lambda_smooth * energy),
            "risk_cost_integral": float(opt_risk.cost),
            "lambda_risk_cost": float(lambda_risk * opt_risk.cost),
            "original_objective_approx": float(lambda_smooth * energy + lambda_risk * opt_risk.cost),
            "optimizer_min_distance_m": float(opt_risk.min_distance),
            "optimizer_active_sample_count": int(opt_risk.active_sample_count),
        },
        "dense_audit": {
            "risk_cost_integral": float(dense_risk.cost),
            "min_distance_m": float(dense_risk.min_distance),
            "nearest_link": dense_risk.nearest_link,
            "active_sample_count": int(dense_risk.active_sample_count),
        },
        "minimal_change_metrics": metrics,
    }


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# 6.5.2 Objective-Term Audit",
        "",
        "This audit explains why the original CCRO-NUBS objective can prefer an overpass route.",
        "",
        "| candidate | accepted | approx original objective | smooth term | risk term | dense min / m | max TCP z dev / m | TCP path length / m | joint length / rad |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in payload["plans"]:
        terms = item["optimizer_objective_terms"]
        dense = item["dense_audit"]
        metrics = item["minimal_change_metrics"]
        lines.append(
            f"| `{item['name']}` | {str(item['accepted_for_real_execution'])} | "
            f"{terms['original_objective_approx']:.6f} | "
            f"{terms['lambda_smooth_energy']:.6f} | "
            f"{terms['lambda_risk_cost']:.6f} | "
            f"{dense['min_distance_m']:.4f} | "
            f"{metrics['max_tcp_z_deviation_m']:.4f} | "
            f"{metrics['tcp_path_length_m']:.4f} | "
            f"{metrics['joint_path_length_rad']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- The original objective is dominated by joint-space smooth energy once both candidates are outside the hard acceptance distance.",
            "- It has no direct penalty for lifting the TCP or deviating from the intended tabletop path.",
            "- Therefore an overpass route can be mathematically optimal under the original objective, even when a planar/lateral route is more appropriate for the real tabletop task.",
            "- The corrected 6.5.2 policy is: dense safety gate first, then choose the strict accepted candidate with smaller task-space deviation and bounded TCP height change.",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    trial_dir = args.trial_dir.resolve()
    config = _load(trial_dir / "config_used.yaml")
    surface_model = make_surface_model(config)
    evaluator, _, _ = make_evaluator_and_verifier(config, surface_model)
    plan_dirs = [p.resolve() for p in args.plan_dirs] if args.plan_dirs else [
        p.resolve() for p in sorted(trial_dir.glob("ccro_nubs_jointspace_plan*")) if (p / "summary.json").exists()
    ]
    plans = [summarize_plan(plan_dir, config, surface_model, evaluator, args) for plan_dir in plan_dirs]
    payload = {"trial_dir": str(trial_dir), "plans": plans}
    output_dir = (args.output or (trial_dir / "objective_audit")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "objective_terms_summary.json", payload)
    (output_dir / "objective_terms_report.md").write_text(render_report(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-dir", type=Path, default=DEFAULT_TRIAL_DIR)
    parser.add_argument("--plan-dirs", nargs="*", type=Path, default=[])
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--tcp-link", default="gripper_base_link")
    parser.add_argument("--samples", type=int, default=241)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
