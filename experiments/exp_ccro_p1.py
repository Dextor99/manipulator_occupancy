"""Monte-Carlo robustness statistics for stage-three CCRO-NUBS trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import yaml

from experiments.exp_ccro_stage3 import _baseline, _limits, _states, make_forecast
from planning.nubs_trajectory import NUBSTrajectory6D
from planning.obstacle_forecast import (
    CompositeForecast,
    ConstantVelocitySphereForecast,
    ObstacleForecast,
)
from planning.robot_surface_model import RobotSurfaceModel


ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _perturb(forecast: ObstacleForecast, rng: np.random.Generator, cfg: dict):
    if isinstance(forecast, CompositeForecast):
        return CompositeForecast([_perturb(item, rng, cfg) for item in forecast.forecasts])
    if not isinstance(forecast, ConstantVelocitySphereForecast):
        raise TypeError(f"unsupported forecast type: {type(forecast).__name__}")
    radius = max(0.005, forecast.radius + rng.normal(0.0, cfg["radius_sigma"]))
    return ConstantVelocitySphereForecast(
        forecast.center + rng.normal(0.0, cfg["center_sigma"], 3),
        forecast.velocity + rng.normal(0.0, cfg["velocity_sigma"], 3),
        radius,
        forecast.valid_horizon,
        object_id=forecast.object_id,
        margin=forecast.margin,
        uncertainty=forecast.uncertainty,
        uncertainty_growth=forecast.uncertainty_growth,
        velocity_radius_scale=forecast.velocity_radius_scale,
        beyond_horizon=forecast.beyond_horizon,
    )


def _trajectory_from_points(points, head, tail, durations):
    return NUBSTrajectory6D().generate(
        np.asarray(points, dtype=np.float64), head, tail, durations
    )


def _precompute_surfaces(model, trajectories, times, density):
    output = {}
    for name, trajectory in trajectories.items():
        q_samples = trajectory.sample(times, max_derivative=0).q
        output[name] = [model.surface(q, density) for q in q_samples]
    return output


def _evaluate(points_by_time, forecast, times, d_stop):
    distances = np.full(len(times), np.inf)
    for index, (tau, points) in enumerate(zip(times, points_by_time)):
        occupancy = forecast.occupancy_at(float(tau))
        for sphere in occupancy.spheres:
            distance = np.maximum(
                np.linalg.norm(points - sphere.center[None, :], axis=1) - sphere.radius,
                0.0,
            )
            distances[index] = min(distances[index], float(np.min(distance)))
    below = (distances < d_stop).astype(float)
    return {
        "min_distance": float(np.min(distances)),
        "accepted": bool(np.min(distances) >= d_stop),
        "time_below_d_stop": float(np.trapezoid(below, times)),
    }


def _aggregate(rows):
    distances = np.asarray([row["min_distance"] for row in rows])
    return {
        "n": len(rows),
        "pass_rate": float(np.mean([row["accepted"] for row in rows])),
        "d_min_mean": float(np.mean(distances)),
        "d_min_p05": float(np.percentile(distances, 5)),
        "d_min_min": float(np.min(distances)),
        "time_below_d_stop_mean": float(np.mean([row["time_below_d_stop"] for row in rows])),
    }


def run(config_path, output_override=None):
    config_path = Path(config_path).resolve()
    config = _load(config_path)
    stage_config = _load(ROOT / config["source_config"])
    stage_metrics = json.loads((ROOT / config["source_metrics"]).read_text(encoding="utf-8"))
    p0 = json.loads((ROOT / config["p0_metrics"]).read_text(encoding="utf-8"))
    output = Path(output_override or config["output_dir"])
    if not output.is_absolute():
        output = ROOT / output
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output / "config.yaml")

    robot, surface = stage_config["robot"], stage_config["surface"]
    model = RobotSurfaceModel(
        ROOT / robot["urdf_path"], robot["joint_names"], surface["density_totals"],
        seed=surface["random_seed"], min_points_per_link=surface["min_points_per_link"],
        cache_dir=surface["cache_dir"], geometry=surface["geometry"],
    )
    head, tail, durations = _states(stage_config)
    baseline_result = _baseline(stage_config, head, tail, durations, _limits(stage_config))
    baseline = baseline_result.trajectory
    dt = float(config["evaluation"]["time_step"])
    times = np.linspace(0.0, baseline.total_duration, int(np.ceil(baseline.total_duration / dt)) + 1)
    all_rows = []
    by_scenario_method = {}
    body_counterexamples = []
    for scenario_name, scenario_metrics in stage_metrics["scenarios"].items():
        nominal, _ = make_forecast(stage_config, scenario_name, model, baseline)
        trajectories = {"baseline": baseline}
        for method in ("current_full", "temporal_ee", "temporal_full"):
            trajectories[method] = _trajectory_from_points(
                scenario_metrics["methods"][method]["p_inner"], head, tail, durations
            )
        surfaces = _precompute_surfaces(
            model, trajectories, times, config["evaluation"]["density"]
        )
        for trial in range(int(config["trials"])):
            seed = int(config["random_seed"]) + trial * 101 + ord(scenario_name)
            truth = _perturb(nominal, np.random.default_rng(seed), config["perturbation"])
            trial_results = {}
            for method, points in surfaces.items():
                result = _evaluate(
                    points, truth, times, float(config["evaluation"]["d_stop"])
                )
                row = {
                    "scenario": scenario_name,
                    "trial": trial,
                    "seed": seed,
                    "method": method,
                    **result,
                }
                all_rows.append(row)
                by_scenario_method.setdefault((scenario_name, method), []).append(row)
                trial_results[method] = result
            if scenario_name == "B":
                body_counterexamples.append(
                    (not trial_results["temporal_ee"]["accepted"])
                    and trial_results["temporal_full"]["accepted"]
                )

    aggregate = {
        scenario: {
            method: _aggregate(by_scenario_method[(scenario, method)])
            for method in ("baseline", "current_full", "temporal_ee", "temporal_full")
        }
        for scenario in stage_metrics["scenarios"]
    }
    pooled = {
        method: _aggregate([row for row in all_rows if row["method"] == method])
        for method in ("baseline", "current_full", "temporal_ee", "temporal_full")
    }
    current_failure = 1.0 - pooled["current_full"]["pass_rate"]
    body_rate = float(np.mean(body_counterexamples))
    thresholds = config["acceptance"]
    checks = {
        "p0_accepted": bool(p0["accepted"]),
        "nominal_stage3_accepted": bool(stage_metrics["accepted"]),
        "temporal_full_pass_rate": pooled["temporal_full"]["pass_rate"] >= thresholds["temporal_full_pass_rate_min"],
        "baseline_pass_rate": pooled["baseline"]["pass_rate"] <= thresholds["baseline_pass_rate_max"],
        "current_failure_rate": current_failure >= thresholds["current_failure_rate_min"],
        "body_counterexample_rate": body_rate >= thresholds["body_counterexample_rate_min"],
    }
    metrics = {
        "accepted": bool(all(checks.values())),
        "checks": checks,
        "p0_source_sha256": p0["source_sha256"],
        "trials_per_scenario": int(config["trials"]),
        "total_method_evaluations": len(all_rows),
        "evaluation_density": config["evaluation"]["density"],
        "pooled": pooled,
        "by_scenario": aggregate,
        "current_failure_rate": current_failure,
        "body_counterexample_rate": body_rate,
    }
    (output / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output / "trials.jsonl").open("w", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    lines = [
        "| method | n | pass rate | D_min mean / m | D_min p05 / m | D_min min / m | time<stop mean / s |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method, item in pooled.items():
        lines.append(
            f"| {method} | {item['n']} | {item['pass_rate']:.4f} | "
            f"{item['d_min_mean']:.6f} | {item['d_min_p05']:.6f} | "
            f"{item['d_min_min']:.6f} | {item['time_below_d_stop_mean']:.4f} |"
        )
    lines.extend([
        "",
        f"Current-frame failure rate: **{current_failure:.4f}**  ",
        f"Body-counterexample rate: **{body_rate:.4f}**  ",
        f"Overall: **{'PASS' if metrics['accepted'] else 'FAIL'}**",
        "",
    ])
    (output / "table_p1.md").write_text("\n".join(lines), encoding="utf-8")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config" / "ccro_p1.yaml"))
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    metrics = run(args.config, args.output)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if not metrics["accepted"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
