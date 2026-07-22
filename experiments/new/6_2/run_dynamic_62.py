"""Run revised Chapter 6.2 dynamic warning experiments."""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from perception.occupancy_tracker import OccupancyTracker

from . import config_62 as cfg
from .common_62 import (
    ensure_output_tree,
    lead_time,
    load_surface_model,
    make_reference_trajectory,
    min_distance_to_sphere,
    sample_sphere_surface,
    save_reference_trajectory,
    scenario_distance_diagnostics,
    summarize_leads,
    write_json,
)
from .dynamic_methods_62 import METHODS, METHOD_NAMES, DynamicMethodState, evaluate_method, observe_object
from .dynamic_scenarios_62 import make_dynamic_scenario, make_leave_scenario, make_static_safe_scenario


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run revised 6.2 dynamic STRO warning experiments.")
    parser.add_argument("--output", default=str(cfg.DEFAULT_OUTPUT))
    parser.add_argument("--trials-per-speed", type=int, default=cfg.DYNAMIC_TRIALS_PER_SPEED)
    parser.add_argument("--calibration-trials", type=int, default=cfg.CALIBRATION_TRIALS)
    parser.add_argument("--seed", type=int, default=cfg.RANDOM_SEED)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def true_risk_time(surface, trajectory, scenario) -> tuple[float | None, list[dict[str, Any]]]:
    rows = []
    t_risk = None
    for t in trajectory.times:
        timestamp = float(t)
        q = trajectory.sample(timestamp)
        center = scenario.center_at(timestamp)
        distance, link, _ = min_distance_to_sphere(surface, q, center, scenario.obstacle_radius, density="dense")
        rows.append({"time": timestamp, "D_gt": float(distance), "nearest_link_gt": link})
        if t_risk is None and distance <= cfg.DYNAMIC_ALARM_DISTANCE:
            t_risk = timestamp
    return t_risk, rows


def scenario_truth_diagnostics(surface, trajectory, scenario) -> dict[str, Any]:
    _, rows = true_risk_time(surface, trajectory, scenario)
    return scenario_distance_diagnostics([row["D_gt"] for row in rows], [row["time"] for row in rows])


def valid_dynamic_scenario(surface, trajectory, scene: str, speed: float, seed: int):
    for attempt in range(300):
        scenario = make_dynamic_scenario(surface, trajectory, scene, speed, seed + attempt * 10000)
        normal = scenario.crossing_center - scenario.target_point
        normal = normal / max(float(np.linalg.norm(normal)), 1.0e-12)
        for clearance in np.linspace(0.09, 0.32, 24):
            tuned = replace(
                scenario,
                crossing_center=scenario.target_point + normal * (scenario.obstacle_radius + float(clearance)),
                clearance=float(clearance),
            )
            diagnostics = scenario_truth_diagnostics(surface, trajectory, tuned)
            t_risk = diagnostics["t_risk"]
            if (
                diagnostics["first_contact_time"] is None
                and diagnostics["valid_min_clearance"]
                and t_risk is not None
                and 1.0 <= t_risk <= cfg.TOTAL_DURATION - 1.0
            ):
                return tuned
    raise RuntimeError(f"failed to generate a valid {scene} scenario at speed={speed}")


def run_trial(surface, trajectory, scenario, output_dir: Path) -> dict[str, Any]:
    rng = np.random.default_rng(scenario.seed)
    trackers = {
        method: OccupancyTracker(
            association_distance=0.25,
            alpha=0.45,
            pos_alpha=0.65,
            motion_gate=0.002,
            velocity_dead_zone=0.005,
        )
        for method in METHODS
    }
    states = {method: DynamicMethodState() for method in METHODS}
    t_risk, truth_rows = true_risk_time(surface, trajectory, scenario)
    truth_by_time = {round(row["time"], 8): row for row in truth_rows}
    first_alarm = {method: None for method in METHODS}
    alarm_details = {method: None for method in METHODS}
    frame_rows: list[dict[str, Any]] = []
    runtime_values = {method: [] for method in METHODS}

    for frame_index, t in enumerate(trajectory.times):
        timestamp = float(t)
        gt_center = scenario.center_at(timestamp)
        gt_velocity = scenario.velocity_at(timestamp)
        points = sample_sphere_surface(rng, gt_center, scenario.obstacle_radius, cfg.OBSTACLE_POINTS)
        taus, q_future = trajectory.future(timestamp, cfg.H_MON, cfg.PREDICTION_STEP)
        observed_by_method = {
            method: observe_object(points, timestamp, trackers[method])
            for method in METHODS
        }

        method_results = {}
        for method in METHODS:
            start = time.perf_counter()
            result = evaluate_method(method, surface, q_future, taus, states[method], observed_by_method[method])
            runtime = (time.perf_counter() - start) * 1000.0
            runtime_values[method].append(runtime)
            method_results[method] = result
            if result["risk"] and first_alarm[method] is None:
                first_alarm[method] = timestamp
                alarm_details[method] = {
                    "D_gt_at_alarm": truth_by_time[round(timestamp, 8)]["D_gt"],
                    "alarm_tau": result.get("alarm_tau"),
                    "inflated_radius_at_alarm": result.get("inflated_radius_at_min"),
                    "history_count_at_alarm": result.get("history_count"),
                }

        obs = observed_by_method["stro"]
        stro_pred = method_results["stro"]["prediction_center_tau_05"]
        gt_tau = scenario.center_at(min(timestamp + 0.5, cfg.TOTAL_DURATION))
        pred_error = None if stro_pred is None else float(np.linalg.norm(np.asarray(stro_pred) - gt_tau))
        row = {
            "frame": frame_index,
            "time": timestamp,
            "q_ref": trajectory.sample(timestamp).tolist(),
            "obstacle_gt_center": gt_center.tolist(),
            "obstacle_gt_velocity": gt_velocity.tolist(),
            "obstacle_observed_center": None if obs is None else obs.center.tolist(),
            "obstacle_estimated_velocity": None if obs is None else obs.velocity.tolist(),
            "prediction_center_tau_05": stro_pred,
            "prediction_error_tau_05": pred_error,
            "D_gt": truth_by_time[round(timestamp, 8)]["D_gt"],
        }
        for method in METHODS:
            row[f"risk_{method}"] = method_results[method]["risk"]
            row[f"distance_{method}"] = method_results[method]["distance"]
            row[f"alarm_tau_{method}"] = method_results[method].get("alarm_tau")
            row[f"inflated_radius_{method}"] = method_results[method].get("inflated_radius_at_min")
            row[f"history_count_{method}"] = method_results[method].get("history_count")
            row[f"runtime_{method}_ms"] = runtime_values[method][-1]
        frame_rows.append(row)

    trial_id = f"{scenario.scene}_{scenario.speed:.2f}_{scenario.seed}"
    write_json(output_dir / "trials" / f"{trial_id}.json", frame_rows)
    leads = {method: lead_time(t_risk, first_alarm[method]) for method in METHODS}
    diagnostics = scenario_distance_diagnostics([row["D_gt"] for row in frame_rows], [row["time"] for row in frame_rows])
    false_alarm_time_ratio = {
        method: (
            float(sum(1 for row in frame_rows if row[f"risk_{method}"]) / max(len(frame_rows), 1))
            if t_risk is None
            else 0.0
        )
        for method in METHODS
    }
    prediction_errors = [
        row["prediction_error_tau_05"] ** 2
        for row in frame_rows
        if row["prediction_error_tau_05"] is not None
    ]
    return {
        "trial_id": trial_id,
        "scene": scenario.scene,
        "seed": scenario.seed,
        "speed": scenario.speed,
        "target_link": scenario.target_link,
        "crossing_time": scenario.crossing_time,
        "motion_start_time": scenario.schedule.start_time,
        "path_length": scenario.schedule.path_length,
        "t_risk": t_risk,
        **diagnostics,
        "first_alarm": first_alarm,
        "alarm_details": alarm_details,
        "lead_time": leads,
        "false_alarm_time_ratio": false_alarm_time_ratio,
        "runtime_mean_ms": {method: float(np.mean(runtime_values[method])) for method in METHODS},
        "prediction_rmse_tau_05": float(np.sqrt(np.mean(prediction_errors))) if prediction_errors else math.nan,
    }


def aggregate(trials: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"methods": {}}
    for method in METHODS:
        out["methods"][method] = {}
        for scene in ("approach", "crossing"):
            rows = [trial for trial in trials if trial["scene"] == scene]
            out["methods"][method][scene] = summarize_leads([row["lead_time"][method] for row in rows])
        runtimes = [trial["runtime_mean_ms"][method] for trial in trials if trial["scene"] in {"approach", "crossing"}]
        out["methods"][method]["runtime_ms"] = {
            "mean": float(np.mean(runtimes)) if runtimes else math.nan,
            "std": float(np.std(runtimes, ddof=1)) if len(runtimes) > 1 else 0.0,
        }
        out["methods"][method]["speed_breakdown"] = {}
        for scene in ("approach", "crossing"):
            out["methods"][method]["speed_breakdown"][scene] = {}
            for speed in cfg.DYNAMIC_SPEEDS:
                rows = [
                    trial for trial in trials
                    if trial["scene"] == scene and abs(float(trial["speed"]) - float(speed)) < 1.0e-9
                ]
                out["methods"][method]["speed_breakdown"][scene][f"{speed:.2f}"] = summarize_leads(
                    [row["lead_time"][method] for row in rows]
                )
    return out


def aggregate_calibration(trials: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for method in METHODS:
        ratios = [trial["false_alarm_time_ratio"][method] for trial in trials]
        out[method] = {
            "false_alarm_time_ratio_mean": float(np.mean(ratios)) if ratios else math.nan,
            "false_alarm_time_ratio_max": float(np.max(ratios)) if ratios else math.nan,
            "sequence_count": len(ratios),
            "below_5_percent": bool(ratios and max(ratios) <= 0.05),
        }
    return out


def write_table(path: Path, metrics: dict[str, Any]) -> None:
    lines = [
        "| 方法 | 接近提前量 / s（漏检/30） | 横穿提前量 / s（漏检/30） | 每帧风险评价时间 / ms |",
        "| --- | ---: | ---: | ---: |",
    ]
    for method in METHODS:
        item = metrics["methods"][method]
        app = item["approach"]
        cro = item["crossing"]
        run = item["runtime_ms"]
        lines.append(
            f"| {METHOD_NAMES[method]} | {app['mean']:.3f} ± {app['std']:.3f}（{app['misses']}/{app['total']}） "
            f"| {cro['mean']:.3f} ± {cro['std']:.3f}（{cro['misses']}/{cro['total']}） "
            f"| {run['mean']:.3f} ± {run['std']:.3f} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_speed_table(path: Path, metrics: dict[str, Any]) -> None:
    lines = [
        "| 场景 | 速度 / m/s | Current-frame / s | OctoMap-like / s | STRO / s |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for scene in ("approach", "crossing"):
        for speed in cfg.DYNAMIC_SPEEDS:
            key = f"{speed:.2f}"
            cells = []
            for method in METHODS:
                item = metrics["methods"][method]["speed_breakdown"][scene][key]
                cells.append(f"{item['mean']:.3f} ± {item['std']:.3f}（{item['misses']}/{item['total']}）")
            lines.append(f"| {scene} | {speed:.2f} | {cells[0]} | {cells[1]} | {cells[2]} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    paths = ensure_output_tree(output)
    trajectory = make_reference_trajectory()
    save_reference_trajectory(output / "reference_trajectory_62.npz", trajectory)
    surface = load_surface_model()
    trials = []
    seed = int(args.seed)
    for scene in ("approach", "crossing"):
        for speed in cfg.DYNAMIC_SPEEDS:
            for index in range(args.trials_per_speed):
                scenario = valid_dynamic_scenario(surface, trajectory, scene, speed, seed + len(trials) + index)
                trials.append(run_trial(surface, trajectory, scenario, paths["dynamic"]))
    calibration = []
    for index in range(args.calibration_trials):
        calibration.append(run_trial(surface, trajectory, make_static_safe_scenario(seed + 900 + index), paths["calibration"]))
        calibration.append(run_trial(surface, trajectory, make_leave_scenario(surface, trajectory, seed + 950 + index), paths["calibration"]))
    metrics = aggregate(trials)
    calibration_metrics = aggregate_calibration(calibration)
    write_json(
        paths["dynamic"] / "summary.json",
        {
            "trials": trials,
            "metrics": metrics,
            "calibration": calibration,
            "calibration_metrics": calibration_metrics,
        },
    )
    write_table(paths["paper"] / "table_6_2_dynamic.md", metrics)
    write_speed_table(paths["paper"] / "table_6_2_dynamic_by_speed.md", metrics)


if __name__ == "__main__":
    main()
