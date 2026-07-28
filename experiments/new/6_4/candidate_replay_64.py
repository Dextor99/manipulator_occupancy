"""Stage-A candidate-planning replay for the Chapter 6.4 boundary experiment.

This script isolates candidate generation from the closed-loop pending state
machine.  It replays fixed trigger/switch states and compares Critical-NUBS
and CCRO-NUBS under identical forecasts, budgets, continuity checks, and
online/dense validation.
"""

from __future__ import annotations

import argparse
import math
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from planning.obstacle_forecast import ShiftedForecast
from planning.verifier import DynamicTrajectoryVerifier

from . import config_64 as cfg
from .common_64 import (
    constant_forecast,
    git_commit_hash,
    load_stage4_config,
    load_surface_model,
    make_critical_risk_stack,
    make_reference,
    make_risk_stack,
    optimize_candidate,
    write_json,
)
from .scenarios_64 import generate_instances, obstacle_center_at, obstacle_velocity_at


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "-"
    return f"{number:.{digits}f}"


def _reference_local_min_distance(evaluator, reference, forecast, switch_tau: float, duration: float) -> float:
    count = max(3, int(math.ceil(float(duration) / cfg.DT)) + 1)
    best = math.inf
    for delta in np.linspace(0.0, float(duration), count):
        q = reference.evaluate(min(reference.total_duration, float(switch_tau) + float(delta)))
        risk = evaluator.configuration(q, forecast, float(delta), density=cfg.SURFACE_DENSITY_VERIFY, with_gradient=False)
        best = min(best, float(risk.min_distance))
    return float(best)


def _observed_at_trigger(instance: dict[str, Any], trigger_time: float) -> tuple[np.ndarray, np.ndarray, float]:
    gt_center0 = np.asarray(instance["gt_center0"], dtype=np.float64)
    gt_velocity = np.asarray(instance["gt_velocity"], dtype=np.float64)
    gt_radius = float(instance["gt_radius"])
    motion_start_time = float(instance.get("motion_start_time", 0.0))
    pre_motion_center = (
        None
        if instance.get("pre_motion_center") is None
        else np.asarray(instance["pre_motion_center"], dtype=np.float64)
    )
    center = obstacle_center_at(gt_center0, gt_velocity, trigger_time, motion_start_time, pre_motion_center)
    velocity = obstacle_velocity_at(gt_velocity, trigger_time, motion_start_time)
    rng = np.random.default_rng(int(instance["observation_seed"]) + int(round(100.0 * trigger_time)))
    if float(np.linalg.norm(velocity)) < 1.0e-9:
        measured_velocity = np.zeros(3, dtype=np.float64)
    else:
        measured_velocity = velocity + rng.normal(0.0, cfg.OBS_VEL_SIGMA, size=3)
    return (
        center + rng.normal(0.0, cfg.OBS_POS_SIGMA, size=3),
        measured_velocity,
        max(0.025, float(gt_radius + rng.normal(0.0, cfg.OBS_RADIUS_SIGMA))),
    )


def replay_one(
    config: dict[str, Any],
    reference,
    limits,
    evaluator,
    verifier,
    dense_verifier,
    instance: dict[str, Any],
    method: str,
    lead_label: str,
    lead_time: float,
) -> dict[str, Any]:
    reference_risk_time = instance.get("reference_risk_time")
    if reference_risk_time is None:
        raise ValueError("candidate replay requires reference_risk_time")
    trigger_time = max(0.0, float(reference_risk_time) - float(lead_time))
    planned_switch_delay = cfg.PLANNED_SWITCH_DELAY
    switch_tau = min(reference.total_duration, trigger_time + planned_switch_delay)
    resume_tau = min(reference.total_duration, switch_tau + cfg.LOCAL_REPLAN_HORIZON)
    remaining_duration = min(
        max(float(resume_tau - switch_tau), 1.2),
        max(1.0, cfg.FORECAST_HORIZON - planned_switch_delay),
    )
    q_now = reference.evaluate(switch_tau)
    qd_now = reference.evaluate(switch_tau, derivative_order=1)
    qdd_now = reference.evaluate(switch_tau, derivative_order=2)
    q_goal = reference.evaluate(resume_tau)
    qd_goal = reference.evaluate(resume_tau, derivative_order=1)
    qdd_goal = reference.evaluate(resume_tau, derivative_order=2)
    obs_center, obs_velocity, obs_radius = _observed_at_trigger(instance, trigger_time)
    forecast = constant_forecast(obs_center, obs_velocity, obs_radius)
    local_forecast = ShiftedForecast(forecast, planned_switch_delay, remaining_duration)
    started = time.perf_counter()
    candidate = optimize_candidate(
        config,
        evaluator,
        limits,
        local_forecast,
        q_now=q_now,
        qd_now=qd_now,
        qdd_now=qdd_now,
        q_goal=q_goal,
        qd_goal=qd_goal,
        qdd_goal=qdd_goal,
        remaining_duration=remaining_duration,
        verifier=verifier,
        warm_start_trajectory=reference,
        warm_start_tau=switch_tau,
        optimization_budget_s=min(cfg.PLANNING_BUDGET, max(0.5, planned_switch_delay - 0.5)),
    )
    dense = dense_verifier.verify(
        candidate["trajectory"],
        local_forecast,
        current_q=q_now,
        current_qd=qd_now,
        current_qdd=qdd_now,
        q_goal=q_goal,
        solver_success=True,
    )
    reference_min = _reference_local_min_distance(evaluator, reference, local_forecast, switch_tau, remaining_duration)
    online_min = float(candidate["verification"]["min_distance"])
    return {
        "instance_id": instance["instance_id"],
        "scenario_type": instance["scenario_type"],
        "method": method,
        "lead_label": lead_label,
        "lead_time_requested": float(lead_time),
        "lead_time_actual": float(float(reference_risk_time) - trigger_time),
        "speed_group": instance.get("speed_group"),
        "trigger_time": float(trigger_time),
        "reference_risk_time": float(reference_risk_time),
        "planned_switch_delay": float(planned_switch_delay),
        "switch_tau": float(switch_tau),
        "resume_tau": float(resume_tau),
        "planner_finished": True,
        "within_budget": float(candidate["optimization"]["elapsed_ms"]) <= 1000.0 * cfg.PLANNING_BUDGET,
        "optimizer_converged": bool(candidate["optimization"]["success"]),
        "online_feasible": bool(candidate["verification"]["accepted"]),
        "dense_feasible": bool(dense.accepted),
        "continuity_pass": bool(candidate["verification"]["checks"].get("continuity_q_ok", False)),
        "online_distance_pass": bool(candidate["verification"]["checks"].get("distance_ok", False)),
        "dense_distance_pass": bool(dense.checks.get("distance_ok", False)),
        "reference_min_distance_online": float(reference_min),
        "candidate_min_distance_online": online_min,
        "candidate_min_distance_dense": float(dense.min_distance),
        "delta_min_distance_online": float(online_min - reference_min),
        "beneficial": bool(online_min >= reference_min + cfg.SWITCH_IMPROVEMENT_MARGIN),
        "elapsed_ms": float(candidate["optimization"]["elapsed_ms"]),
        "wall_elapsed_ms": float((time.perf_counter() - started) * 1000.0),
        "reasons": candidate["verification"]["reasons"],
        "dense_reasons": dense.reasons,
        "optimization": candidate["optimization"],
    }


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["scenario_type"], row["lead_label"], row["method"]), []).append(row)
    table = []
    for (scenario_type, lead_label, method), items in sorted(grouped.items()):
        table.append(
            {
                "scenario_type": scenario_type,
                "lead_label": lead_label,
                "method": method,
                "n": len(items),
                "within_budget": float(np.mean([item["within_budget"] for item in items])),
                "optimizer_converged": float(np.mean([item["optimizer_converged"] for item in items])),
                "online_feasible": float(np.mean([item["online_feasible"] for item in items])),
                "dense_feasible": float(np.mean([item["dense_feasible"] for item in items])),
                "beneficial": float(np.mean([item["beneficial"] for item in items])),
                "delta_min_distance_online": float(np.mean([item["delta_min_distance_online"] for item in items])),
                "planner_ms_p95": float(np.percentile([item["elapsed_ms"] for item in items], 95)),
            }
        )
    return table


def write_replay_table(summary: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "| scenario | lead | method | n | within budget | converged | online feasible | dense feasible | beneficial | delta Dmin online / m | planner p95 / ms |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['scenario_type']} | {row['lead_label']} | {cfg.METHOD_NAMES.get(row['method'], row['method'])} | "
            f"{row['n']} | {_fmt(row['within_budget'], 2)} | {_fmt(row['optimizer_converged'], 2)} | "
            f"{_fmt(row['online_feasible'], 2)} | {_fmt(row['dense_feasible'], 2)} | {_fmt(row['beneficial'], 2)} | "
            f"{_fmt(row['delta_min_distance_online'])} | {_fmt(row['planner_ms_p95'], 1)} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(Path("results/new/6_4_candidate_replay")))
    parser.add_argument("--config", default=str(cfg.STAGE4_CONFIG))
    parser.add_argument("--scenario", choices=["D1", "D2M"], default="D1")
    parser.add_argument("--method", choices=["critical_point_nubs", "ccro_nubs"], default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    trials_dir = output / "candidate_trials"
    paper_dir = output / "paper"
    trials_dir.mkdir(parents=True, exist_ok=True)
    paper_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__).with_name("config_64.yaml"), output / "config_64.yaml")
    config = load_stage4_config(Path(args.config))
    model = load_surface_model(config)
    reference, _, _, _ = make_reference(config)
    evaluator, verifier, limits = make_risk_stack(config, model, None)
    critical_evaluator, critical_verifier = make_critical_risk_stack(config, model)
    dense_verifier = DynamicTrajectoryVerifier(
        evaluator,
        limits,
        d_stop=cfg.D_STOP,
        time_step=cfg.DT,
        density="dense",
        epsilon_goal=1.0e-2,
        epsilon_continuity_q=5.0e-3,
        epsilon_continuity_qd=3.0e-3,
        epsilon_continuity_qdd=3.0e-3,
        limit_tolerance=1.0e-8,
    )
    instances = generate_instances(model, reference, output / "instances", smoke=args.smoke, gate=args.smoke)
    instances = [item for item in instances if item["instance_id"].startswith(args.scenario + "_")]
    methods = (args.method,) if args.method else ("critical_point_nubs", "ccro_nubs")
    rows: list[dict[str, Any]] = []
    for instance in instances:
        for lead_label, lead_time in cfg.LEAD_TIME_GROUPS.items():
            for method in methods:
                path = trials_dir / f"{instance['instance_id']}_{lead_label}_{method}.json"
                if args.resume and path.exists():
                    row = json.loads(path.read_text(encoding="utf-8"))
                else:
                    method_evaluator = critical_evaluator if method == "critical_point_nubs" else evaluator
                    method_verifier = critical_verifier if method == "critical_point_nubs" else verifier
                    row = replay_one(
                        config,
                        reference,
                        limits,
                        method_evaluator,
                        method_verifier,
                        dense_verifier,
                        instance,
                        method,
                        lead_label,
                        lead_time,
                    )
                    write_json(path, row)
                rows.append(row)
                print(
                    f"[6.4 replay] {row['instance_id']} {row['lead_label']} {cfg.METHOD_NAMES.get(method, method)} "
                    f"online={row['online_feasible']} dense={row['dense_feasible']} "
                    f"dD={row['delta_min_distance_online']:.3f}"
                )
    summary = _aggregate(rows)
    metrics = {
        "experiment": "6.4 stage-A candidate replay",
        "scope": "candidate generation only; no closed-loop execution",
        "git_commit": git_commit_hash(),
        "scenario": args.scenario,
        "methods": list(methods),
        "lead_time_groups": cfg.LEAD_TIME_GROUPS,
        "trial_count": len(rows),
        "summary": summary,
        "trials": rows,
    }
    write_json(output / "candidate_replay_64.json", metrics)
    write_replay_table(summary, paper_dir / "table_6_4_candidate_replay.md")
    print(f"[6.4 replay] saved {output / 'candidate_replay_64.json'}")
    print(f"[6.4 replay] saved {paper_dir / 'table_6_4_candidate_replay.md'}")


if __name__ == "__main__":
    main()
