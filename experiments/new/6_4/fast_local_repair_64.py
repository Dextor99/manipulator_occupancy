"""Fast 1 s local repair experiment for Chapter 6.4.

This is the replacement online experiment for the old seconds-level dynamic
NUBS optimizer.  It keeps time allocation fixed, repairs only a few local
interpolation points, uses one high-risk gradient query per repair step, and
reports end-to-end wall-clock timing.
"""

from __future__ import annotations

import argparse
import math
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from planning.nubs_trajectory import NUBSTrajectory6D
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
    write_json,
)
from .scenarios_64 import _tangent_direction


FAST_METHOD_NAMES = {
    "critical_fast_repair": "Critical-fast-repair",
    "ccro_fast_repair": "CCRO-fast-repair",
}


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


def _make_local_reference(reference, tau_start: float) -> tuple[NUBSTrajectory6D, np.ndarray, np.ndarray, np.ndarray]:
    horizon = cfg.FAST_LOCAL_HORIZON
    segments = cfg.FAST_LOCAL_SEGMENTS
    durations = np.full(segments, horizon / segments, dtype=np.float64)
    knot_times = tau_start + np.cumsum(durations)[:-1]
    p_inner = np.vstack([reference.evaluate(float(tau)) for tau in knot_times])
    head = NUBSTrajectory6D.make_boundary_state(
        reference.evaluate(tau_start),
        reference.evaluate(tau_start, derivative_order=1),
        reference.evaluate(tau_start, derivative_order=2),
    )
    tau_goal = tau_start + horizon
    tail = NUBSTrajectory6D.make_boundary_state(
        reference.evaluate(tau_goal),
        reference.evaluate(tau_goal, derivative_order=1),
        reference.evaluate(tau_goal, derivative_order=2),
    )
    return NUBSTrajectory6D().generate(p_inner, head, tail, durations), p_inner, head, tail


def _surface_outward(model, q: np.ndarray, links: tuple[str, ...], rng: np.random.Generator) -> tuple[str, np.ndarray, np.ndarray]:
    link = links[int(rng.integers(0, len(links)))]
    points = model.surface_by_link(q, cfg.SURFACE_DENSITY_LOOP, {link})[link]
    point = points[int(rng.integers(0, len(points)))]
    centroid = points.mean(axis=0)
    outward = point - centroid
    norm = float(np.linalg.norm(outward))
    if norm < 1.0e-9:
        outward = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        outward = outward / norm
    return link, point, outward


def make_fast_instances(model, reference, *, scenario: str, smoke: bool, g1: bool) -> list[dict[str, Any]]:
    rng = np.random.default_rng(cfg.RANDOM_SEED + (8100 if scenario == "D1" else 9100) + int(g1) * 1000)
    links = cfg.BODY_LINKS if scenario == "D1" else cfg.EE_LINKS
    scenario_type = "body_crossing_fast" if scenario == "D1" else "ee_crossing_fast"
    if g1:
        speeds = (0.15,)
        conflict_groups = {"Long": cfg.FAST_CONFLICT_TIME_GROUPS["Long"]}
        repeats = 10
    else:
        speeds = cfg.SPEED_GROUPS
        conflict_groups = cfg.FAST_CONFLICT_TIME_GROUPS
        repeats = 1 if smoke else 5
    instances: list[dict[str, Any]] = []
    index = 0
    tau_low, tau_high = cfg.FAST_TAU_START_RANGE
    tau_high = min(tau_high, reference.total_duration - cfg.FAST_LOCAL_HORIZON - 0.05)
    for speed in speeds:
        for lead_label, conflict_time in conflict_groups.items():
            for repeat in range(repeats):
                tau_start = float(rng.uniform(tau_low, tau_high))
                local_ref, _, _, _ = _make_local_reference(reference, tau_start)
                q_conflict = local_ref.evaluate(float(conflict_time))
                link, surface_point, outward = _surface_outward(model, q_conflict, links, rng)
                radius = float(rng.uniform(0.04, 0.055))
                clearance = float(rng.uniform(0.025, 0.065) if not g1 else rng.uniform(0.035, 0.060))
                crossing_center = surface_point + (radius + clearance) * outward
                direction = _tangent_direction(outward, rng)
                velocity = float(speed) * direction
                center_at_start = crossing_center - velocity * float(conflict_time)
                instances.append(
                    {
                        "instance_id": f"{scenario}F_{index:03d}",
                        "scenario_type": scenario_type,
                        "scenario": scenario,
                        "speed_group": float(speed),
                        "lead_label": lead_label,
                        "conflict_time_after_start": float(conflict_time),
                        "tau_start": tau_start,
                        "tau_goal": tau_start + cfg.FAST_LOCAL_HORIZON,
                        "target_link": link,
                        "obstacle_center0": center_at_start.tolist(),
                        "obstacle_velocity": velocity.tolist(),
                        "obstacle_radius": radius,
                        "reference_clearance_at_conflict": clearance,
                        "seed": int(cfg.RANDOM_SEED + index),
                        "repeat_index": int(repeat),
                    }
                )
                index += 1
    return instances


def _trajectory_min(evaluator, trajectory, forecast, *, density: str) -> float:
    times = np.arange(0.0, trajectory.total_duration + 0.5 * cfg.FAST_SAMPLE_DT, cfg.FAST_SAMPLE_DT)
    risk = evaluator.trajectory(trajectory, forecast, times, density=density, with_gradient=False)
    return float(risk.min_distance)


def _worst_risk(evaluator, trajectory, forecast):
    times = np.arange(0.0, trajectory.total_duration + 0.5 * cfg.FAST_SAMPLE_DT, cfg.FAST_SAMPLE_DT)
    worst = None
    for tau in times:
        q = trajectory.evaluate(float(tau))
        risk = evaluator.configuration(q, forecast, float(tau), density=cfg.SURFACE_DENSITY_LOOP, with_gradient=False)
        if worst is None or risk.min_distance < worst["min_distance"]:
            worst = {"tau": float(tau), "q": q, "min_distance": float(risk.min_distance), "nearest_link": risk.nearest_link}
    return worst


def _repair_points(points: np.ndarray, direction: np.ndarray, tau: float, durations: np.ndarray, step: float, limits) -> np.ndarray:
    if points.size == 0:
        return points.copy()
    knot_times = np.cumsum(durations)[:-1]
    sigma = max(0.18, 0.25 * cfg.FAST_LOCAL_HORIZON)
    weights = np.exp(-0.5 * ((knot_times - float(tau)) / sigma) ** 2)
    if float(np.max(weights)) > 0.0:
        weights = weights / float(np.max(weights))
    repaired = points + float(step) * weights[:, None] * direction[None, :]
    return np.minimum(np.maximum(repaired, limits.q_min[None, :]), limits.q_max[None, :])


def fast_repair(
    reference,
    model,
    evaluator,
    online_verifier,
    dense_verifier,
    limits,
    instance: dict[str, Any],
) -> dict[str, Any]:
    local_ref, p_inner, head, tail = _make_local_reference(reference, float(instance["tau_start"]))
    durations = local_ref.durations
    forecast = constant_forecast(
        np.asarray(instance["obstacle_center0"], dtype=np.float64),
        np.asarray(instance["obstacle_velocity"], dtype=np.float64),
        float(instance["obstacle_radius"]),
    )
    started = time.perf_counter()
    risk_scan_ms = 0.0
    repair_ms = 0.0
    points = p_inner.copy()
    trajectory = local_ref
    repair_steps = 0
    for step_index in range(cfg.FAST_REPAIR_STEPS + 1):
        t0 = time.perf_counter()
        worst = _worst_risk(evaluator, trajectory, forecast)
        risk_scan_ms += (time.perf_counter() - t0) * 1000.0
        if worst is None or worst["min_distance"] >= cfg.D_ONLINE_ACCEPT or step_index == cfg.FAST_REPAIR_STEPS:
            break
        t1 = time.perf_counter()
        gradient_risk = evaluator.configuration(
            worst["q"],
            forecast,
            worst["tau"],
            density=cfg.SURFACE_DENSITY_LOOP,
            with_gradient=True,
        )
        gradient = np.zeros(6) if gradient_risk.gradient_q is None else np.asarray(gradient_risk.gradient_q, dtype=np.float64)
        direction = -gradient
        norm = float(np.linalg.norm(direction))
        if norm < 1.0e-10 or not np.all(np.isfinite(direction)):
            break
        points = _repair_points(points, direction / norm, worst["tau"], durations, cfg.FAST_REPAIR_STEP_SIZE, limits)
        trajectory = NUBSTrajectory6D().generate(points, head, tail, durations)
        repair_ms += (time.perf_counter() - t1) * 1000.0
        repair_steps += 1
    t_val = time.perf_counter()
    online = online_verifier.verify(
        trajectory,
        forecast,
        current_q=head[:, 0],
        current_qd=head[:, 1],
        current_qdd=head[:, 2],
        q_goal=tail[:, 0],
        solver_success=True,
    )
    dense = dense_verifier.verify(
        trajectory,
        forecast,
        current_q=head[:, 0],
        current_qd=head[:, 1],
        current_qdd=head[:, 2],
        q_goal=tail[:, 0],
        solver_success=True,
    )
    validation_ms = (time.perf_counter() - t_val) * 1000.0
    total_ms = (time.perf_counter() - started) * 1000.0
    ref_dense_min = _trajectory_min(dense_verifier.risk_evaluator, local_ref, forecast, density="dense")
    cand_dense_min = float(dense.min_distance)
    return {
        "reference_trajectory": local_ref,
        "candidate_trajectory": trajectory,
        "metrics": {
            "reference_dense_min_distance": ref_dense_min,
            "candidate_dense_min_distance": cand_dense_min,
            "delta_dense_min_distance": float(cand_dense_min - ref_dense_min),
            "online_feasible": bool(online.accepted),
            "dense_geometry_only_feasible": bool(dense.accepted),
            "usable_candidate": bool(total_ms <= cfg.FAST_REPAIR_ACCEPT_MS and dense.accepted),
            "hard_realtime_ok": bool(total_ms <= cfg.FAST_REPAIR_HARD_MAX_MS),
            "beneficial": bool(cand_dense_min >= ref_dense_min + cfg.SWITCH_IMPROVEMENT_MARGIN),
            "repair_steps": repair_steps,
            "risk_scan_ms": float(risk_scan_ms),
            "repair_ms": float(repair_ms),
            "validation_ms": float(validation_ms),
            "total_ms": float(total_ms),
            "online_reasons": online.reasons,
            "dense_reasons": dense.reasons,
        },
    }


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            row["scenario_type"],
            f"{float(row['speed_group']):.2f}",
            row["lead_label"],
            row["method"],
        )
        groups.setdefault(key, []).append(row)
    summary = []
    for (scenario, speed, lead, method), items in sorted(groups.items()):
        summary.append(
            {
                "scenario_type": scenario,
                "speed_group": speed,
                "lead_label": lead,
                "method": method,
                "n": len(items),
                "usable": float(np.mean([item["usable_candidate"] for item in items])),
                "dense_feasible": float(np.mean([item["dense_geometry_only_feasible"] for item in items])),
                "beneficial": float(np.mean([item["beneficial"] for item in items])),
                "delta_dense": float(np.mean([item["delta_dense_min_distance"] for item in items])),
                "total_ms_mean": float(np.mean([item["total_ms"] for item in items])),
                "total_ms_p95": float(np.percentile([item["total_ms"] for item in items], 95)),
                "total_ms_max": float(np.max([item["total_ms"] for item in items])),
                "validation_ms_p95": float(np.percentile([item["validation_ms"] for item in items], 95)),
            }
        )
    return summary


def write_table(summary: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "| scenario | speed | conflict | method | n | usable | dense feasible | beneficial | delta Dmin dense / m | total mean / ms | total p95 / ms | total max / ms | validation p95 / ms |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['scenario_type']} | {row['speed_group']} | {row['lead_label']} | {FAST_METHOD_NAMES.get(row['method'], row['method'])} | "
            f"{row['n']} | {_fmt(row['usable'], 2)} | {_fmt(row['dense_feasible'], 2)} | {_fmt(row['beneficial'], 2)} | "
            f"{_fmt(row['delta_dense'])} | {_fmt(row['total_ms_mean'], 1)} | {_fmt(row['total_ms_p95'], 1)} | "
            f"{_fmt(row['total_ms_max'], 1)} | {_fmt(row['validation_ms_p95'], 1)} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(Path("results/new/6_4_fast_local_repair")))
    parser.add_argument("--config", default=str(cfg.STAGE4_CONFIG))
    parser.add_argument("--scenario", choices=["D1", "D2M"], default="D1")
    parser.add_argument("--method", choices=["critical_fast_repair", "ccro_fast_repair"], default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--g1", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    trials_dir = output / "trials"
    paper_dir = output / "paper"
    trials_dir.mkdir(parents=True, exist_ok=True)
    paper_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__).with_name("config_64.yaml"), output / "config_64.yaml")
    config = load_stage4_config(Path(args.config))
    model = load_surface_model(config)
    reference, _, _, _ = make_reference(config)
    ccro_evaluator, online_verifier, limits = make_risk_stack(config, model, None)
    critical_evaluator, _ = make_critical_risk_stack(config, model)
    dense_verifier = DynamicTrajectoryVerifier(
        ccro_evaluator,
        limits,
        d_stop=cfg.D_STOP,
        time_step=cfg.FAST_SAMPLE_DT,
        density="dense",
        epsilon_goal=1.0e-2,
        epsilon_continuity_q=5.0e-3,
        epsilon_continuity_qd=3.0e-3,
        epsilon_continuity_qdd=3.0e-3,
        limit_tolerance=1.0e-8,
    )
    instances = make_fast_instances(model, reference, scenario=args.scenario, smoke=args.smoke, g1=args.g1)
    methods = (args.method,) if args.method else ("critical_fast_repair", "ccro_fast_repair")
    rows: list[dict[str, Any]] = []
    for instance in instances:
        for method in methods:
            evaluator = critical_evaluator if method == "critical_fast_repair" else ccro_evaluator
            repaired = fast_repair(reference, model, evaluator, online_verifier, dense_verifier, limits, instance)
            row = {
                **{key: value for key, value in instance.items() if key not in {"obstacle_center0", "obstacle_velocity"}},
                "method": method,
                **repaired["metrics"],
            }
            rows.append(row)
            write_json(trials_dir / f"{instance['instance_id']}_{method}.json", row)
            print(
                f"[6.4 fast] {instance['instance_id']} {instance['lead_label']} {FAST_METHOD_NAMES[method]} "
                f"usable={row['usable_candidate']} dense={row['dense_geometry_only_feasible']} "
                f"dD={row['delta_dense_min_distance']:.3f} total={row['total_ms']:.1f} ms"
            )
    summary = _aggregate(rows)
    metrics = {
        "experiment": "6.4 Fast CCRO-NUBS local repair",
        "scope": "1 s local repair; no seconds-level trajectory optimization",
        "git_commit": git_commit_hash(),
        "scenario": args.scenario,
        "mode": "g1" if args.g1 else ("smoke" if args.smoke else "formal_stage_a_fast"),
        "timing_targets_ms": {
            "p95_total": cfg.FAST_REPAIR_ACCEPT_MS,
            "hard_max": cfg.FAST_REPAIR_HARD_MAX_MS,
        },
        "trial_count": len(rows),
        "summary": summary,
        "trials": rows,
    }
    write_json(output / "fast_local_repair_64.json", metrics)
    write_table(summary, paper_dir / "table_6_4_fast_local_repair.md")
    print(f"[6.4 fast] saved {output / 'fast_local_repair_64.json'}")
    print(f"[6.4 fast] saved {paper_dir / 'table_6_4_fast_local_repair.md'}")


if __name__ == "__main__":
    main()
