"""Post-run audit tables for Chapter 6.4 dynamic replanning results."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from . import config_64 as cfg
from .common_64 import (
    load_stage4_config,
    load_surface_model,
    min_distance_to_sphere,
    write_json,
)
from .scenarios_64 import obstacle_center_at


def _stat(values: list[float]) -> dict[str, float | None]:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not clean:
        return {"n": 0, "mean": None, "std": None, "min": None, "p05": None, "p50": None, "p95": None, "max": None}
    return {
        "n": len(clean),
        "mean": float(np.mean(clean)),
        "std": float(np.std(clean, ddof=1)) if len(clean) > 1 else 0.0,
        "min": float(np.min(clean)),
        "p05": float(np.percentile(clean, 5)),
        "p50": float(np.percentile(clean, 50)),
        "p95": float(np.percentile(clean, 95)),
        "max": float(np.max(clean)),
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


def load_trials(output: Path) -> list[dict[str, Any]]:
    trials = []
    for path in sorted((output / "trials").glob("*.json")):
        trials.append(json.loads(path.read_text(encoding="utf-8")))
    return trials


def load_instances(output: Path) -> dict[str, dict[str, Any]]:
    return {
        path.stem: json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((output / "instances").glob("*.json"))
    }


def candidate_audit(trials: list[dict[str, Any]]) -> dict[str, Any]:
    async_methods = {"ccro_nubs", "critical_point_nubs"}
    events = [
        {**event, "scenario_type": trial["scenario_type"], "method": trial["method"], "trial_id": trial["trial_id"]}
        for trial in trials
        if trial["method"] in async_methods
        for event in trial.get("events", [])
    ]
    reasons = Counter(
        "none" if not event.get("rejection_reasons") else "+".join(event["rejection_reasons"])
        for event in events
    )
    accepted = [event for event in events if event.get("candidate_accepted")]
    rejected = [event for event in events if not event.get("candidate_accepted")]
    by_scenario: dict[str, dict[str, Any]] = {}
    for scenario in sorted({f"{event['scenario_type']}::{event['method']}" for event in events}):
        scenario_type, method = scenario.split("::", 1)
        rows = [event for event in events if event["scenario_type"] == scenario_type and event["method"] == method]
        accepted_rows = [event for event in rows if event.get("candidate_accepted")]
        by_scenario[scenario] = {
            "scenario_type": scenario_type,
            "method": method,
            "events": len(rows),
            "accepted": len(accepted_rows),
            "optimizer_converged": int(
                sum(bool(row.get("optimizer_converged", row.get("solver_success"))) for row in rows)
            ),
            "elapsed_ms": _stat([row.get("elapsed_ms") for row in rows]),
            "candidate_min_distance": _stat([row.get("candidate_min_distance") for row in rows]),
            "accepted_candidate_min_distance": _stat([row.get("candidate_min_distance") for row in accepted_rows]),
        }
    return {
        "events": len(events),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "accepted_rate": len(accepted) / len(events) if events else None,
        "optimizer_converged": int(
            sum(bool(event.get("optimizer_converged", event.get("solver_success"))) for event in events)
        ),
        "optimizer_converged_rate": (
            sum(bool(event.get("optimizer_converged", event.get("solver_success"))) for event in events) / len(events)
            if events else None
        ),
        "elapsed_ms": _stat([event.get("elapsed_ms") for event in events]),
        "accepted_elapsed_ms": _stat([event.get("elapsed_ms") for event in accepted]),
        "candidate_min_distance": _stat([event.get("candidate_min_distance") for event in events]),
        "accepted_candidate_min_distance": _stat([event.get("candidate_min_distance") for event in accepted]),
        "rejection_reasons": dict(reasons),
        "by_scenario_method": by_scenario,
        "interpretation": (
            "candidate_accepted means online validation and switch gating passed under the "
            "experiment acceptance checks; optimizer_converged is retained as a separate "
            "optimizer convergence flag and is not used alone for switching."
        ),
    }


def d4_hold_audit(trials: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [trial for trial in trials if trial["scenario_type"] == "initial_high_risk" and trial["method"] == "ccro_nubs"]
    return {
        "n": len(rows),
        "immediate_hold": int(sum(trial["first_safety_hold_time"] == 0.0 for trial in rows)),
        "no_replan": int(sum(trial["replan_count"] == 0 for trial in rows)),
        "no_candidate_accept": int(sum(trial["accepted_count"] == 0 for trial in rows)),
        "goal_completion_expected": False,
        "hold_time_s": _stat([trial["first_safety_hold_time"] for trial in rows]),
        "min_distance_gt": _stat([trial["min_distance_gt"] for trial in rows]),
    }


def dense_gt_audit(
    output: Path,
    trials: list[dict[str, Any]],
    instances: dict[str, dict[str, Any]],
    *,
    density: str,
    time_step: float,
) -> dict[str, Any]:
    config = load_stage4_config(cfg.STAGE4_CONFIG)
    model = load_surface_model(config)
    per_trial = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for trial in trials:
        instance = instances[trial["instance_id"]]
        gt_center0 = np.asarray(instance["gt_center0"], dtype=np.float64)
        gt_velocity = np.asarray(instance["gt_velocity"], dtype=np.float64)
        gt_radius = float(instance["gt_radius"])
        motion_start_time = float(instance.get("motion_start_time", 0.0))
        pre_motion_center = (
            None
            if instance.get("pre_motion_center") is None
            else np.asarray(instance["pre_motion_center"], dtype=np.float64)
        )
        best = math.inf
        best_time = None
        best_link = None
        checked = 0
        last_time = -math.inf
        for row in trial.get("timeline", []):
            timestamp = float(row["time"])
            if timestamp + 1.0e-9 < last_time + time_step:
                continue
            q = np.asarray(row["q"], dtype=np.float64)
            center = obstacle_center_at(
                gt_center0,
                gt_velocity,
                timestamp,
                motion_start_time,
                pre_motion_center,
            )
            distance, link = min_distance_to_sphere(model, q, center, gt_radius, density)
            checked += 1
            last_time = timestamp
            if distance < best:
                best = distance
                best_time = timestamp
                best_link = link
        item = {
            "trial_id": trial["trial_id"],
            "scenario_type": trial["scenario_type"],
            "method": trial["method"],
            "checked_samples": checked,
            "min_distance_gt_recheck": float(best),
            "min_time": best_time,
            "nearest_link": best_link,
            "violation": bool(best < cfg.D_STOP),
        }
        per_trial.append(item)
        grouped[(trial["scenario_type"], trial["method"])].append(item)
    groups = []
    for (scenario_type, method), rows in sorted(grouped.items()):
        groups.append(
            {
                "scenario_type": scenario_type,
                "method": method,
                "n": len(rows),
                "checked_samples": int(sum(row["checked_samples"] for row in rows)),
                "violation_rate": float(np.mean([row["violation"] for row in rows])),
                "min_distance_gt_recheck": _stat([row["min_distance_gt_recheck"] for row in rows]),
            }
        )
    return {
        "density": density,
        "time_step": time_step,
        "d_stop": cfg.D_STOP,
        "groups": groups,
        "per_trial": per_trial,
    }


def write_candidate_table(audit: dict[str, Any], path: Path) -> None:
    c = audit["candidate"]
    reasons = ", ".join(f"{name}: {count}" for name, count in c["rejection_reasons"].items())
    lines = [
        "| scope | events | accepted | accepted rate | optimizer converged | optimizer converged rate | Dmin accepted min / m | planner p95 / ms | rejection reasons |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
        (
            f"| all async NUBS candidates | {c['events']} | {c['accepted']} | {_fmt(c['accepted_rate'], 3)} | "
            f"{c['optimizer_converged']} | {_fmt(c['optimizer_converged_rate'], 3)} | "
            f"{_fmt(c['accepted_candidate_min_distance']['min'])} | "
            f"{_fmt(c['accepted_elapsed_ms']['p95'], 1)} | {reasons} |"
        ),
    ]
    for _, row in c["by_scenario_method"].items():
        lines.append(
            f"| {row['scenario_type']} / {row['method']} | {row['events']} | {row['accepted']} | "
            f"{_fmt(row['accepted'] / row['events'] if row['events'] else None, 3)} | "
            f"{row['optimizer_converged']} | {_fmt(row['optimizer_converged'] / row['events'] if row['events'] else None, 3)} | "
            f"{_fmt(row['accepted_candidate_min_distance']['min'])} | {_fmt(row['elapsed_ms']['p95'], 1)} | - |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_dense_table(audit: dict[str, Any], path: Path) -> None:
    dense = audit["dense_gt_recheck"]
    lines = [
        f"GT recheck density: `{dense['density']}`, time step: `{dense['time_step']}` s, d_stop: `{dense['d_stop']}` m.",
        "",
        "| scenario | method | n | checked samples | violation rate | Dmin mean / m | Dmin min / m | Dmin p05 / m |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in dense["groups"]:
        stat = row["min_distance_gt_recheck"]
        lines.append(
            f"| {row['scenario_type']} | {row['method']} | {row['n']} | {row['checked_samples']} | "
            f"{_fmt(row['violation_rate'], 2)} | {_fmt(stat['mean'])} | {_fmt(stat['min'])} | {_fmt(stat['p05'])} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_d4_table(audit: dict[str, Any], path: Path) -> None:
    d4 = audit["d4_hold"]
    lines = [
        "| n | immediate hold | no replan | no candidate accepted | first hold mean / s | Dmin mean / m | interpretation |",
        "|---:|---:|---:|---:|---:|---:|---|",
        (
            f"| {d4['n']} | {d4['immediate_hold']} | {d4['no_replan']} | {d4['no_candidate_accept']} | "
            f"{_fmt(d4['hold_time_s']['mean'])} | {_fmt(d4['min_distance_gt']['mean'])} | "
            "initial high risk is expected to enter safety hold rather than finish the task |"
        ),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(cfg.DEFAULT_OUTPUT))
    parser.add_argument("--density", default="dense", choices=["coarse", "medium", "dense"])
    parser.add_argument("--time-step", type=float, default=0.08)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output).resolve()
    trials = load_trials(output)
    instances = load_instances(output)
    audit = {
        "candidate": candidate_audit(trials),
        "d4_hold": d4_hold_audit(trials),
        "dense_gt_recheck": dense_gt_audit(
            output,
            trials,
            instances,
            density=args.density,
            time_step=float(args.time_step),
        ),
    }
    write_json(output / "audit_64.json", audit)
    write_candidate_table(audit, output / "paper" / "table_6_4_candidate_validation_audit.md")
    write_dense_table(audit, output / "paper" / "table_6_4_gt_dense_recheck.md")
    write_d4_table(audit, output / "paper" / "table_6_4_initial_high_risk_hold.md")
    print(f"[6.4 audit] saved {output / 'audit_64.json'}")
    print(f"[6.4 audit] saved paper audit tables under {output / 'paper'}")


if __name__ == "__main__":
    main()
