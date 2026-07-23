"""Summaries and paper tables for Chapter 6.4 dynamic replanning."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from . import config_64 as cfg


def _stat(values: list[float]) -> dict[str, float | None]:
    clean = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not clean:
        return {"mean": None, "std": None, "min": None, "max": None}
    return {
        "mean": float(np.mean(clean)),
        "std": float(np.std(clean, ddof=1)) if len(clean) > 1 else 0.0,
        "min": float(np.min(clean)),
        "max": float(np.max(clean)),
    }


def aggregate(trials: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, dict[str, Any]] = {}
    for trial in trials:
        key = f"{trial['scenario_type']}::{trial['method']}"
        groups.setdefault(key, {"scenario_type": trial["scenario_type"], "method": trial["method"], "trials": []})
        groups[key]["trials"].append(trial)
    rows = []
    for group in groups.values():
        items = group["trials"]
        rows.append(
            {
                "scenario_type": group["scenario_type"],
                "method": group["method"],
                "method_name": cfg.METHOD_NAMES.get(group["method"], group["method"]),
                "n": len(items),
                "success_rate": float(np.mean([bool(item["success"]) for item in items])),
                "task_safe_success_rate": float(np.mean([bool(item.get("task_safe_success", item["success"])) for item in items])),
                "replan_success_rate": float(np.mean([bool(item.get("replan_success", item["accepted_count"] >= 1)) for item in items])),
                "finish_rate": float(np.mean([bool(item["finished"]) for item in items])),
                "safety_violation_rate": float(np.mean([item["safety_violation_time_s"] > 0.0 for item in items])),
                "min_distance_gt": _stat([item["min_distance_gt"] for item in items]),
                "goal_error": _stat([item["goal_error"] for item in items]),
                "replan_count": _stat([item["replan_count"] for item in items]),
                "accepted_count": _stat([item["accepted_count"] for item in items]),
                "first_replan_time": _stat([item["first_replan_time"] for item in items if item["first_replan_time"] is not None]),
                "first_safety_hold_time": _stat([item["first_safety_hold_time"] for item in items if item["first_safety_hold_time"] is not None]),
                "planning_cycles": _stat([item["planning_control_cycles"] for item in items]),
                "bridge_min_distance": _stat([item["bridge_min_distance"] for item in items if item.get("bridge_min_distance") is not None]),
                "planner_elapsed_ms": _stat(
                    [
                        event["elapsed_ms"]
                        for item in items
                        for event in item.get("events", [])
                        if event.get("elapsed_ms") is not None
                    ]
                ),
                "false_replans": int(sum(bool(item.get("false_replan")) for item in items)),
            }
        )
    return {
        "trial_count": len(trials),
        "accepted": bool(all(row["success_rate"] >= 0.0 for row in rows)),
        "groups": sorted(rows, key=lambda row: (row["scenario_type"], row["method"])),
    }


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "-"
    return f"{number:.{digits}f}"


def write_paper_table(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "| scenario | method | n | task safe | replan success | finish | violation | Dmin GT / m | bridge Dmin / m | replans | accepted | planner ms |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["groups"]:
        lines.append(
            "| {scenario_type} | {method_name} | {n} | {task:.2f} | {replan_success:.2f} | {finish:.2f} | {violation:.2f} | "
            "{dmin} | {bridge} | {replans} | {accepted} | {planner} |".format(
                scenario_type=row["scenario_type"],
                method_name=row["method_name"],
                n=row["n"],
                task=row["task_safe_success_rate"],
                replan_success=row["replan_success_rate"],
                finish=row["finish_rate"],
                violation=row["safety_violation_rate"],
                dmin=_fmt(row["min_distance_gt"]["mean"]),
                bridge=_fmt(row["bridge_min_distance"]["mean"]),
                replans=_fmt(row["replan_count"]["mean"], 2),
                accepted=_fmt(row["accepted_count"]["mean"], 2),
                planner=_fmt(row["planner_elapsed_ms"]["mean"], 1),
            )
        )
    lines.extend(
        [
            "",
            "Notes:",
            "",
            "- `violation` is GT safety-distance violation rate under the executed closed loop.",
            "- `initial_high_risk` is a safety-hold test: `finish=0` and `violation=1` are expected because the obstacle is initialized inside the hold region; acceptance is judged by immediate hold, zero replans, and zero candidate switches.",
            "- `task safe` reports task completion without GT safety violation; `replan success` reports at least one accepted candidate switch after a trigger.",
            "- Candidate switching uses online medium validation and is followed by dense GT offline audit; optimizer convergence flags are reported separately in `table_6_4_candidate_validation_audit.md`.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
