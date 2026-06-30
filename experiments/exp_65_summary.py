"""Build Chapter 6.5 dynamic virtual-loop and rolling-replanning tables."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


SOURCES = {
    "virtual": Path("data/results/ch4_5_virtual/metrics.json"),
    "stage4": Path("data/results/ccro_stage4/metrics.json"),
    "p4": Path("data/results/ccro_p4/metrics.json"),
    "p5": Path("data/results/ccro_p5/metrics.json"),
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def require_sources() -> dict[str, dict[str, Any]]:
    missing = [str(path) for path in SOURCES.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing required source files:\n" + "\n".join(missing))
    return {name: load_json(path) for name, path in SOURCES.items()}


def build_summary() -> dict[str, Any]:
    data = require_sources()
    virtual = data["virtual"]
    stage4 = data["stage4"]
    p4 = data["p4"]
    p5 = data["p5"]

    virtual_rows = []
    for method in ("ssm", "apf", "ours_scale", "ours_rep", "ours_full"):
        row = virtual[method]
        virtual_rows.append(
            {
                "method": method,
                "n_trials": row["n_trials"],
                "R_avoid": row["R_avoid"],
                "R_timeout": row["R_timeout"],
                "D_min_ref": row["D_min_ref"],
                "T_viol": row["T_viol"],
                "J_q_rms": row["J_q_rms"],
                "T_task": row["T_task"],
            }
        )

    replan_rows = []
    for scenario, row in stage4["scenarios"].items():
        passive = row["passive"]
        active = row["active"]
        replan_rows.append(
            {
                "scenario": scenario,
                "name": row["name"],
                "expectation": row["expectation"],
                "passive_D_min": passive["min_actual_distance"],
                "active_D_min": active["min_actual_distance"],
                "improvement": active["min_actual_distance"] - passive["min_actual_distance"],
                "replan_count": active["replan_count"],
                "accepted_count": active["accepted_count"],
                "hold_time": active["hold_time"],
                "safety_events": len(active["safety_events"]),
                "finished": active["finished"],
                "goal_error": active["goal_error"],
            }
        )

    p4_rows = []
    for scenario, row in p4["scenarios"].items():
        p4_rows.append(
            {
                "scenario": scenario,
                "expected": row["expected"],
                "A4_D_min": row["A4"]["min_distance"],
                "A5_D_min": row["A5"]["min_distance"],
                "A6_D_min": row["A6"]["min_distance"],
                "A5_replans": row["A5"].get("accepted_count", 0),
                "A6_time_below_stop": row["A6"]["time_below_stop"],
                "A6_finished": row["A6"]["finished"],
                "A6_goal_error": row["A6"]["goal_error"],
                "A6_control_p95_ms": row["A6"]["control_p95_ms"],
                "A6_state_mismatch_holds": row["A6"]["state_mismatch_holds"],
            }
        )

    timing = p5["timing"]
    timing_row = {
        "runs": timing["runs"],
        "planner_mean_ms": timing["mean_ms"],
        "planner_p95_ms": timing["p95_ms"],
        "planner_max_ms": timing["max_ms"],
        "timeout_rate": timing["timeout_rate"],
        "dense_feasible_accept_rate": timing["dense_feasible_accept_rate"],
        "strict_accept_rate": timing["strict_accept_rate"],
        "solver_convergence_rate": timing["solver_convergence_rate"],
        "control_p95_ms": timing["control_p95_ms"],
    }

    return {
        "source": "Chapter 6.5 summary from virtual closed-loop and CCRO rolling replanning results",
        "source_files": {name: str(path) for name, path in SOURCES.items()},
        "virtual_loop": virtual_rows,
        "rolling_replan": replan_rows,
        "unified_executor": p4_rows,
        "realtime_budget": timing_row,
        "accepted": bool(stage4["accepted"] and p4["accepted"] and p5["accepted"]),
        "notes": [
            "Scenario D is an emergency safety-takeover case; unfinished goal is expected and should not be counted as task failure.",
            "P4 scenario A A6 now finishes after fail-soft state-mismatch recovery; state_mismatch_holds should remain zero in the current run.",
            "P5 reports finite-budget dense-feasible acceptance separately from strict solver convergence.",
        ],
    }


def markdown(headers: list[str], rows: list[list[str]]) -> str:
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
            *["| " + " | ".join(row) + " |" for row in rows],
        ]
    )


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if math.isinf(value):
            return "inf"
        return f"{value:.6g}"
    return str(value)


def table_virtual(summary: dict[str, Any]) -> str:
    rows = []
    for row in summary["virtual_loop"]:
        rows.append(
            [
                row["method"],
                str(row["n_trials"]),
                fmt(row["R_avoid"]),
                fmt(row["D_min_ref"]),
                fmt(row["T_viol"]),
                fmt(row["J_q_rms"]),
                fmt(row["R_timeout"]),
            ]
        )
    return markdown(["method", "n", "R_avoid", "D_min_ref/m", "T_viol/s", "J_q_rms", "R_timeout"], rows)


def table_replan(summary: dict[str, Any]) -> str:
    rows = []
    for row in summary["rolling_replan"]:
        rows.append(
            [
                row["scenario"],
                row["name"],
                row["expectation"],
                fmt(row["passive_D_min"]),
                fmt(row["active_D_min"]),
                fmt(row["improvement"]),
                str(row["replan_count"]),
                str(row["accepted_count"]),
                fmt(row["hold_time"]),
                str(row["safety_events"]),
                str(row["finished"]),
            ]
        )
    return markdown(["scenario", "name", "expected", "passive D_min", "active D_min", "gain", "replans", "accepted", "hold/s", "safety events", "finished"], rows)


def table_unified(summary: dict[str, Any]) -> str:
    rows = []
    for row in summary["unified_executor"]:
        rows.append(
            [
                row["scenario"],
                row["expected"],
                fmt(row["A4_D_min"]),
                fmt(row["A5_D_min"]),
                fmt(row["A6_D_min"]),
                str(row["A5_replans"]),
                fmt(row["A6_time_below_stop"]),
                str(row["A6_finished"]),
                fmt(row["A6_goal_error"]),
                fmt(row["A6_control_p95_ms"]),
                str(row["A6_state_mismatch_holds"]),
            ]
        )
    return markdown(["scenario", "expected", "A4 static D_min", "A5 replan D_min", "A6 executor D_min", "A5 accepted", "A6 below stop/s", "A6 finished", "A6 goal error", "A6 control p95/ms", "state holds"], rows)


def table_realtime(summary: dict[str, Any]) -> str:
    row = summary["realtime_budget"]
    rows = [
        ["planner runs", fmt(row["runs"])],
        ["planner mean / ms", fmt(row["planner_mean_ms"])],
        ["planner p95 / ms", fmt(row["planner_p95_ms"])],
        ["planner max / ms", fmt(row["planner_max_ms"])],
        ["timeout rate", fmt(row["timeout_rate"])],
        ["dense feasible accept rate", fmt(row["dense_feasible_accept_rate"])],
        ["strict accept rate", fmt(row["strict_accept_rate"])],
        ["solver convergence rate", fmt(row["solver_convergence_rate"])],
        ["control p95 / ms", fmt(row["control_p95_ms"])],
    ]
    return markdown(["metric", "value"], rows)


def notes(summary: dict[str, Any]) -> str:
    return "\n".join(f"- {line}" for line in summary["notes"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Chapter 6.5 summary.")
    parser.add_argument("--output", default="data/results/ch6_5")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    summary = build_summary()
    (output / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tables = {
        "table_6_5_virtual_loop.md": table_virtual(summary),
        "table_6_5_rolling_replan.md": table_replan(summary),
        "table_6_5_unified_executor.md": table_unified(summary),
        "table_6_5_realtime_budget.md": table_realtime(summary),
        "notes.md": notes(summary),
    }
    for name, text in tables.items():
        (output / name).write_text(text + "\n", encoding="utf-8")
    print(tables["table_6_5_virtual_loop.md"])
    print()
    print(tables["table_6_5_rolling_replan.md"])
    print(f"\n[exp_65] saved summary to {output}")


if __name__ == "__main__":
    main()
