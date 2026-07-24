"""Mechanism gates for the near-final Chapter 6.4 experiment."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from .common_64 import (
    constant_forecast,
    load_stage4_config,
    load_surface_model,
    make_critical_risk_stack,
    make_reference,
    make_risk_stack,
    write_json,
)


def benchmark_risk_queries(repeats: int) -> dict[str, object]:
    config = load_stage4_config()
    model = load_surface_model(config)
    reference, *_ = make_reference(config)
    mesh_evaluator, _, _ = make_risk_stack(config, model, None)
    critical_evaluator, _ = make_critical_risk_stack(config, model)
    q = reference.evaluate(0.45 * reference.total_duration)
    forecast = constant_forecast(
        np.array([0.4, 0.0, 0.5], dtype=np.float64),
        np.array([0.08, 0.02, 0.0], dtype=np.float64),
        0.05,
    )
    rows = {}
    for name, evaluator in [("critical", critical_evaluator), ("ccro", mesh_evaluator)]:
        for _ in range(5):
            evaluator.configuration(q, forecast, 0.5, density="coarse", with_gradient=True)
        started = time.perf_counter()
        for _ in range(repeats):
            evaluator.configuration(q, forecast, 0.5, density="coarse", with_gradient=True)
        rows[name] = (time.perf_counter() - started) * 1000.0 / repeats
    return {
        "critical_gradient_query_ms": float(rows["critical"]),
        "ccro_gradient_query_ms": float(rows["ccro"]),
        "critical_point_count": int(critical_evaluator.critical_point_count()),
        "critical_faster_than_ccro": bool(rows["critical"] < rows["ccro"]),
    }


def _stat(values: list[float]) -> dict[str, float | None]:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not clean:
        return {"mean": None, "p95": None, "max": None}
    return {
        "mean": float(np.mean(clean)),
        "p95": float(np.percentile(clean, 95)),
        "max": float(np.max(clean)),
    }


def _trial_summary(trials_dir: Path, methods: tuple[str, ...]) -> dict[str, object]:
    trials = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(trials_dir.glob("*.json"))
        if any(path.name.endswith(f"_{method}.json") for method in methods)
    ]
    events = [event for trial in trials for event in trial.get("events", [])]
    deadline_miss = [event for event in events if "planning_budget" in event.get("rejection_reasons", [])]
    continuity_reject = [
        event for event in events
        if any(reason.startswith("continuity") for reason in event.get("rejection_reasons", []))
    ]
    verification_reject = [
        event for event in events
        if "distance_ok" in event.get("rejection_reasons", [])
    ]
    safe_with_switch = [
        trial for trial in trials
        if trial.get("task_safe_success") and int(trial.get("accepted_count", 0)) > 0
    ]
    safe_without_switch = [
        trial for trial in trials
        if trial.get("task_safe_success") and int(trial.get("accepted_count", 0)) == 0
    ]
    unsafe = [trial for trial in trials if not trial.get("task_safe_success")]
    tau_errors = [
        event.get("tau_prediction_error_at_switch")
        for event in events
        if event.get("tau_prediction_error_at_switch") is not None
    ]
    planner_ms = [
        event.get("elapsed_ms")
        for event in events
        if event.get("elapsed_ms") is not None
    ]
    return {
        "trials": len(trials),
        "task_safe": int(sum(bool(trial.get("task_safe_success")) for trial in trials)),
        "replan_success": int(sum(int(trial.get("accepted_count", 0)) > 0 for trial in trials)),
        "deadline_miss": len(deadline_miss),
        "continuity_rejections": len(continuity_reject),
        "online_distance_rejections": len(verification_reject),
        "safe_with_switch": len(safe_with_switch),
        "safe_without_switch": len(safe_without_switch),
        "unsafe_or_unfinished": len(unsafe),
        "planner_ms": _stat(planner_ms),
        "tau_prediction_error": _stat(tau_errors),
    }


def run_mechanism_gate(output: Path, scenario: str, methods: tuple[str, ...]) -> dict[str, object]:
    run_dir = output / f"mechanism_{scenario.lower()}"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    cmd = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        "py310",
        "python",
        "-u",
        "-m",
        "experiments.new.6_4.run_dynamic_replanning",
        "--output",
        str(run_dir),
        "--gate",
        "--scenario",
        scenario,
    ]
    for method in methods:
        # Run one method at a time so a method-specific failure is obvious.
        method_cmd = [*cmd, "--method", method]
        subprocess.run(method_cmd, cwd=Path(__file__).resolve().parents[3], check=True)
    summary = _trial_summary(run_dir / "trials", methods)
    if scenario == "D1":
        summary["gate_pass"] = bool(
            int(summary["task_safe"]) >= 5
            and int(summary["replan_success"]) >= 4
            and int(summary["deadline_miss"]) <= 1
            and int(summary["online_distance_rejections"]) <= 1
            and int(summary["continuity_rejections"]) == 0
            and (summary["planner_ms"]["p95"] is not None and float(summary["planner_ms"]["p95"]) <= 5500.0)
            and int(summary["safe_with_switch"]) >= 4
        )
    elif scenario == "D2M":
        summary["gate_pass"] = bool(
            int(summary["task_safe"]) >= 4
            and int(summary["replan_success"]) >= 3
            and int(summary["deadline_miss"]) <= 1
            and int(summary["continuity_rejections"]) == 0
            and (summary["planner_ms"]["p95"] is not None and float(summary["planner_ms"]["p95"]) <= 5500.0)
        )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/new/6_4_final_gate")
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--mechanism", action="store_true")
    parser.add_argument("--scenario", choices=["D1", "D2M"], default="D1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output).resolve()
    result = {
        "risk_query_benchmark": benchmark_risk_queries(max(1, int(args.repeats))),
        "module": importlib.import_module(__package__ + ".config_64").__name__,
    }
    if args.mechanism:
        result["mechanism_gate"] = run_mechanism_gate(
            output,
            args.scenario,
            ("ccro_nubs",),
        )
    write_json(output / "gate_64.json", result)
    print(result)


if __name__ == "__main__":
    main()
