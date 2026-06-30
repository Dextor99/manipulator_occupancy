"""Build Chapter 6.7 ablation, timing, and sim-real consistency tables."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


SOURCES = {
    "p1": Path("data/results/ccro_p1/metrics.json"),
    "p5": Path("data/results/ccro_p5/metrics.json"),
    "p6": Path("data/results/ccro_p6/metrics.json"),
    "ch4_6_ablation": Path("data/results/ch4_6/ablation.json"),
    "ch4_6_timing": Path("data/results/ch4_6/timing.json"),
    "ch4_6_quality": Path("data/results/ch4_6/quality_check.json"),
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
    p1 = data["p1"]
    p5 = data["p5"]
    p6 = data["p6"]
    ch_ablation = data["ch4_6_ablation"]
    ch_timing = data["ch4_6_timing"]
    quality = data["ch4_6_quality"]

    risk_rows = []
    for method, row in p1["pooled"].items():
        risk_rows.append(
            {
                "method": method,
                "n": row["n"],
                "pass_rate": row["pass_rate"],
                "d_min_mean": row["d_min_mean"],
                "d_min_p05": row["d_min_p05"],
                "d_min_min": row["d_min_min"],
                "time_below_d_stop_mean": row["time_below_d_stop_mean"],
            }
        )

    control_rows = []
    for method, row in ch_ablation.items():
        control_rows.append(
            {
                "method": method,
                "T_lead": row["T_lead"],
                "D_min_ref": row["D_min_ref"],
                "T_viol": row["T_viol"],
                "R_avoid": row["R_avoid"],
                "R_timeout": row["R_timeout"],
                "J_q_rms": row["J_q_rms"],
            }
        )

    timing_rows = []
    for name, row in ch_timing["timing"].items():
        if name == "_meta":
            continue
        timing_rows.append(
            {
                "module": name,
                "mean_ms": row["mean"],
                "p95_ms": row["p95"],
                "ratio": row["ratio"],
                "nonzero_rows": row["nonzero_rows"],
            }
        )

    realtime = p5["timing"]
    lambda_rows = p6["lambda_time_sweep"]
    representation = p6["representation"]

    return {
        "source": "Chapter 6.7 ablation and timing summary",
        "source_files": {name: str(path) for name, path in SOURCES.items()},
        "risk_ablation": risk_rows,
        "control_ablation": control_rows,
        "timing_modules": timing_rows,
        "realtime_planning": {
            "runs": realtime["runs"],
            "mean_ms": realtime["mean_ms"],
            "p95_ms": realtime["p95_ms"],
            "max_ms": realtime["max_ms"],
            "timeout_rate": realtime["timeout_rate"],
            "dense_feasible_accept_rate": realtime["dense_feasible_accept_rate"],
            "strict_accept_rate": realtime["strict_accept_rate"],
            "solver_convergence_rate": realtime["solver_convergence_rate"],
            "control_p95_ms": realtime["control_p95_ms"],
        },
        "representation": representation,
        "lambda_time_sweep": lambda_rows,
        "quality_checks": quality["checks"],
        "accepted": bool(p1["accepted"] and p5["accepted"] and p6["accepted"]),
        "consistency_notes": [
            "Simulation and real replay agree that temporal full-body risk is necessary for body-link obstacles.",
            "Current real timing logs contain controller-side timing but not complete RGB-D preprocessing timing; use 6.6/6.7 timing claims conservatively.",
            "P5 optimized planner timing is the best current source for online replanning budget.",
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


def table_risk(summary: dict[str, Any]) -> str:
    rows = []
    for row in summary["risk_ablation"]:
        rows.append(
            [
                row["method"],
                str(row["n"]),
                fmt(row["pass_rate"]),
                fmt(row["d_min_mean"]),
                fmt(row["d_min_p05"]),
                fmt(row["d_min_min"]),
                fmt(row["time_below_d_stop_mean"]),
            ]
        )
    return markdown(["method", "n", "pass rate", "D_min mean/m", "D_min p05/m", "D_min min/m", "time<stop mean/s"], rows)


def table_control(summary: dict[str, Any]) -> str:
    rows = []
    for row in summary["control_ablation"]:
        rows.append(
            [
                row["method"],
                fmt(row["T_lead"]),
                fmt(row["D_min_ref"]),
                fmt(row["T_viol"]),
                fmt(row["R_avoid"]),
                fmt(row["R_timeout"]),
                fmt(row["J_q_rms"]),
            ]
        )
    return markdown(["method", "T_lead/s", "D_min_ref/m", "T_viol/s", "R_avoid", "R_timeout", "J_q_rms"], rows)


def table_timing(summary: dict[str, Any]) -> str:
    rows = [
        [row["module"], fmt(row["mean_ms"]), fmt(row["p95_ms"]), fmt(row["ratio"]), str(row["nonzero_rows"])]
        for row in summary["timing_modules"]
    ]
    return markdown(["module", "mean/ms", "p95/ms", "ratio", "nonzero rows"], rows)


def table_realtime(summary: dict[str, Any]) -> str:
    row = summary["realtime_planning"]
    rows = [
        ["runs", fmt(row["runs"])],
        ["planner mean/ms", fmt(row["mean_ms"])],
        ["planner p95/ms", fmt(row["p95_ms"])],
        ["planner max/ms", fmt(row["max_ms"])],
        ["timeout rate", fmt(row["timeout_rate"])],
        ["dense feasible accept rate", fmt(row["dense_feasible_accept_rate"])],
        ["strict accept rate", fmt(row["strict_accept_rate"])],
        ["solver convergence rate", fmt(row["solver_convergence_rate"])],
        ["control p95/ms", fmt(row["control_p95_ms"])],
    ]
    return markdown(["metric", "value"], rows)


def table_lambda(summary: dict[str, Any]) -> str:
    rows = [
        [fmt(row["lambda_time"]), str(row["success"]), fmt(row["duration"]), fmt(row["jerk_energy"]), fmt(row["objective"])]
        for row in summary["lambda_time_sweep"]
    ]
    return markdown(["lambda_time", "success", "duration/s", "jerk energy", "objective"], rows)


def notes(summary: dict[str, Any]) -> str:
    lines = ["Consistency notes:"]
    lines.extend(f"- {item}" for item in summary["consistency_notes"])
    lines.append("")
    lines.append("Quality checks:")
    for check in summary["quality_checks"]:
        status = "PASS" if check["passed"] else "NOTE"
        lines.append(f"- {status}: {check['name']} - {check['detail']}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Chapter 6.7 summary.")
    parser.add_argument("--output", default="data/results/ch6_7")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    summary = build_summary()
    (output / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tables = {
        "table_6_7_risk_ablation.md": table_risk(summary),
        "table_6_7_control_ablation.md": table_control(summary),
        "table_6_7_timing_modules.md": table_timing(summary),
        "table_6_7_realtime_planning.md": table_realtime(summary),
        "table_6_7_lambda_time.md": table_lambda(summary),
        "notes.md": notes(summary),
    }
    for name, text in tables.items():
        (output / name).write_text(text + "\n", encoding="utf-8")
    print(tables["table_6_7_risk_ablation.md"])
    print()
    print(tables["table_6_7_control_ablation.md"])
    print(f"\n[exp_67] saved summary to {output}")


if __name__ == "__main__":
    main()
