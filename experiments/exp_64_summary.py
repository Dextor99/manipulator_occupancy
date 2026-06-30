"""Build Chapter 6.4 NUBS trajectory-optimization summary tables."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


SOURCES = {
    "stage1": Path("data/results/ccro_stage1/metrics.json"),
    "stage2": Path("data/results/ccro_stage2/metrics.json"),
    "p2": Path("data/results/ccro_p2/metrics.json"),
    "p6": Path("data/results/ccro_p6/metrics.json"),
    "external": Path("data/results/ch6_4_external/metrics.json"),
}

METHOD_NAMES_STAGE2 = {
    "baseline": "NUBS-base",
    "ee_only": "NUBS-EEF-risk",
    "full_body": "CCRO-NUBS",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def require_sources() -> dict[str, dict[str, Any]]:
    required = {name: path for name, path in SOURCES.items() if name != "external"}
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing required source files:\n" + "\n".join(missing))
    data = {name: load_json(path) for name, path in required.items()}
    if SOURCES["external"].exists():
        data["external"] = load_json(SOURCES["external"])
    return data


def build_summary() -> dict[str, Any]:
    data = require_sources()
    stage1 = data["stage1"]
    stage2 = data["stage2"]
    p2 = data["p2"]
    p6 = data["p6"]
    external = data.get("external")

    initial = stage1["optimization"]["initial_energy"]
    final = stage1["optimization"]["final_energy"]
    basic = {
        "accepted": stage1["accepted"],
        "start_error": stage1["boundary_errors"]["q_start"],
        "goal_error": stage1["boundary_errors"]["q_goal"],
        "waypoint_error": stage1["waypoint_error"],
        "gradient_relative_error": stage1["gradient_check"]["relative_error"],
        "gradient_cosine": stage1["gradient_check"]["cosine_similarity"],
        "initial_energy": initial,
        "final_energy": final,
        "energy_reduction_rate": (initial - final) / initial if initial else None,
        "max_q_violation": stage1["limits"]["max_q_violation"],
        "max_qd_violation": stage1["limits"]["max_qd_violation"],
        "max_qdd_violation": stage1["limits"]["max_qdd_violation"],
        "optimization_ms": stage1["optimization"]["elapsed_ms"],
        "construct_mean_ms": stage1["benchmark"]["construct_mean_ms"],
        "sample_101_mean_ms": stage1["benchmark"]["sample_101_mean_ms"],
    }

    time_rows = []
    for method in ("fixed", "total", "segment"):
        row = p2[method]
        time_rows.append(
            {
                "method": method,
                "success": row["success"],
                "duration": row["total_duration"],
                "jerk_energy": row["jerk_energy"],
                "objective": row.get("final_cost"),
                "elapsed_ms": row["elapsed_ms"],
                "max_q_violation": row.get("max_q_violation", 0.0),
                "max_qd_violation": row.get("max_qd_violation", 0.0),
                "max_qdd_violation": row.get("max_qdd_violation", 0.0),
            }
        )

    static_rows = []
    for scenario, scenario_data in stage2["scenarios"].items():
        for method in ("baseline", "ee_only", "full_body"):
            row = scenario_data["methods"][method]
            verification = row["verification"]
            optimization = row.get("optimization", {})
            static_rows.append(
                {
                    "scenario": scenario,
                    "method": method,
                    "method_name": METHOD_NAMES_STAGE2[method],
                    "solver_success": row.get("solver_success"),
                    "accepted": verification.get("accepted"),
                    "D_min_dense": verification.get("min_distance"),
                    "nearest_link": verification.get("nearest_link"),
                    "goal_error": verification.get("goal_error"),
                    "full_body_risk": row.get("full_body_risk_cost"),
                    "jerk_energy": optimization.get("final_energy"),
                    "optimization_ms": optimization.get("elapsed_ms", 0.0),
                }
            )

    representation = p6.get("representation", [])
    lambda_sweep = p6.get("lambda_time_sweep", [])
    external_rows = []
    rrt_stats_rows = []
    if external is not None:
        for scenario, payload in external["scenarios"].items():
            for method, row in payload["methods"].items():
                verification = row["verification"]
                optimization = row.get("optimization", {})
                external_rows.append(
                    {
                        "scenario": scenario,
                        "method": {
                            "minco_base": "MINCO-base",
                            "minco_risk": "MINCO-risk",
                            "rrt_connect_smooth": "RRT-Connect + smoothing",
                        }.get(method, method),
                        "solver_success": row.get("solver_success"),
                        "accepted": verification.get("accepted"),
                        "D_min_dense": verification.get("min_distance"),
                        "full_body_risk": row.get("full_body_risk_cost"),
                        "jerk_energy": optimization.get("final_energy"),
                        "goal_error": verification.get("goal_error"),
                        "nearest_link": row.get("nearest_link"),
                        "elapsed_ms": optimization.get("elapsed_ms", 0.0),
                    }
                )
                if method == "rrt_connect_smooth" and row.get("statistics"):
                    stats = row["statistics"]
                    rrt_stats_rows.append(
                        {
                            "scenario": scenario,
                            "n": stats.get("n"),
                            "success_rate": stats.get("success_rate"),
                            "accepted_rate": stats.get("accepted_rate"),
                            "D_min_mean": stats.get("D_min", {}).get("mean"),
                            "D_min_std": stats.get("D_min", {}).get("std"),
                            "D_min_ci95": stats.get("D_min", {}).get("ci95"),
                            "J_smooth_mean": stats.get("J_smooth", {}).get("mean"),
                            "J_smooth_std": stats.get("J_smooth", {}).get("std"),
                            "elapsed_ms_mean": stats.get("elapsed_ms", {}).get("mean"),
                            "elapsed_ms_std": stats.get("elapsed_ms", {}).get("std"),
                        }
                    )
    missing_baselines = []
    if external is None:
        missing_baselines = [
            {
                "baseline": "MINCO-base",
                "status": "not yet implemented in this repository",
                "required_action": "run experiments.exp_64_external_baselines and cite GCOPTER/MINCO source",
            },
            {
                "baseline": "MINCO-risk",
                "status": "not yet implemented in this repository",
                "required_action": "reuse same J_risk samples and dense verifier; separate representation effect from risk-model effect",
            },
            {
                "baseline": "RRT-Connect + smoothing",
                "status": "not yet implemented in this repository",
                "required_action": "lock planner version, collision model, time parameterization, smoothing and random seeds",
            },
        ]

    return {
        "source": "Chapter 6.4 summary from frozen CCRO-NUBS stage results",
        "source_files": {name: str(path) for name, path in SOURCES.items()},
        "basic_nubs": basic,
        "time_optimization": time_rows,
        "static_risk": static_rows,
        "external_baselines": external_rows,
        "rrt_multiseed": rrt_stats_rows,
        "external_references": None if external is None else external.get("references"),
        "representation": representation,
        "lambda_time_sweep": lambda_sweep,
        "missing_external_baselines": missing_baselines,
        "accepted": bool(stage1.get("accepted") and stage2.get("accepted") and p2.get("accepted") and p6.get("accepted")),
    }


def table_basic(summary: dict[str, Any]) -> str:
    b = summary["basic_nubs"]
    rows = [
        ["start q error", fmt(b["start_error"])],
        ["goal q error", fmt(b["goal_error"])],
        ["waypoint error", fmt(b["waypoint_error"])],
        ["gradient relative error", fmt(b["gradient_relative_error"])],
        ["gradient cosine", fmt(b["gradient_cosine"])],
        ["initial jerk energy", fmt(b["initial_energy"])],
        ["final jerk energy", fmt(b["final_energy"])],
        ["jerk reduction rate", fmt(b["energy_reduction_rate"])],
        ["max q / qd / qdd violation", f"{fmt(b['max_q_violation'])} / {fmt(b['max_qd_violation'])} / {fmt(b['max_qdd_violation'])}"],
        ["optimization time / ms", fmt(b["optimization_ms"])],
        ["construct mean / ms", fmt(b["construct_mean_ms"])],
        ["sample 101 mean / ms", fmt(b["sample_101_mean_ms"])],
    ]
    return markdown(["metric", "value"], rows)


def table_time(summary: dict[str, Any]) -> str:
    rows = []
    for row in summary["time_optimization"]:
        rows.append(
            [
                row["method"],
                str(row["success"]),
                fmt(row["duration"]),
                fmt(row["jerk_energy"]),
                fmt(row["objective"]),
                fmt(row["elapsed_ms"]),
                f"{fmt(row['max_q_violation'])}/{fmt(row['max_qd_violation'])}/{fmt(row['max_qdd_violation'])}",
            ]
        )
    return markdown(["method", "success", "duration/s", "jerk", "objective", "time/ms", "limit viol q/qd/qdd"], rows)


def table_static(summary: dict[str, Any]) -> str:
    rows = []
    for row in summary["static_risk"]:
        rows.append(
            [
                row["scenario"],
                row["method_name"],
                str(row["solver_success"]),
                str(row["accepted"]),
                fmt(row["D_min_dense"]),
                fmt(row["full_body_risk"]),
                fmt(row["jerk_energy"]),
                fmt(row["goal_error"]),
                row.get("nearest_link") or "-",
                fmt(row["optimization_ms"]),
            ]
        )
    return markdown(["scenario", "method", "solver", "accepted", "D_min dense/m", "J_risk full", "J_smooth", "goal error", "nearest link", "time/ms"], rows)


def table_representation(summary: dict[str, Any]) -> str:
    rows = [
        [row["method"], fmt(row["duration"]), fmt(row["jerk_energy"])]
        for row in summary["representation"]
    ]
    return markdown(["method", "duration/s", "jerk energy"], rows)


def table_lambda(summary: dict[str, Any]) -> str:
    rows = [
        [fmt(row["lambda_time"]), str(row["success"]), fmt(row["duration"]), fmt(row["jerk_energy"]), fmt(row["objective"])]
        for row in summary["lambda_time_sweep"]
    ]
    return markdown(["lambda_time", "success", "duration/s", "jerk energy", "objective"], rows)


def table_missing(summary: dict[str, Any]) -> str:
    if not summary["missing_external_baselines"]:
        return "All Chapter 6.4 external baselines are implemented in `data/results/ch6_4_external/`.\n"
    rows = [
        [row["baseline"], row["status"], row["required_action"]]
        for row in summary["missing_external_baselines"]
    ]
    return markdown(["baseline", "status", "required action before paper-final comparison"], rows)


def table_external(summary: dict[str, Any]) -> str:
    rows = []
    for row in summary["external_baselines"]:
        rows.append(
            [
                row["scenario"],
                row["method"],
                str(row["solver_success"]),
                str(row["accepted"]),
                fmt(row["D_min_dense"]),
                fmt(row["full_body_risk"]),
                fmt(row["jerk_energy"]),
                fmt(row["goal_error"]),
                row.get("nearest_link") or "-",
                fmt(row["elapsed_ms"]),
            ]
        )
    if not rows:
        return "External baselines have not been generated yet.\n"
    return markdown(["scenario", "method", "solver", "accepted", "D_min dense/m", "J_risk full", "J_smooth", "goal error", "nearest link", "time/ms"], rows)


def table_rrt_multiseed(summary: dict[str, Any]) -> str:
    rows = []
    for row in summary["rrt_multiseed"]:
        rows.append(
            [
                row["scenario"],
                fmt(row["n"]),
                fmt(row["success_rate"]),
                fmt(row["accepted_rate"]),
                fmt(row["D_min_mean"]),
                fmt(row["D_min_std"]),
                fmt(row["D_min_ci95"]),
                fmt(row["J_smooth_mean"]),
                fmt(row["J_smooth_std"]),
                fmt(row["elapsed_ms_mean"]),
                fmt(row["elapsed_ms_std"]),
            ]
        )
    if not rows:
        return "RRT multi-seed statistics have not been generated yet.\n"
    return markdown(["scenario", "n", "success rate", "accepted rate", "D_min mean", "D_min std", "D_min ci95", "J_smooth mean", "J_smooth std", "time mean/ms", "time std/ms"], rows)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Chapter 6.4 NUBS optimization summary.")
    parser.add_argument("--output", default="data/results/ch6_4")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    summary = build_summary()
    (output / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tables = {
        "table_6_4_basic_nubs.md": table_basic(summary),
        "table_6_4_time_optimization.md": table_time(summary),
        "table_6_4_static_risk.md": table_static(summary),
        "table_6_4_external_baselines.md": table_external(summary),
        "table_6_4_rrt_multiseed.md": table_rrt_multiseed(summary),
        "table_6_4_representation.md": table_representation(summary),
        "table_6_4_lambda_sweep.md": table_lambda(summary),
        "missing_external_baselines.md": table_missing(summary),
    }
    for name, text in tables.items():
        (output / name).write_text(text + "\n", encoding="utf-8")
    print(tables["table_6_4_basic_nubs.md"])
    print()
    print(tables["table_6_4_static_risk.md"])
    print(f"\n[exp_64] saved summary to {output}")


if __name__ == "__main__":
    main()
