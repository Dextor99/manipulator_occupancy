"""Summarize currently reusable E2 static-planning benchmark results."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


SCENARIO_LABELS = {"A": "P1", "B": "P2", "C": "P3"}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if math.isinf(value):
            return "inf"
        return f"{value:.6g}"
    return str(value)


def row_from_stage2(scenario: str, row: dict[str, Any]) -> list[str]:
    ver = row["verification"]
    opt = row.get("optimization", {})
    return [
        SCENARIO_LABELS.get(scenario, scenario),
        scenario,
        "Ours CCRO-NUBS",
        str(row["solver_success"]),
        str(ver["accepted"]),
        fmt(ver["min_distance"]),
        fmt(row["full_body_risk_cost"]),
        fmt(opt.get("final_energy")),
        fmt(ver.get("goal_error")),
        fmt(opt.get("elapsed_ms", 0.0)),
        row.get("nearest_link") or "-",
    ]


def row_from_external(scenario: str, method_label: str, row: dict[str, Any]) -> list[str]:
    ver = row["verification"]
    opt = row.get("optimization", {})
    accepted = ver.get("accepted")
    if method_label == "RRT-Connect + smoothing":
        accepted = f"{row['statistics']['accepted_rate']:.4f}"
    return [
        SCENARIO_LABELS.get(scenario, scenario),
        scenario,
        method_label,
        str(row["solver_success"]),
        str(accepted),
        fmt(ver.get("min_distance")),
        fmt(row.get("full_body_risk_cost")),
        fmt(opt.get("final_energy")),
        fmt(ver.get("goal_error")),
        fmt(opt.get("elapsed_ms")),
        row.get("nearest_link") or "-",
    ]


def build_current_table(stage2: dict[str, Any], external: dict[str, Any]) -> str:
    headers = [
        "scene",
        "source_scene",
        "method",
        "success",
        "accepted",
        "D_min",
        "J_risk",
        "J_smooth",
        "goal_error",
        "T_plan_ms",
        "nearest_link",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for scenario in ("A", "B", "C"):
        ext_methods = external["scenarios"][scenario]["methods"]
        lines.append("| " + " | ".join(row_from_external(scenario, "MINCO-risk", ext_methods["minco_risk"])) + " |")
        lines.append(
            "| "
            + " | ".join(
                row_from_external(
                    scenario,
                    "RRT-Connect + smoothing",
                    ext_methods["rrt_connect_smooth"],
                )
            )
            + " |"
        )
        lines.append(
            "| "
            + " | ".join(
                row_from_stage2(
                    scenario,
                    stage2["scenarios"][scenario]["methods"]["full_body"],
                )
            )
            + " |"
        )
    return "\n".join(lines)


def build_reuse_decision_table() -> str:
    rows = [
        ["CCRO-NUBS full-body", "use", "Our method result; dense verifier accepted in A/B/C."],
        ["NUBS-base", "ablation", "Internal no-risk variant; not an external benchmark."],
        ["NUBS-EEF-risk", "ablation", "Internal end-effector-only risk variant."],
        ["MINCO-risk", "use", "Continuous polynomial trajectory optimization baseline."],
        ["MINCO-base", "auxiliary", "No risk term; useful only as lower baseline."],
        ["RRT-Connect + smoothing", "use", "Sampling baseline; already has 30 seeds per scenario."],
        ["CHOMP / TrajOpt", "missing", "Add at least one optimization-style baseline."],
        ["GPMP2", "missing", "Add GPMP2-style continuous-time optimizer or document omission."],
    ]
    return "\n".join(
        [
            "| material | decision | reason |",
            "|---|---|---|",
            *["| " + " | ".join(row) + " |" for row in rows],
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/results/ch6_e1_e5/E2_static_planning_benchmark")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    final = root / "final_current"
    final.mkdir(parents=True, exist_ok=True)
    stage2 = load_json(root / "reuse" / "ccro_stage2" / "metrics.json")
    external = load_json(root / "reuse" / "ch6_4_external" / "metrics.json")
    (final / "table_E2_current_reusable_benchmark.md").write_text(
        build_current_table(stage2, external) + "\n", encoding="utf-8"
    )
    (final / "table_E2_reuse_decision.md").write_text(
        build_reuse_decision_table() + "\n", encoding="utf-8"
    )
    note = "\n".join(
        [
            "# E2 Current Benchmark Note",
            "",
            "This is a reduced E2 benchmark generated from existing reusable results.",
            "It includes MINCO-risk, RRT-Connect + smoothing, and Ours CCRO-NUBS.",
            "It does not yet include CHOMP, TrajOpt, or GPMP2-style optimizers required by the expanded E2 plan.",
            "",
        ]
    )
    (final / "E2_current_note.md").write_text(note, encoding="utf-8")
    print(f"[E2] current reusable summary saved to {final}")


if __name__ == "__main__":
    main()
