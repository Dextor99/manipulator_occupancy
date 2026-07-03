"""Summarize method weaknesses, runtime bottlenecks, and fast-mode evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


E1_MODE_DIRS = [
    ("coarse", "critical_point_apf_coarse_mesh"),
    ("medium", "critical_point_apf_medium_mesh"),
    ("dense", "critical_point_apf"),
]
E1_SCENES = ("ee_near", "body_near", "dynamic_future")
E2_SCENES = ("A", "B", "C")
E2_LABELS = {"A": "P1", "B": "P2", "C": "P3"}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: float | int | None, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def table(lines: list[list[str]]) -> str:
    headers, *rows = lines
    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    out.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(out)


def e1_density_table(e1_root: Path) -> str:
    rows = [["scene", "mesh_mode", "R_detect", "R_future", "T_ours_ms", "T_apf_ms", "speed_gap"]]
    for mode, rel in E1_MODE_DIRS:
        data = load_json(e1_root / rel / "metrics.json")
        for scene in E1_SCENES:
            ours = data["scenes"][scene]["methods"]["ours_ccro"]
            apf = data["scenes"][scene]["methods"]["critical_point_apf"]
            t_ours = float(ours["T_ms_mean"])
            t_apf = float(apf["T_ms_mean"])
            rows.append(
                [
                    scene,
                    mode,
                    fmt(ours["R_detect"]),
                    fmt(ours.get("R_future")),
                    fmt(t_ours),
                    fmt(t_apf),
                    f"{t_ours / max(t_apf, 1.0e-9):.2f}x",
                ]
            )
    return table(rows)


def e2_runtime_modes_table(e2_root: Path) -> str:
    main = load_json(e2_root / "reuse" / "ccro_stage2" / "metrics.json")
    fast = load_json(e2_root / "ours_fast_mode" / "metrics.json")
    rows = [
        [
            "scene",
            "mode",
            "accepted",
            "D_min",
            "J_smooth",
            "T_plan_ms",
            "speedup_vs_full",
        ]
    ]
    for scenario in E2_SCENES:
        full_row = main["scenarios"][scenario]["methods"]["full_body"]
        fast_row = fast["scenarios"][scenario]["methods"]["full_body"]
        full_time = float(full_row["optimization"]["elapsed_ms"])
        fast_time = float(fast_row["optimization"]["elapsed_ms"])
        for mode, row, speedup in [
            ("full-medium", full_row, 1.0),
            ("fast-coarse", fast_row, full_time / max(fast_time, 1.0e-9)),
        ]:
            rows.append(
                [
                    E2_LABELS[scenario],
                    mode,
                    str(row["verification"]["accepted"]),
                    fmt(row["verification"]["min_distance"], 5),
                    fmt(row["optimization"]["final_energy"], 5),
                    fmt(row["optimization"]["elapsed_ms"], 2),
                    f"{speedup:.2f}x",
                ]
            )
    return table(rows)


def e2_method_tradeoff_table(e2_root: Path) -> str:
    main = load_json(e2_root / "reuse" / "ccro_stage2" / "metrics.json")
    external = load_json(e2_root / "reuse" / "ch6_4_external" / "metrics.json")
    classical = load_json(e2_root / "classical_optimizers" / "metrics.json")
    rows = [["scene", "audit_item", "evidence", "recommended_framing"]]
    for scenario in E2_SCENES:
        ours = main["scenarios"][scenario]["methods"]["full_body"]
        minco = external["scenarios"][scenario]["methods"]["minco_risk"]
        chomp = classical["scenarios"][scenario]["methods"]["chomp_style"]
        d_values = {
            "ours": ours["verification"]["min_distance"],
            "minco": minco["verification"]["min_distance"],
            "chomp": chomp["verification"]["min_distance"],
        }
        best_name = max(d_values, key=d_values.get)
        rows.append(
            [
                E2_LABELS[scenario],
                "Ours is not the maximum-clearance method",
                f"best={best_name}, D_ours={fmt(d_values['ours'], 5)}, D_best={fmt(d_values[best_name], 5)}",
                "Report D_min as safety pass margin, and use J_smooth plus robustness as the main advantage.",
            ]
        )
        rows.append(
            [
                E2_LABELS[scenario],
                "Full mode is slower than lightweight baselines",
                f"T_ours={fmt(ours['optimization']['elapsed_ms'], 1)} ms, T_minco={fmt(minco['optimization']['elapsed_ms'], 1)} ms",
                "Separate low-frequency trajectory optimization from high-frequency risk monitoring.",
            ]
        )
    return table(rows)


def report_text(e1_root: Path, e2_root: Path) -> str:
    occupancy = load_json(e1_root / "occupancy_sim" / "metrics.json")
    crossing = occupancy["scenes"]["crossing"]
    approach = occupancy["scenes"]["approach"]
    return "\n".join(
        [
            "# Chapter 6 Weakness and Runtime Review",
            "",
            "## Main Weaknesses To State Honestly",
            "",
            "- E1: Ours-STRO has longer warning lead time, but also higher conservative false-trigger time.",
            f"  In approach, Ours false-time={fmt(approach['ours_stro']['R_false_time'])}, current-frame={fmt(approach['current_clustering']['R_false_time'])}.",
            f"  In crossing, Ours false-time={fmt(crossing['ours_stro']['R_false_time'])}, current-frame={fmt(crossing['current_clustering']['R_false_time'])}.",
            "- E1: Dense Ours-CCRO mesh risk is slower than Critical-point APF; the advantage is future/full-body detection, not raw per-frame speed.",
            "- E2: Full CCRO-NUBS is not always the highest-clearance or fastest planner.",
            "- E2: The correct advantage is the combination of dense safety acceptance, very low smoothness cost, full-body risk coupling, and stable perturbation results.",
            "",
            "## Runtime Framing",
            "",
            "- High-frequency layer: occupancy/STRO and coarse or medium mesh risk monitoring.",
            "- Low-frequency layer: CCRO-NUBS candidate generation when risk is triggered.",
            "- Acceptance layer: dense verifier for final safety gate. This can be slower because it is not the continuous control-rate loop.",
            "",
            "## Practical Improvement Already Verified",
            "",
            "- `config/ccro_stage2_fast.yaml` uses coarse optimizer mesh, fewer risk samples, fewer dynamic samples, and a looser convergence tolerance.",
            "- It keeps dense validation unchanged and passes P1/P2/P3.",
            "- It should be presented as `Ours-fast` or `fast candidate generation`, not as a replacement for the full high-fidelity result.",
            "",
            "## Thesis Wording",
            "",
            "Use this wording idea: the full method is a conservative high-fidelity planner, while the fast mode is the online candidate generator; both are filtered by the same dense safety gate.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e1-root", default="data/results/ch6_e1_e5/E1_occupancy_risk_final")
    parser.add_argument("--e2-root", default="data/results/ch6_e1_e5/E2_static_planning_benchmark")
    args = parser.parse_args()

    e1_root = Path(args.e1_root)
    e2_root = Path(args.e2_root)
    out = e2_root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    (out / "table_E1_mesh_density_runtime.md").write_text(
        e1_density_table(e1_root) + "\n", encoding="utf-8"
    )
    (out / "table_E2_ours_runtime_modes.md").write_text(
        e2_runtime_modes_table(e2_root) + "\n", encoding="utf-8"
    )
    (out / "table_E2_method_tradeoff_audit.md").write_text(
        e2_method_tradeoff_table(e2_root) + "\n", encoding="utf-8"
    )
    (out / "CH6_method_weakness_runtime_review.md").write_text(
        report_text(e1_root, e2_root), encoding="utf-8"
    )
    print(f"[CH6] weakness and runtime review saved to {out}")


if __name__ == "__main__":
    main()
