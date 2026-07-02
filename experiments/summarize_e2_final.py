"""Build final E2 tables and figures after adding classical optimizer baselines."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


SCENARIOS = ("A", "B", "C")
SCENE_LABELS = {"A": "P1", "B": "P2", "C": "P3"}
METHOD_ORDER = [
    ("rrt_connect_smooth", "RRT-Connect + smoothing", "external"),
    ("official_tesseract_trajopt", "Official TrajOpt/Tesseract", "official"),
    ("chomp_style", "CHOMP-style", "classical"),
    ("trajopt_style", "TrajOpt-style", "classical"),
    ("gpmp2_style", "GPMP2-style", "classical"),
    ("minco_risk", "MINCO-risk", "external"),
    ("full_body", "Ours CCRO-NUBS", "stage2"),
]


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


def row_for_method(
    scenario: str,
    method_key: str,
    label: str,
    source: str,
    stage2: dict[str, Any],
    external: dict[str, Any],
    classical: dict[str, Any],
    official: dict[str, Any] | None = None,
) -> list[str]:
    if source == "stage2":
        row = stage2["scenarios"][scenario]["methods"][method_key]
        ver = row["verification"]
        opt = row.get("optimization", {})
        accepted = ver["accepted"]
    elif source == "external":
        row = external["scenarios"][scenario]["methods"][method_key]
        ver = row["verification"]
        opt = row.get("optimization", {})
        accepted = ver.get("accepted")
        if method_key == "rrt_connect_smooth":
            accepted = f"{row['statistics']['accepted_rate']:.4f}"
    elif source == "official":
        if official is None:
            return [SCENE_LABELS[scenario], scenario, label, "-", "-", "-", "-", "-", "-", "-", "-"]
        row = official["scenarios"][scenario]["methods"][method_key]
        ver = row["verification"]
        opt = row.get("optimization", {})
        accepted = ver["accepted"]
    else:
        row = classical["scenarios"][scenario]["methods"][method_key]
        ver = row["verification"]
        opt = row.get("optimization", {})
        accepted = ver["accepted"]
    return [
        SCENE_LABELS[scenario],
        scenario,
        label,
        str(row["solver_success"]),
        str(accepted),
        fmt(ver.get("min_distance")),
        fmt(row.get("full_body_risk_cost")),
        fmt(opt.get("final_energy")),
        fmt(ver.get("goal_error")),
        fmt(opt.get("elapsed_ms")),
        row.get("nearest_link") or "-",
    ]


def build_table(
    stage2: dict[str, Any],
    external: dict[str, Any],
    classical: dict[str, Any],
    official: dict[str, Any] | None = None,
) -> str:
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
    for scenario in SCENARIOS:
        for method_key, label, source in METHOD_ORDER:
            lines.append(
                "| "
                + " | ".join(
                    row_for_method(
                        scenario, method_key, label, source, stage2, external, classical, official
                    )
                )
                + " |"
            )
    return "\n".join(lines)


def summary_note(
    stage2: dict[str, Any],
    external: dict[str, Any],
    classical: dict[str, Any],
    official: dict[str, Any] | None = None,
) -> str:
    p2_ours = stage2["scenarios"]["B"]["methods"]["full_body"]["verification"]["min_distance"]
    p2_rrt = external["scenarios"]["B"]["methods"]["rrt_connect_smooth"]["verification"]["min_distance"]
    p2_chomp = classical["scenarios"]["B"]["methods"]["chomp_style"]["verification"]["min_distance"]
    return "\n".join(
        [
            "# E2 Final Summary",
            "",
            "## Completed Baselines",
            "",
            "- RRT-Connect + smoothing",
            "- Official TrajOpt/Tesseract",
            "- CHOMP-style",
            "- TrajOpt-style",
            "- GPMP2-style",
            "- MINCO-risk",
            "- Ours CCRO-NUBS",
            "",
            "## Notes",
            "",
            "- Official TrajOpt/Tesseract is bound through the `tesseract-robotics` PyPI package and evaluated by the shared dense verifier.",
            "- CHOMP/TrajOpt/GPMP2 are lightweight style reproductions; keep their `*-style` labels.",
            "- The official TrajOpt run uses Tesseract joint-space optimization; E2 point-cloud risk is not injected into Tesseract and is applied only by the common verifier.",
            "- All methods are evaluated with the same dense `TrajectoryVerifier`.",
            "- Supplemental perturbation statistics use 10 perturbed obstacle point clouds for each P1/P2/P3 scene and coarse risk evaluation at 0.1 s; they support robustness analysis but do not replace the dense final acceptance table.",
            "- Representative P2 time-series visualization is saved as `final/figures/fig_E2_P2_Dmin_curve.png`.",
            f"- In P2, Ours CCRO-NUBS reaches `D_min={fmt(p2_ours)}`, RRT reaches `D_min={fmt(p2_rrt)}`, and CHOMP-style reaches `D_min={fmt(p2_chomp)}`.",
            "- Internal `NUBS-base` and `NUBS-EEF-risk` should remain in ablation rather than the main external table.",
            "",
        ]
    )


def plot_bar(
    stage2: dict[str, Any],
    external: dict[str, Any],
    classical: dict[str, Any],
    official: dict[str, Any] | None,
    output: Path,
) -> None:
    import matplotlib.pyplot as plt

    labels = [label for _, label, _ in METHOD_ORDER]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for ax, scenario in zip(axes, SCENARIOS):
        dmins = []
        for method_key, label, source in METHOD_ORDER:
            row = row_for_method(
                scenario, method_key, label, source, stage2, external, classical, official
            )
            try:
                dmins.append(float(row[5]))
            except ValueError:
                dmins.append(0.0)
        ax.bar(x, dmins)
        ax.axhline(0.035, color="tab:red", linestyle="--", linewidth=1.2, label="d_stop")
        ax.set_title(f"{SCENE_LABELS[scenario]} / {scenario}")
        ax.set_xticks(x, labels, rotation=35, ha="right")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("D_min (m)")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=240)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/results/ch6_e1_e5/E2_static_planning_benchmark")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    final = root / "final"
    figures = final / "figures"
    final.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    stage2 = load_json(root / "reuse" / "ccro_stage2" / "metrics.json")
    external = load_json(root / "reuse" / "ch6_4_external" / "metrics.json")
    classical = load_json(root / "classical_optimizers" / "metrics.json")
    official_path = root / "official_tesseract_trajopt" / "metrics.json"
    official = load_json(official_path) if official_path.exists() else None
    (final / "table_E2_static_planning_final.md").write_text(
        build_table(stage2, external, classical, official) + "\n", encoding="utf-8"
    )
    (final / "E2_final_summary.md").write_text(
        summary_note(stage2, external, classical, official), encoding="utf-8"
    )
    plot_bar(stage2, external, classical, official, figures / "fig_E2_Dmin_methods.png")
    print(f"[E2] final tables and figures saved to {final}")


if __name__ == "__main__":
    main()
