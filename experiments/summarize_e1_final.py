"""Build thesis-ready E1 tables and figures from completed E1 outputs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


OCC_METHODS = [
    ("current_clustering", "Current-frame occupancy"),
    ("voxel_occupancy", "Voxel occupancy"),
    ("octomap_decay", "OctoMap-like occupancy"),
    ("ours_stro", "Ours-STRO"),
]
APF_METHODS = [
    ("critical_point_apf", "Critical-point APF"),
    ("ours_ccro", "Ours-CCRO Mesh"),
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if math.isinf(value):
            return "inf"
        return f"{value:.4f}"
    return str(value)


def table_occupancy(metrics: dict[str, Any]) -> str:
    headers = ["scene", "method", "R_det", "T_lead", "R_false_time", "R_keep", "R_over", "T_ms"]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for scene in ("static_safe", "approach", "crossing", "leave"):
        data = metrics["scenes"].get(scene, {})
        for method, label in OCC_METHODS:
            vals = data.get(method, {})
            lines.append(
                "| "
                + " | ".join(
                    [
                        scene,
                        label,
                        fmt(vals.get("R_det")),
                        fmt(vals.get("T_lead")),
                        fmt(vals.get("R_false_time")),
                        fmt(vals.get("R_keep")),
                        fmt(vals.get("R_over")),
                        fmt(vals.get("T_process_ms_mean")),
                    ]
                )
                + " |"
            )
    return "\n".join(lines)


def table_whole_body(metrics: dict[str, Any]) -> str:
    headers = [
        "scene",
        "method",
        "trials",
        "body-only",
        "future-only",
        "R_detect",
        "R_body",
        "R_future",
        "D_mean",
        "D_min",
        "T_ms",
    ]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for scene in ("ee_near", "body_near", "dynamic_future"):
        data = metrics["scenes"].get(scene, {})
        for method, label in APF_METHODS:
            vals = data.get("methods", {}).get(method, {})
            lines.append(
                "| "
                + " | ".join(
                    [
                        scene,
                        label,
                        str(data.get("trials", "-")),
                        str(data.get("body_only_events", "-")),
                        str(data.get("future_only_events", "-")),
                        fmt(vals.get("R_detect")),
                        fmt(vals.get("R_body")),
                        fmt(vals.get("R_future")),
                        fmt(vals.get("D_mean")),
                        fmt(vals.get("D_min")),
                        fmt(vals.get("T_ms_mean")),
                    ]
                )
                + " |"
            )
    return "\n".join(lines)


def final_notes(occupancy: dict[str, Any], apf: dict[str, Any]) -> str:
    approach = occupancy["scenes"]["approach"]
    crossing = occupancy["scenes"]["crossing"]
    dynamic = apf["scenes"]["dynamic_future"]["methods"]
    return "\n".join(
        [
            "# E1 Final Summary",
            "",
            "## Key Findings",
            "",
            f"- In `approach`, Ours-STRO reaches `T_lead={fmt(approach['ours_stro']['T_lead'])}s`, while current-frame occupancy reaches `T_lead={fmt(approach['current_clustering']['T_lead'])}s`.",
            f"- In `crossing`, Ours-STRO reaches `T_lead={fmt(crossing['ours_stro']['T_lead'])}s`, while current-frame occupancy reaches `T_lead={fmt(crossing['current_clustering']['T_lead'])}s`.",
            f"- The conservative prediction trade-off is visible in `R_false_time`: Ours-STRO is higher than current-frame methods in dynamic scenes.",
            f"- In `dynamic_future`, Critical-point APF detects future-only risk at `R_future={fmt(dynamic['critical_point_apf']['R_future'])}`, while Ours-CCRO Mesh reaches `R_future={fmt(dynamic['ours_ccro']['R_future'])}`.",
            "",
            "## Recommended Thesis Use",
            "",
            "- Use `table_E1_occupancy_final.md` as the main risk/occupancy table.",
            "- Use `table_E1_whole_body_apf_final.md` as the main whole-body APF comparison table.",
            "- Use `fig_E1_dynamic_warning.png` and `fig_E1_whole_body_apf.png` as the two core E1 figures.",
            "- Keep the earlier EEF-only and Body-current table as auxiliary evidence or E5 ablation.",
            "",
        ]
    )


def plot_dynamic_warning(root: Path, output: Path) -> None:
    import matplotlib.pyplot as plt

    method_labels = {
        "current_clustering": "Current",
        "voxel_occupancy": "Voxel",
        "octomap_decay": "OctoMap-like",
        "ours_stro": "Ours-STRO",
    }
    scenes = [("approach", "Dynamic approach"), ("crossing", "Dynamic crossing")]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), sharey=True)
    for ax, (scene, title) in zip(axes, scenes):
        trial = load_json(root / "occupancy_sim" / f"trial_{scene}_00.json")
        ref = trial["reference"]
        ts = np.asarray([row["timestamp"] for row in ref], dtype=float)
        d_true = np.asarray([row["d_true"] for row in ref], dtype=float)
        d_true = np.where(np.isfinite(d_true), d_true, np.nan)
        ax.plot(ts, d_true, color="black", linewidth=2.0, label="true D_min")
        for method, label in method_labels.items():
            states = [row["state"] for row in trial["series"][method]]
            first = next((i for i, state in enumerate(states) if state != "SAFE"), None)
            if first is not None:
                ax.axvline(ts[first], linestyle="--", alpha=0.75, label=f"{label} trigger")
        ax.axhline(trial["parameters"]["danger_threshold"], color="tab:red", linestyle=":", label="danger")
        ax.set_title(title)
        ax.set_xlabel("time (s)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("distance (m)")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
    fig.tight_layout(rect=[0, 0.16, 1, 1])
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=240)
    plt.close(fig)


def plot_whole_body_apf(metrics: dict[str, Any], output: Path) -> None:
    import matplotlib.pyplot as plt

    scenes = ["ee_near", "body_near", "dynamic_future"]
    x = np.arange(len(scenes))
    width = 0.34
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    for offset, method, label in [
        (-width / 2, "critical_point_apf", "Critical-point APF"),
        (width / 2, "ours_ccro", "Ours-CCRO Mesh"),
    ]:
        detect = [metrics["scenes"][scene]["methods"][method]["R_detect"] for scene in scenes]
        future = [
            metrics["scenes"][scene]["methods"][method]["R_future"]
            if metrics["scenes"][scene]["methods"][method]["R_future"] is not None
            else np.nan
            for scene in scenes
        ]
        axes[0].bar(x + offset, detect, width, label=label)
        axes[1].bar(x + offset, future, width, label=label)
    axes[0].set_title("Risk detection")
    axes[1].set_title("Future-only detection")
    for ax in axes:
        ax.set_xticks(x, scenes, rotation=18)
        ax.set_ylim(0.0, 1.05)
        ax.set_ylabel("rate")
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=240)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/results/ch6_e1_e5/E1_occupancy_risk_final")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    final_dir = root / "final"
    figures = final_dir / "figures"
    final_dir.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    occupancy = load_json(root / "occupancy_sim" / "metrics.json")
    apf = load_json(root / "critical_point_apf" / "metrics.json")
    (final_dir / "table_E1_occupancy_final.md").write_text(
        table_occupancy(occupancy) + "\n", encoding="utf-8"
    )
    (final_dir / "table_E1_whole_body_apf_final.md").write_text(
        table_whole_body(apf) + "\n", encoding="utf-8"
    )
    (final_dir / "E1_final_summary.md").write_text(
        final_notes(occupancy, apf), encoding="utf-8"
    )
    plot_dynamic_warning(root, figures / "fig_E1_dynamic_warning.png")
    plot_whole_body_apf(apf, figures / "fig_E1_whole_body_apf.png")
    print(f"[E1] final tables and figures saved to {final_dir}")


if __name__ == "__main__":
    main()
