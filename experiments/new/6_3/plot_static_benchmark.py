"""Generate revised Chapter 6.3 Figure 5 from static benchmark outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def select_representative_instance(metrics: dict[str, Any]) -> str:
    for row in metrics.get("scenarios", {}).get("B", {}).get("instances", []):
        if row.get("all_main_accepted"):
            return row["id"]
    for scenario in ["B", "C", "A"]:
        instances = metrics.get("scenarios", {}).get(scenario, {}).get("instances", [])
        if instances:
            return instances[0]["id"]
    raise ValueError("no instances available for figure selection")


def jerk_norm_curve(samples: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(samples["times"], dtype=float)
    jerk = np.asarray(samples["jerk"], dtype=float)
    return times, np.linalg.norm(jerk, axis=1)


def plot(input_dir: str | Path, output_dir: str | Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    metrics = json.loads((input_path / "metrics.json").read_text(encoding="utf-8"))
    instance_id = select_representative_instance(metrics)
    scenario, _ = instance_id.split("_", 1)
    trial = json.loads((input_path / "trials" / f"{instance_id}.json").read_text(encoding="utf-8"))
    frozen = json.loads((input_path / "frozen_instances" / f"{instance_id}.json").read_text(encoding="utf-8"))
    obstacle = np.asarray(frozen["gt_dense_points"], dtype=float)

    fig = plt.figure(figsize=(11.5, 4.5))
    ax0 = fig.add_subplot(1, 2, 1, projection="3d")
    if len(obstacle):
        keep = np.linspace(0, len(obstacle) - 1, min(len(obstacle), 1200)).round().astype(int)
        shown = obstacle[keep]
        ax0.scatter(shown[:, 0], shown[:, 1], shown[:, 2], s=2, c="black", alpha=0.35, label="GT obstacle")
    colors = {
        "critical_point_nubs": "tab:orange",
        "ccro_nubs": "tab:blue",
    }
    for method, color in colors.items():
        samples = trial.get(method, {}).get("plot_samples")
        if not samples:
            continue
        q = np.asarray(samples["q"], dtype=float)
        ax0.plot(q[:, 0], q[:, 1], q[:, 2], color=color, linewidth=2.0, label=method)
        marker_idx = np.linspace(0, len(q) - 1, min(6, len(q))).round().astype(int)
        ax0.scatter(q[marker_idx, 0], q[marker_idx, 1], q[marker_idx, 2], color=color, s=12)
    ax0.set_title(f"{scenario} representative trajectory")
    ax0.set_xlabel("q1 / rad")
    ax0.set_ylabel("q2 / rad")
    ax0.set_zlabel("q3 / rad")
    ax0.legend(fontsize=8)

    ax1 = fig.add_subplot(1, 2, 2)
    for method in ["rrt_connect_smooth", "minco_risk", "critical_point_nubs", "ccro_nubs"]:
        samples = trial.get(method, {}).get("plot_samples")
        if not samples:
            continue
        times, values = jerk_norm_curve(samples)
        ax1.plot(times, values, label=method)
    ax1.set_xlabel("time / s")
    ax1.set_ylabel(r"$\|q^{(3)}(t)\|_2$")
    ax1.grid(True, alpha=0.25)
    ax1.legend(fontsize=8)
    fig.tight_layout()
    target = output_path / "figure_5_static_trajectory_and_jerk.png"
    fig.savefig(target, dpi=220)
    plt.close(fig)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="result/new/6_3")
    parser.add_argument("--output", default="result/new/6_3/paper")
    args = parser.parse_args()
    target = plot(args.input, args.output)
    print(f"[6_3_plot] saved {target}")


if __name__ == "__main__":
    main()
