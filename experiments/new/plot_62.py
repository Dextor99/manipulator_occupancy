"""Generate revised 6.2 diagnostic figures from frozen outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from . import config_62 as cfg
from .body_coverage_62 import build_critical_points
from .common_62 import ensure_output_tree, load_surface_model, read_json


PRETTY_METHOD = {
    "current_frame": "Current-frame",
    "octomap_like": "OctoMap-like",
    "stro": "STRO (ours)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot revised 6.2 figures.")
    parser.add_argument("--output", default=str(cfg.DEFAULT_OUTPUT))
    return parser.parse_args()


def _select_dynamic_trial(output: Path, scene: str) -> tuple[dict, list[dict]] | None:
    summary_path = output / "dynamic" / "summary.json"
    if not summary_path.exists():
        return None
    summary = read_json(summary_path)
    candidates = [
        trial for trial in summary["trials"]
        if trial["scene"] == scene
        and trial["first_contact_time"] is None
        and trial["valid_min_clearance"]
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda item: abs(item["min_D_gt"] - 0.105))
    trial = candidates[0]
    rows = read_json(output / "dynamic" / "trials" / f"{trial['trial_id']}.json")
    return trial, rows


def plot_dynamic(output: Path, paper: Path) -> None:
    selected = [_select_dynamic_trial(output, scene) for scene in ("approach", "crossing")]
    selected = [item for item in selected if item is not None]
    if not selected:
        return
    fig, axes = plt.subplots(1, len(selected), figsize=(6.8 * len(selected), 4.0), squeeze=False)
    for panel_index, (ax, (trial, rows)) in enumerate(zip(axes[0], selected)):
        times = np.asarray([row["time"] for row in rows], dtype=float)
        d_gt = np.asarray([row["D_gt"] for row in rows], dtype=float)
        ax.plot(times, d_gt, color="black", linewidth=1.8, label="$D_{gt}$")
        ax.axhline(cfg.DYNAMIC_ALARM_DISTANCE, color="red", linestyle="--", linewidth=1.2, label="0.14 m")
        if trial["t_risk"] is not None:
            ax.axvline(trial["t_risk"], color="0.25", linewidth=1.1, alpha=0.9, label="$t_{risk}$")
        top_y = max(float(np.max(d_gt)), cfg.DYNAMIC_ALARM_DISTANCE) + 0.025
        for method, color, marker in (
            ("current_frame", "tab:blue", "o"),
            ("octomap_like", "tab:orange", "s"),
            ("stro", "tab:green", "^"),
        ):
            alarm = trial["first_alarm"][method]
            if alarm is not None:
                alarm_time = float(alarm)
                linestyle = ":" if method != "octomap_like" else "-."
                ax.axvline(alarm_time, color=color, linestyle=linestyle, linewidth=1.4, label=PRETTY_METHOD[method])
                ax.scatter([alarm_time], [top_y], color=color, marker=marker, s=28, zorder=4)
        stro_alarm = trial["first_alarm"]["stro"]
        if stro_alarm is not None and trial["t_risk"] is not None:
            y = cfg.DYNAMIC_ALARM_DISTANCE + 0.045
            ax.annotate(
                "",
                xy=(trial["t_risk"], y),
                xytext=(stro_alarm, y),
                arrowprops={"arrowstyle": "<->", "color": "tab:green", "linewidth": 1.5},
            )
            ax.text((trial["t_risk"] + stro_alarm) * 0.5, y + 0.008, "STRO lead", ha="center", color="tab:green")
        panel = "(a)" if panel_index == 0 else "(b)"
        title_scene = "Approach" if trial["scene"] == "approach" else "Crossing"
        ax.set_title(f"{panel} {title_scene}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Surface clearance (m)")
        ax.set_ylim(bottom=min(0.0, float(np.min(d_gt)) - 0.02))
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(paper / "figure_3.png", dpi=220)
    plt.close(fig)


def plot_body(output: Path, paper: Path) -> None:
    summary = output / "body" / "summary.json"
    if not summary.exists():
        return
    rows = read_json(summary)["samples"]
    misses = [row for row in rows if row["risk_gt"] and not row["risk_critical"] and row["risk_ccro"]]
    if not misses:
        misses = [row for row in rows if row["risk_gt"]]
    if not misses:
        return
    row = misses[0]
    surface = load_surface_model()
    q = np.asarray(row["q"], dtype=float)
    center = np.asarray(row["obstacle_center"], dtype=float)
    obstacle_radius = float(row["obstacle_radius"])
    region_links = set(cfg.BODY_REGIONS[row["region"]])
    link_points = surface.surface(q, density="medium", links=region_links)
    critical_points = build_critical_points(surface, q)
    nearest_ccro = np.asarray(row["nearest_surface_point_ccro"], dtype=float)
    nearest_cp = next((cp for cp in critical_points if cp.name == row["nearest_point_critical"]), None)

    fig, ax = plt.subplots(figsize=(6, 5))
    if len(link_points):
        ax.scatter(link_points[:, 0], link_points[:, 1], s=3, c="lightgray", label="target region mesh")
    labeled_cp = False
    labeled_radius = False
    for cp in critical_points:
        if cp.link not in region_links:
            continue
        ax.scatter(
            [cp.position[0]],
            [cp.position[1]],
            s=28,
            c="tab:blue",
            label=None if labeled_cp else "Critical point",
        )
        labeled_cp = True
        circle = plt.Circle(
            (cp.position[0], cp.position[1]),
            cp.radius,
            fill=False,
            color="tab:blue",
            alpha=0.35,
            label=None if labeled_radius else "Equivalent coverage radius",
        )
        ax.add_patch(circle)
        labeled_radius = True
    obstacle = plt.Circle((center[0], center[1]), obstacle_radius, color="tab:red", alpha=0.35, label="obstacle sphere")
    ax.add_patch(obstacle)
    ax.scatter([center[0]], [center[1]], c="tab:red", s=35)
    ax.scatter([nearest_ccro[0]], [nearest_ccro[1]], c="tab:green", s=45, label="CCRO nearest surface")
    ax.plot([center[0], nearest_ccro[0]], [center[1], nearest_ccro[1]], "k--", linewidth=1.0)
    if nearest_cp is not None:
        direction = center - nearest_cp.position
        norm = max(float(np.linalg.norm(direction)), 1.0e-12)
        unit = direction / norm
        cp_boundary = nearest_cp.position + unit * nearest_cp.radius
        obstacle_boundary = center - unit * obstacle_radius
        ax.plot(
            [cp_boundary[0], obstacle_boundary[0]],
            [cp_boundary[1], obstacle_boundary[1]],
            color="tab:blue",
            linestyle=":",
            linewidth=1.4,
            label="Critical-point distance",
        )
    ax.text(
        0.02,
        0.98,
        f"$D_{{gt}}$={row['D_gt']:.3f} m\n"
        f"$D_{{critical}}$={row['D_critical']:.3f} m\n"
        f"$D_{{CCRO}}$={row['D_ccro']:.3f} m",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.7", "alpha": 0.9},
    )
    region_label = row["region"].replace("_", "-")
    ax.set_title(f"Critical-point miss on the {region_label} surface")
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.axis("equal")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(paper / "figure_4.png", dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    paths = ensure_output_tree(output)
    plot_dynamic(output, paths["paper"])
    plot_body(output, paths["paper"])


if __name__ == "__main__":
    main()
