#!/usr/bin/env python3
"""Generate publication figures from archived 6.5.3 trials.

This is an offline-only post-processing tool.  It never connects to the robot
or camera and does not alter experiment code.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


def trial_dir(root: Path, repeat: str) -> Path:
    return root / repeat / "core_live" / "trials" / f"D2_opposing_approach_{repeat}"


def finite(a):
    a = np.asarray(a, dtype=float)
    return np.isfinite(a)


def distance_figure(root: Path, output: Path, cases=("r27", "r29")):
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.5), sharex=False)
    for ax, rep in zip(axes, cases):
        td = trial_dir(root, rep)
        rows = read_csv(td / "frames.csv")
        t = np.array([num(x.get("t_s")) for x in rows])
        cur = np.array([num(x.get("nearest_distance_m")) for x in rows])
        pred = np.array([num(x.get("predicted_distance_m")) for x in rows])
        guard = np.array([num(x.get("guard_distance_m")) for x in rows])
        speed = np.array([num(x.get("max_track_speed_m_s")) for x in rows])
        for y, label, color in ((cur, "current CCRO", "tab:blue"),
                                (pred, "predicted CCRO", "tab:orange"),
                                (guard, "raw point-cloud guard", "tab:green")):
            m = finite(t) & finite(y)
            if np.any(m):
                ax.plot(t[m], y[m], label=label, color=color, lw=1.6)
        ax.axhline(.14, ls="--", lw=1, color="tab:orange", label="STRO 0.14 m")
        ax.axhline(.10, ls="--", lw=1, color="tab:green", label="raw hard guard 0.10 m")
        ax.axhline(.09, ls="--", lw=1, color="tab:red", label="candidate 0.09 m")
        summary = json.loads((td / "summary.json").read_text())
        for ev in summary.get("events", []):
            if "t_s" in ev:
                x = float(ev["t_s"])
                ax.axvline(x, color="0.35", alpha=.45, lw=.8)
                if ev.get("type") in {"TRIGGER", "IMMEDIATE_STOP_BEFORE_LIVE_REPLAN"}:
                    ax.text(x, .98, ev.get("type", ""), rotation=90,
                            transform=ax.get_xaxis_transform(), va="top", fontsize=7)
        ax2 = ax.twinx()
        m = finite(t) & finite(speed)
        if np.any(m):
            ax2.plot(t[m], speed[m], color="tab:purple", alpha=.45, lw=1,
                     label="track speed")
        ax.set_title(f"{rep}: {summary.get('status', '')}", fontsize=10)
        ax.set_ylabel("clearance / m")
        ax2.set_ylabel("speed / m/s", color="tab:purple")
        ax.grid(alpha=.22)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=7, ncol=2, loc="best")
    axes[-1].set_xlabel("trial time / s")
    fig.suptitle("6.5.3 dynamic risk and safety evidence", y=.995)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def fast_figure(root: Path, output: Path, repeat="r27"):
    td = trial_dir(root, repeat) / "candidate"
    ref = read_csv(td / "fast_reference_risk_profile.csv")
    cand = read_csv(td / "fast_candidate_risk_profile.csv")
    active_path = td / "fast_active_distance_profile.csv"
    active = read_csv(active_path) if active_path.exists() else []
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for rows, label, color in ((ref, "reference", "tab:gray"),
                               (cand, "Fast candidate", "tab:red"),
                               (active, "active execution profile", "tab:blue")):
        if not rows or "tau" not in rows[0]:
            continue
        x = np.array([num(r.get("tau")) for r in rows])
        y = np.array([num(r.get("distance_m")) for r in rows])
        m = finite(x) & finite(y)
        if np.any(m):
            ax.plot(x[m], y[m], lw=1.8, label=label, color=color)
    ax.axhline(.09, color="tab:red", ls="--", lw=1, label="authorization 0.09 m")
    ax.set(xlabel="local horizon $\\tau$ / s", ylabel="clearance / m",
           title=f"Fast local repair risk profile ({repeat})")
    ax.grid(alpha=.25); ax.legend(fontsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(output, dpi=220); fig.savefig(output.with_suffix('.pdf'))
    plt.close(fig)


def pointcloud_figure(root: Path, output: Path, repeat="r27"):
    td = trial_dir(root, repeat)
    pts = read_csv(td / "stationary_confirmed_cluster_points.csv")
    keys = {k.lower(): k for k in pts[0]}
    def col(name):
        k = keys.get(name, name)
        return np.array([num(r.get(k)) for r in pts])
    x, y, z = col("x"), col("y"), col("z")
    geo = json.loads((td / "stationary_confirmed_multisphere.json").read_text())
    centers = np.array(geo["component_centers"], dtype=float)
    radii = np.array(geo["component_base_radii"], dtype=float)
    fig = plt.figure(figsize=(7.5, 6)); ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x, y, z, s=7, alpha=.55, label="stationary RGB-D cluster")
    u, v = np.mgrid[0:2*np.pi:24j, 0:np.pi:14j]
    for i, (c, r) in enumerate(zip(centers, radii), 1):
        xs = c[0] + r*np.cos(u)*np.sin(v)
        ys = c[1] + r*np.sin(u)*np.sin(v)
        zs = c[2] + r*np.cos(v)
        ax.plot_wireframe(xs, ys, zs, rstride=3, cstride=3, alpha=.24,
                          color="tab:red", label="multisphere" if i == 1 else None)
        ax.scatter(*c, color="black", s=22)
    ax.set(xlabel="X / m", ylabel="Y / m", zlabel="Z / m")
    ax.set_title(f"Stationary obstacle point cloud and two-sphere model ({repeat})")
    ax.legend(fontsize=8); fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220); fig.savefig(output.with_suffix('.pdf'))
    plt.close(fig)


def tracking_figure(root: Path, output: Path, repeat="r27"):
    td = trial_dir(root, repeat)
    e = json.loads((td / "event_replan_summary.json").read_text())
    terminal = e.get("terminal_execution") or {}
    samples = terminal.get("feedback_samples") or []
    if not samples:
        return False
    t = np.array([num(s.get("t_s")) for s in samples])
    actual = np.array([s.get("actual_joint_rad", [math.nan]*6) for s in samples], dtype=float)
    goal = np.array((terminal.get("candidate_start_joint_rad") or [math.nan]*6), dtype=float)
    # terminal goal is stored in the trajectory CSV; use its final joint vector.
    traj = td / "stationary_fast_terminal_bypass" / "authorized_terminal_goal.csv"
    if traj.exists():
        tr = read_csv(traj); goal = np.array([num(tr[-1].get(f"q{i}_rad")) for i in range(1,7)])
    err = np.abs(actual - goal[None, :])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i in range(6): ax.plot(t, err[:, i], lw=1.2, label=f"q{i+1}")
    ax.set(xlabel="terminal execution time / s", ylabel="absolute joint error / rad",
           title=f"Terminal tracking error ({repeat})")
    ax.grid(alpha=.25); ax.legend(ncol=3, fontsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(output, dpi=220); fig.savefig(output.with_suffix('.pdf'))
    plt.close(fig); return True


def make_table(root: Path, output: Path, repeats=("r27", "r28", "r29", "r30")):
    rows = []
    for rep in repeats:
        td = trial_dir(root, rep)
        if not (td / "summary.json").exists(): continue
        s = json.loads((td / "summary.json").read_text())
        outer_path = root / rep / "summary.json"
        outer = json.loads(outer_path.read_text()) if outer_path.exists() else {}
        epath = td / "event_replan_summary.json"
        e = json.loads(epath.read_text()) if epath.exists() else {}
        bypass = e.get("stationary_fast_terminal_bypass") or {}
        status = outer.get("status", s.get("status"))
        commanded = outer.get("robot_commanded", s.get("robot_commanded"))
        rows.append({"case": rep, "status": status,
                     "robot_commanded": commanded,
                     "terminal_mode": e.get("terminal_planner_mode"),
                     "terminal_authorized": bypass.get("authorized"),
                     "terminal_min_clearance_m": bypass.get("verification_min_distance_m"),
                     "terminal_elapsed_ms": bypass.get("stationary_terminal_total_elapsed_ms"),
                     "goal_reached": status == "SIMPLE_DYNAMIC_NUBS_RECOVERED_AND_GOAL_REACHED"})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["case"])
        w.writeheader(); w.writerows(rows)
    (output.with_suffix(".json")).write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("results/new/6_5/6_5_3/d2_approach_hold_complete_live"))
    p.add_argument("--output", type=Path, default=Path("results/new/6_5/6_5_3/paper_figures"))
    args = p.parse_args(); out = args.output
    distance_figure(args.root, out / "fig_distance_cases.png")
    fast_figure(args.root, out / "fig_fast_local_profile.png")
    pointcloud_figure(args.root, out / "fig_stationary_pointcloud_multisphere.png")
    tracking_figure(args.root, out / "fig_terminal_tracking_error.png")
    make_table(args.root, out / "table_653_results.csv")
    print(out)


if __name__ == "__main__":
    main()
