#!/usr/bin/env python3
"""Prepare figures and a guarded playback package for a 6.5.3 dynamic candidate.

The input trial is a ``moving-shadow-stop`` run.  Its Fast CCRO-NUBS candidate
was generated after the robot was stopped and validated in shadow mode.  This
script does not connect to hardware.  It creates:

* ``candidate_preview_package/summary.json``
* ``candidate_preview_package/ccro_nubs_candidate_trajectory.csv``
* joint, TCP, and top-view preview figures

The package is intended for offline inspection and optional low-speed motion
shape playback.  It is not evidence of online dynamic trajectory switching.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
EXP652 = ROOT / "experiments" / "new" / "6_5" / "6_5_2"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(EXP652) not in sys.path:
    sys.path.insert(0, str(EXP652))

from experiments.exp_ccro_stage2 import _load  # noqa: E402
from run_652_static_avoidance import make_surface_model  # noqa: E402


DEFAULT_TRIAL = (
    ROOT
    / "results"
    / "new"
    / "6_5"
    / "6_5_3"
    / "dynamic_repair_pilot"
    / "trials"
    / "D1_crossing_body_r14"
)


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")


def load_candidate_csv(path: Path) -> tuple[np.ndarray, np.ndarray, list[dict[str, str]]]:
    times: list[float] = []
    qs: list[list[float]] = []
    rows: list[dict[str, str]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(dict(row))
            times.append(float(row["t_s"]))
            qs.append([float(row[f"q{i}_rad"]) for i in range(1, 7)])
    if len(qs) < 2:
        raise RuntimeError(f"candidate has too few rows: {path}")
    return np.asarray(times, dtype=np.float64), np.asarray(qs, dtype=np.float64), rows


def joint_dict(surface_model: Any, q: np.ndarray) -> dict[str, float]:
    return {name: float(q[i]) for i, name in enumerate(surface_model.joint_names)}


def tcp_path(surface_model: Any, qs: np.ndarray, tcp_link: str) -> np.ndarray:
    out = []
    for q in qs:
        fk = surface_model.urdf.link_transforms(joint_dict(surface_model, q))
        out.append(np.asarray(fk[tcp_link][:3, 3], dtype=np.float64))
    return np.vstack(out)


def sphere_points(center: np.ndarray, radius: float, n: int = 320) -> np.ndarray:
    rng = np.random.default_rng(20260810)
    phi = rng.uniform(0.0, 2.0 * math.pi, n)
    cos_theta = rng.uniform(-1.0, 1.0, n)
    sin_theta = np.sqrt(1.0 - cos_theta**2)
    unit = np.c_[sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta]
    return center[None, :] + radius * unit


def plot_joint_curves(path: Path, times: np.ndarray, qs: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 2, figsize=(9.2, 6.8), sharex=True, dpi=180)
    for j, ax in enumerate(axes.ravel()):
        ax.plot(times, np.rad2deg(qs[:, j]), linewidth=1.8)
        ax.set_ylabel(f"q{j+1} / deg")
        ax.grid(True, alpha=0.25)
    axes[-1, 0].set_xlabel("t / s")
    axes[-1, 1].set_xlabel("t / s")
    fig.suptitle("6.5.3 accepted Fast CCRO-NUBS local candidate")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def plot_tcp_xyz(path: Path, times: np.ndarray, tcp: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(8.6, 6.4), sharex=True, dpi=180)
    labels = ["X", "Y", "Z"]
    for i, ax in enumerate(axes):
        ax.plot(times, tcp[:, i], linewidth=1.8)
        ax.set_ylabel(f"{labels[i]} / m")
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("t / s")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def plot_top_view(path: Path, tcp: np.ndarray, obstacle_center: np.ndarray, obstacle_radius: float, q_now_tcp: np.ndarray | None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    theta = np.linspace(0.0, 2.0 * math.pi, 240)
    circle = np.c_[
        obstacle_center[0] + obstacle_radius * np.cos(theta),
        obstacle_center[1] + obstacle_radius * np.sin(theta),
    ]
    fig, ax = plt.subplots(figsize=(7.4, 6.6), dpi=200)
    ax.plot(tcp[:, 0], tcp[:, 1], color="#1f77b4", linewidth=2.4, label="accepted local candidate TCP")
    ax.scatter(tcp[0, 0], tcp[0, 1], c="#2ca02c", s=52, label="candidate start")
    ax.scatter(tcp[-1, 0], tcp[-1, 1], c="#9467bd", s=52, label="candidate end")
    if q_now_tcp is not None:
        ax.scatter(q_now_tcp[0], q_now_tcp[1], c="#111111", marker="x", s=70, label="trigger TCP")
    ax.fill(circle[:, 0], circle[:, 1], color="#d62728", alpha=0.16, label="predicted obstacle sphere")
    ax.plot(circle[:, 0], circle[:, 1], color="#d62728", linewidth=1.4)
    ax.scatter(obstacle_center[0], obstacle_center[1], c="#d62728", s=45, label="obstacle center")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("X / m")
    ax.set_ylabel("Y / m")
    ax.set_title("Top view: 6.5.3 accepted dynamic-repair candidate")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def run(args: argparse.Namespace) -> dict[str, Any]:
    trial_dir = args.trial_dir.resolve()
    candidate_dir = trial_dir / "candidate"
    candidate_summary_path = candidate_dir / "candidate_summary.json"
    candidate_csv = candidate_dir / "fast_ccro_nubs_candidate.csv"
    trial_summary = json.loads((trial_dir / "summary.json").read_text(encoding="utf-8"))
    candidate_summary = json.loads(candidate_summary_path.read_text(encoding="utf-8"))
    accepted_for_switch = bool(candidate_summary.get("accepted_for_switch", False))
    accepted_steps = int(candidate_summary.get("accepted_steps", 0))
    repair_step_ok = bool(candidate_summary.get("repair_step_ok", accepted_steps > 0))

    output_dir = (args.output_dir or (trial_dir / "candidate_preview_package")).resolve()
    figures = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(candidate_csv, output_dir / "ccro_nubs_candidate_trajectory.csv")

    times, qs, _ = load_candidate_csv(candidate_csv)
    config = _load(args.config)
    surface_model = make_surface_model(config)
    tcp = tcp_path(surface_model, qs, args.tcp_link)
    q_now_tcp = tcp_path(surface_model, np.asarray([candidate_summary["q_now"]], dtype=np.float64), args.tcp_link)[0]
    obstacle_center = np.asarray(candidate_summary["obstacle_center"], dtype=np.float64)
    obstacle_radius = float(candidate_summary["obstacle_radius"])
    obstacle_points = sphere_points(obstacle_center, obstacle_radius)

    plot_joint_curves(figures / "joint_trajectory_preview.png", times, qs)
    plot_tcp_xyz(figures / "tcp_xyz_preview.png", times, tcp)
    plot_top_view(figures / "top_view_dynamic_candidate.png", tcp, obstacle_center, obstacle_radius, q_now_tcp)
    np.savez_compressed(
        output_dir / "preview_geometry.npz",
        times=times,
        qs=qs,
        tcp=tcp,
        obstacle_center=obstacle_center,
        obstacle_radius=obstacle_radius,
        obstacle_points=obstacle_points,
    )

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_trial_dir": str(trial_dir),
        "source_candidate_summary": str(candidate_summary_path),
        "status": "DYNAMIC_SHADOW_CANDIDATE_PREVIEW_PACKAGE",
        "accepted_for_real_execution": bool(accepted_for_switch and repair_step_ok),
        "accepted_scope_note": (
            "Accepted only for guarded low-speed preview of this validated local candidate. "
            "This does not prove online dynamic trajectory switching. If accepted_steps=0, "
            "the trajectory is only the local reference continuation and must not be "
            "reported as a repaired avoidance candidate."
        ),
        "accepted_for_switch_from_source": accepted_for_switch,
        "repair_step_ok": repair_step_ok,
        "accepted_steps": accepted_steps,
        "candidate_is_reference_continuation": not repair_step_ok,
        "trajectory_type": "fast_local_CCRO_NUBS_joint_space_candidate",
        "q_start_rad": qs[0].tolist(),
        "q_goal_rad": qs[-1].tolist(),
        "candidate": {
            "dense_verification": {
                "accepted": True,
                "min_distance": candidate_summary.get("verification_min_distance_m"),
                "checks": candidate_summary.get("verification_checks", {}),
                "reasons": candidate_summary.get("verification_reasons", []),
            },
            "fast_elapsed_ms": candidate_summary.get("fast_elapsed_ms"),
            "fast_budget_ms": candidate_summary.get("fast_budget_ms"),
            "obstacle_center": obstacle_center.tolist(),
            "obstacle_radius": obstacle_radius,
            "risk_links": candidate_summary.get("risk_links", []),
        },
        "dynamic_trial": {
            "status": trial_summary.get("status"),
            "trigger_frame": trial_summary.get("trigger_frame"),
            "current_min_distance_m": trial_summary.get("current_min_distance_m"),
            "predicted_min_distance_m": trial_summary.get("predicted_min_distance_m"),
            "guard_min_distance_m": trial_summary.get("guard_min_distance_m"),
        },
        "tcp_stats": {
            "tcp_link": args.tcp_link,
            "start": tcp[0].tolist(),
            "goal": tcp[-1].tolist(),
            "path_length_m": float(np.sum(np.linalg.norm(np.diff(tcp, axis=0), axis=1))),
            "z_min_m": float(np.min(tcp[:, 2])),
            "z_max_m": float(np.max(tcp[:, 2])),
            "z_range_m": float(np.max(tcp[:, 2]) - np.min(tcp[:, 2])),
        },
        "trajectory_csv_stats": {
            "waypoints": int(len(qs)),
            "duration_s": float(times[-1] - times[0]),
            "max_abs_joint_step_rad": float(np.max(np.abs(np.diff(qs, axis=0)))),
        },
        "files": [
            "summary.json",
            "ccro_nubs_candidate_trajectory.csv",
            "preview_geometry.npz",
            "figures/joint_trajectory_preview.png",
            "figures/tcp_xyz_preview.png",
            "figures/top_view_dynamic_candidate.png",
        ],
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps({"output_dir": str(output_dir), "status": summary["status"], "tcp_stats": summary["tcp_stats"]}, indent=2, ensure_ascii=False))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-dir", type=Path, default=DEFAULT_TRIAL)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "ccro_stage4.yaml")
    parser.add_argument("--tcp-link", default="gripper_base_link")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
