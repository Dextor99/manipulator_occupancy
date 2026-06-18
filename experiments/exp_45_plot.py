"""Plot Chapter 4.5 trial curves."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from experiments.exp_45_eval import load_trial


def plot_fig45(trial_path: str | Path, output: str | Path) -> None:
    import matplotlib.pyplot as plt

    trial = load_trial(trial_path)
    frames = trial.get("frames", [])
    if not frames:
        raise RuntimeError("trial has no frames")
    t0 = frames[0]["timestamp"]
    ts = np.array([f["timestamp"] - t0 for f in frames], dtype=float)
    d_ref = np.array([f["d_ref"] for f in frames], dtype=float)
    speed = np.array([f["speed_scale"] for f in frames], dtype=float)
    rep_norm = np.array([np.linalg.norm(f["rep_velocity"]) for f in frames], dtype=float)
    cmd_norm = np.array([np.linalg.norm(f["cmd_velocity"]) for f in frames], dtype=float)
    params = trial.get("parameters", {})
    d_safe = float(params.get("d_safe", 0.15))
    d_stop = float(params.get("d_stop", 0.05))

    fig, axes = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True)
    axes[0].plot(ts, d_ref, label="D_ref")
    axes[0].axhline(d_safe, color="tab:orange", linestyle="--", label="d_safe")
    axes[0].axhline(d_stop, color="tab:red", linestyle="--", label="d_stop")
    axes[0].set_ylabel("distance (m)")
    axes[0].legend(loc="best")
    axes[1].plot(ts, speed, label="speed_scale")
    axes[1].plot(ts, rep_norm, label="||qdot_rep||")
    axes[1].plot(ts, cmd_norm, label="||qdot_cmd||")
    axes[1].set_xlabel("time (s)")
    axes[1].legend(loc="best")
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    print(f"[exp_45_plot] saved {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Chapter 4.5 trial curves.")
    parser.add_argument("--trial", required=True)
    parser.add_argument("--output", default="data/results/ch4_5/fig45.png")
    args = parser.parse_args()
    plot_fig45(args.trial, args.output)


if __name__ == "__main__":
    main()
