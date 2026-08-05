#!/usr/bin/env python3
"""Plot revised 6.5.1 perception capture trials."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


RISK_LEVEL = {"SAFE": 0, "WARNING": 1, "SLOW": 2, "STOP": 3}


def _float(value: str) -> float:
    if value == "":
        return np.nan
    try:
        return float(value)
    except Exception:
        return np.nan


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def plot_trial(trial_dir: Path) -> Path | None:
    frames = trial_dir / "frames.csv"
    if not frames.exists():
        return None
    rows = load_rows(frames)
    if not rows:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t0 = _float(rows[0]["timestamp"])
    t = np.asarray([_float(row["timestamp"]) - t0 for row in rows], dtype=float)
    d_cur = np.asarray([_float(row["nearest_distance_m"]) for row in rows], dtype=float)
    d_pred = np.asarray([_float(row["predicted_distance_m"]) for row in rows], dtype=float)
    speed = np.asarray([_float(row["max_track_speed_m_s"]) for row in rows], dtype=float)
    risk = np.asarray([RISK_LEVEL.get(row["risk_state_predicted"], np.nan) for row in rows], dtype=float)
    phases = [row["phase"] for row in rows]

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(t, d_cur, label="current Dmin", color="tab:blue")
    axes[0].plot(t, d_pred, label="STRO predicted Dmin", color="tab:red", alpha=0.85)
    axes[0].axhline(0.15, color="tab:orange", linestyle="--", linewidth=1, label="d_safe")
    axes[0].axhline(0.05, color="tab:red", linestyle="--", linewidth=1, label="d_stop")
    axes[0].set_ylabel("distance / m")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, speed, color="tab:green")
    axes[1].set_ylabel("speed / m/s")
    axes[1].grid(True, alpha=0.3)

    axes[2].step(t, risk, where="post", color="tab:purple")
    axes[2].set_yticks([0, 1, 2, 3])
    axes[2].set_yticklabels(["SAFE", "WARN", "SLOW", "STOP"])
    axes[2].set_ylabel("risk")
    axes[2].set_xlabel("time / s")
    axes[2].grid(True, alpha=0.3)

    # Phase boundaries.
    last = phases[0]
    for idx, phase in enumerate(phases):
        if phase != last:
            for ax in axes:
                ax.axvline(t[idx], color="gray", linewidth=0.8, alpha=0.5)
            axes[2].text(t[idx], 3.05, phase, rotation=90, va="bottom", fontsize=8)
            last = phase
    axes[2].text(t[0], 3.05, phases[0], rotation=90, va="bottom", fontsize=8)

    fig.suptitle(trial_dir.name)
    fig.tight_layout()
    out = trial_dir / "perception_curve.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/new/6_5/6_5_1/perception_formal"))
    args = parser.parse_args()
    trial_root = args.root / "trials"
    outputs = []
    for trial_dir in sorted(trial_root.glob("*")):
        if trial_dir.is_dir():
            out = plot_trial(trial_dir)
            if out is not None:
                outputs.append(out)
    print(f"[plot_651] generated {len(outputs)} figures under {trial_root}")


if __name__ == "__main__":
    main()
