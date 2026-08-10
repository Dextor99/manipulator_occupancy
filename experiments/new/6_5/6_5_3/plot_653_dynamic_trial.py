#!/usr/bin/env python3
"""Plot one 6.5.3 dynamic repair trial."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np


def parse_float(value: str) -> float:
    if value in ("", None):
        return math.nan
    try:
        return float(value)
    except Exception:
        return math.nan


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def plot_trial(trial_dir: Path, output: Path | None = None) -> Path:
    rows = load_rows(trial_dir / "frames.csv")
    summary_path = trial_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    t = np.asarray([parse_float(r["t_s"]) for r in rows], dtype=float)
    current = np.asarray([parse_float(r.get("nearest_distance_m", "")) for r in rows], dtype=float)
    predicted = np.asarray([parse_float(r.get("predicted_distance_m", "")) for r in rows], dtype=float)
    guard = np.asarray([parse_float(r.get("guard_distance_m", "")) for r in rows], dtype=float)
    speeds = np.asarray([parse_float(r.get("max_track_speed_m_s", "")) for r in rows], dtype=float)
    output = output or (trial_dir / "figures" / "distance_trigger_curve.png")
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(8.0, 5.8), sharex=True)
    axes[0].plot(t, current, label="current distance", linewidth=1.8)
    axes[0].plot(t, predicted, label="predicted distance", linewidth=1.8)
    if np.any(np.isfinite(guard)):
        axes[0].plot(t, guard, label="pointcloud guard", linewidth=2.0)
    axes[0].axhline(0.14, color="tab:orange", linestyle="--", linewidth=1.2, label="replan in")
    axes[0].axhline(0.09, color="tab:red", linestyle="--", linewidth=1.2, label="online accept")
    axes[0].axhline(0.08, color="black", linestyle=":", linewidth=1.2, label="stop")
    for event in summary.get("events", []):
        if "t_s" in event:
            axes[0].axvline(float(event["t_s"]), color="tab:purple", alpha=0.55, linewidth=1.0)
            axes[0].text(float(event["t_s"]), axes[0].get_ylim()[1], event.get("type", ""), rotation=90, va="top", fontsize=7)
    axes[0].set_ylabel("distance / m")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="best", fontsize=8)

    axes[1].plot(t, speeds, color="tab:green", linewidth=1.8)
    axes[1].set_ylabel("track speed / m/s")
    axes[1].set_xlabel("time / s")
    axes[1].grid(True, alpha=0.25)
    title = f"{summary.get('scene', trial_dir.name)} {summary.get('status', '')} candidate={summary.get('candidate_status', '-')}"
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    out = plot_trial(args.trial_dir, args.output)
    print(out)


if __name__ == "__main__":
    main()
