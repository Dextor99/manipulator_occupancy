#!/usr/bin/env python3
"""Summarize 6.5.3 dynamic repair trials."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def load_summary(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def summarize(root: Path) -> dict[str, Any]:
    trials = []
    for path in sorted((root / "trials").glob("*/summary.json")):
        item = load_summary(path)
        if item is not None:
            item["_summary_path"] = str(path)
            trials.append(item)
    by_scene: dict[str, list[dict[str, Any]]] = {}
    for trial in trials:
        by_scene.setdefault(trial.get("scene", "unknown"), []).append(trial)
    scenes = {}
    for scene, items in sorted(by_scene.items()):
        accepted = [t for t in items if t.get("candidate_accepted") is True]
        triggered = [t for t in items if t.get("status") == "TRIGGERED"]
        fast_times = []
        mins_pred = []
        mins_cur = []
        for trial in items:
            if trial.get("predicted_min_distance_m") is not None:
                mins_pred.append(float(trial["predicted_min_distance_m"]))
            if trial.get("current_min_distance_m") is not None:
                mins_cur.append(float(trial["current_min_distance_m"]))
            for ev in trial.get("events", []):
                cand = ev.get("candidate")
                if isinstance(cand, dict) and cand.get("fast_elapsed_ms") is not None:
                    fast_times.append(float(cand["fast_elapsed_ms"]))
        scenes[scene] = {
            "trials": len(items),
            "triggered": len(triggered),
            "candidate_accepted": len(accepted),
            "trigger_rate": len(triggered) / len(items) if items else None,
            "candidate_accept_rate": len(accepted) / len(items) if items else None,
            "current_min_distance_m": None if not mins_cur else float(np.min(mins_cur)),
            "predicted_min_distance_m": None if not mins_pred else float(np.min(mins_pred)),
            "fast_elapsed_ms_mean": None if not fast_times else float(np.mean(fast_times)),
            "fast_elapsed_ms_std": None if not fast_times else float(np.std(fast_times)),
        }
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "trial_count": len(trials),
        "scenes": scenes,
        "trials": trials,
    }


def write_outputs(root: Path, payload: dict[str, Any]) -> None:
    (root / "metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")
    lines = [
        "# 6.5.3 Dynamic Repair Summary",
        "",
        "| Scene | Trials | Triggered | Candidate accepted | Min current / m | Min predicted / m | Fast time / ms |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for scene, row in payload["scenes"].items():
        fast = "-"
        if row["fast_elapsed_ms_mean"] is not None:
            fast = f"{row['fast_elapsed_ms_mean']:.1f} ± {row['fast_elapsed_ms_std']:.1f}"
        lines.append(
            f"| {scene} | {row['trials']} | {row['triggered']} | {row['candidate_accepted']} | "
            f"{fmt(row['current_min_distance_m'])} | {fmt(row['predicted_min_distance_m'])} | {fast} |"
        )
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/new/6_5/6_5_3/dynamic_repair_formal"))
    args = parser.parse_args()
    payload = summarize(args.input)
    write_outputs(args.input, payload)
    print(json.dumps({"trial_count": payload["trial_count"], "summary": str(args.input / "summary.md")}, indent=2))


if __name__ == "__main__":
    main()
