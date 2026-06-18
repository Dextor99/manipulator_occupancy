"""Metrics for Chapter 4.5 safe motion generation experiments."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def load_trial(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def compute_trial_metrics(
    trial: dict[str, Any],
    d_stop: float = 0.05,
    t_max: float = 45.0,
    e_max: float = 0.05,
    t_nominal: float = 15.0,
) -> dict[str, float | bool | None]:
    frames = trial.get("frames", [])
    if not frames:
        return {}
    ts = np.array([f["timestamp"] for f in frames], dtype=float)
    ts = ts - ts[0]
    d_ref = np.array([f["d_ref"] for f in frames], dtype=float)
    qd = np.array([f["cmd_velocity"] for f in frames], dtype=float)
    dt = float(np.median(np.diff(ts))) if len(ts) > 1 else 0.05
    task_time = float(ts[-1] - ts[0]) if len(ts) else 0.0
    d_min = float(np.nanmin(d_ref)) if len(d_ref) else math.inf
    t_viol = float(np.sum(d_ref < d_stop) * dt)
    final_error = trial.get("final_error")
    if final_error is None:
        final_error = 0.0
    timeout = bool(task_time >= t_max)
    success = bool(d_min > d_stop and not timeout and float(final_error) < e_max)
    jerk = 0.0
    if len(qd) > 1 and dt > 0:
        acc = np.diff(qd, axis=0) / dt
        jerk = float(np.sqrt(np.mean(np.sum(acc * acc, axis=1))))
    return {
        "D_min_ref": d_min,
        "T_viol": t_viol,
        "success": success,
        "timeout": timeout,
        "Delta_T": float(task_time - t_nominal) if success else None,
        "E_final": float(final_error),
        "J_q_rms": jerk,
        "T_task": task_time,
    }


def aggregate_trials(paths: list[str | Path]) -> dict[str, Any]:
    metrics = [compute_trial_metrics(load_trial(path)) for path in paths]
    metrics = [m for m in metrics if m]
    if not metrics:
        return {}
    out: dict[str, Any] = {
        "n_trials": len(metrics),
        "R_avoid": float(np.mean([m["success"] for m in metrics])),
        "R_timeout": float(np.mean([m["timeout"] for m in metrics])),
    }
    for key in ("D_min_ref", "T_viol", "Delta_T", "E_final", "J_q_rms", "T_task"):
        vals = [m[key] for m in metrics if m.get(key) is not None]
        out[key] = None if not vals else float(np.mean(vals))
    return out


def table_45(rows: dict[str, Any]) -> str:
    names = {
        "ssm": "SSM",
        "apf": "APF",
        "ours_scale": "Ours-Scale",
        "ours_rep": "Ours-Rep",
        "ours_full": "Ours-Full",
    }
    headers = ["方法", "D_min_ref↑", "T_viol↓", "R_avoid↑", "R_timeout↓", "Delta_T↓", "J_q_rms↓"]
    body = []
    for key in ("ssm", "apf", "ours_scale", "ours_rep", "ours_full"):
        if key not in rows:
            continue
        v = rows[key]
        body.append([
            names[key],
            fmt(v.get("D_min_ref")),
            fmt(v.get("T_viol")),
            fmt(v.get("R_avoid")),
            fmt(v.get("R_timeout")),
            fmt(v.get("Delta_T")),
            fmt(v.get("J_q_rms")),
        ])
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
        *["| " + " | ".join(row) + " |" for row in body],
    ])


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def collect_trials(results_dir: str | Path, scenario: str | None = None) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = {}
    for path in sorted(Path(results_dir).glob("trial_*.json")):
        try:
            trial = load_trial(path)
        except (OSError, json.JSONDecodeError):
            continue
        if scenario is not None and trial.get("scenario") != scenario:
            continue
        controller = trial.get("controller")
        if not controller:
            continue
        grouped.setdefault(str(controller), []).append(path)
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate Chapter 4.5 trial logs.")
    parser.add_argument("--results-dir", default="data/results/ch4_5")
    parser.add_argument("--output", default=None, help="Directory for metrics.json and table_4_6.md. Defaults to --results-dir.")
    parser.add_argument("--scenario", choices=["A", "B", "C"], default=None, help="Only aggregate one scenario.")
    parser.add_argument("--d-stop", type=float, default=0.05)
    parser.add_argument("--t-max", type=float, default=45.0)
    parser.add_argument("--e-max", type=float, default=0.05)
    parser.add_argument("--t-nominal", type=float, default=15.0)
    args = parser.parse_args()

    grouped = collect_trials(args.results_dir, scenario=args.scenario)
    rows: dict[str, Any] = {}
    for controller, paths in grouped.items():
        metrics = [
            compute_trial_metrics(
                load_trial(path),
                d_stop=args.d_stop,
                t_max=args.t_max,
                e_max=args.e_max,
                t_nominal=args.t_nominal,
            )
            for path in paths
        ]
        metrics = [m for m in metrics if m]
        if not metrics:
            continue
        rows[controller] = {
            "n_trials": len(metrics),
            "R_avoid": float(np.mean([m["success"] for m in metrics])),
            "R_timeout": float(np.mean([m["timeout"] for m in metrics])),
        }
        for key in ("D_min_ref", "T_viol", "Delta_T", "E_final", "J_q_rms", "T_task"):
            vals = [m[key] for m in metrics if m.get(key) is not None]
            rows[controller][key] = None if not vals else float(np.mean(vals))

    table = table_45(rows)
    print(table)

    out_dir = Path(args.output or args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.scenario is None else f"_{args.scenario}"
    with (out_dir / f"metrics{suffix}.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, ensure_ascii=False)
    (out_dir / f"table_4_6{suffix}.md").write_text(table + "\n", encoding="utf-8")
    print(f"[exp_45_eval] saved {out_dir / f'metrics{suffix}.json'}")
    print(f"[exp_45_eval] saved {out_dir / f'table_4_6{suffix}.md'}")


if __name__ == "__main__":
    main()
