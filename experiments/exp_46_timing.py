"""Timing analysis for Chapter 4.6."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


TIMING_KEYS = ("T_pre_ms", "T_dec_ms", "T_obj_ms", "T_trk_ms", "T_risk_ms", "T_rep_ms", "T_cmd_ms", "T_frame_ms")


class TimingAnalyzer:
    def __init__(self, log_dir: str | Path):
        self.log_dir = Path(log_dir)

    def load_timing_rows(self) -> list[dict[str, float]]:
        rows = []
        for path in sorted(self.log_dir.glob("trial_*.json")):
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            for frame in data.get("frames", []):
                timing = frame.get("timing", {})
                if timing:
                    rows.append({key: float(timing.get(key, 0.0)) for key in TIMING_KEYS})
        return rows

    def compute_timing_stats(self) -> dict[str, Any]:
        rows = self.load_timing_rows()
        if not rows:
            return {}
        out = {}
        frame = np.array([r["T_frame_ms"] for r in rows], dtype=float)
        total_mean = float(np.mean(frame)) if len(frame) else 0.0
        for key in TIMING_KEYS:
            vals = np.array([r[key] for r in rows], dtype=float)
            out[key] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "p95": float(np.percentile(vals, 95)),
                "ratio": None if total_mean <= 0 else float(np.mean(vals) / total_mean),
            }
        return out

    def compute_e2e_stats(self) -> dict[str, float]:
        stats = self.compute_timing_stats()
        frame = stats.get("T_frame_ms", {})
        p95 = frame.get("p95", 0.0) or 0.0
        mean = frame.get("mean", 0.0) or 0.0
        return {
            "T_e2e_p95_ms": float(p95),
            "f_perc_hz": 0.0 if mean <= 0 else float(1000.0 / mean),
            "T_ctrl_p95_ms": float(stats.get("T_cmd_ms", {}).get("p95", 0.0) or 0.0),
        }


def table_timing(stats: dict[str, Any]) -> str:
    headers = ["模块", "mean(ms)", "std(ms)", "p95(ms)", "ratio"]
    body = []
    for key in TIMING_KEYS:
        if key not in stats:
            continue
        v = stats[key]
        body.append([key, fmt(v["mean"]), fmt(v["std"]), fmt(v["p95"]), fmt(v["ratio"])])
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Chapter 4.6 timing logs.")
    parser.add_argument("--logs", default="data/results/ch4_5")
    parser.add_argument("--output", default="data/results/ch4_6/timing.json")
    args = parser.parse_args()
    analyzer = TimingAnalyzer(args.logs)
    payload = {"timing": analyzer.compute_timing_stats(), "e2e": analyzer.compute_e2e_stats()}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    print(table_timing(payload["timing"]))
    print(f"\n[exp_46_timing] saved {out}")


if __name__ == "__main__":
    main()
