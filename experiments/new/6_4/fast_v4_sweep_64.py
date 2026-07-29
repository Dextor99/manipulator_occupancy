"""Small development sweep for Fast CCRO-NUBS v4 margin calibration."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from . import config_64 as cfg


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return "-"
    return f"{number:.{digits}f}"


def _run_case(output: Path, *, target: float, iterations: int, reward: float, scenario: str) -> dict[str, Any]:
    case_dir = output / f"ccro_v4_t{target:.3f}_i{iterations}_r{reward:.2f}".replace(".", "p")
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "experiments.new.6_4.fast_local_repair_64",
        "--output",
        str(case_dir),
        "--scenario",
        scenario,
        "--g1-near",
        "--method",
        "ccro_fast_v4",
        "--v4-target-clearance",
        f"{target:.6f}",
        "--v4-max-iterations",
        str(iterations),
        "--v4-clearance-reward",
        f"{reward:.6f}",
    ]
    subprocess.run(cmd, cwd=cfg.ROOT, check=True)
    metrics = json.loads((case_dir / "fast_local_repair_64.json").read_text(encoding="utf-8"))
    summary = metrics["summary"][0]
    rows = metrics["trials"]
    return {
        "case_dir": str(case_dir.relative_to(cfg.ROOT)),
        "target_clearance": target,
        "max_iterations": iterations,
        "clearance_reward": reward,
        "n": summary["n"],
        "qp_solved": summary["qp_solved"],
        "repair_success": summary["repair_success"],
        "online_acceptance": summary["online_acceptance"],
        "verified_safety": summary["verified_safety"],
        "time_pass": summary["time_pass"],
        "acceleration_ok": summary["acceleration_ok"],
        "usable": summary["usable"],
        "delta_dense": summary["delta_dense"],
        "medium_dense_gap_mean": summary["medium_dense_gap_mean"],
        "online_margin_min": summary["online_margin_min"],
        "online_margin_mean": float(np.mean([row["online_threshold_margin"] for row in rows])),
        "online_ms_p95": summary["online_ms_p95"],
        "online_ms_max": summary["online_ms_max"],
    }


def _write_table(rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# 6.4 Fast CCRO-NUBS v4 Margin Sweep",
        "",
        "| target | QP iter | reward | n | repair success | online acceptance | verified safety | time pass | accel pass | usable | mean dD dense / m | mean online margin / m | min online margin / m | online p95 / ms | online max / ms | case |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {_fmt(row['target_clearance'], 3)} | {row['max_iterations']} | {_fmt(row['clearance_reward'], 2)} | "
            f"{row['n']} | {_fmt(row['repair_success'], 2)} | {_fmt(row['online_acceptance'], 2)} | "
            f"{_fmt(row['verified_safety'], 2)} | {_fmt(row['time_pass'], 2)} | {_fmt(row['acceleration_ok'], 2)} | "
            f"{_fmt(row['usable'], 2)} | {_fmt(row['delta_dense'], 4)} | {_fmt(row['online_margin_mean'], 4)} | "
            f"{_fmt(row['online_margin_min'], 4)} | {_fmt(row['online_ms_p95'], 1)} | {_fmt(row['online_ms_max'], 1)} | "
            f"`{row['case_dir']}` |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(Path("results/new/6_4_fast_v4_margin_sweep")))
    parser.add_argument("--scenario", choices=["D1", "D2M"], default="D1")
    parser.add_argument("--targets", nargs="+", type=float, default=[0.095, 0.100, 0.105])
    parser.add_argument("--iterations", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--rewards", nargs="+", type=float, default=[0.0, 0.3])
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output).resolve()
    if args.clean and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for target in args.targets:
        for iterations in args.iterations:
            for reward in args.rewards:
                row = _run_case(output, target=float(target), iterations=int(iterations), reward=float(reward), scenario=args.scenario)
                rows.append(row)
                print(
                    f"[6.4 sweep] target={target:.3f} iter={iterations} reward={reward:.2f} "
                    f"repair={row['repair_success']:.2f} online={row['online_acceptance']:.2f} "
                    f"p95={row['online_ms_p95']:.1f} ms"
                )
    payload = {"experiment": "6.4 Fast CCRO-NUBS v4 margin sweep", "rows": rows}
    (output / "fast_v4_margin_sweep.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_table(rows, output / "paper" / "table_6_4_fast_v4_margin_sweep.md")
    print(f"[6.4 sweep] saved {output / 'paper' / 'table_6_4_fast_v4_margin_sweep.md'}")


if __name__ == "__main__":
    main()
