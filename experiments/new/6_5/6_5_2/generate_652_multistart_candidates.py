#!/usr/bin/env python3
"""Generate formal 6.5.2 multi-start CCRO-NUBS candidates.

This wrapper keeps the formal objective consistent.  Any task-space or manual
detour plan is treated only as an initial seed; each formal candidate is then
re-optimized with the original joint-space CCRO-NUBS objective, i.e. all
task-space preference weights are set to zero.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
PYTHON = sys.executable


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def run_command(cmd: list[str]) -> dict[str, Any]:
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def seed_items(args: argparse.Namespace) -> list[tuple[str, Path | None]]:
    items: list[tuple[str, Path | None]] = [("seed_free", None)]
    for item in args.seed:
        if "=" not in item:
            raise ValueError("--seed entries must use name=plan_dir")
        name, value = item.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError("--seed name cannot be empty")
        items.append((name, Path(value).resolve()))
    return items


def build_plan_command(args: argparse.Namespace, trial_dir: Path, name: str, seed_dir: Path | None, output_dir: Path) -> list[str]:
    cmd = [
        PYTHON,
        str(HERE / "plan_652_static_ccro_nubs_from_trial.py"),
        "--trial-dir",
        str(trial_dir),
        "--output",
        str(output_dir / name),
        f"--q-goal-rad={args.q_goal_rad}",
        "--segment-durations",
        args.segment_durations,
        "--max-iterations-override",
        str(args.max_iterations),
        "--lambda-tcp-z",
        "0.0",
        "--lambda-tcp-xy",
        "0.0",
        "--lambda-tcp-xyz",
        "0.0",
        "--lambda-joint-deviation",
        "0.0",
        "--tcp-z-hard-tolerance-m",
        "0.0",
        "--tcp-orientation-hard-deg",
        "0.0",
    ]
    if seed_dir is not None:
        cmd.extend(["--initial-plan-dir", str(seed_dir)])
    return cmd


def build_select_command(args: argparse.Namespace, trial_dir: Path, plan_dirs: list[Path], output_dir: Path) -> list[str]:
    return [
        PYTHON,
        str(HERE / "select_652_candidate_family.py"),
        "--trial-dir",
        str(trial_dir),
        "--output",
        str(output_dir / "candidate_selection_layered_fixed"),
        "--plan-dirs",
        *[str(p) for p in plan_dirs],
        "--clearance-m",
        str(args.clearance_m),
        "--clearance-pref-m",
        str(args.clearance_pref_m),
        "--time-near-optimal-ratio",
        str(args.time_near_optimal_ratio),
    ]


def run(args: argparse.Namespace) -> dict[str, Any]:
    trial_dir = args.trial_dir.resolve()
    output_dir = (args.output or (trial_dir / "multistart_formal_ccro")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    plan_results = []
    plan_dirs = []
    for name, seed_dir in seed_items(args):
        plan_dir = output_dir / name
        cmd = build_plan_command(args, trial_dir, name, seed_dir, output_dir)
        result = run_command(cmd)
        plan_results.append(
            {
                "candidate": name,
                "formal_plan_dir": str(plan_dir),
                "seed_plan_dir": None if seed_dir is None else str(seed_dir),
                "formal_objective": "joint_space_CCRO_NUBS",
                "result": result,
            }
        )
        if result["returncode"] == 0 and (plan_dir / "summary.json").exists():
            plan_dirs.append(plan_dir)

    selection_result = None
    if plan_dirs:
        selection_result = run_command(build_select_command(args, trial_dir, plan_dirs, output_dir))

    payload = {
        "trial_dir": str(trial_dir),
        "output_dir": str(output_dir),
        "formal_objective": "joint_space_CCRO_NUBS_for_all_candidates",
        "note": (
            "Seed plans may come from task-space preference or manual detours, "
            "but all formal candidates are re-optimized with lambda_tcp_z=0, "
            "lambda_tcp_xy=0, lambda_joint_deviation=0."
        ),
        "plans": plan_results,
        "selection": selection_result,
    }
    write_json(output_dir / "multistart_generation_summary.json", payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--seed",
        action="append",
        default=[],
        help="Optional seed candidate as name=plan_dir. The seed is used only as initial_inner.",
    )
    parser.add_argument(
        "--q-goal-rad",
        default="-0.36184728145599365,-0.22320318222045898,1.315380573272705,-0.03216493874788284,1.5707743167877197,-0.3618289530277252",
    )
    parser.add_argument("--segment-durations", default="2.0,2.0,2.0,2.0")
    parser.add_argument("--max-iterations", type=int, default=300)
    parser.add_argument("--clearance-m", type=float, default=0.08)
    parser.add_argument("--clearance-pref-m", type=float, default=0.11)
    parser.add_argument("--time-near-optimal-ratio", type=float, default=1.05)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
