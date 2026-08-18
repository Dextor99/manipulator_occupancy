#!/usr/bin/env python3
"""Replay the D2-AH stationary Full CCRO-NUBS plan without robot commands."""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import numpy as np

from experiments.exp_ccro_stage2 import _load


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source-trial", type=Path, required=True)
    p.add_argument("--config", type=Path, default=Path("config/ccro_stage4.yaml"))
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    static = importlib.import_module("experiments.new.6_5.6_5_2.run_652_static_avoidance")
    planner = importlib.import_module("experiments.new.6_5.6_5_3.stationary_terminal_ccro")
    trial = args.source_trial.resolve()
    geometry = json.loads((trial / "event_local_05" / "fresh5_multisphere.json").read_text())
    authorization = json.loads((trial / "terminal_goal_authorization" / "authorization_summary.json").read_text())
    config = _load(args.config)
    model = static.make_surface_model(config)
    payload, trajectory = planner.plan_stationary_terminal_ccro(
        config=config,
        model=model,
        q_start=np.asarray(authorization["q_start_rad"], dtype=float),
        q_goal=np.asarray(authorization["q_goal_rad"], dtype=float),
        geometry=geometry,
        output_dir=args.output.resolve(),
    )
    args.output.resolve().mkdir(parents=True, exist_ok=True)
    (args.output.resolve() / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    raise SystemExit(0 if payload.get("authorized") else 2)


if __name__ == "__main__":
    main()
