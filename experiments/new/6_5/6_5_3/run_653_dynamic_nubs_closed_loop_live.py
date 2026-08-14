#!/usr/bin/env python3
"""V3 dynamic NUBS closed-loop protected shadow entry point.

V3 removes fixed-X scene control, uses adaptive PCA 1--4 sphere STRO/Fast/Fresh
geometry, and treats the 0.11 m coarse target and 3 mm improvement as ranking
diagnostics.  The four execution safety conditions remain unchanged.

Real execution remains intentionally blocked.  This phase extends the single
persistent perception stream through the 0.35 s pre-command interval and a
1 s virtual candidate playback; only a later reviewed phase may command the
candidate.  Archived V2/r09 behavior is unaffected.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

trial = importlib.import_module("experiments.new.6_5.6_5_3.run_653_dynamic_repair_trial")
live = importlib.import_module("experiments.new.6_5.6_5_3.run_653_simple_dynamic_nubs_live")
event = importlib.import_module("experiments.new.6_5.6_5_3.run_653_simple_dynamic_nubs_event_replan_live")
v3 = importlib.import_module("experiments.new.6_5.6_5_3.dynamic_nubs_v3")

DEFAULT_OUTPUT = ROOT / "results/new/6_5/6_5_3/dynamic_nubs_closed_loop_v3_shadow"
SCENE_PHRASE = "CCRO_653_DYNAMIC_NUBS_V3_OPPOSING_SCENE_CONFIRMED"


def build_parser() -> argparse.ArgumentParser:
    p = event.build_parser()
    p.description = __doc__
    p.set_defaults(
        output=DEFAULT_OUTPUT,
        task_geometry_id="D2_DYNAMIC_NUBS_CLOSED_LOOP_V3_XP10",
        planning_robust_target_m=0.11,
    )
    p.add_argument("--scene-operator-phrase", default="")
    return p


def validate(args: argparse.Namespace) -> None:
    if args.scene_operator_phrase != SCENE_PHRASE:
        raise RuntimeError(f"bad V3 scene phrase; required: {SCENE_PHRASE}")
    if args.execute:
        raise RuntimeError(
            "V3 real execution is not authorized yet: the protected pre-command "
            "and virtual-playback shadow must pass and be reviewed first"
        )
    if abs(float(args.planning_robust_target_m) - 0.11) > 1.0e-12:
        raise RuntimeError("V3 preferred seed target remains frozen at 0.11 m")


def run(args: argparse.Namespace) -> dict[str, Any]:
    validate(args)
    original_predictor = trial.RISK_SPHERE_PREDICTOR
    original_trigger_gate = trial.RISK_TRIGGER_REQUIRES_DYNAMIC_TRACK
    original_execution_forecast = trial.constant_multisphere_forecast
    original_worker_factory = trial.PERSISTENT_OBSTACLE_WORKER_FACTORY
    original_latest_state_policy = trial.LATEST_STATE_AUTHORIZATION_POLICY
    original_playback_shadow = trial.POST_AUTHORIZATION_PLAYBACK_SHADOW
    original_adapter = live.fixed_two_sphere_adapter
    original_factory = live.make_r06_fast_wrapper
    try:
        trial.RISK_SPHERE_PREDICTOR = v3.adaptive_multisphere_predictor
        trial.RISK_TRIGGER_REQUIRES_DYNAMIC_TRACK = False
        trial.constant_multisphere_forecast = v3.v3_execution_multisphere_forecast
        trial.PERSISTENT_OBSTACLE_WORKER_FACTORY = (
            v3.make_persistent_perception_worker
        )
        trial.LATEST_STATE_AUTHORIZATION_POLICY = (
            v3.latest_state_authorize_with_one_replan
        )
        trial.POST_AUTHORIZATION_PLAYBACK_SHADOW = (
            v3.run_virtual_candidate_playback_shadow
        )
        live.fixed_two_sphere_adapter = v3.adaptive_geometry_adapter
        live.make_r06_fast_wrapper = v3.make_v3_fast_factory(original_factory)
        result = event.run(args)
    finally:
        trial.RISK_SPHERE_PREDICTOR = original_predictor
        trial.RISK_TRIGGER_REQUIRES_DYNAMIC_TRACK = original_trigger_gate
        trial.constant_multisphere_forecast = original_execution_forecast
        trial.PERSISTENT_OBSTACLE_WORKER_FACTORY = original_worker_factory
        trial.LATEST_STATE_AUTHORIZATION_POLICY = original_latest_state_policy
        trial.POST_AUTHORIZATION_PLAYBACK_SHADOW = original_playback_shadow
        live.fixed_two_sphere_adapter = original_adapter
        live.make_r06_fast_wrapper = original_factory
    result["v3_protocol"] = v3.V3_PROTOCOL
    output = Path(result["output"])
    trial.write_json(output / "v3_protocol.json", v3.V3_PROTOCOL)
    trial.write_json(output / "summary.json", result)
    return result


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
