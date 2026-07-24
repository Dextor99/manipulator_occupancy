"""Run Chapter 6.4 dynamic-obstacle CCRO-NUBS virtual closed-loop experiments."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

from . import config_64 as cfg
from .common_64 import (
    git_commit_hash,
    load_stage4_config,
    load_surface_model,
    make_critical_risk_stack,
    make_reference,
    make_risk_stack,
    write_json,
)
from .scenarios_64 import generate_instances
from .summarize_64 import aggregate, write_paper_table, write_stratified_tables
from .virtual_loop_64 import run_trial


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(cfg.DEFAULT_OUTPUT))
    parser.add_argument("--config", default=str(cfg.STAGE4_CONFIG))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--scenario", choices=["D1", "D2M", "D2S", "D2", "D3", "D4"], default=None)
    parser.add_argument("--method", choices=cfg.METHODS, default=None)
    parser.add_argument("--instance", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def planned_methods(instance: dict[str, Any]) -> tuple[str, ...]:
    if instance["scenario_type"] in {"far_safe", "initial_high_risk"}:
        return ("ccro_nubs",)
    return cfg.MAIN_METHODS


def main() -> None:
    args = parse_args()
    output = Path(args.output).resolve()
    instances_dir = output / "instances"
    trials_dir = output / "trials"
    paper_dir = output / "paper"
    output.mkdir(parents=True, exist_ok=True)
    trials_dir.mkdir(parents=True, exist_ok=True)
    paper_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__).with_name("config_64.yaml"), output / "config_64.yaml")

    config = load_stage4_config(Path(args.config))
    model = load_surface_model(config)
    reference, head, tail, durations = make_reference(config)
    evaluator, verifier, limits = make_risk_stack(config, model, None)
    critical_evaluator, critical_verifier = make_critical_risk_stack(config, model)
    instances = generate_instances(model, reference, instances_dir, smoke=args.smoke, gate=args.gate)
    if args.scenario:
        prefixes = ("D2S_",) if args.scenario == "D2" else (args.scenario + "_",)
        instances = [item for item in instances if item["instance_id"].startswith(prefixes)]
    if args.instance:
        instances = [item for item in instances if item["instance_id"] == args.instance]

    trial_summaries: list[dict[str, Any]] = []
    for instance in instances:
        methods = planned_methods(instance)
        if args.method:
            methods = tuple(method for method in methods if method == args.method)
        for method in methods:
            trial_path = trials_dir / f"{instance['instance_id']}_{method}.json"
            if args.resume and trial_path.exists():
                import json

                payload = json.loads(trial_path.read_text(encoding="utf-8"))
            else:
                payload = run_trial(
                    config=config,
                    model=model,
                    reference=reference,
                    tail=tail,
                    durations=durations,
                    evaluator=evaluator,
                    verifier=verifier,
                    critical_evaluator=critical_evaluator,
                    critical_verifier=critical_verifier,
                    limits=limits,
                    instance=instance,
                    method=method,
                )
                write_json(trial_path, payload)
            compact = {key: value for key, value in payload.items() if key != "timeline"}
            trial_summaries.append(compact)
            print(
                f"[6.4] {payload['trial_id']}: success={payload['success']} "
                f"Dmin={payload['min_distance_gt']:.3f} replans={payload['replan_count']} "
                f"accepted={payload['accepted_count']}"
            )

    summary = aggregate(trial_summaries)
    metrics = {
        "experiment": "6.4 dynamic obstacle CCRO-NUBS virtual closed loop",
        "scope": "software simulation only; no real robot commands",
        "git_commit": git_commit_hash(),
        "mode": "smoke" if args.smoke else "formal",
        "output_dir": str(output),
        "config_source": str(Path(args.config).resolve()),
        "trial_count": len(trial_summaries),
        "instances": [item["instance_id"] for item in instances],
        "summary": summary,
        "trials": trial_summaries,
    }
    write_json(output / "metrics.json", metrics)
    write_json(output / "manifest.json", {k: metrics[k] for k in ("experiment", "scope", "git_commit", "mode", "trial_count", "output_dir")})
    write_paper_table(summary, paper_dir / "table_6_4_dynamic_replanning.md")
    write_stratified_tables(summary, paper_dir)
    print(f"[6.4] saved metrics to {output / 'metrics.json'}")
    print(f"[6.4] saved paper table to {paper_dir / 'table_6_4_dynamic_replanning.md'}")


if __name__ == "__main__":
    main()
