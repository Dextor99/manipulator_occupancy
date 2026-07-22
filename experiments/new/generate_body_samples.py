"""Generate frozen non-end-effector body risk samples for revised 6.2."""

from __future__ import annotations

import argparse
from pathlib import Path

from . import config_62 as cfg
from .body_coverage_62 import make_body_sample
from .common_62 import (
    ensure_output_tree,
    load_surface_model,
    make_reference_trajectory,
    save_reference_trajectory,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate frozen revised 6.2 body samples.")
    parser.add_argument("--output", default=str(cfg.DEFAULT_OUTPUT))
    parser.add_argument("--seed", type=int, default=cfg.RANDOM_SEED + 1200)
    return parser.parse_args()


def generate_samples(surface, trajectory, seed: int) -> list[dict]:
    samples = []
    sample_index = 0
    for config_index, time_value in enumerate(cfg.BODY_CONFIG_TIMES):
        q = trajectory.sample(time_value)
        config_id = f"q{chr(ord('A') + config_index)}"
        for region in cfg.BODY_REGIONS:
            for local_index in range(cfg.BODY_SAMPLES_PER_REGION):
                is_risk = local_index < cfg.BODY_RISK_PER_REGION
                samples.append(
                    make_body_sample(
                        surface,
                        q,
                        config_id,
                        region,
                        sample_index,
                        is_risk,
                        seed + sample_index,
                    )
                )
                sample_index += 1
    return samples


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    ensure_output_tree(output)
    trajectory = make_reference_trajectory()
    save_reference_trajectory(output / "reference_trajectory_62.npz", trajectory)
    surface = load_surface_model()
    samples = generate_samples(surface, trajectory, int(args.seed))
    write_json(output / "body" / "body_samples_62.json", samples)


if __name__ == "__main__":
    main()
