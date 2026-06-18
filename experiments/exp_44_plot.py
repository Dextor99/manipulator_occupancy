"""Generate Chapter 4.4 representative repulsive-vector figure."""
from __future__ import annotations

import argparse
from pathlib import Path

from experiments.exp_44_main import RepulsionEvaluator44


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Chapter 4.4 representative risk frames.")
    parser.add_argument("--record-dir", required=True)
    parser.add_argument("--empty-record-dir", default=None)
    parser.add_argument("--output", default="data/results/ch4_4/fig44.png")
    parser.add_argument("--config", default="config")
    parser.add_argument("--urdf", default="urdf/aubo_i16_gripper.urdf")
    parser.add_argument("--delta-r", type=float, default=0.05)
    parser.add_argument("--bg-eps", type=float, default=0.03)
    parser.add_argument("--mesh-samples", type=int, default=50000)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--max-background-points", type=int, default=500000)
    args = parser.parse_args()

    evaluator = RepulsionEvaluator44(
        config_dir=args.config,
        urdf_path=args.urdf,
        delta_r=args.delta_r,
        bg_eps=args.bg_eps,
        mesh_samples=args.mesh_samples,
        max_background_points=args.max_background_points,
    )
    ok = evaluator.make_fig44(args.record_dir, args.empty_record_dir, Path(args.output), max_frames=args.max_frames)
    if not ok:
        raise SystemExit("[exp_44_plot] no suitable EE/body risk frame found")
    print(f"[exp_44_plot] saved {Path(args.output)}")


if __name__ == "__main__":
    main()
