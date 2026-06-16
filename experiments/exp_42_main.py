"""Command-line entry point for Chapter 4.2 decoupling experiments."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from experiments.decoupling_eval import (
    METHODS,
    DecouplingEvaluator,
    markdown_table,
    parse_omega,
    parse_omegas,
    save_results,
)


def _method_list(value: str) -> list[str]:
    if value == "all":
        return list(METHODS)
    methods = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [item for item in methods if item not in METHODS]
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown methods: {unknown}; choose from {METHODS} or all")
    return methods


def _float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def run_once(args: argparse.Namespace) -> None:
    evaluator = DecouplingEvaluator(
        config_dir=args.config,
        urdf_path=args.urdf,
        delta_r=args.delta_r,
        delta_eval=args.delta_eval,
        voxel_size=args.voxel_size,
        max_raw_points=args.max_raw_points,
        mesh_samples=args.mesh_samples,
        remove_planes=args.remove_planes,
    )
    omegas = parse_omegas(args.omegas) if args.omegas else []
    omega = parse_omega(args.omega) if args.omega else None
    if not omegas and omega is not None:
        omegas = [omega]
    result = evaluator.evaluate_recording(
        args.record_dir,
        methods=args.methods,
        scene=args.scene,
        omegas=omegas,
        n_min_obj=args.n_min_obj,
    )
    stem = f"scene_{result['scene']}_delta_{args.delta_r:.3f}".replace(".", "p")
    out_path = save_results(result, args.output, stem)

    print(markdown_table(result["metrics"], result["scene"]))
    print(f"\n[exp_42] saved metrics: {out_path}")


def run_sweep(args: argparse.Namespace) -> None:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    omegas = parse_omegas(args.omegas) if args.omegas else []
    omega = parse_omega(args.omega) if args.omega else None
    if not omegas and omega is not None:
        omegas = [omega]
    rows = []

    for delta_r in args.sweep_delta_r:
        print(f"\n[exp_42] sweep delta_r={delta_r:.3f}")
        evaluator = DecouplingEvaluator(
            config_dir=args.config,
            urdf_path=args.urdf,
            delta_r=delta_r,
            delta_eval=args.delta_eval,
            voxel_size=args.voxel_size,
            max_raw_points=args.max_raw_points,
            mesh_samples=args.mesh_samples,
            remove_planes=args.remove_planes,
        )
        result = evaluator.evaluate_recording(
            args.record_dir,
            methods=args.methods,
            scene=args.scene,
            omegas=omegas,
            n_min_obj=args.n_min_obj,
        )
        stem = f"sweep_scene_{result['scene']}_delta_{delta_r:.3f}".replace(".", "p")
        save_results(result, output, stem)
        print(markdown_table(result["metrics"], result["scene"]))

        for method, vals in result["metrics"].items():
            row = {"scene": result["scene"], "delta_r": delta_r, "method": method}
            row.update(vals)
            rows.append(row)

    csv_path = output / f"sweep_scene_{args.scene}.csv"
    keys = sorted({key for row in rows for key in row.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[exp_42] saved sweep CSV: {csv_path}")

    try:
        import matplotlib.pyplot as plt

        ours = [row for row in rows if row["method"] == "ours"]
        if ours:
            xs = np.array([row["delta_r"] for row in ours], dtype=float)
            fig, ax1 = plt.subplots(figsize=(7, 4))
            if any("R_res" in row and row["R_res"] is not None for row in ours):
                ax1.plot(xs, [row.get("R_res", np.nan) for row in ours], "o-", label="R_res")
                ax1.set_ylabel("R_res")
            if any("R_keep" in row and row["R_keep"] is not None for row in ours):
                ax2 = ax1.twinx()
                ax2.plot(xs, [row.get("R_keep", np.nan) for row in ours], "s-", color="tab:orange", label="R_keep")
                ax2.set_ylabel("R_keep")
            ax1.set_xlabel("delta_R (m)")
            ax1.grid(True, alpha=0.3)
            fig.tight_layout()
            fig_path = output / f"sweep_scene_{args.scene}.png"
            fig.savefig(fig_path, dpi=200)
            print(f"[exp_42] saved sweep figure: {fig_path}")
    except Exception as exc:
        print(f"[exp_42] skip plotting sweep figure: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Chapter 4.2 external occupancy decoupling evaluation.")
    parser.add_argument("--record-dir", required=True)
    parser.add_argument("--output", default="data/results/ch4_2")
    parser.add_argument("--config", default="config")
    parser.add_argument("--urdf", default="urdf/aubo_i16_gripper.urdf")
    parser.add_argument("--scene", choices=["A", "B", "B2", "C"], required=True)
    parser.add_argument("--method", type=_method_list, default=list(METHODS), dest="methods")
    parser.add_argument("--omega", default=None, help="Scene B weak truth AABB: x0,x1,y0,y1,z0,z1")
    parser.add_argument(
        "--omegas",
        default=None,
        help='Multiple obstacle AABBs separated by semicolons: "x0,x1,y0,y1,z0,z1;x0,x1,y0,y1,z0,z1"',
    )
    parser.add_argument("--n-min-obj", type=int, default=30)
    parser.add_argument("--delta-r", type=float, default=0.05)
    parser.add_argument("--delta-eval", type=float, default=0.10)
    parser.add_argument("--voxel-size", type=float, default=None)
    parser.add_argument(
        "--max-raw-points",
        type=int,
        default=100000,
        help="Deterministically thin each raw frame before transform; set <=0 to disable.",
    )
    parser.add_argument("--mesh-samples", type=int, default=50000)
    parser.add_argument("--remove-planes", action="store_true")
    parser.add_argument(
        "--sweep-delta-r",
        type=_float_list,
        default=None,
        help="Comma-separated delta_R values, e.g. 0.02,0.03,0.04,0.05,0.06,0.08",
    )
    args = parser.parse_args()
    if args.max_raw_points <= 0:
        args.max_raw_points = None

    if args.scene in {"B", "B2"} and args.omega is None and args.omegas is None:
        raise SystemExit("--omega or --omegas is required for scene B/B2 metrics")

    if args.sweep_delta_r:
        run_sweep(args)
    else:
        run_once(args)


if __name__ == "__main__":
    main()
