"""Official Tesseract/TrajOpt baseline for E2 static planning.

This script binds the PyPI ``tesseract-robotics`` package and evaluates its
TrajOpt output with the same dense verifier used by CCRO-NUBS. Tesseract's C++
bindings may segfault on invalid plugin/resource configuration, so each
official planner call runs in a subprocess and returns diagnostics instead of
bringing down the whole experiment.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np

if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz  # type: ignore[attr-defined]

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.exp_ccro_stage2 import (  # noqa: E402
    _baseline,
    _limits,
    _load,
    _states,
    make_scenario_obstacle,
)
from planning.mesh_risk import MeshRiskEvaluator  # noqa: E402
from planning.nubs_trajectory import NUBSTrajectory6D  # noqa: E402
from planning.robot_surface_model import RobotSurfaceModel  # noqa: E402
from planning.verifier import TrajectoryVerifier  # noqa: E402


METHOD_KEY = "official_tesseract_trajopt"
METHOD_LABEL = "Official TrajOpt/Tesseract"
TRAJOPT_NAMESPACE = "TrajOptMotionPlannerTask"
TRAJOPT_PROFILE = "TEST_PROFILE"


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, NUBSTrajectory6D):
        return "<NUBSTrajectory6D>"
    raise TypeError(type(value).__name__)


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if math.isinf(value):
            return "inf"
        return f"{value:.6g}"
    return str(value)


def markdown(headers: list[str], rows: list[list[str]]) -> str:
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
            *["| " + " | ".join(row) + " |" for row in rows],
        ]
    )


def _write_tesseract_urdf(src_urdf: Path, dst_urdf: Path) -> None:
    tree = ET.parse(src_urdf)
    root = tree.getroot()
    for link in root.findall("link"):
        for visual in list(link.findall("visual")):
            link.remove(visual)
    for mesh in root.findall(".//mesh"):
        filename = mesh.attrib.get("filename", "")
        if filename and not filename.startswith("/") and "://" not in filename:
            mesh.set("filename", str((src_urdf.parent / filename).resolve()))
    tree.write(dst_urdf, encoding="utf-8", xml_declaration=True)


def _write_tesseract_srdf(
    path: Path,
    joint_names: list[str],
    q_start: np.ndarray,
    plugin_path: Path,
    collision_links: list[str],
) -> None:
    joint_lines = "\n".join(
        f'    <joint name="{name}" value="{float(value):.12g}" />'
        for name, value in zip(joint_names, q_start)
    )
    disabled = [
        (collision_links[i], collision_links[j], "InternalRobotPair")
        for i in range(len(collision_links))
        for j in range(i + 1, len(collision_links))
    ]
    disabled_lines = "\n".join(
        f'  <disable_collisions link1="{a}" link2="{b}" reason="{reason}" />'
        for a, b, reason in disabled
    )
    path.write_text(
        "\n".join(
            [
                '<?xml version="1.0" ?>',
                '<robot name="aubo_i16_gripper">',
                '  <group name="manipulator">',
                '    <chain base_link="base_link" tip_link="wrist3_Link" />',
                '  </group>',
                '  <group_tcps group="manipulator">',
                '    <tcp name="wrist3_Link" xyz="0 0 0" wxyz="1 0 0 0" />',
                '  </group_tcps>',
                f'  <contact_managers_plugin_config filename="{plugin_path}" />',
                '  <collision_margins default_margin="0.04" />',
                '  <group_state name="home" group="manipulator">',
                joint_lines,
                '  </group_state>',
                disabled_lines,
                "</robot>",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_contact_plugin_yaml(path: Path) -> None:
    # Schema follows Tesseract's PluginInfoContainer convention. The class names
    # are the factories bundled in the official tesseract-robotics wheel.
    path.write_text(
        "\n".join(
            [
                "contact_manager_plugins:",
                "  search_paths: []",
                "  search_libraries:",
                "    - tesseract_collision_bullet",
                "    - tesseract_collision_fcl",
                "  discrete_plugins:",
                "    default: BulletDiscreteBVHManager",
                "    plugins:",
                "      BulletDiscreteBVHManager:",
                "        class: BulletDiscreteBVHManagerFactory",
                "      BulletDiscreteSimpleManager:",
                "        class: BulletDiscreteSimpleManagerFactory",
                "      FCLDiscreteBVHManager:",
                "        class: FCLDiscreteBVHManagerFactory",
                "  continuous_plugins:",
                "    default: BulletCastBVHManager",
                "    plugins:",
                "      BulletCastBVHManager:",
                "        class: BulletCastBVHManagerFactory",
                "      BulletCastSimpleManager:",
                "        class: BulletCastSimpleManagerFactory",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _prepare_tesseract_model(
    output: Path, robot_urdf: Path, joint_names: list[str], q_start: np.ndarray
) -> tuple[Path, Path, Path]:
    model_dir = output / "tesseract_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    urdf_path = model_dir / "aubo_i16_gripper_tesseract.urdf"
    srdf_path = model_dir / "aubo_i16_gripper_tesseract.srdf"
    plugin_path = model_dir / "contact_manager_plugins.yaml"
    _write_contact_plugin_yaml(plugin_path)
    _write_tesseract_urdf(robot_urdf, urdf_path)
    collision_tree = ET.parse(urdf_path)
    collision_links = [
        link.attrib["name"]
        for link in collision_tree.getroot().findall("link")
        if link.attrib.get("name") and link.findall("collision")
    ]
    _write_tesseract_srdf(
        srdf_path, joint_names, q_start, plugin_path.resolve(), collision_links
    )
    return urdf_path, srdf_path, plugin_path


def _resample_positions(q_waypoints: np.ndarray, count: int) -> np.ndarray:
    q_waypoints = np.asarray(q_waypoints, dtype=np.float64)
    if q_waypoints.ndim != 2 or q_waypoints.shape[0] < 2:
        raise ValueError("expected at least two joint-space waypoints")
    old = np.linspace(0.0, 1.0, q_waypoints.shape[0])
    new = np.linspace(0.0, 1.0, count)
    return np.column_stack([np.interp(new, old, q_waypoints[:, j]) for j in range(q_waypoints.shape[1])])


def _trajectory_from_waypoints(
    q_waypoints: np.ndarray, head: np.ndarray, tail: np.ndarray, durations: np.ndarray
) -> NUBSTrajectory6D:
    control_count = len(durations) + 1
    controls = _resample_positions(q_waypoints, control_count)
    controls[0] = head[:, 0]
    controls[-1] = tail[:, 0]
    return NUBSTrajectory6D().generate(controls[1:-1], head, tail, durations)


def _extract_result_positions(worker_npz: Path) -> tuple[bool, np.ndarray | None, dict[str, Any]]:
    if not worker_npz.exists():
        return False, None, {"message": "worker did not create result npz"}
    data = np.load(worker_npz, allow_pickle=True)
    payload = json.loads(str(data["payload"].item()))
    q = data["q"] if "q" in data.files else None
    return bool(payload.get("successful", False)), q, payload


def _run_worker(
    urdf_path: Path,
    srdf_path: Path,
    joint_names: list[str],
    seed_waypoints: np.ndarray,
    worker_npz: Path,
    timeout_s: float,
) -> tuple[bool, dict[str, Any]]:
    request_npz = worker_npz.with_name(worker_npz.stem + "_request.npz")
    np.savez_compressed(
        request_npz,
        joint_names=np.asarray(joint_names, dtype=object),
        seed_waypoints=np.asarray(seed_waypoints, dtype=np.float64),
    )
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--_worker",
        "--urdf",
        str(urdf_path),
        "--srdf",
        str(srdf_path),
        "--worker-input",
        str(request_npz),
        "--worker-output",
        str(worker_npz),
    ]
    started = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        timeout=timeout_s,
        env={**os.environ, "TRAJOPT_LOG_THRESH": "ERROR"},
        check=False,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    ok, q, payload = _extract_result_positions(worker_npz)
    payload.update(
        {
            "worker_returncode": proc.returncode,
            "worker_elapsed_ms": elapsed_ms,
            "worker_stdout": proc.stdout[-3000:],
            "worker_stderr": proc.stderr[-5000:],
            "q_waypoint_count": 0 if q is None else int(q.shape[0]),
        }
    )
    return bool(ok and proc.returncode == 0), payload


def _worker_main(args: argparse.Namespace) -> None:
    from tesseract_robotics.tesseract_environment import Environment
    from tesseract_robotics.tesseract_common import (
        FilesystemPath,
        GeneralResourceLocator,
        ManipulatorInfo,
    )
    from tesseract_robotics.tesseract_command_language import (
        CompositeInstruction,
        InstructionPoly_as_MoveInstructionPoly,
        MoveInstruction,
        MoveInstructionPoly_wrap_MoveInstruction,
        MoveInstructionType_FREESPACE,
        ProfileDictionary,
        StateWaypoint,
        StateWaypointPoly_wrap_StateWaypoint,
        WaypointPoly_as_JointWaypointPoly,
        WaypointPoly_as_StateWaypointPoly,
    )
    from tesseract_robotics.tesseract_motion_planners import PlannerRequest, formatProgram
    from tesseract_robotics.tesseract_motion_planners_trajopt import (
        ProfileDictionary_addProfile_TrajOptCompositeProfile,
        ProfileDictionary_addProfile_TrajOptPlanProfile,
        TrajOptDefaultCompositeProfile,
        TrajOptDefaultPlanProfile,
        TrajOptMotionPlanner,
    )

    data = np.load(args.worker_input, allow_pickle=True)
    joint_names = [str(x) for x in data["joint_names"].tolist()]
    seed_waypoints = np.asarray(data["seed_waypoints"], dtype=np.float64)
    payload: dict[str, Any] = {
        "successful": False,
        "message": "",
        "official_package": "tesseract-robotics",
        "planner": "TrajOptMotionPlanner",
    }
    try:
        locator = GeneralResourceLocator()
        env = Environment()
        ok = env.init(FilesystemPath(args.urdf), FilesystemPath(args.srdf), locator)
        if not ok:
            raise RuntimeError("Tesseract Environment.init returned False")
        env.setState(joint_names, seed_waypoints[0])
        manip = ManipulatorInfo()
        manip.manipulator = "manipulator"
        manip.working_frame = "base_link"
        manip.tcp_frame = "wrist3_Link"
        program = CompositeInstruction(TRAJOPT_PROFILE)
        program.setManipulatorInfo(manip)
        for q in seed_waypoints:
            waypoint = StateWaypoint(joint_names, np.asarray(q, dtype=np.float64))
            instruction = MoveInstruction(
                StateWaypointPoly_wrap_StateWaypoint(waypoint),
                MoveInstructionType_FREESPACE,
                TRAJOPT_PROFILE,
            )
            instruction.setManipulatorInfo(manip)
            instruction.setProfile(TRAJOPT_PROFILE)
            instruction.setPathProfile(TRAJOPT_PROFILE)
            program.appendMoveInstruction(MoveInstructionPoly_wrap_MoveInstruction(instruction))
        profiles = ProfileDictionary()
        ProfileDictionary_addProfile_TrajOptPlanProfile(
            profiles, TRAJOPT_NAMESPACE, TRAJOPT_PROFILE, TrajOptDefaultPlanProfile()
        )
        ProfileDictionary_addProfile_TrajOptCompositeProfile(
            profiles, TRAJOPT_NAMESPACE, TRAJOPT_PROFILE, TrajOptDefaultCompositeProfile()
        )
        request = PlannerRequest()
        request.instructions = program
        request.env = env
        request.env_state = env.getState()
        request.profiles = profiles
        started = time.perf_counter()
        response = TrajOptMotionPlanner(TRAJOPT_NAMESPACE).solve(request)
        solve_ms = (time.perf_counter() - started) * 1000.0
        q_rows: list[np.ndarray] = []
        flattened = response.results.flatten()
        payload.update(
            {
                "response_successful": bool(getattr(response, "successful", False)),
                "response_message": str(getattr(response, "message", "")),
                "solve_ms": solve_ms,
                "flattened_instruction_count": int(len(flattened)),
                "succeeded_instruction_count": int(len(getattr(response, "succeeded_instructions", []))),
                "failed_instruction_count": int(len(getattr(response, "failed_instructions", []))),
            }
        )
        try:
            payload["formatted_program"] = str(formatProgram(response.results, env.getState(), env))
        except Exception as exc:
            payload["formatted_program_error"] = f"{type(exc).__name__}: {exc}"
        for instruction in flattened:
            move_instruction = InstructionPoly_as_MoveInstructionPoly(instruction)
            waypoint = move_instruction.getWaypoint()
            if waypoint.isStateWaypoint():
                state_waypoint = WaypointPoly_as_StateWaypointPoly(waypoint)
                q_rows.append(np.asarray(state_waypoint.getPosition(), dtype=np.float64).ravel())
            elif waypoint.isJointWaypoint():
                joint_waypoint = WaypointPoly_as_JointWaypointPoly(waypoint)
                q_rows.append(np.asarray(joint_waypoint.getPosition(), dtype=np.float64).ravel())
        if len(q_rows) < 2:
            raise RuntimeError(
                "TrajOpt response did not contain enough joint/state waypoints; "
                f"response_successful={payload.get('response_successful')}, "
                f"flattened_instruction_count={payload.get('flattened_instruction_count')}"
            )
        q = np.vstack(q_rows)
        payload.update(
            {
                "successful": bool(response.successful),
                "message": "official TrajOptMotionPlanner.solve completed",
                "solve_ms": solve_ms,
                "result_waypoints": int(q.shape[0]),
            }
        )
        np.savez_compressed(args.worker_output, q=q, payload=json.dumps(payload))
    except Exception as exc:
        payload.update({"successful": False, "message": f"{type(exc).__name__}: {exc}"})
        np.savez_compressed(args.worker_output, payload=json.dumps(payload))
        raise


def _official_metrics(
    official_success: bool,
    worker_payload: dict[str, Any],
    q_waypoints: np.ndarray | None,
    seed_waypoints: np.ndarray,
    evaluator: MeshRiskEvaluator,
    obstacle,
    verifier: TrajectoryVerifier,
    head: np.ndarray,
    tail: np.ndarray,
    durations: np.ndarray,
    sample_times: np.ndarray,
) -> dict[str, Any]:
    used_official_output = bool(official_success and q_waypoints is not None)
    source_waypoints = q_waypoints if used_official_output else seed_waypoints
    trajectory = _trajectory_from_waypoints(source_waypoints, head, tail, durations)
    risk = evaluator.trajectory(trajectory, obstacle, sample_times, with_gradient=False)
    verification = verifier.verify(
        trajectory,
        obstacle,
        current_q=head[:, 0],
        current_qd=head[:, 1],
        current_qdd=head[:, 2],
        q_goal=tail[:, 0],
        solver_success=used_official_output,
    )
    return {
        "method": METHOD_KEY,
        "method_label": METHOD_LABEL,
        "solver_success": used_official_output,
        "official_output_used": used_official_output,
        "optimized_links": None,
        "risk_cost_for_method": risk.cost,
        "full_body_risk_cost": risk.cost,
        "optimization_sample_min_distance": risk.min_distance,
        "nearest_link": risk.nearest_link,
        "verification": asdict(verification),
        "optimization": {
            "elapsed_ms": worker_payload.get("worker_elapsed_ms"),
            "solve_ms": worker_payload.get("solve_ms"),
            "final_energy": trajectory.energy(),
            "result_waypoints": worker_payload.get("q_waypoint_count"),
            "worker_returncode": worker_payload.get("worker_returncode"),
            "message": worker_payload.get("message"),
        },
        "diagnostics": worker_payload,
        "q_waypoints": source_waypoints.tolist(),
    }


def run(
    config_path: str | Path,
    output_override: str | Path | None = None,
    *,
    timeout_s: float = 60.0,
    seed_waypoints: int = 9,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    config = _load(config_path)
    output = Path(output_override or "data/results/ch6_e1_e5/E2_static_planning_benchmark/official_tesseract_trajopt")
    if not output.is_absolute():
        output = ROOT / output
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output / "source_ccro_stage2.yaml")

    robot = config["robot"]
    joint_names = list(robot["joint_names"])
    head, tail, durations = _states(config)
    limits = _limits(config)
    baseline_result = _baseline(config, head, tail, durations)
    baseline = baseline_result.trajectory
    baseline_samples = baseline.sample(np.linspace(0.0, baseline.total_duration, seed_waypoints)).q
    urdf_path, srdf_path, plugin_path = _prepare_tesseract_model(
        output, (ROOT / robot["urdf_path"]).resolve(), joint_names, head[:, 0]
    )

    surface_cfg = config["surface"]
    surface_model = RobotSurfaceModel(
        ROOT / robot["urdf_path"],
        joint_names,
        surface_cfg["density_totals"],
        seed=surface_cfg["random_seed"],
        min_points_per_link=surface_cfg["min_points_per_link"],
        cache_dir=surface_cfg["cache_dir"],
        geometry=surface_cfg["geometry"],
    )
    risk_cfg = config["risk"]
    evaluator = MeshRiskEvaluator(
        surface_model,
        d_safe=risk_cfg["d_safe"],
        d_activate=risk_cfg["d_activate"],
        fd_epsilon_q=risk_cfg["fd_epsilon_q"],
        density=risk_cfg["optimizer_density"],
    )
    verifier = TrajectoryVerifier(
        evaluator,
        limits,
        d_stop=risk_cfg["d_stop"],
        time_step=config["validation"]["dense_time_step"],
        density=risk_cfg["validation_density"],
        epsilon_goal=config["validation"]["epsilon_goal"],
        epsilon_continuity_q=config["validation"]["epsilon_continuity_q"],
        epsilon_continuity_qd=config["validation"]["epsilon_continuity_qd"],
        epsilon_continuity_qdd=config["validation"]["epsilon_continuity_qdd"],
        limit_tolerance=config["validation"]["limit_tolerance"],
    )
    rng = np.random.default_rng(int(config["experiment"]["random_seed"]))
    sample_times = np.linspace(0.0, baseline.total_duration, 41)
    metrics: dict[str, Any] = {
        "source": "official Tesseract/TrajOpt baseline for E2",
        "method_key": METHOD_KEY,
        "method_label": METHOD_LABEL,
        "official_package": "tesseract-robotics",
        "notes": {
            "model_adapter": "URDF visuals removed and mesh paths made absolute for Tesseract",
            "planner_isolation": "TrajOptMotionPlanner is executed in subprocess to isolate native crashes",
            "evaluation": "official output is evaluated with the shared dense TrajectoryVerifier",
            "point_cloud_obstacles": "E2 point-cloud risk field is not injected into Tesseract; it is used by the common verifier",
        },
        "tesseract_files": {
            "urdf": str(urdf_path),
            "srdf": str(srdf_path),
            "contact_plugins": str(plugin_path),
        },
        "scenarios": {},
    }
    for scenario_name in config["experiment"]["scenarios"]:
        obstacle, obstacle_info = make_scenario_obstacle(
            config, scenario_name, surface_model, baseline, rng
        )
        with tempfile.TemporaryDirectory(prefix=f"e2_tesseract_{scenario_name}_") as tmp:
            worker_npz = Path(tmp) / "worker_result.npz"
            try:
                official_success, worker_payload = _run_worker(
                    urdf_path,
                    srdf_path,
                    joint_names,
                    baseline_samples,
                    worker_npz,
                    timeout_s,
                )
            except subprocess.TimeoutExpired as exc:
                official_success = False
                worker_payload = {
                    "successful": False,
                    "message": f"worker timeout after {timeout_s}s",
                    "worker_stdout": (exc.stdout or "")[-3000:],
                    "worker_stderr": (exc.stderr or "")[-5000:],
                    "worker_returncode": None,
                    "worker_elapsed_ms": timeout_s * 1000.0,
                }
            ok, q_waypoints, payload_from_npz = _extract_result_positions(worker_npz)
            if payload_from_npz:
                worker_payload.update(payload_from_npz)
            scenario_metrics = _official_metrics(
                official_success and ok,
                worker_payload,
                q_waypoints,
                baseline_samples,
                evaluator,
                obstacle,
                verifier,
                head,
                tail,
                durations,
                sample_times,
            )
        metrics["scenarios"][scenario_name] = {
            "obstacle": obstacle_info,
            "methods": {METHOD_KEY: scenario_metrics},
        }
    metrics["accepted"] = all(
        payload["methods"][METHOD_KEY]["solver_success"] for payload in metrics["scenarios"].values()
    )
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )
    table_rows = []
    for scenario, payload in metrics["scenarios"].items():
        row = payload["methods"][METHOD_KEY]
        opt = row["optimization"]
        ver = row["verification"]
        table_rows.append(
            [
                scenario,
                METHOD_LABEL,
                str(row["solver_success"]),
                str(ver["accepted"]),
                fmt(ver["min_distance"]),
                fmt(row["full_body_risk_cost"]),
                fmt(opt.get("final_energy")),
                fmt(ver.get("goal_error")),
                row.get("nearest_link") or "-",
                fmt(opt.get("elapsed_ms")),
                str(opt.get("worker_returncode")),
                opt.get("message") or "-",
            ]
        )
    table = markdown(
        [
            "scenario",
            "method",
            "solver",
            "accepted",
            "D_min dense/m",
            "J_risk",
            "J_smooth",
            "goal error",
            "nearest link",
            "time/ms",
            "worker code",
            "message",
        ],
        table_rows,
    )
    (output / "table_E2_official_tesseract_trajopt.md").write_text(table + "\n", encoding="utf-8")
    print(table)
    print(f"\n[exp_65_official_tesseract] saved results to {output}")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config" / "ccro_stage2.yaml"))
    parser.add_argument(
        "--output",
        default="data/results/ch6_e1_e5/E2_static_planning_benchmark/official_tesseract_trajopt",
    )
    parser.add_argument("--timeout-s", type=float, default=60.0)
    parser.add_argument("--seed-waypoints", type=int, default=9)
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--urdf", help=argparse.SUPPRESS)
    parser.add_argument("--srdf", help=argparse.SUPPRESS)
    parser.add_argument("--worker-input", help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args._worker:
        _worker_main(args)
        return
    run(args.config, args.output, timeout_s=args.timeout_s, seed_waypoints=args.seed_waypoints)


if __name__ == "__main__":
    main()
