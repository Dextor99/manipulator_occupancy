from __future__ import annotations

import importlib
import inspect
import csv
import json
import time
from types import SimpleNamespace

import numpy as np
import pytest

from planning.nubs_trajectory import NUBSTrajectory6D
from planning.optimizer import JointLimits
from planning.obstacle_forecast import ConstantVelocitySphereForecast
from planning.spatiotemporal_risk import SpatioTemporalRiskEvaluator


trial = importlib.import_module("experiments.new.6_5.6_5_3.run_653_dynamic_repair_trial")
prepare = importlib.import_module("experiments.new.6_5.6_5_3.prepare_653_reference")
repair_v3 = importlib.import_module("experiments.new.6_4.repair.repair_v3")
linearization = importlib.import_module("experiments.new.6_4.repair.nubs_linearization")
calibration = importlib.import_module("experiments.new.6_5.6_5_3.calibrate_653_local_offline_track")
alignment = importlib.import_module("experiments.new.6_5.6_5_3.align_653_authorized_start")
candidate_return = importlib.import_module("experiments.new.6_5.6_5_3.return_653_local_candidate_start")
delayed_calibration = importlib.import_module(
    "experiments.new.6_5.6_5_3.calibrate_653_local_delayed_rejoin"
)
delayed_resume = importlib.import_module(
    "experiments.new.6_5.6_5_3.resume_653_from_delayed_rejoin"
)
hold_resume = importlib.import_module(
    "experiments.new.6_5.6_5_3.resume_653_from_hold"
)
d2_sweep = importlib.import_module("experiments.new.6_5.6_5_3.offline_d2_geometry_sweep")
rolling_replay = importlib.import_module(
    "experiments.new.6_5.6_5_3.offline_rolling_local_replay"
)
rolling_virtual = importlib.import_module(
    "experiments.new.6_5.6_5_3.shadow_653_rolling_local_virtual"
)
static_geometry_ab = importlib.import_module(
    "experiments.new.6_5.6_5_3.offline_static_geometry_ab"
)
static_distance_ledger = importlib.import_module(
    "experiments.new.6_5.6_5_3.offline_static_distance_ledger"
)
static20_closure = importlib.import_module(
    "experiments.new.6_5.6_5_3.offline_static20_fast_closure_replay"
)
offset_rollout = importlib.import_module(
    "experiments.new.6_5.6_5_3.offline_static20_offset_preserving_rollout"
)
rolling_common = importlib.import_module(
    "experiments.new.6_5.6_5_3.goal_directed_rolling_common"
)
static20_shadow = importlib.import_module(
    "experiments.new.6_5.6_5_3.shadow_653_static20_goal_directed_virtual"
)
simple_bypass = importlib.import_module(
    "experiments.new.6_5.6_5_3.simple_bypass_planner"
)
simple_dynamic = importlib.import_module(
    "experiments.new.6_5.6_5_3.run_653_simple_dynamic_nubs_avoidance"
)


def test_track_geometry_uses_one_track_for_center_velocity_and_radius():
    obj = SimpleNamespace(id=7, center=np.array([1.0, 2.0, 3.0]), velocity=np.array([0.1, 0.0, 0.0]), radius=0.06)
    clusters = [SimpleNamespace(center=np.array([1.01, 2.0, 3.0]), points=np.array([[1.07, 2.0, 3.0], [0.95, 2.0, 3.0]]))]
    geometry = trial.track_geometry(obj, clusters, 0.055)
    assert geometry["track_id"] == 7
    assert np.isclose(geometry["raw_radius"], 0.06)
    assert np.isclose(geometry["inflated_radius"], 0.06)
    assert geometry["associated_cluster_index"] == 0
    assert geometry["association_error_m"] < 0.011


def test_recorded_reference_state_after_uses_future_reference_not_velocity_extrapolation():
    times = np.array([0.0, 1.0, 2.0])
    q = np.zeros((3, 6))
    q[:, 0] = [0.0, 0.3, 1.0]
    qd = np.zeros_like(q)
    ref = trial.RecordedReference(times, q, qd)
    ref.locate(np.array([0.3, 0, 0, 0, 0, 0]), y_actual=None, max_forward_step=5)
    q_future, _, _ = ref.state_after(1.0)
    assert ref.index == 1
    assert q_future[0] == 1.0


def test_recorded_reference_remainder_is_zero_based_and_endpoint_inclusive():
    times = np.array([0.0, 1.0, 2.0, 3.0])
    q = np.zeros((4, 6))
    q[:, 0] = times
    ref = trial.RecordedReference(times, q, np.ones_like(q))
    remainder_t, remainder_q, start_qd = ref.remainder_after(1.25)
    assert remainder_t[0] == 0.0
    assert remainder_t[-1] == 1.75
    assert remainder_q[0, 0] == 1.25
    assert remainder_q[-1, 0] == 3.0
    assert start_qd.shape == (6,)


def test_authorized_rejoin_is_located_from_full_trajectory_endpoint(tmp_path):
    times = np.array([0.0, 1.0, 2.0])
    q = np.zeros((3, 6))
    q[:, 0] = [0.0, 0.1, 0.2]
    ref = trial.RecordedReference(times, q, np.zeros_like(q))
    path = tmp_path / "authorized_repair_rejoin.csv"
    trial.save_joint_waypoint_csv(path, np.array([0.0, 1.25]), np.vstack([np.zeros(6), q[-1]]))
    match = trial.locate_authorized_rejoin_on_reference(ref, path)
    assert match["index"] == 2
    assert match["time_s"] == 2.0
    assert match["max_abs_error_rad"] == 0.0


def test_future_risk_uses_q_reference_at_each_tau():
    class SurfaceModel:
        def surface_by_link(self, q, density):
            return {"link": np.array([[q[0], 0.0, 0.0]])}

    times = np.array([0.0, 1.0])
    q = np.zeros((2, 6))
    q[1, 0] = 1.0
    ref = trial.RecordedReference(times, q, np.zeros_like(q))
    sphere = trial.RiskSphere(center=np.array([1.0, 0.0, 0.0]), radius=0.1, object_id=3, tau=1.0)
    best = trial.future_reference_sphere_distance(SurfaceModel(), ref, [sphere], density="coarse")
    assert best["object_id"] == 3
    assert best["tau"] == 1.0
    assert np.isclose(best["distance"], -0.1)


def test_recorded_reference_loader_rejects_incomplete_y_span(tmp_path):
    path = tmp_path / "reference_feedback.csv"
    fields = ["t_s", *[f"q{j}_rad" for j in range(1, 7)], *[f"qd{j}_rad_s" for j in range(1, 7)], "pose_y"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i, y in enumerate((-0.393, -0.400)):
            writer.writerow({"t_s": i, **{f"q{j}_rad": 0 for j in range(1, 7)}, **{f"qd{j}_rad_s": 0 for j in range(1, 7)}, "pose_y": y})
    try:
        trial.RecordedReference.load(path)
    except ValueError as exc:
        assert "covers only" in str(exc)
    else:
        raise AssertionError("incomplete reference must be rejected")


def test_reference_completeness_requires_full_stroke():
    args = SimpleNamespace(
        y_start=0.4,
        y_goal=-0.4,
        minimum_reference_rows=3,
        reference_endpoint_tolerance_m=0.015,
        minimum_reference_span_fraction=0.95,
        nonzero_qd_threshold=1.0e-4,
        minimum_moving_rows=2,
    )
    rows = []
    for i, y in enumerate((0.4, 0.0, -0.4)):
        rows.append({"t_s": str(i), "pose_y": str(y), **{f"qd{j}_rad_s": "0.01" for j in range(1, 7)}})
    assert prepare.reference_completeness(rows, args)["accepted"]
    rows[0]["pose_y"] = "-0.39"
    result = prepare.reference_completeness(rows, args)
    assert not result["accepted"]
    assert "starts_near_reference_start" in result["reasons"]


def test_reference_locator_clamps_large_forward_jump():
    times = np.arange(20, dtype=float) * 0.04
    q = np.zeros((20, 6))
    q[:, 0] = np.arange(20)
    y = np.linspace(0.4, -0.4, 20)
    ref = trial.RecordedReference(times, q, np.zeros_like(q), y)
    audit = ref.locate(q[-1], y_actual=float(y[-1]), max_forward_step=5, joint_refine_window=2)
    assert audit["index"] == 5
    assert audit["step_was_clamped"]


def test_static_large_track_is_blocked_by_motion_not_radius_identity():
    obj = SimpleNamespace(id=9, center=np.array([0.59, -0.59, 0.36]), velocity=np.zeros(3), radius=0.203, age=8)
    cluster = SimpleNamespace(
        center=np.array([0.60, -0.59, 0.36]),
        points=np.array([[0.803, -0.59, 0.36], [0.397, -0.59, 0.36]]),
    )
    args = SimpleNamespace(
        default_obstacle_radius_m=0.055,
        dynamic_speed_window=3,
        min_track_age=3,
        min_dynamic_trigger_speed_m_s=0.08,
        dynamic_radius_min_m=0.03,
        dynamic_radius_max_m=0.10,
        max_track_cluster_association_m=0.08,
        dynamic_valid_streak_frames=2,
    )
    valid, audits = trial.update_dynamic_track_validity([obj], [cluster], {}, {}, args)
    assert valid == []
    assert not audits[9]["checks"]["speed_ok"]
    assert "radius_ok" not in audits[9]["checks"]


def test_dynamic_tracker_accepts_all_external_clusters_and_logs_legacy_radius_band():
    small = SimpleNamespace(center=np.zeros(3), points=np.array([[0.05, 0, 0], [-0.05, 0, 0]]))
    large = SimpleNamespace(center=np.zeros(3), points=np.array([[0.22, 0, 0], [-0.22, 0, 0]]))
    args = SimpleNamespace(dynamic_radius_min_m=0.03, dynamic_radius_max_m=0.10)
    selected, audits = trial.dynamic_cluster_inputs([small, large], args)
    assert selected == [small, large]
    assert [item["accepted"] for item in audits] == [True, True]
    assert [item["radius_in_legacy_band"] for item in audits] == [True, False]


def test_large_connected_component_can_be_dynamic_when_motion_checks_pass():
    args = SimpleNamespace(
        default_obstacle_radius_m=0.055, dynamic_speed_window=5,
        min_track_age=3, min_dynamic_trigger_speed_m_s=0.08,
        dynamic_exit_speed_m_s=0.04, dynamic_exit_streak_frames=3,
        max_track_cluster_association_m=0.08, dynamic_valid_streak_frames=1,
    )
    history, streak, state, low = {}, {}, {}, {}
    valid = []
    for i in range(5):
        x = 0.012 * i
        center = np.array([x, 0.0, 0.0])
        obj = SimpleNamespace(id=12, center=center, velocity=np.zeros(3), radius=0.22, age=i + 1, timestamp=i * 0.1)
        cluster = SimpleNamespace(center=center.copy(), points=np.array([[x - 0.20, 0, 0], [x + 0.20, 0, 0]]))
        valid, audits = trial.update_dynamic_track_validity([obj], [cluster], history, streak, args, state, low, i * 0.1)
    assert valid == [obj]
    assert "radius_ok" not in audits[12]["checks"]


def test_dynamic_track_audit_mode_is_non_motion_mode():
    args = trial.build_parser().parse_args(["--scene", "D1", "--mode", "dynamic-track-audit"])
    assert args.operator_phrase == ""
    assert args.dynamic_tracker_association_distance_m == 0.12
    assert args.cluster_eps == 0.05
    assert args.temporal_denoise
    assert args.moving_shadow_current_stop_m == 0.12
    assert args.guided_hard_stop_m == 0.10


def test_reference_preparation_defaults_match_formal_motion_protocol():
    args = prepare.build_parser().parse_args([])
    assert args.line_velocity_m_s == trial.FORMAL_PROTOCOL["line_velocity_m_s"]
    assert args.line_acc_m_s2 == trial.FORMAL_PROTOCOL["line_acc_m_s2"]
    assert not args.stop_after_start


def test_delayed_rejoin_calibration_is_noncommanding_by_default():
    args = delayed_calibration.build_parser().parse_args(["--repeat", "1"])
    assert not args.execute
    assert delayed_calibration.PHRASE == "CCRO_653_EMPTY_SCENE_LOCAL_DELAYED_REJOIN_APPROVED"


def test_delayed_rejoin_resume_recovery_is_noncommanding_by_default():
    args = delayed_resume.build_parser().parse_args(["--repeat", "1"])
    assert not args.execute
    assert delayed_resume.PHRASE == "CCRO_653_DELAYED_REJOIN_RESUME_APPROVED"


def test_d2_geometry_sweep_defaults_to_offline_lateral_scan():
    args = d2_sweep.build_parser().parse_args([])
    assert args.axis == "x"
    assert args.offset_min_m == -0.20
    assert args.offset_max_m == 0.20
    assert args.offset_step_m == 0.01


def test_d2_geometry_sweep_reports_contiguous_feasible_intervals():
    rows = [
        {"offset_m": 0.10, "formal_scene_feasible": False, "candidate_clearance_m": 0.08},
        {"offset_m": 0.11, "formal_scene_feasible": True, "candidate_clearance_m": 0.091},
        {"offset_m": 0.12, "formal_scene_feasible": True, "candidate_clearance_m": 0.095},
        {"offset_m": 0.13, "formal_scene_feasible": False, "candidate_clearance_m": 0.10},
        {"offset_m": 0.14, "formal_scene_feasible": True, "candidate_clearance_m": 0.092},
    ]
    intervals = d2_sweep.contiguous_intervals(rows, 0.01)
    assert len(intervals) == 2
    assert intervals[0]["sample_count"] == 2
    assert intervals[0]["midpoint_offset_m"] == pytest.approx(0.115)


def test_delayed_rejoin_calibration_allows_only_untracked_result_artifacts(monkeypatch):
    monkeypatch.setattr(
        delayed_calibration.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="?? results/new/trial/\n M planning/optimizer.py\n?? notes.txt\n",
        ),
    )
    assert delayed_calibration.execution_blocking_worktree_entries() == [
        " M planning/optimizer.py",
        "?? notes.txt",
    ]


def test_d1_and_d2_share_one_scene_independent_formal_protocol():
    calibrated_time = ["--candidate-playback-duration-s", "1.0"]
    d1 = trial.build_parser().parse_args(["--scene", "D1", "--mode", "moving-shadow-stop", *calibrated_time])
    d2 = trial.build_parser().parse_args(["--scene", "D2", "--mode", "moving-shadow-stop", *calibrated_time])
    assert trial.formal_protocol_violations(d1) == []
    assert trial.formal_protocol_violations(d2) == []
    assert trial.formal_protocol_signature(d1) == trial.formal_protocol_signature(d2)
    assert "risk_links" not in trial.SCENARIOS["D1"]
    assert "risk_links" not in trial.SCENARIOS["D2"]
    assert trial.FORMAL_PROTOCOL_ID == "653_unified_d1_d2_v2"


def test_delayed_rejoin_bridge_is_c2_at_local_tail_and_reference_target():
    q0 = np.zeros(6)
    q_tail = np.array([0.03, -0.02, 0.01, 0.0, 0.0, 0.0])
    qd_tail = np.array([0.004, -0.003, 0.002, 0.0, 0.0, 0.0])
    qdd_tail = np.array([0.001, 0.0, -0.001, 0.0, 0.0, 0.0])
    repair = NUBSTrajectory6D().generate(
        np.empty((0, 6)),
        NUBSTrajectory6D.make_boundary_state(q0),
        NUBSTrajectory6D.make_boundary_state(q_tail, qd_tail, qdd_tail),
        np.array([1.0]),
    )
    q_goal = q_tail + np.array([0.01, 0.01, -0.005, 0.0, 0.0, 0.0])
    qd_goal = np.array([0.002, 0.001, 0.0, 0.0, 0.0, 0.0])
    qdd_goal = np.zeros(6)
    bridge = trial.make_rejoin_bridge(repair, (q_goal, qd_goal, qdd_goal), 0.75)
    np.testing.assert_allclose(bridge.evaluate(0.0), q_tail, atol=1.0e-10)
    np.testing.assert_allclose(bridge.evaluate(0.0, 1), qd_tail, atol=1.0e-10)
    np.testing.assert_allclose(bridge.evaluate(0.0, 2), qdd_tail, atol=1.0e-10)
    np.testing.assert_allclose(bridge.evaluate(bridge.total_duration), q_goal, atol=1.0e-10)
    np.testing.assert_allclose(bridge.evaluate(bridge.total_duration, 1), qd_goal, atol=1.0e-10)
    np.testing.assert_allclose(bridge.evaluate(bridge.total_duration, 2), qdd_goal, atol=1.0e-10)


def test_formal_protocol_rejects_scene_specific_threshold_tuning():
    args = trial.build_parser().parse_args(
        ["--scene", "D2", "--mode", "moving-shadow-stop", "--min-dynamic-trigger-speed-m-s", "0.05"]
    )
    violations = trial.formal_protocol_violations(args)
    assert any(item.startswith("min_dynamic_trigger_speed_m_s=") for item in violations)


def test_formal_protocol_freezes_calibrated_candidate_execution_time_scale():
    args = trial.build_parser().parse_args(
        ["--scene", "D1", "--mode", "moving-shadow-stop", "--candidate-playback-duration-s", "1.0"]
    )
    assert trial.formal_protocol_violations(args) == []

    args.candidate_playback_duration_s = 0.0
    violations = trial.formal_protocol_violations(args)
    assert any(item.startswith("candidate_playback_duration_s=") for item in violations)


def test_formal_robot_protocol_keeps_lateral_warm_start_offline_only():
    args = trial.build_parser().parse_args(
        ["--scene", "D2", "--mode", "live-stop-replan-execute", "--candidate-playback-duration-s", "1.0"]
    )
    assert trial.formal_protocol_violations(args) == []
    args.fast_warm_start = "lateral"
    violations = trial.formal_protocol_violations(args)
    assert any(item.startswith("fast_warm_start=") for item in violations)


def test_formal_protocol_freezes_rolling_fast_safety_envelope():
    args = trial.build_parser().parse_args(
        ["--scene", "D2", "--mode", "live-stop-replan-execute", "--candidate-playback-duration-s", "1.0"]
    )
    assert trial.formal_protocol_violations(args) == []
    args.rolling_fast_max_s = 10.0
    assert any(item.startswith("rolling_fast_max_s=") for item in trial.formal_protocol_violations(args))


def test_rolling_multisphere_translation_preserves_rigid_shape():
    geometry = {
        "component_centers": np.array([[1.0, 2.0, 3.0], [1.2, 2.1, 3.0]]),
        "component_base_radii": np.array([0.1, 0.2]),
    }
    moved = trial.translated_multisphere_geometry(
        geometry, np.array([1.0, 2.0, 3.0]), np.array([1.3, 1.8, 3.1])
    )
    np.testing.assert_allclose(
        moved["component_centers"],
        geometry["component_centers"] + np.array([0.3, -0.2, 0.1]),
    )
    np.testing.assert_allclose(moved["component_base_radii"], geometry["component_base_radii"])


def test_nonformal_moving_trial_is_blocked_before_robot_setup(tmp_path):
    args = trial.build_parser().parse_args(
        [
            "--scene", "D1", "--repeat", "99", "--mode", "moving-shadow-stop",
            "--output", str(tmp_path), "--moving-shadow-replan-in-m", "0.18",
        ]
    )
    try:
        trial.run(args)
    except RuntimeError as exc:
        assert "formal D1/D2 protocol mismatch" in str(exc)
    else:
        raise AssertionError("nonformal moving trial must fail closed")
    summary = json.loads((tmp_path / "trials" / "D1_crossing_body_r99" / "summary.json").read_text())
    assert summary["status"] == "BLOCKED_NONFORMAL_PROTOCOL"
    assert not summary["robot_commanded"]


def test_fast_repair_past_deadline_fails_before_risk_scan():
    q0 = np.zeros(6)
    q1 = np.full(6, 0.01)
    durations = np.full(5, 0.2)
    head = NUBSTrajectory6D.make_boundary_state(q0)
    tail = NUBSTrajectory6D.make_boundary_state(q1)
    inner = NUBSTrajectory6D.linear_inner_points(q0, q1, durations)
    limits = JointLimits.from_arrays([-6] * 6, [6] * 6, [1] * 6, [2] * 6)
    result = repair_v3.run_repair_v3(
        None, None, limits, inner, head, tail, durations,
        dense_active=True, v4_mode=True, deadline_perf=time.perf_counter() - 1.0,
    )
    assert result.budget_exhausted
    assert result.accepted_steps == 0
    assert result.risk_scan_ms == 0.0
    assert any("budget exhausted before risk scan" in message for message in result.messages)


def test_elastic_tail_sensitivity_controls_position_but_keeps_tail_derivatives_fixed():
    q0 = np.zeros(6)
    q1 = np.full(6, 0.02)
    durations = np.full(2, 0.5)
    head = NUBSTrajectory6D.make_boundary_state(q0)
    tail = NUBSTrajectory6D.make_boundary_state(q1, np.full(6, 0.01), np.full(6, -0.02))
    inner = NUBSTrajectory6D.linear_inner_points(q0, q1, durations)
    sensitivity = linearization.build_local_sensitivity(
        inner, head, tail, durations, np.array([0.0, 0.5, 1.0]), elastic_tail_position=True,
    )
    assert sensitivity.variable_count == inner.size + 6
    tail_columns = sensitivity.sq[-1, :, inner.size:]
    np.testing.assert_allclose(tail_columns, np.eye(6), atol=1.0e-7)
    np.testing.assert_allclose(sensitivity.sqd[-1, :, inner.size:], 0.0, atol=1.0e-6)
    np.testing.assert_allclose(sensitivity.sqdd[-1, :, inner.size:], 0.0, atol=1.0e-5)


def test_dynamic_audit_buffers_exist_even_when_no_clusters_are_seen():
    cluster_rows, track_rows, centers, timestamp, per_track = trial.new_dynamic_audit_buffers()
    assert cluster_rows == []
    assert track_rows == []
    assert centers == []
    assert timestamp is None
    assert per_track == {}


def test_oversized_audit_cluster_saves_exact_points_and_track_metadata(tmp_path):
    points = np.array([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0], [0.10, 0.02, 0.0]])
    cluster = SimpleNamespace(center=np.mean(points, axis=0), points=points)
    audits = {7: {"associated_cluster_center": cluster.center.copy()}}
    saved = trial.save_anomalous_audit_clusters(
        tmp_path, 12, 3.5, [cluster], audits, max_bbox_m=0.20, max_radius_m=0.12
    )
    assert saved == 1
    files = list((tmp_path / "anomalous_clusters").glob("*.npz"))
    assert len(files) == 1
    with np.load(files[0], allow_pickle=False) as data:
        np.testing.assert_allclose(data["points"], points)
        assert int(data["frame"]) == 12
        assert int(data["track_id"]) == 7


def test_audit_visualization_flags_do_not_change_formal_protocol():
    base = trial.build_parser().parse_args(["--scene", "D1", "--mode", "dynamic-track-audit"])
    visual = trial.build_parser().parse_args(
        ["--scene", "D1", "--mode", "dynamic-track-audit", "--visualize-audit", "--show-filtered", "--show-noise"]
    )
    assert trial.formal_protocol_signature(base) == trial.formal_protocol_signature(visual)


def test_fresh_obstacle_fit_recovers_linear_motion_and_conservative_radius():
    velocity = np.array([0.12, -0.03, 0.01])
    samples = []
    for index, timestamp in enumerate((10.0, 10.15, 10.30, 10.45)):
        samples.append(
            {
                "timestamp": timestamp,
                "center": np.array([0.5, -0.2, 0.4]) + velocity * (timestamp - 10.0),
                "radius": 0.06 + 0.002 * index,
                "association_error_m": 0.01,
            }
        )
    result = trial.fit_fresh_obstacle_motion(samples, minimum_frames=3, minimum_span_s=0.25)
    assert result["accepted"]
    np.testing.assert_allclose(result["velocity"], velocity, atol=1.0e-10)
    assert np.isclose(result["radius"], 0.066)


def test_fresh_obstacle_fit_fails_closed_on_short_capture():
    samples = [
        {"timestamp": 1.0, "center": np.zeros(3), "radius": 0.06, "association_error_m": 0.0},
        {"timestamp": 1.1, "center": np.ones(3) * 0.01, "radius": 0.06, "association_error_m": 0.0},
    ]
    result = trial.fit_fresh_obstacle_motion(samples, minimum_frames=3, minimum_span_s=0.25)
    assert not result["accepted"]
    assert result["reason"] == "insufficient_fresh_frames"


def test_r34_fresh_bootstrap_replay():
    trigger_timestamp = 1786433371.408877
    trigger_raw_cluster_center = np.array([0.532074, -0.293245, 0.323890])
    trigger_velocity = np.array([-0.12168755184164848, -0.08369360107180769, -0.04977319265044797])
    replay = [
        (1786433371.8314953, [0.5406672963405228, -0.2780102538420974, 0.3332786570948649]),
        (1786433371.9615479, [0.5233946411604286, -0.2758505165741201, 0.3255380943953102]),
        (1786433372.0912097, [0.5019498506374160, -0.2783695640209339, 0.3276268681055270]),
        (1786433372.2214808, [0.4994716108695837, -0.27374477138165965, 0.3216477724811940]),
        (1786433372.3547300, [0.4912779928036916, -0.28198743924259667, 0.31825513114650317]),
    ]
    samples = []
    first_association = None
    for timestamp, center in replay:
        association = trial.associate_fresh_cluster(
            [np.asarray(center)],
            samples,
            timestamp=timestamp,
            trigger_cluster_center=trigger_raw_cluster_center,
            trigger_velocity=trigger_velocity,
            trigger_timestamp=trigger_timestamp,
            bootstrap_threshold_m=0.12,
            continuity_threshold_m=0.08,
        )
        assert association["associated"]
        if first_association is None:
            first_association = association
        samples.append(
            {
                "timestamp": timestamp,
                "center": np.asarray(center),
                "radius": 0.10,
                "association_error_m": association["association_error_m"],
            }
        )

    assert first_association is not None
    assert first_association["bootstrap_model"] == "stopped_or_decelerated"
    assert first_association["error_hold_m"] < first_association["error_cv_m"]
    result = trial.fit_fresh_obstacle_motion(samples, minimum_frames=3, minimum_span_s=0.25)
    assert result["accepted"]
    assert result["reason"] == "fresh_obstacle_ready"
    assert result["sample_count"] == 5


def test_clearance_only_query_matches_full_dynamic_risk_geometry():
    class SurfaceModel:
        def surface_by_link(self, q, density="medium", links=None):
            del density
            surfaces = {
                "upperArm_Link": np.array([[q[0], 0.0, 0.0], [q[0], 0.1, 0.0]]),
                "foreArm_Link": np.array([[q[0] + 0.2, 0.0, 0.0]]),
            }
            return surfaces if links is None else {name: surfaces[name] for name in links}

    evaluator = SpatioTemporalRiskEvaluator(SurfaceModel(), d_safe=0.09, d_activate=0.14)
    forecast = ConstantVelocitySphereForecast(
        np.array([0.35, 0.02, 0.0]), np.array([-0.05, 0.01, 0.0]), 0.06, 1.0, object_id=17
    )
    for tau in np.linspace(0.0, 1.0, 11):
        q = np.array([0.03 * tau, 0.0, 0.0, 0.0, 0.0, 0.0])
        full = evaluator.configuration(q, forecast, float(tau), density="medium", with_gradient=False)
        clearance = evaluator.configuration_clearance(q, forecast, float(tau), density="medium")
        assert clearance.min_distance == full.min_distance
        assert clearance.nearest_link == full.nearest_link
        assert clearance.nearest_object_id == full.nearest_object_id
        assert clearance.extrapolated == full.extrapolated


def test_pca_multisphere_covers_every_point_and_splits_elongated_cluster():
    x = np.linspace(-0.20, 0.20, 81)
    points = np.column_stack((x, 0.02 * np.sin(10 * x), 0.015 * np.cos(8 * x)))
    geometry = trial.fit_pca_multisphere(points, fit_margin_m=0.005, max_components=4)
    assert geometry["covered"]
    assert geometry["coverage_ratio"] == 1.0
    assert 2 <= geometry["component_count"] <= 4
    assert geometry["multi_sphere_max_radius"] < geometry["single_sphere_radius"]


def test_multisphere_forecast_keeps_component_count_and_shared_velocity():
    common = importlib.import_module("experiments.new.6_4.common_64")
    centers = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    velocity = np.array([0.2, -0.1, 0.0])
    forecast = common.constant_multisphere_forecast(centers, np.array([0.03, 0.04]), velocity, object_id=7)
    occupancy = forecast.occupancy_at(0.5)
    assert len(occupancy.spheres) == 2
    assert {sphere.object_id for sphere in occupancy.spheres} == {7}
    np.testing.assert_allclose(occupancy.spheres[0].center, centers[0] + 0.5 * velocity)
    np.testing.assert_allclose(occupancy.spheres[1].center, centers[1] + 0.5 * velocity)


def test_dynamic_window_speed_and_hysteresis_tolerate_one_slow_sample():
    args = SimpleNamespace(
        default_obstacle_radius_m=0.055, dynamic_speed_window=5,
        min_track_age=3, min_dynamic_trigger_speed_m_s=0.08,
        dynamic_exit_speed_m_s=0.04, dynamic_exit_streak_frames=3,
        dynamic_radius_min_m=0.03, dynamic_radius_max_m=0.10,
        max_track_cluster_association_m=0.08, dynamic_valid_streak_frames=2,
    )
    history, streak, state, low = {}, {}, {}, {}
    audits = None
    for i, x in enumerate((0.00, 0.012, 0.024, 0.028, 0.052)):
        obj = SimpleNamespace(id=4, center=np.array([x, 0.0, 0.0]), velocity=np.zeros(3), radius=0.07, age=i + 3, timestamp=i * 0.1)
        cluster = SimpleNamespace(center=obj.center.copy(), points=np.array([[x - 0.05, 0, 0], [x + 0.05, 0, 0]]))
        _, audits = trial.update_dynamic_track_validity([obj], [cluster], history, streak, args, state, low, i * 0.1)
    assert audits is not None
    assert audits[4]["window_speed_m_s"] >= 0.08
    assert audits[4]["dynamic_state"]


def test_prediction_ready_precedes_two_frame_dynamic_valid_and_uses_window_velocity():
    args = SimpleNamespace(
        default_obstacle_radius_m=0.055, dynamic_speed_window=5,
        min_track_age=3, min_dynamic_trigger_speed_m_s=0.08,
        dynamic_exit_speed_m_s=0.04, dynamic_exit_streak_frames=3,
        max_track_cluster_association_m=0.08, dynamic_valid_streak_frames=2,
    )
    history, streak, state, low = {}, {}, {}, {}
    objects = []
    audits = {}
    valid = []
    for i in range(5):
        center = np.array([0.012 * i, 0.004 * i, 0.0])
        obj = SimpleNamespace(id=6, center=center, velocity=np.zeros(3), radius=0.07, age=i + 1, timestamp=i * 0.1)
        cluster = SimpleNamespace(center=center.copy(), points=np.array([[center[0] - 0.05, center[1], 0], [center[0] + 0.05, center[1], 0]]))
        valid, audits = trial.update_dynamic_track_validity([obj], [cluster], history, streak, args, state, low, i * 0.1)
        objects = [obj]
    assert valid == []
    assert audits[6]["prediction_ready"]
    assert not audits[6]["valid"]
    ready = trial.make_prediction_ready_objects(objects, audits)
    assert len(ready) == 1
    np.testing.assert_allclose(ready[0].velocity, [0.12, 0.04, 0.0])
    np.testing.assert_allclose(objects[0].velocity, np.zeros(3))


def test_dynamic_window_accumulates_before_track_reaches_minimum_age():
    args = SimpleNamespace(
        default_obstacle_radius_m=0.055, dynamic_speed_window=5,
        min_track_age=3, min_dynamic_trigger_speed_m_s=0.08,
        dynamic_exit_speed_m_s=0.04, dynamic_exit_streak_frames=3,
        dynamic_radius_min_m=0.03, dynamic_radius_max_m=0.10,
        max_track_cluster_association_m=0.08, dynamic_valid_streak_frames=1,
    )
    history, streak, state, low = {}, {}, {}, {}
    valid = []
    for i in range(5):
        center = np.array([0.012 * i, 0.0, 0.0])
        obj = SimpleNamespace(id=8, center=center, velocity=np.zeros(3), radius=0.07, age=i + 1, timestamp=i * 0.1)
        cluster = SimpleNamespace(center=center.copy(), points=np.array([[center[0] - 0.05, 0, 0], [center[0] + 0.05, 0, 0]]))
        valid, _ = trial.update_dynamic_track_validity([obj], [cluster], history, streak, args, state, low, i * 0.1)
    assert valid == [obj]


def test_track_geometry_separates_cluster_tracked_and_risk_radius():
    obj = SimpleNamespace(id=2, center=np.zeros(3), velocity=np.zeros(3), radius=0.09)
    cluster = SimpleNamespace(center=np.zeros(3), points=np.array([[-0.05, 0, 0], [0.05, 0, 0]]))
    geometry = trial.track_geometry(obj, [cluster], 0.055)
    assert np.isclose(geometry["raw_radius"], 0.05)
    assert np.isclose(geometry["track_radius"], 0.09)
    assert np.isclose(geometry["inflated_radius"], 0.09)


def test_time_scaled_trajectory_preserves_geometry_and_scales_derivatives():
    q0 = np.zeros(6)
    q1 = np.array([0.03, -0.02, 0.01, 0.0, 0.0, 0.0])
    source = NUBSTrajectory6D().generate(
        np.empty((0, 6)),
        NUBSTrajectory6D.make_boundary_state(q0),
        NUBSTrajectory6D.make_boundary_state(q1),
        np.array([1.0]),
    )
    scaled = trial.TimeScaledTrajectory6D(source, 2.0)
    native = source.sample(np.array([0.0, 0.5, 1.0]))
    stretched = scaled.sample(np.array([0.0, 1.0, 2.0]))
    np.testing.assert_allclose(stretched.q, native.q)
    np.testing.assert_allclose(stretched.qd, native.qd / 2.0)
    np.testing.assert_allclose(stretched.qdd, native.qdd / 4.0)
    assert scaled.total_duration == 2.0


def test_saved_nubs_reconstruction_matches_saved_candidate(tmp_path):
    q0 = np.zeros(6)
    q1 = np.array([0.04, -0.03, 0.02, 0.0, 0.0, 0.0])
    durations = np.full(5, 0.2)
    source = NUBSTrajectory6D().generate(
        NUBSTrajectory6D.linear_inner_points(q0, q1, durations),
        NUBSTrajectory6D.make_boundary_state(q0),
        NUBSTrajectory6D.make_boundary_state(q1),
        durations,
    )
    path = tmp_path / "candidate.csv"
    trial.save_trajectory_csv(path, source, dt=0.01)
    reconstructed = trial.reconstruct_saved_nubs_candidate(path, segments=5)
    times = np.linspace(0.0, 1.0, 51)
    np.testing.assert_allclose(reconstructed.sample(times).q, source.sample(times).q, atol=1.0e-8)
    np.testing.assert_allclose(reconstructed.sample(times).qd, source.sample(times).qd, atol=1.0e-7)


def test_guarded_candidate_wait_stops_on_existing_hard_guard(monkeypatch):
    class Robot:
        def __init__(self):
            self.stopped = False

        def get_joint(self):
            return np.zeros(6)

        def move_stop(self, *args):
            self.stopped = True
            return 0

    robot = Robot()
    monkeypatch.setattr(trial, "execution_hard_guard_distance", lambda processor, denoiser, args: 0.05)
    result, samples = trial.wait_for_candidate_goal_guarded(
        robot,
        np.ones(6),
        processor=object(),
        denoiser=None,
        args=SimpleNamespace(guided_hard_stop_m=0.10),
        goal_tolerance_rad=0.01,
        min_execution_wait_s=0.0,
        motion_timeout_s=1.0,
        poll_s=0.0,
        min_motion_rad=0.001,
    )
    assert result["guard_stopped"]
    assert result["hard_guard_distance_m"] == 0.05
    assert robot.stopped
    assert len(samples) == 1


def test_executor_rejects_authorized_csv_playback_time_mismatch(tmp_path):
    q0 = np.zeros(6)
    q1 = np.ones(6) * 0.01
    trajectory = NUBSTrajectory6D().generate(
        np.empty((0, 6)),
        NUBSTrajectory6D.make_boundary_state(q0),
        NUBSTrajectory6D.make_boundary_state(q1),
        np.array([1.0]),
    )
    path = tmp_path / "authorized_local_repair.csv"
    trial.save_trajectory_csv(path, trajectory, dt=0.01)
    with pytest.raises(RuntimeError, match="time axis does not match"):
        trial.execute_fast_candidate_offline_track(
            object(),
            path,
            SimpleNamespace(candidate_playback_duration_s=6.0),
            processor=object(),
            denoiser=None,
        )


def test_generic_executor_accepts_native_full_trajectory_time_before_robot_check(tmp_path):
    q0 = np.zeros(6)
    q1 = np.ones(6) * 0.01
    trajectory = NUBSTrajectory6D().generate(
        np.empty((0, 6)),
        NUBSTrajectory6D.make_boundary_state(q0),
        NUBSTrajectory6D.make_boundary_state(q1),
        np.array([1.25]),
    )
    path = tmp_path / "authorized_repair_rejoin.csv"
    trial.save_trajectory_csv(path, trajectory, dt=0.01)
    args = SimpleNamespace(
        candidate_controller_waypoint_period_s=0.005,
        candidate_max_waypoints=0,
        candidate_min_execution_wait_s=0.0,
        candidate_joint_velc=0.006,
        candidate_joint_acc=0.012,
    )
    with pytest.raises(RuntimeError, match="does not expose offline_track_execute_joints"):
        trial.execute_authorized_trajectory_offline_track(
            object(), path, args, processor=object(), denoiser=None, playback_duration_s=None
        )


def test_fresh3_reference_resume_requires_current_future_and_hard_guard(monkeypatch):
    class Evaluator:
        def configuration(self, q, forecast, tau, density, with_gradient):
            return SimpleNamespace(min_distance=0.16, nearest_link="upperArm_Link")

    monkeypatch.setattr(trial, "make_risk_stack", lambda *a, **k: (Evaluator(), None, None))
    monkeypatch.setattr(trial, "constant_multisphere_forecast", lambda *a, **k: object())
    args = SimpleNamespace(
        prediction_horizon_s=0.5,
        prediction_step_s=0.1,
        guided_hard_stop_m=0.10,
        moving_shadow_current_stop_m=0.12,
        moving_shadow_replan_in_m=0.14,
    )
    fresh = {"accepted": True, "velocity": [0.1, 0.0, 0.0]}
    geometry = {"component_centers": [[0.5, 0.0, 0.5]], "component_base_radii": [0.05]}
    times = np.array([0.0, 1.0])
    qs = np.zeros((2, 6))
    accepted = trial.authorize_reference_resume_after_fresh3(
        args, {}, object(), fresh3=fresh, fresh3_geometry=geometry,
        remainder_times=times, remainder_q=qs, hard_guard_distance_m=0.20,
    )
    assert accepted["authorized"]

    held = trial.authorize_reference_resume_after_fresh3(
        args, {}, object(), fresh3=fresh, fresh3_geometry=geometry,
        remainder_times=times, remainder_q=qs, hard_guard_distance_m=0.08,
    )
    assert not held["authorized"]
    assert not held["checks"]["hard_guard_safe"]


def test_fresh3_scene_clear_accepts_three_valid_unassociated_safe_frames():
    class SurfaceModel:
        def surface_by_link(self, q, density):
            return {"link": np.array([[0.0, 0.0, 0.0]])}

    args = SimpleNamespace(
        post_stop_recheck_min_frames=3,
        prediction_horizon_s=0.5,
        prediction_step_s=0.1,
        prediction_margin_m=0.035,
        prediction_uncertainty_m=0.02,
        moving_shadow_current_stop_m=0.12,
        moving_shadow_replan_in_m=0.14,
        guided_hard_stop_m=0.10,
    )
    frames = [
        {
            "timestamp": float(i), "frame_valid": True, "associated": False,
            "raw_guard_distance_m": float("inf"), "all_external_clusters": [],
        }
        for i in range(3)
    ]
    result = trial.authorize_fresh3_scene_clear(
        args, SurfaceModel(), fresh3_frames=frames,
        remainder_times=np.array([0.0, 1.0]), remainder_q=np.zeros((2, 6)),
    )
    assert result["accepted"]
    assert result["status"] == "FRESH3_SCENE_CLEAR"


def test_fresh3_scene_clear_rejects_unassociated_but_unsafe_external_cluster():
    class SurfaceModel:
        def surface_by_link(self, q, density):
            return {"link": np.array([[0.0, 0.0, 0.0]])}

    args = SimpleNamespace(
        post_stop_recheck_min_frames=3,
        prediction_horizon_s=0.5,
        prediction_step_s=0.1,
        prediction_margin_m=0.035,
        prediction_uncertainty_m=0.02,
        moving_shadow_current_stop_m=0.12,
        moving_shadow_replan_in_m=0.14,
        guided_hard_stop_m=0.10,
    )
    frames = [
        {
            "timestamp": float(i), "frame_valid": True, "associated": False,
            "raw_guard_distance_m": 0.20,
            "all_external_clusters": [{"center": [0.10, 0.0, 0.0], "radius_m": 0.02}],
        }
        for i in range(3)
    ]
    result = trial.authorize_fresh3_scene_clear(
        args, SurfaceModel(), fresh3_frames=frames,
        remainder_times=np.array([0.0, 1.0]), remainder_q=np.zeros((2, 6)),
    )
    assert not result["accepted"]
    assert not result["frames"][-1]["checks"]["remaining_reference_0p5s_safe"]


def test_candidate_tracking_metrics_use_authorized_time_axis():
    command_times = np.array([0.0, 0.5, 1.0])
    command_q = np.zeros((3, 6))
    command_q[:, 0] = [0.0, 0.01, 0.02]
    feedback = [
        {"t_s": 0.0, "actual_joint_rad": [0.0] * 6, "max_motion_from_start_rad": 0.0},
        {"t_s": 0.5, "actual_joint_rad": [0.01, 0, 0, 0, 0, 0], "max_motion_from_start_rad": 0.01},
        {"t_s": 1.0, "actual_joint_rad": [0.02, 0, 0, 0, 0, 0], "max_motion_from_start_rad": 0.02},
    ]
    metrics = trial.candidate_tracking_metrics(command_times, command_q, feedback, minimum_motion_rad=0.003)
    assert metrics["requested_duration_s"] == 1.0
    assert metrics["command_to_last_feedback_duration_s"] == 1.0
    assert metrics["observed_motion_duration_s"] == 0.5
    assert metrics["duration_error_s"] == 0.0
    assert metrics["tracking_rmse_rad"] == 0.0
    assert metrics["tracking_max_error_rad"] == 0.0


def test_full_rejoin_authorization_rejects_failed_fast_before_verification(tmp_path):
    result = trial.authorize_candidate_execution(
        SimpleNamespace(),
        {},
        None,
        local_repair_ready=False,
        local_artifacts={},
        fresh_geometry={},
        fresh_velocity=np.zeros(3),
        rejoin_goals=[],
        trial_dir=tmp_path,
    )
    assert result["status"] == "NOT_ELIGIBLE_FAST_REPAIR_FAILED"
    assert not result["execution_authorized"]
    assert result["reason"] == "local_repair_not_ready"
    saved = json.loads((tmp_path / "post_plan_authorization" / "authorization_summary.json").read_text())
    assert not saved["execution_authorized"]


def test_raw_cluster_distance_does_not_require_a_tracked_object():
    robot_points = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    cluster = SimpleNamespace(
        center=np.array([0.2, 0.0, 0.0]),
        points=np.array([[0.2, 0.0, 0.0], [0.3, 0.0, 0.0]]),
    )
    distance, obj, obj_id, robot_point, obstacle_point = (
        trial._find_nearest_cluster_distance_detail(robot_points, [cluster], [])
    )
    assert distance == pytest.approx(0.2)
    assert obj is None
    assert obj_id is None
    np.testing.assert_allclose(robot_point, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(obstacle_point, [0.2, 0.0, 0.0])


def test_authorized_timing_check_rejects_compressed_waypoint_playback():
    feedback = [
        {"t_s": 0.0, "goal_max_abs_error_rad": 1.0},
        {"t_s": 3.8, "goal_max_abs_error_rad": 0.001},
        {"t_s": 24.0, "goal_max_abs_error_rad": 0.0},
    ]
    result = trial.authorized_execution_timing_check(
        26.8, feedback, valid_completion_time_s=3.8, goal_tolerance_rad=0.012
    )
    assert not result["accepted"]
    assert result["actual_to_requested_ratio"] < 0.2


def test_authorized_timing_check_accepts_calibrated_repair_duration():
    feedback = [
        {"t_s": 0.0, "goal_max_abs_error_rad": 1.0},
        {"t_s": 1.19, "goal_max_abs_error_rad": 0.004},
    ]
    result = trial.authorized_execution_timing_check(
        1.25, feedback, valid_completion_time_s=1.19, goal_tolerance_rad=0.012
    )
    assert result["accepted"]


def test_r05_short_bridge_uses_valid_completion_not_first_tolerance_entry():
    feedback = [
        {"t_s": 0.0005, "goal_max_abs_error_rad": 0.015689},
        {"t_s": 0.17789388, "goal_max_abs_error_rad": 0.010851},
        {"t_s": 0.3491, "goal_max_abs_error_rad": 0.005437},
        {"t_s": 0.522888058, "goal_max_abs_error_rad": 0.000149},
    ]
    result = trial.authorized_execution_timing_check(
        0.5,
        feedback,
        valid_completion_time_s=0.522888058,
        goal_tolerance_rad=0.012,
    )
    assert result["accepted"]
    assert result["first_goal_tolerance_time_s"] == pytest.approx(0.17789388)
    assert result["valid_completion_time_s"] == pytest.approx(0.522888058)
    assert result["completion_to_requested_ratio"] == pytest.approx(1.045776116)


def test_empty_scene_calibration_defaults_to_noncommanding_dry_run(tmp_path):
    args = calibration.build_parser().parse_args(
        ["--playback-duration-s", "1.0", "--repeat", "1", "--output", str(tmp_path)]
    )
    result = calibration.run(args)
    assert result["status"] == "DRY_RUN_ONLY"
    assert not result["robot_commanded"]


def test_hold_resume_defaults_to_noncommanding_r04_dry_run(tmp_path):
    args = hold_resume.build_parser().parse_args(["--output", str(tmp_path)])
    result = hold_resume.run(args)
    assert result["status"] == "DRY_RUN_READY"
    assert not result["robot_commanded"]
    assert result["source_status"] == "TRIGGERED_AND_REPAIR_REJOIN_EXECUTED_HOLD"
    assert result["source_resume_status"] == "REFERENCE_RESUME_HOLD"


def test_hold_resume_freezes_rejoin_state_tolerance_at_0p02_rad():
    assert hold_resume.REJOIN_STATE_TOLERANCE_RAD == pytest.approx(0.020)


def test_rolling_local_policy_preempts_full_first_only_when_enabled():
    assert trial.select_dynamic_execution_path(
        local_authorized=True, full_authorized=True, rolling_local_enabled=False
    ) == "FULL_FIRST"
    assert trial.select_dynamic_execution_path(
        local_authorized=True, full_authorized=True, rolling_local_enabled=True
    ) == "ROLLING_LOCAL_FIRST"
    assert trial.select_dynamic_execution_path(
        local_authorized=False, full_authorized=True, rolling_local_enabled=True
    ) is None


def test_avoidance_side_lock_rejects_material_reversal_but_allows_refinement():
    first = np.array([0.01, 0, 0, 0, 0, 0])
    initialized = trial.avoidance_side_consistent(
        None, first, opposite_projection_tolerance_rad=0.002
    )
    assert initialized["accepted"]
    assert trial.avoidance_side_consistent(
        initialized["locked_tail_delta_q"],
        np.array([0.004, 0.005, 0, 0, 0, 0]),
        opposite_projection_tolerance_rad=0.002,
    )["accepted"]
    reversed_result = trial.avoidance_side_consistent(
        initialized["locked_tail_delta_q"],
        np.array([-0.008, 0, 0, 0, 0, 0]),
        opposite_projection_tolerance_rad=0.002,
    )
    assert not reversed_result["accepted"]
    assert reversed_result["reason"] == "opposite_avoidance_side"


def test_rolling_local_reference_schedule_advances_from_each_real_segment():
    schedule = trial.rolling_local_reference_schedule(
        12.0, local_horizon_s=1.0, max_segments=3, reference_end_time_s=40.0
    )
    assert [row["reference_plan_start_time_s"] for row in schedule] == [12.0, 13.0, 14.0]
    assert [row["reference_goal_time_s"] for row in schedule] == [13.0, 14.0, 15.0]


def test_recorded_reference_absolute_state_does_not_mutate_online_index():
    times = np.array([0.0, 1.0, 2.0])
    q = np.repeat(times[:, None], 6, axis=1)
    reference = trial.RecordedReference(times, q, np.ones_like(q))
    reference.index = 1
    state = reference.state_at(0.5)
    np.testing.assert_allclose(state[0], 0.5)
    assert reference.index == 1


def test_rolling_local_segment_gate_does_not_repair_a_safe_reference():
    result = trial.rolling_local_segment_gate(
        reference_min_distance_m=0.427,
        local_repair_ready=True,
        side_consistent=True,
        fresh_authorized=True,
        replan_threshold_m=0.14,
    )
    assert not result["advance"]
    assert result["status"] == "REFERENCE_SAFE_FOR_REJOIN"


def test_rolling_local_segment_gate_requires_all_authorizations_when_risky():
    accepted = trial.rolling_local_segment_gate(
        reference_min_distance_m=0.08,
        local_repair_ready=True,
        side_consistent=True,
        fresh_authorized=True,
        replan_threshold_m=0.14,
    )
    assert accepted["advance"]
    rejected = trial.rolling_local_segment_gate(
        reference_min_distance_m=0.08,
        local_repair_ready=True,
        side_consistent=False,
        fresh_authorized=True,
        replan_threshold_m=0.14,
    )
    assert not rejected["advance"]


def test_rolling_virtual_shadow_parser_has_no_execution_option():
    parser = rolling_virtual.build_parser()
    destinations = {action.dest for action in parser._actions}
    assert "execute" not in destinations
    assert "allow_live_candidate_execution" not in destinations
    args = parser.parse_args(["--repeat", "1"])
    assert args.seed_timeout_s == pytest.approx(8.0)
    assert args.task_geometry_id == "D2C_COMPACT_TABLETOP_XP10"
    assert args.obstacle_nominal_size_m == "0.10,0.10,0.10"
    assert args.obstacle_motion_mode == "dynamic"


def test_rolling_virtual_retries_same_segment_after_valid_fresh_rejection():
    action = rolling_virtual.retry_action(
        {"advance": False, "status": "ROLLING_LOCAL_SEGMENT_REJECTED"},
        fresh_accepted=True,
        has_points=True,
        has_geometry=True,
    )
    assert action == "retry_same_segment"


def test_rolling_virtual_retry_transitions_remain_fail_closed():
    assert rolling_virtual.retry_action(
        {"advance": True, "status": "ROLLING_LOCAL_SEGMENT_AUTHORIZED"},
        fresh_accepted=True,
        has_points=True,
        has_geometry=True,
    ) == "advance"
    assert rolling_virtual.retry_action(
        {"advance": False, "status": "REFERENCE_SAFE_FOR_REJOIN"},
        fresh_accepted=True,
        has_points=True,
        has_geometry=True,
    ) == "reference_safe"
    assert rolling_virtual.retry_action(
        {"advance": False, "status": "ROLLING_LOCAL_SEGMENT_REJECTED"},
        fresh_accepted=False,
        has_points=False,
        has_geometry=False,
    ) == "safe_hold"


def test_geometry_quality_audit_never_changes_planning_geometry():
    compact = rolling_virtual.geometry_quality_audit(
        {"axial_length_m": 0.10, "component_base_radii": [0.06, 0.07]},
        axial_limit_m=0.16,
        component_radius_limit_m=0.12,
    )
    elongated = rolling_virtual.geometry_quality_audit(
        {"axial_length_m": 0.22, "component_base_radii": [0.13]},
        axial_limit_m=0.16,
        component_radius_limit_m=0.12,
    )
    assert compact["compact_scene_quality_ok"]
    assert compact["planning_geometry_unchanged"]
    assert not elongated["compact_scene_quality_ok"]
    assert elongated["planning_geometry_unchanged"]


def test_static_obstacle_uses_zero_velocity_without_losing_measurement():
    fresh = {
        "accepted": True,
        "velocity": [0.02, -0.01, 0.03],
        "speed_m_s": 0.04,
        "center": [1.0, 2.0, 3.0],
    }
    static = rolling_virtual.obstacle_state_for_mode(fresh, "static")
    assert static["velocity"] == [0.0, 0.0, 0.0]
    assert static["speed_m_s"] == 0.0
    assert static["measured_velocity"] == fresh["velocity"]
    assert static["measured_speed_m_s"] == fresh["speed_m_s"]
    assert static["center"] == fresh["center"]
    assert fresh["velocity"] != static["velocity"]


def test_virtual_shadow_has_no_live_execution_or_raw_guard_claim():
    source = inspect.getsource(rolling_virtual.run)
    assert "raw_hard_guard_applicable\": False" in source
    assert "physical_robot_remains_at_start_in_virtual_shadow" in source
    assert "hard_guard_distance_m=math.inf" in source


def test_point_obb_signed_distance_is_exact_for_axis_aligned_box():
    points = np.asarray(
        [
            [2.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [2.0, 3.0, 0.0],
        ]
    )
    distances = static_geometry_ab.point_obb_signed_distance(
        points,
        np.zeros(3),
        np.eye(3),
        np.ones(3),
    )
    np.testing.assert_allclose(distances, [1.0, 0.0, -1.0, np.sqrt(5.0)])


def test_static_geometry_ab_parser_is_offline_only():
    parser = static_geometry_ab.build_parser()
    destinations = {action.dest for action in parser._actions}
    assert "execute" not in destinations
    assert "allow_live_candidate_execution" not in destinations


def test_static_distance_ledger_parser_is_offline_only():
    parser = static_distance_ledger.build_parser()
    destinations = {action.dest for action in parser._actions}
    assert "execute" not in destinations
    assert "allow_live_candidate_execution" not in destinations


def test_static20_closure_parser_is_offline_only():
    parser = static20_closure.build_parser()
    destinations = {action.dest for action in parser._actions}
    assert "execute" not in destinations
    assert "allow_live_candidate_execution" not in destinations


def test_static20_forecast_is_time_invariant_and_adds_only_observation_inflation():
    geometry = {
        "component_centers": [[1.0, 2.0, 3.0]],
        "component_base_radii": [0.08],
    }
    forecast = static20_closure._static20_forecast(geometry, 0.02, 5.0)
    at_zero = forecast.occupancy_at(0.0).spheres[0]
    at_later = forecast.occupancy_at(4.0).spheres[0]
    np.testing.assert_allclose(at_zero.center, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(at_later.center, at_zero.center)
    assert np.isclose(at_zero.radius, 0.10)
    assert np.isclose(at_later.radius, 0.10)


def test_static20_safe_suffix_requires_every_later_sample_to_pass():
    rows = [
        {"distance_m": 0.11},
        {"distance_m": 0.08},
        {"distance_m": 0.095},
        {"distance_m": 0.10},
    ]
    result, earliest = static20_closure._safe_suffix(rows, 0.09)
    assert [row["safe_suffix"] for row in result] == [False, False, True, True]
    assert earliest is result[2]
    assert np.isclose(earliest["suffix_min_distance_m"], 0.095)


def test_offset_preserving_rollout_parser_is_offline_only():
    parser = offset_rollout.build_parser()
    destinations = {action.dest for action in parser._actions}
    assert "execute" not in destinations
    assert "allow_live_candidate_execution" not in destinations


def test_transported_goal_preserves_joint_offset_and_reference_increment():
    times = np.asarray([0.0, 1.0, 2.0])
    q = np.zeros((3, 6))
    q[:, 0] = [0.0, 0.2, 0.5]
    q[:, 1] = [0.0, -0.1, -0.3]
    qd = np.zeros_like(q)
    reference = trial.RecordedReference(times, q, qd)
    offset = np.asarray([0.04, 0.02, 0.0, 0.0, 0.0, 0.0])
    q_now = q[1] + offset
    goal, audit = offset_rollout.transported_reference_goal(reference, q_now, 1.0, 1.0)
    expected_increment = q[2] - q[1]
    np.testing.assert_allclose(goal[0], q_now + expected_increment)
    np.testing.assert_allclose(goal[0] - q[2], offset)
    np.testing.assert_allclose(audit["offset_from_reference_at_goal_rad"], offset)


def test_bounded_terminal_goal_advances_toward_exact_goal_without_overshoot():
    q_now = np.zeros(6)
    q_final = np.asarray([0.09, -0.03, 0.015, 0.0, 0.0, 0.0])
    goal, audit = offset_rollout.bounded_terminal_goal(
        q_now, q_final, max_step_rad=0.03
    )
    assert np.isclose(np.max(np.abs(goal[0] - q_now)), 0.03)
    np.testing.assert_allclose(goal[0], q_final / 3.0)
    assert np.isclose(audit["terminal_step_scale"], 1.0 / 3.0)


def test_bounded_terminal_goal_uses_exact_goal_inside_step_bound():
    q_now = np.zeros(6)
    q_final = np.asarray([0.01, -0.02, 0.0, 0.0, 0.0, 0.0])
    goal, audit = offset_rollout.bounded_terminal_goal(
        q_now, q_final, max_step_rad=0.03
    )
    np.testing.assert_allclose(goal[0], q_final)
    assert np.isclose(audit["terminal_step_scale"], 1.0)


def test_static20_goal_directed_shadow_parser_has_no_execution_switch():
    destinations = {action.dest for action in static20_shadow.build_parser()._actions}
    assert "execute" not in destinations
    assert "allow_live_candidate_execution" not in destinations
    assert "max_segments" not in destinations


def test_forecast_override_gate_allows_only_offline_or_static_shadow():
    offline = SimpleNamespace(mode="live-stop-replan-execute")
    shadow = SimpleNamespace(mode="shadow")
    live = SimpleNamespace(mode="live-stop-replan-execute")
    moving = SimpleNamespace(mode="moving-shadow-stop")
    assert trial.forecast_override_authorized(
        offline, {"offline_forecast_override_authorized": True}
    )
    assert trial.forecast_override_authorized(
        shadow, {"static20_shadow_forecast_override_authorized": True}
    )
    assert not trial.forecast_override_authorized(
        live, {"static20_shadow_forecast_override_authorized": True}
    )
    assert not trial.forecast_override_authorized(
        moving, {"static20_shadow_forecast_override_authorized": True}
    )
    assert not trial.forecast_override_authorized(shadow, {})


def test_shared_static20_forecast_is_stationary_and_has_twenty_mm_inflation():
    forecast = rolling_common.make_static20_forecast(
        {"component_centers": [[0.1, 0.2, 0.3]], "component_base_radii": [0.08]},
        observation_inflation_m=0.02,
        valid_horizon_s=2.0,
    )
    first = forecast.occupancy_at(0.0).spheres[0]
    later = forecast.occupancy_at(1.5).spheres[0]
    np.testing.assert_allclose(first.center, later.center)
    assert np.isclose(first.radius, 0.10)
    assert np.isclose(later.radius, 0.10)


def test_terminal_side_lock_releases_only_after_complete_verifier_pass():
    assert rolling_common.terminal_side_release_allowed("terminal_goal", True)
    assert not rolling_common.terminal_side_release_allowed("terminal_goal", False)
    assert not rolling_common.terminal_side_release_allowed("reference_transport", True)


def test_static20_shadow_wall_budget_starts_from_rolling_epoch():
    # A long operator setup before this epoch is deliberately irrelevant.
    rolling_started = 300.0
    assert not static20_shadow.rolling_wall_expired(rolling_started, 240.0, now=300.1)
    assert not static20_shadow.rolling_wall_expired(rolling_started, 240.0, now=539.9)
    assert static20_shadow.rolling_wall_expired(rolling_started, 240.0, now=540.0)


def test_simple_dynamic_nubs_parser_is_shadow_only():
    parser = simple_dynamic.build_parser()
    parsed = parser.parse_args(["--repeat", "1"])
    assert parsed.mode == "shadow"
    assert parsed.trigger_timeout_s == 15.0
    assert parsed.max_demo_speed_m_s == 0.12
    destinations = {action.dest for action in parser._actions}
    assert "execute" not in destinations
    assert "allow_live_candidate_execution" not in destinations
    mode_action = next(action for action in parser._actions if action.dest == "mode")
    assert tuple(mode_action.choices) == ("shadow",)


def test_simple_bypass_side_is_orthogonal_to_task_and_points_away():
    task, side, _ = simple_bypass.task_and_side_directions(
        np.zeros(3),
        np.asarray([0.0, 1.0, 0.0]),
        np.asarray([1.0, 0.0, 0.0]),
        np.zeros(3),
    )
    np.testing.assert_allclose(task, [0.0, 1.0, 0.0])
    np.testing.assert_allclose(side, [1.0, 0.0, 0.0])
    assert abs(float(np.dot(task, side))) < 1.0e-12


def test_simple_bypass_generates_six_bounded_joint_goals():
    class Model:
        @staticmethod
        def point_jacobian(q, link, point):
            return np.hstack([np.eye(3), np.zeros((3, 3))])

    rows, _ = simple_bypass.bypass_goal_candidates(
        Model(),
        np.zeros(6),
        tcp_position=np.zeros(3),
        goal_position=np.asarray([0.0, 1.0, 0.0]),
        risk_position=np.asarray([1.0, 0.0, 0.0]),
        predicted_obstacle_position=np.zeros(3),
        forward_m=0.05,
        side_lengths_m=(0.04, 0.06, 0.08),
        max_joint_delta_rad=0.12,
    )
    assert len(rows) == 6
    assert {row["side_sign"] for row in rows} == {-1, 1}
    assert {row["side_m"] for row in rows} == {0.04, 0.06, 0.08}
    assert all(np.max(np.abs(row["q_goal"])) <= 0.12 + 1.0e-12 for row in rows)


def test_risk_link_bypass_generates_three_away_only_goals():
    class Urdf:
        @staticmethod
        def link_transforms(joints):
            return {"left_link": np.eye(4), "gripper_base_link": np.eye(4)}

    class Model:
        joint_names = tuple(f"j{index}" for index in range(6))
        urdf = Urdf()

        @staticmethod
        def point_jacobian(q, link, point):
            if link == "left_link":
                return np.hstack([np.eye(3), np.zeros((3, 3))])
            jacobian = np.zeros((3, 6))
            jacobian[1, 3] = 1.0
            return jacobian

    rows, audit = simple_bypass.risk_link_bypass_goal_candidates(
        Model(),
        np.zeros(6),
        tcp_position=np.zeros(3),
        goal_position=np.asarray([0.0, 1.0, 0.0]),
        risk_link="left_link",
        risk_position=np.asarray([1.0, 0.0, 0.0]),
        predicted_obstacle_position=np.zeros(3),
        forward_m=0.05,
        side_lengths_m=(0.04, 0.06, 0.08),
        max_joint_delta_rad=0.12,
    )
    assert len(rows) == 3
    assert {row["side_sign"] for row in rows} == {1}
    assert [row["side_m"] for row in rows] == [0.04, 0.06, 0.08]
    assert audit["risk_link"] == "left_link"
    assert audit["candidate_side_policy"] == "away_only"
    assert all(row["mapping"]["linearized_risk_delta_m"][0] > 0.0 for row in rows)
    assert all(row["mapping"]["linearized_task_progress_m"] > 0.0 for row in rows)


def test_fixed_pca_two_sphere_has_two_components_and_full_coverage():
    points = np.asarray(
        [
            [-0.10, -0.01, 0.0],
            [-0.08, 0.01, 0.0],
            [-0.04, -0.01, 0.0],
            [-0.02, 0.01, 0.0],
            [0.02, -0.01, 0.0],
            [0.04, 0.01, 0.0],
            [0.08, -0.01, 0.0],
            [0.10, 0.01, 0.0],
        ]
    )
    geometry = simple_dynamic.fit_fixed_pca_two_sphere(points)
    assert geometry["component_count"] == 2
    assert geometry["fit_policy"] == "fixed_pca_two_sphere"
    assert geometry["covered"]
    assert geometry["coverage_ratio"] == 1.0


def test_robust_candidate_gate_rejects_sub_target_rows():
    rows = [
        {"candidate": 1, "task_progress_ok": True, "coarse_min_distance_m": 0.109},
        {"candidate": 2, "task_progress_ok": True, "coarse_min_distance_m": 0.071},
        {"candidate": 3, "task_progress_ok": False, "coarse_min_distance_m": 0.20},
    ]
    assert simple_dynamic.select_robust_candidate(rows, 0.11) is None
    rows[0]["coarse_min_distance_m"] = 0.111
    assert simple_dynamic.select_robust_candidate(rows, 0.11)["candidate"] == 1


def test_post_trigger_speed_uses_exit_hysteresis_not_enter_threshold():
    assert simple_dynamic.post_trigger_speed_in_hold_domain(0.07746, 0.04, 0.12)
    assert not simple_dynamic.post_trigger_speed_in_hold_domain(0.039, 0.04, 0.12)
    assert not simple_dynamic.post_trigger_speed_in_hold_domain(0.121, 0.04, 0.12)


def test_simple_dynamic_summary_allows_missing_fresh_verification():
    result = {
        "status": "SIMPLE_DYNAMIC_NUBS_FAST_HOLD",
        "bypass_generation": {"selected_coarse_clearance_m": 0.047},
        "fresh_candidate_verification": None,
    }
    bypass_summary = result.get("bypass_generation") or {}
    fresh_summary = result.get("fresh_candidate_verification") or {}
    fresh_verification = fresh_summary.get("verification") or {}
    assert bypass_summary["selected_coarse_clearance_m"] == 0.047
    assert fresh_verification.get("min_distance") is None


def test_point_obb_distance_reports_nearest_surface_point():
    distances, nearest = static_distance_ledger.point_obb_signed_distance_and_nearest(
        np.asarray([[2.0, 0.5, 0.0], [0.0, 0.0, 0.0]]),
        np.zeros(3),
        np.eye(3),
        np.ones(3),
    )
    np.testing.assert_allclose(distances, [1.0, -1.0])
    np.testing.assert_allclose(nearest[0], [1.0, 0.5, 0.0])
    assert np.isclose(np.linalg.norm(nearest[1]), 1.0)


def test_authorized_start_alignment_uses_recorded_reference_in_reverse():
    reference = np.zeros((6, 6))
    reference[:, 0] = np.linspace(0.0, 0.05, 6)
    actual = reference[5] + np.array([1.0e-4, 0, 0, 0, 0, 0])
    target = reference[2] + np.array([-1.0e-4, 0, 0, 0, 0, 0])
    segment, audit = alignment.matched_reference_segment(
        reference, actual, target, match_tolerance_rad=0.001
    )
    assert audit["current_reference_index"] == 5
    assert audit["target_reference_index"] == 2
    assert audit["reference_direction"] == "reverse"
    np.testing.assert_allclose(segment[0], actual)
    np.testing.assert_allclose(segment[-1], target)
    assert len(segment) == 4


def test_v2_alignment_defaults_to_formal_d1_local_candidate():
    args = alignment.build_parser().parse_args(["--repeat", "1"])
    assert "dynamic_repair_formal/trials/D1_crossing_body_r01" in str(args.trajectory_csv)
    assert not args.execute


def test_alignment_allows_only_untracked_result_artifacts(monkeypatch):
    monkeypatch.setattr(
        alignment.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="?? results/new/alignment/\n M config/ccro_stage4.yaml\n",
        ),
    )
    assert alignment.execution_blocking_worktree_entries() == [" M config/ccro_stage4.yaml"]


def test_candidate_return_reverses_only_authorized_waypoint_geometry():
    times = np.array([0.0, 0.5, 1.0])
    qs = np.zeros((3, 6))
    qs[:, 0] = [0.0, 0.02, 0.05]
    command_times, command_q = candidate_return.reverse_authorized_waypoints(
        times, qs, return_duration_s=1.0, controller_period_s=0.5
    )
    np.testing.assert_allclose(command_times, [0.0, 0.5, 1.0])
    np.testing.assert_allclose(command_q[0], qs[-1])
    np.testing.assert_allclose(command_q[-1], qs[0])
