from __future__ import annotations

import importlib
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


def test_d1_and_d2_share_one_scene_independent_formal_protocol():
    d1 = trial.build_parser().parse_args(["--scene", "D1", "--mode", "moving-shadow-stop"])
    d2 = trial.build_parser().parse_args(["--scene", "D2", "--mode", "moving-shadow-stop"])
    assert trial.formal_protocol_violations(d1) == []
    assert trial.formal_protocol_violations(d2) == []
    assert trial.formal_protocol_signature(d1) == trial.formal_protocol_signature(d2)
    assert "risk_links" not in trial.SCENARIOS["D1"]
    assert "risk_links" not in trial.SCENARIOS["D2"]


def test_formal_protocol_rejects_scene_specific_threshold_tuning():
    args = trial.build_parser().parse_args(
        ["--scene", "D2", "--mode", "moving-shadow-stop", "--min-dynamic-trigger-speed-m-s", "0.05"]
    )
    violations = trial.formal_protocol_violations(args)
    assert any(item.startswith("min_dynamic_trigger_speed_m_s=") for item in violations)


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


def test_empty_scene_calibration_defaults_to_noncommanding_dry_run(tmp_path):
    args = calibration.build_parser().parse_args(
        ["--playback-duration-s", "1.0", "--repeat", "1", "--output", str(tmp_path)]
    )
    result = calibration.run(args)
    assert result["status"] == "DRY_RUN_ONLY"
    assert not result["robot_commanded"]


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
