from __future__ import annotations

import importlib
import csv
import json
from types import SimpleNamespace

import numpy as np


trial = importlib.import_module("experiments.new.6_5.6_5_3.run_653_dynamic_repair_trial")
prepare = importlib.import_module("experiments.new.6_5.6_5_3.prepare_653_reference")


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


def test_dynamic_audit_buffers_exist_even_when_no_clusters_are_seen():
    cluster_rows, track_rows, centers, timestamp, per_track = trial.new_dynamic_audit_buffers()
    assert cluster_rows == []
    assert track_rows == []
    assert centers == []
    assert timestamp is None
    assert per_track == {}


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
