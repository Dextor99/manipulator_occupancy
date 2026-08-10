from __future__ import annotations

import importlib
import csv
from types import SimpleNamespace

import numpy as np


trial = importlib.import_module("experiments.new.6_5.6_5_3.run_653_dynamic_repair_trial")
prepare = importlib.import_module("experiments.new.6_5.6_5_3.prepare_653_reference")


def test_track_geometry_uses_one_track_for_center_velocity_and_radius():
    obj = SimpleNamespace(id=7, center=np.array([1.0, 2.0, 3.0]), velocity=np.array([0.1, 0.0, 0.0]), radius=0.06)
    clusters = [SimpleNamespace(center=np.array([1.01, 2.0, 3.0]))]
    geometry = trial.track_geometry(obj, clusters, 0.055)
    assert geometry["track_id"] == 7
    assert geometry["raw_radius"] == 0.06
    assert geometry["inflated_radius"] == 0.06
    assert geometry["associated_cluster_index"] == 0
    assert geometry["association_error_m"] < 0.011


def test_recorded_reference_state_after_uses_future_reference_not_velocity_extrapolation():
    times = np.array([0.0, 1.0, 2.0])
    q = np.zeros((3, 6))
    q[:, 0] = [0.0, 0.3, 1.0]
    qd = np.zeros_like(q)
    ref = trial.RecordedReference(times, q, qd)
    ref.locate(np.array([0.3, 0, 0, 0, 0, 0]))
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
