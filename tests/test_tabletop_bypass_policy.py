import importlib

import numpy as np
from types import SimpleNamespace


bypass = importlib.import_module(
    "experiments.new.6_5.6_5_3.simple_bypass_planner"
)
live = importlib.import_module(
    "experiments.new.6_5.6_5_3.run_653_simple_dynamic_nubs_live"
)
event = importlib.import_module(
    "experiments.new.6_5.6_5_3.run_653_simple_dynamic_nubs_event_replan_live"
)
trial = importlib.import_module(
    "experiments.new.6_5.6_5_3.run_653_dynamic_repair_trial"
)


def test_tabletop_parallel_side_is_horizontal_and_lateral():
    side = bypass.tabletop_parallel_lateral_direction(
        np.array([0.0, -1.0, 0.0]),
        np.array([-0.15, 0.01, -0.06]),
    )
    assert abs(float(side[2])) < 1.0e-12
    assert abs(float(np.dot(side, [0.0, -1.0, 0.0]))) < 1.0e-12
    assert side[0] < 0.0


def test_tabletop_parallel_side_is_strictly_horizontal_for_tilted_task():
    side = bypass.tabletop_parallel_lateral_direction(
        np.array([-0.0012, -0.999996, -0.0024]),
        np.array([-0.15, 0.01, -0.06]),
    )
    assert abs(float(side[2])) < 1.0e-12


def test_tabletop_gate_filters_unsafe_fast_seed():
    rows = [
        {"candidate": 1, "tabletop_feasible": False, "coarse_min_distance_m": 0.20,
         "task_progress_ok": True, "coarse_closest_approach_before_tail": True,
         "coarse_end_minus_min_clearance_m": 0.1, "guide_clearance_m": 0.2,
         "task_progress_m": 0.05},
        {"candidate": 2, "tabletop_feasible": True, "coarse_min_distance_m": 0.13,
         "task_progress_ok": True, "coarse_closest_approach_before_tail": True,
         "coarse_end_minus_min_clearance_m": 0.1, "guide_clearance_m": 0.1,
         "task_progress_m": 0.04},
    ]
    selected = live.select_planning_seed(
        rows, robust_target_m=0.11, coarse_gate_is_hard=False,
        tabletop_gate_is_hard=True,
    )
    assert selected["candidate"] == 2


def test_tabletop_gate_filters_unsafe_continuation_even_diagnostic():
    rows = [
        {"candidate": 1, "tabletop_feasible": False, "coarse_min_distance_m": 0.20,
         "task_progress_ok": True, "task_progress_m": 0.05, "goal_distance_m": 0.1},
        {"candidate": 2, "tabletop_feasible": True, "coarse_min_distance_m": 0.08,
         "task_progress_ok": True, "task_progress_m": 0.04, "goal_distance_m": 0.2},
    ]
    selected = event.select_goal_directed_continuation(
        rows, robust_target_m=0.11, diagnostic_only=True,
        tabletop_gate_is_hard=True,
    )
    assert selected["candidate"] == 2


def test_formal_protocol_freezes_tabletop_floor():
    assert trial.FORMAL_PROTOCOL["gripper_base_min_z_m"] == 0.46


def test_fast_height_shape_policy_falls_back_to_verified_seed(monkeypatch, tmp_path):
    class FakeTrajectory:
        total_duration = 1.0

        def evaluate(self, _tau):
            return np.zeros(6)

    candidate = FakeTrajectory()
    seed = FakeTrajectory()
    guards = iter([
        {"passed": True, "min_gripper_base_z_m": 0.526},
        {"passed": True, "min_gripper_base_z_m": 0.539},
    ])
    monkeypatch.setattr(trial, "gripper_base_workspace_guard", lambda *a, **k: next(guards))

    class Verifier:
        def verify(self, *args, **kwargs):
            return SimpleNamespace(accepted=True, min_distance=0.12, checks={}, reasons=[])

    monkeypatch.setattr(trial, "make_risk_stack", lambda *a, **k: (None, Verifier(), None))
    monkeypatch.setattr(trial, "save_trajectory_csv", lambda *a, **k: None)
    result = live.apply_tabletop_height_shape_policy(
        result={"local_repair_ready": True},
        artifacts_out={"candidate_trajectory": candidate, "reference_trajectory": seed},
        runtime_args=SimpleNamespace(gripper_base_min_z_m=0.46, online_accept_m=0.09),
        config={}, model=object(), forecast=object(), q_now=np.zeros(6), qd_now=np.zeros(6),
        trial_dir=tmp_path, max_drop_m=0.005,
    )
    assert result["height_preserving_seed_fallback"] is True
    assert result["selected_execution_candidate_source"] == "VERIFIED_TABLETOP_BYPASS_SEED"
