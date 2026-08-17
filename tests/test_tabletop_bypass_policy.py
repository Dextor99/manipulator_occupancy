import importlib

import numpy as np


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
