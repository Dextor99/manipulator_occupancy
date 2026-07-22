from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml


bench = importlib.import_module("experiments.new.6_3.run_static_benchmark")
plotter = importlib.import_module("experiments.new.6_3.plot_static_benchmark")
cfg62 = importlib.import_module("experiments.new.6_2.config_62")
body62 = importlib.import_module("experiments.new.6_2.body_coverage_62")


def test_validation_accept_distance_matches_paper_config() -> None:
    config = yaml.safe_load(Path("config/ccro_stage2.yaml").read_text(encoding="utf-8"))
    assert bench.validation_accept_distance(config) == pytest.approx(0.08)


def test_budget_fields_do_not_overwrite_dense_feasibility() -> None:
    row = {
        "verification": {"accepted": True},
        "optimization": {"elapsed_ms": 12_345.0},
    }
    annotated = bench._annotate_budget(row)
    assert annotated["dense_feasible"] is True
    assert annotated["within_time_budget"] is False
    assert annotated["budgeted_accepted"] is False
    assert annotated["optimization"]["elapsed_ms_raw"] == pytest.approx(12_345.0)


def test_aggregate_uses_dense_feasible_for_quality_and_budget_separately() -> None:
    rows = [
        {
            "dense_feasible": True,
            "budgeted_accepted": False,
            "timeout": True,
            "verification": {"accepted": True, "min_distance": 0.09},
            "optimization": {"elapsed_ms_raw": 12_000.0},
            "post": {"J_smooth": 1.0},
        },
        {
            "dense_feasible": False,
            "budgeted_accepted": False,
            "timeout": False,
            "verification": {"accepted": False, "min_distance": 0.02},
            "optimization": {"elapsed_ms_raw": 100.0},
            "post": {"J_smooth": 99.0},
        },
    ]
    summary = bench.aggregate_method_rows(rows)
    assert summary["dense_feasible_rate"] == pytest.approx(0.5)
    assert summary["budgeted_accepted_rate"] == pytest.approx(0.0)
    assert summary["D_min"]["mean"] == pytest.approx(0.09)
    assert summary["J_smooth"]["mean"] == pytest.approx(1.0)
    assert summary["timeout_count"] == 1


class FakeSurface:
    link_names = tuple({link for links in cfg62.BODY_REGIONS.values() for link in links})

    def surface_by_link(self, q, density="coarse", links=None):
        selected = set(self.link_names) if links is None else set(links)
        out = {}
        for index, link in enumerate(sorted(selected)):
            base = float(index + 1)
            out[link] = np.asarray(
                [
                    [base, 0.0, 0.0],
                    [base, 0.2, 0.0],
                    [base, 0.0, 0.3],
                ],
                dtype=float,
            )
        return out


def test_critical_point_evaluator_reuses_ch6_2_definition() -> None:
    surface = FakeSurface()
    expected = body62.build_critical_points(surface, np.zeros(6))
    evaluator = bench.CriticalPointRiskEvaluator(
        surface,
        d_safe=0.12,
        d_activate=0.18,
        fd_epsilon_q=1.0e-4,
    )
    selected = evaluator._critical_points_by_link(np.zeros(6))
    flattened = [point for points in selected.values() for point in points]
    assert len(flattened) == len(expected)
    assert sorted(point.radius for point in flattened) == sorted(point.radius for point in expected)


def test_rrt_uses_validation_accept_distance_for_connect_and_shortcut(monkeypatch) -> None:
    ext = importlib.import_module("experiments.exp_64_external_baselines")
    seen = {"connect": [], "shortcut": []}

    def fake_rrt_connect(q_start, q_goal, limits, evaluator, obstacle, d_stop, rng):
        seen["connect"].append(d_stop)
        return np.vstack([q_start, q_goal]), {}

    def fake_shortcut(path, evaluator, obstacle, d_stop, rng):
        seen["shortcut"].append(d_stop)
        return path

    def fake_rrt_to_trajectory(path, head, tail, duration):
        return object()

    monkeypatch.setattr(ext, "_rrt_connect", fake_rrt_connect)
    monkeypatch.setattr(ext, "_shortcut_path", fake_shortcut)
    monkeypatch.setattr(ext, "_rrt_to_trajectory", fake_rrt_to_trajectory)

    def fake_row_for_trajectory(**kwargs):
        return bench._annotate_budget(
            {
                "method": kwargs["method"],
                "verification": {"accepted": True, "min_distance": 0.08},
                "optimization": {"elapsed_ms": 1.0},
                "post": {"J_smooth": 1.0},
            }
        )

    monkeypatch.setattr(bench, "_row_for_trajectory", fake_row_for_trajectory)
    context = {
        "head": np.zeros((6, 3)),
        "tail": np.ones((6, 3)),
        "durations": np.ones(1),
        "limits": SimpleNamespace(),
        "evaluator": SimpleNamespace(),
        "verifier": SimpleNamespace(),
        "sample_times": np.linspace(0.0, 1.0, 3),
        "config": {
            "validation": {"d_accept": 0.08},
            "experiment": {"random_seed": 1},
        },
    }
    instance = {
        "observed_points": np.zeros((3, 3)),
        "gt_dense_points": np.zeros((3, 3)),
        "index": 0,
        "scenario": "A",
    }
    row = bench.run_rrt_for_instance(context, instance)
    assert seen["connect"] == [pytest.approx(0.08)]
    assert seen["shortcut"] == [pytest.approx(0.08)]
    assert row["planning_clearance_m"] == pytest.approx(0.08)


def test_plot_selects_instance_with_all_four_dense_feasible(tmp_path: Path) -> None:
    root = tmp_path
    (root / "trials").mkdir()
    metrics = {
        "scenarios": {
            "B": {
                "instances": [
                    {"id": "B_00", "trial_path": "trials/B_00.json"},
                    {"id": "B_01", "trial_path": "trials/B_01.json"},
                ]
            },
            "C": {"instances": []},
            "A": {"instances": []},
        }
    }
    partial = {
        "rrt_connect_smooth": {"dense_feasible": False},
        "minco_risk": {"dense_feasible": True},
        "critical_point_nubs": {"dense_feasible": True},
        "ccro_nubs": {"dense_feasible": True},
    }
    full = {
        "rrt_connect_smooth": {"dense_feasible": True},
        "minco_risk": {"dense_feasible": True},
        "critical_point_nubs": {"dense_feasible": True},
        "ccro_nubs": {"dense_feasible": True},
    }
    (root / "trials/B_00.json").write_text(json.dumps(partial), encoding="utf-8")
    (root / "trials/B_01.json").write_text(json.dumps(full), encoding="utf-8")
    assert plotter.select_representative_instance(root, metrics) == "B_01"


def test_official_ch6_3_result_tree_has_30_frozen_instances_and_trials() -> None:
    root = Path("data/results/6_3")
    if not root.exists():
        pytest.skip("official 6.3 result directory is not present")
    assert len(list((root / "frozen_instances").glob("*.json"))) == 30
    assert len(list((root / "trials").glob("*.json"))) == 30
