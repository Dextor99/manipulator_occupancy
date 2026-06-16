"""Ablation aggregation for Chapter 4.6."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from experiments.exp_45_eval import aggregate_trials


class AblationMetrics:
    def __init__(self, results_43: str | Path, results_45: str | Path):
        self.results_43 = Path(results_43)
        self.results_45 = Path(results_45)

    def load_43_metrics(self) -> dict[str, Any]:
        path = self.results_43 / "metrics.json"
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle).get("metrics", {})

    def load_45_method(self, controller: str) -> dict[str, Any]:
        paths = sorted(self.results_45.glob(f"trial_*_{controller}_*.json"))
        return aggregate_trials(paths) if paths else {}

    def build_ablation_table(self) -> dict[str, Any]:
        m43 = self.load_43_metrics()
        full45 = self.load_45_method("ours_full")
        scale45 = self.load_45_method("ours_scale")
        rep45 = self.load_45_method("ours_rep")
        lead_full = _get(m43, "ours", "T_lead")
        lead_wo_temporal = _get(m43, "ours_wo_temporal", "T_lead")
        return {
            "Full": merge({"T_lead": lead_full}, full45),
            "w/o Temporal Risk": merge({"T_lead": lead_wo_temporal}, full45),
            "w/o Repulsive Vector": merge({"T_lead": lead_full}, scale45),
            "w/o Safety Filter": merge({"T_lead": lead_full}, rep45),
        }


def _get(data: dict[str, Any], method: str, key: str):
    return data.get(method, {}).get(key)


def merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out = dict(b)
    out.update(a)
    return out


def table_46_ablation(rows: dict[str, Any]) -> str:
    headers = ["版本", "T_lead↑", "D_min_ref↑", "T_viol↓", "R_avoid↑", "R_timeout↓", "J_q_rms↓"]
    body = []
    for name, v in rows.items():
        body.append([
            name,
            fmt(v.get("T_lead")),
            fmt(v.get("D_min_ref")),
            fmt(v.get("T_viol")),
            fmt(v.get("R_avoid")),
            fmt(v.get("R_timeout")),
            fmt(v.get("J_q_rms")),
        ])
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
        *["| " + " | ".join(row) + " |" for row in body],
    ])


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)
