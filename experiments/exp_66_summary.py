"""Build Chapter 6.6 real-system response and readiness summary tables."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


SOURCES = {
    "real_occupancy": Path("data/results/ch6_2_real/metrics.json"),
    "real_body_risk": Path("data/results/ch6_3_real/metrics.json"),
    "ch4_6_timing": Path("data/results/ch4_6/timing.json"),
    "ch4_6_quality": Path("data/results/ch4_6/quality_check.json"),
    "p7": Path("data/results/ccro_p7/metrics.json"),
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def require_sources() -> dict[str, dict[str, Any]]:
    missing = [str(path) for path in SOURCES.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing required source files:\n" + "\n".join(missing))
    return {name: load_json(path) for name, path in SOURCES.items()}


def build_summary() -> dict[str, Any]:
    data = require_sources()
    real_occ = data["real_occupancy"]
    real_body = data["real_body_risk"]
    timing = data["ch4_6_timing"]
    quality = data["ch4_6_quality"]
    p7 = data["p7"]

    response_rows = []
    warning = real_occ["warning"]
    for scene, scene_data in warning.items():
        ours = scene_data["metrics"]["ours"]
        response_rows.append(
            {
                "scene": scene,
                "trials": scene_data["trials"],
                "T_lead": ours["T_lead"],
                "R_miss": ours["R_miss"],
                "R_false_time": ours["R_false_time"],
                "D_trigger_ref": ours["D_trigger_ref"],
                "T_recover": ours["T_recover"],
                "N_switch": ours["N_switch"],
            }
        )

    body_rows = []
    for scene, scene_data in real_body["scenes"].items():
        ours = scene_data["metrics"]["ours"]
        frame = scene_data["frame_summary"]
        body_rows.append(
            {
                "scene": scene,
                "trials": scene_data["trials"],
                "sampled_frames": frame["sampled_frames"],
                "ee_risk_frames": frame["ee_risk_frames"],
                "body_risk_frames": frame["body_risk_frames"],
                "C_grad_D": ours["C_grad_D"],
                "G_rep": ours["G_rep"],
                "R_body": ours["R_body"],
                "active_frames": ours["active_frames"],
                "top_links": ", ".join(f"{k}:{v}" for k, v in list(frame["nearest_links"].items())[:3]),
            }
        )

    timing_rows = []
    for name, row in timing["timing"].items():
        if name == "_meta":
            continue
        timing_rows.append(
            {
                "module": name,
                "mean_ms": row["mean"],
                "p95_ms": row["p95"],
                "ratio": row["ratio"],
                "nonzero_rows": row["nonzero_rows"],
            }
        )

    readiness_rows = []
    for mode, decision in p7["decisions"].items():
        readiness_rows.append(
            {
                "mode": mode,
                "allowed": decision["allowed"],
                "reason": decision["reason"],
                "failed_checks": ", ".join(decision["failed_checks"]) if decision["failed_checks"] else "-",
            }
        )

    return {
        "source": "Chapter 6.6 real-system evidence summary",
        "source_files": {name: str(path) for name, path in SOURCES.items()},
        "real_dynamic_response": response_rows,
        "real_body_risk": body_rows,
        "timing": timing_rows,
        "timing_e2e": timing["e2e"],
        "quality_checks": quality["checks"],
        "readiness": readiness_rows,
        "dry_run_accepted": p7["dry_run_accepted"],
        "unsafe_switch_blocked": p7["unsafe_switch_blocked"],
        "real_robot_complete": p7["real_robot_complete"],
        "accepted": bool(p7["dry_run_accepted"] and p7["unsafe_switch_blocked"]),
        "paper_claim_boundary": "Use as real sensing/replay + software readiness evidence. Do not claim completed real robot online trajectory switching until P7 pending checks pass.",
    }


def markdown(headers: list[str], rows: list[list[str]]) -> str:
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
            *["| " + " | ".join(row) + " |" for row in rows],
        ]
    )


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if math.isinf(value):
            return "inf"
        return f"{value:.6g}"
    return str(value)


def table_response(summary: dict[str, Any]) -> str:
    rows = []
    for row in summary["real_dynamic_response"]:
        rows.append(
            [
                row["scene"],
                str(row["trials"]),
                fmt(row["T_lead"]),
                fmt(row["R_miss"]),
                fmt(row["R_false_time"]),
                fmt(row["D_trigger_ref"]),
                fmt(row["T_recover"]),
                fmt(row["N_switch"]),
            ]
        )
    return markdown(["scene", "trials", "T_lead/s", "R_miss", "R_false_time", "D_trigger_ref/m", "T_recover/s", "N_switch"], rows)


def table_body(summary: dict[str, Any]) -> str:
    rows = []
    for row in summary["real_body_risk"]:
        rows.append(
            [
                row["scene"],
                str(row["trials"]),
                str(row["sampled_frames"]),
                str(row["ee_risk_frames"]),
                str(row["body_risk_frames"]),
                fmt(row["C_grad_D"]),
                fmt(row["G_rep"]),
                fmt(row["R_body"]),
                str(row["active_frames"]),
                row["top_links"],
            ]
        )
    return markdown(["scene", "trials", "sampled", "ee-risk", "body-risk", "C_grad_D", "G_rep", "R_body", "active", "top links"], rows)


def table_timing(summary: dict[str, Any]) -> str:
    rows = [
        [row["module"], fmt(row["mean_ms"]), fmt(row["p95_ms"]), fmt(row["ratio"]), str(row["nonzero_rows"])]
        for row in summary["timing"]
    ]
    return markdown(["module", "mean/ms", "p95/ms", "ratio", "nonzero rows"], rows)


def table_readiness(summary: dict[str, Any]) -> str:
    rows = [
        [row["mode"], str(row["allowed"]), row["reason"], row["failed_checks"]]
        for row in summary["readiness"]
    ]
    return markdown(["mode", "allowed", "reason", "failed checks"], rows)


def notes(summary: dict[str, Any]) -> str:
    lines = [
        f"dry_run_accepted: {summary['dry_run_accepted']}",
        f"unsafe_switch_blocked: {summary['unsafe_switch_blocked']}",
        f"real_robot_complete: {summary['real_robot_complete']}",
        f"claim_boundary: {summary['paper_claim_boundary']}",
        "",
        "Quality checks:",
    ]
    for check in summary["quality_checks"]:
        status = "PASS" if check["passed"] else "NOTE"
        lines.append(f"- {status}: {check['name']} - {check['detail']}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Chapter 6.6 summary.")
    parser.add_argument("--output", default="data/results/ch6_6")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    summary = build_summary()
    (output / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tables = {
        "table_6_6_real_dynamic_response.md": table_response(summary),
        "table_6_6_real_body_risk.md": table_body(summary),
        "table_6_6_timing.md": table_timing(summary),
        "table_6_6_readiness_gate.md": table_readiness(summary),
        "notes.md": notes(summary),
    }
    for name, text in tables.items():
        (output / name).write_text(text + "\n", encoding="utf-8")
    print(tables["table_6_6_real_dynamic_response.md"])
    print()
    print(tables["table_6_6_readiness_gate.md"])
    print(f"\n[exp_66] saved summary to {output}")


if __name__ == "__main__":
    main()
