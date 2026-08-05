#!/usr/bin/env python3
"""Summarize reusable real RGB-D perception/risk evidence for revised 6.5.1.

The revised 6.5.1 only validates environment perception and risk recognition
with a static robot.  This script audits existing real recordings/results and
exports concise tables showing what can be reused and what still needs to be
recorded.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT = ROOT / "results" / "new" / "6_5" / "6_5_1" / "perception_reuse_analysis"

DECOUPLING_SOURCES = {
    "empty_robot": ROOT / "data/results/ch4_2/scene_A_delta_0p050.json",
    "single_static": ROOT / "data/results/ch4_2/scene_B_delta_0p050.json",
    "multi_static": ROOT / "data/results/ch4_2/scene_B2_delta_0p050.json",
}

WARNING_SOURCES = {
    "static_safe": ROOT / "data/results/ch4_3/final_static_A/metrics.json",
    "dynamic_approach": ROOT / "data/results/ch4_3/final_dynamic/metrics.json",
    "approach_hold_leave": ROOT / "data/results/ch4_3/final_recover/metrics.json",
}

BODY_RISK_SOURCE = ROOT / "data/results/ch6_3_real/metrics.json"

FIGURE_SOURCES = {
    "static_safe_warning.png": ROOT / "data/results/ch4_3/final_figures/fig43_static_A.png",
    "dynamic_approach_warning.png": ROOT / "data/results/ch4_3/final_figures/fig43_dynamic.png",
    "approach_hold_leave_warning.png": ROOT / "data/results/ch4_3/final_figures/fig43_recover.png",
    "ee_body_risk.png": ROOT / "data/results/ch6_3_real/figures/fig44_final_ee_body.png",
}


SCENE_NAMES = {
    "empty_robot": "E0 empty robot",
    "single_static": "Static single obstacle",
    "multi_static": "Static multi obstacle",
    "static_safe": "Static safe obstacle",
    "dynamic_approach": "Dynamic approach",
    "approach_hold_leave": "Approach-hold-leave",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if math.isinf(value):
            return "inf"
        return f"{value:.{digits}f}"
    return str(value)


def markdown(headers: list[str], rows: list[list[str]]) -> str:
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
            *["| " + " | ".join(row) + " |" for row in rows],
        ]
    )


def recording_manifest(record_dir: str | None) -> dict[str, Any]:
    if not record_dir:
        return {}
    path = Path(record_dir)
    if not path.is_absolute():
        path = ROOT / path
    manifest_path = path / "manifest.json"
    if not manifest_path.exists():
        return {"record_dir": str(path), "exists": False}
    manifest = load_json(manifest_path)
    return {
        "record_dir": str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path),
        "exists": True,
        "scene": manifest.get("scene"),
        "obstacle_desc": manifest.get("obstacle_desc"),
        "frames": manifest.get("frames"),
        "duration_wall_s": manifest.get("duration_wall_s"),
        "requested_camera": manifest.get("requested_camera"),
        "joint_names": manifest.get("joint_names"),
        "save_images": manifest.get("save_images"),
    }


def build_summary() -> dict[str, Any]:
    missing = [
        str(path)
        for path in [*DECOUPLING_SOURCES.values(), *WARNING_SOURCES.values(), BODY_RISK_SOURCE]
        if not path.exists()
    ]
    decoupling = {}
    for name, path in DECOUPLING_SOURCES.items():
        if path.exists():
            raw = load_json(path)
            decoupling[name] = {
                "source_path": str(path.relative_to(ROOT)),
                "recording": recording_manifest(raw.get("record_dir")),
                "metrics": raw.get("metrics", {}),
            }

    warning = {}
    for name, path in WARNING_SOURCES.items():
        if path.exists():
            raw = load_json(path)
            warning[name] = {
                "source_path": str(path.relative_to(ROOT)),
                "trials": raw.get("trials"),
                "metrics": raw.get("metrics", {}),
            }

    body_risk = load_json(BODY_RISK_SOURCE) if BODY_RISK_SOURCE.exists() else {}
    return {
        "purpose": "revised 6.5.1 perception-only reuse audit",
        "missing_sources": missing,
        "decoupling": decoupling,
        "warning": warning,
        "body_risk": body_risk,
        "reuse_decision": {
            "can_reuse_for_651": bool(decoupling and warning and body_risk),
            "usable_evidence": [
                "real RealSense recordings with AUBO joint states exist",
                "empty/static decoupling results cover robot self-filtering and static obstacle detection",
                "dynamic approach/recover results cover obstacle tracking, STRO warning and risk recovery",
                "whole-body risk replay covers end-effector and middle-link nearest-link evidence",
            ],
            "gaps": [
                "no dedicated revised-6.5.1 static S1/S2/S3 dataset with elbow/forearm/wrist labels and 3 repeats each",
                "existing dynamic trials are reused from Chapter 4.3/4.4 and are not named D1/D2 with path markers",
                "existing metrics summarize object-level warning and whole-body risk separately; a unified per-frame 6.5.1 log may still be useful",
            ],
            "recommendation": (
                "Use existing results as preliminary/reuse evidence for 6.5.1, then optionally add a minimal "
                "perception-only recording set: E0 x2 poses, S1-S3 x3 static trials, D1-D2 x5 dynamic trials."
            ),
        },
    }


def table_recordings(summary: dict[str, Any]) -> str:
    headers = ["item", "source", "frames/trials", "real RGB-D", "robot state", "6.5.1 use"]
    rows: list[list[str]] = []
    for name, data in summary["decoupling"].items():
        rec = data.get("recording", {})
        camera = rec.get("requested_camera", {})
        rows.append(
            [
                SCENE_NAMES[name],
                data["source_path"],
                f"{rec.get('frames', '-')} frames",
                "yes" if camera.get("source") == "realsense" else "-",
                "yes" if rec.get("joint_names") else "-",
                "self-filter/static detection",
            ]
        )
    for name, data in summary["warning"].items():
        rows.append(
            [
                SCENE_NAMES[name],
                data["source_path"],
                f"{data.get('trials', '-')} trials",
                "yes",
                "yes",
                "STRO warning/recovery",
            ]
        )
    scenes = summary.get("body_risk", {}).get("scenes", {})
    for scene, data in scenes.items():
        rows.append(
            [
                "CCRO " + scene.replace("_", " "),
                data.get("source_dir", "-"),
                f"{data.get('trials', '-')} trials",
                "yes",
                "yes",
                "nearest link / full-body risk",
            ]
        )
    return markdown(headers, rows)


def table_static_detection(summary: dict[str, Any]) -> str:
    headers = ["scene", "method", "frames", "R_det", "R_keep", "R_over", "sigma_c", "T_dec(ms)"]
    rows: list[list[str]] = []
    for scene in ("empty_robot", "single_static", "multi_static"):
        data = summary["decoupling"].get(scene)
        if not data:
            continue
        frames = data.get("recording", {}).get("frames", "-")
        metrics = data["metrics"].get("ours", {})
        rows.append(
            [
                SCENE_NAMES[scene],
                "Ours self-filter",
                str(frames),
                fmt(metrics.get("R_det")),
                fmt(metrics.get("R_keep")),
                fmt(metrics.get("R_over")),
                fmt(metrics.get("sigma_c")),
                fmt(metrics.get("T_dec_ms_mean")),
            ]
        )
    return markdown(headers, rows)


def table_dynamic_warning(summary: dict[str, Any]) -> str:
    headers = ["scene", "trials", "T_lead(s)", "R_miss", "R_false_time", "D_trigger_ref(m)", "T_recover(s)", "N_switch"]
    rows: list[list[str]] = []
    for scene in ("static_safe", "dynamic_approach", "approach_hold_leave"):
        data = summary["warning"].get(scene)
        if not data:
            continue
        vals = data["metrics"].get("ours", {})
        rows.append(
            [
                SCENE_NAMES[scene],
                str(data.get("trials", "-")),
                fmt(vals.get("T_lead")),
                fmt(vals.get("R_miss")),
                fmt(vals.get("R_false_time")),
                fmt(vals.get("D_trigger_ref")),
                fmt(vals.get("T_recover")),
                fmt(vals.get("N_switch")),
            ]
        )
    return markdown(headers, rows)


def table_body_risk(summary: dict[str, Any]) -> str:
    headers = ["scene", "trials", "sampled frames", "EE risk frames", "body risk frames", "top nearest links"]
    rows: list[list[str]] = []
    for scene, data in summary.get("body_risk", {}).get("scenes", {}).items():
        fs = data.get("frame_summary", {})
        top_links = ", ".join(f"{k}:{v}" for k, v in list(fs.get("nearest_links", {}).items())[:4])
        rows.append(
            [
                scene.replace("_", " "),
                str(data.get("trials", "-")),
                str(fs.get("sampled_frames", "-")),
                str(fs.get("ee_risk_frames", "-")),
                str(fs.get("body_risk_frames", "-")),
                top_links,
            ]
        )
    return markdown(headers, rows)


def write_report(output: Path, summary: dict[str, Any], tables: dict[str, str]) -> None:
    lines = [
        "# Revised 6.5.1 Perception Reuse Analysis",
        "",
        "This audit checks whether existing real RGB-D recordings/results can support the revised perception-only 6.5.1.",
        "",
        "## Reuse Decision",
        "",
        f"Can reuse existing evidence: **{summary['reuse_decision']['can_reuse_for_651']}**",
        "",
        "Usable evidence:",
        *[f"- {item}" for item in summary["reuse_decision"]["usable_evidence"]],
        "",
        "Gaps:",
        *[f"- {item}" for item in summary["reuse_decision"]["gaps"]],
        "",
        f"Recommendation: {summary['reuse_decision']['recommendation']}",
        "",
        "## Source Inventory",
        "",
        tables["table_651_reuse_inventory.md"],
        "",
        "## Static / Self-Filtering Evidence",
        "",
        tables["table_651_static_detection.md"],
        "",
        "## Dynamic Warning / Recovery Evidence",
        "",
        tables["table_651_dynamic_warning.md"],
        "",
        "## Whole-Body Risk / Nearest-Link Evidence",
        "",
        tables["table_651_body_risk.md"],
        "",
    ]
    (output / "reuse_analysis.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--copy-figures", action="store_true", default=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    summary = build_summary()
    tables = {
        "table_651_reuse_inventory.md": table_recordings(summary),
        "table_651_static_detection.md": table_static_detection(summary),
        "table_651_dynamic_warning.md": table_dynamic_warning(summary),
        "table_651_body_risk.md": table_body_risk(summary),
    }
    (output / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    for name, text in tables.items():
        (output / name).write_text(text + "\n", encoding="utf-8")
    write_report(output, summary, tables)

    if args.copy_figures:
        fig_dir = output / "figures"
        fig_dir.mkdir(exist_ok=True)
        for name, src in FIGURE_SOURCES.items():
            if src.exists():
                shutil.copy2(src, fig_dir / name)

    print(tables["table_651_reuse_inventory.md"])
    print(f"\n[summarize_651] saved to {output}")


if __name__ == "__main__":
    main()
