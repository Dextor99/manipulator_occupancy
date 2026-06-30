"""Build Chapter 6.3 real-data summary from existing Chapter 4.4 results."""
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any


SOURCES = {
    "ee_risk": Path("data/results/ch4_4/final_ee"),
    "body_risk": Path("data/results/ch4_4/final_body_09_11"),
}

METHOD_NAMES = {
    "apf": "APF",
    "ours_ee_only": "Ours-EE only",
    "ours_wo_temporal": "Ours-w/o Temporal",
    "ours": "Ours-CCRO",
}

SCENE_NAMES = {
    "ee_risk": "End-effector near-field",
    "body_risk": "Middle-link near-field",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def collect_trials(folder: Path) -> list[dict[str, Any]]:
    return [load_json(path) for path in sorted(folder.glob("trial_*.json"))]


def build_summary() -> dict[str, Any]:
    missing = [str(path) for path in SOURCES.values() if not (path / "metrics.json").exists()]
    if missing:
        raise FileNotFoundError("missing source result folders:\n" + "\n".join(missing))
    scenes = {}
    for name, folder in SOURCES.items():
        metrics = load_json(folder / "metrics.json")
        trials = collect_trials(folder)
        frame_summary = aggregate_frame_summary(trials)
        scenes[name] = {
            "source_dir": str(folder),
            "metrics": metrics["metrics"],
            "trials": metrics.get("trials", len(trials)),
            "frame_summary": frame_summary,
            "record_dirs": sorted({trial.get("record_dir", "") for trial in trials if trial.get("record_dir")}),
        }
    return {
        "source": "reused Chapter 4.4 real replay results for Chapter 6.3",
        "scenes": scenes,
    }


def aggregate_frame_summary(trials: list[dict[str, Any]]) -> dict[str, Any]:
    out = {
        "sampled_frames": 0,
        "ee_risk_frames": 0,
        "body_risk_frames": 0,
        "nearest_links": {},
    }
    for trial in trials:
        fs = trial.get("frame_summary", {})
        out["sampled_frames"] += int(fs.get("sampled_frames", 0))
        out["ee_risk_frames"] += int(fs.get("ee_risk_frames", 0))
        out["body_risk_frames"] += int(fs.get("body_risk_frames", 0))
        for link, count in fs.get("nearest_links", {}).items():
            out["nearest_links"][link] = out["nearest_links"].get(link, 0) + int(count)
    out["nearest_links"] = dict(sorted(out["nearest_links"].items(), key=lambda item: (-item[1], item[0])))
    return out


def table_frame_summary(summary: dict[str, Any]) -> str:
    headers = ["scene", "trials", "sampled", "ee-risk", "body-risk", "top nearest links", "record dirs"]
    rows = []
    for scene, data in summary["scenes"].items():
        fs = data["frame_summary"]
        top_links = ", ".join(f"{k}:{v}" for k, v in list(fs["nearest_links"].items())[:4])
        rows.append(
            [
                SCENE_NAMES[scene],
                str(data["trials"]),
                str(fs["sampled_frames"]),
                str(fs["ee_risk_frames"]),
                str(fs["body_risk_frames"]),
                top_links,
                str(len(data["record_dirs"])),
            ]
        )
    return markdown(headers, rows)


def table_method_summary(summary: dict[str, Any]) -> str:
    headers = ["scene", "method", "C_grad_D", "G_rep", "R_body", "active", "ee-active", "body-active"]
    rows = []
    for scene, data in summary["scenes"].items():
        for method in ("apf", "ours_ee_only", "ours_wo_temporal", "ours"):
            vals = data["metrics"][method]
            rows.append(
                [
                    SCENE_NAMES[scene],
                    METHOD_NAMES[method],
                    fmt(vals.get("C_grad_D")),
                    fmt(vals.get("G_rep")),
                    fmt(vals.get("R_body")),
                    str(vals.get("active_frames", "-")),
                    str(vals.get("active_ee_frames", "-")),
                    str(vals.get("active_body_frames", "-")),
                ]
            )
    return markdown(headers, rows)


def table_reuse(summary: dict[str, Any]) -> str:
    headers = ["item", "source", "recordings", "reuse decision"]
    rows = []
    for scene, data in summary["scenes"].items():
        rows.append(
            [
                SCENE_NAMES[scene],
                data["source_dir"],
                "; ".join(data["record_dirs"]),
                "reuse for 6.3 real whole-body risk evidence",
            ]
        )
    return markdown(headers, rows)


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
        return f"{value:.4f}"
    return str(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Chapter 6.3 real-data summary.")
    parser.add_argument("--output", default="data/results/ch6_3_real")
    parser.add_argument("--copy-figures", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    summary = build_summary()
    with (output / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    tables = {
        "table_6_3_real_frames.md": table_frame_summary(summary),
        "table_6_3_real_methods.md": table_method_summary(summary),
        "reuse_audit.md": table_reuse(summary),
    }
    for name, text in tables.items():
        with (output / name).open("w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    if args.copy_figures:
        fig_dir = output / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        for scene, folder in SOURCES.items():
            for fig in ("fig44.png",):
                src = folder / fig
                if src.exists():
                    shutil.copy2(src, fig_dir / f"{scene}_{fig}")
        merged = Path("data/results/ch4_4/fig44_final_ee_body.png")
        if merged.exists():
            shutil.copy2(merged, fig_dir / merged.name)
    print(tables["table_6_3_real_frames.md"])
    print()
    print(tables["table_6_3_real_methods.md"])
    print(f"\n[exp_63_real] saved summary to {output}")


if __name__ == "__main__":
    main()
