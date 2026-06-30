"""Build Chapter 6.2 real-data summary tables from existing replay results."""
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any


DECOUPLING_SOURCES = {
    "empty_robot": Path("data/results/ch4_2/scene_A_delta_0p050.json"),
    "single_static": Path("data/results/ch4_2/scene_B_delta_0p050.json"),
    "multi_static": Path("data/results/ch4_2/scene_B2_delta_0p050.json"),
}

WARNING_SOURCES = {
    "static_safe": Path("data/results/ch4_3/final_static_A/metrics.json"),
    "dynamic_approach": Path("data/results/ch4_3/final_dynamic/metrics.json"),
    "recover": Path("data/results/ch4_3/final_recover/metrics.json"),
}

FIGURE_SOURCES = {
    "fig43_static_A.png": Path("data/results/ch4_3/final_figures/fig43_static_A.png"),
    "fig43_dynamic.png": Path("data/results/ch4_3/final_figures/fig43_dynamic.png"),
    "fig43_recover.png": Path("data/results/ch4_3/final_figures/fig43_recover.png"),
}

METHOD_NAMES_42 = {
    "workspace": "Workspace",
    "ksi_like": "KSI-like",
    "ours": "Ours",
}

METHOD_NAMES_43 = {
    "dsa": "DSA",
    "ssm": "SSM",
    "ours_wo_temporal": "Ours-w/o Temporal",
    "ours": "Ours-STRO",
}

SCENE_NAMES_42 = {
    "empty_robot": "A empty robot",
    "single_static": "B single static",
    "multi_static": "B2 multi static",
}

SCENE_NAMES_43 = {
    "static_safe": "A static safe",
    "dynamic_approach": "B dynamic approach",
    "recover": "C approach-hold-leave",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def recording_stats(record_dir: str | None) -> dict[str, Any]:
    if not record_dir:
        return {}
    path = Path(record_dir)
    frames = sorted((path / "frames").glob("*.npz")) if (path / "frames").exists() else []
    manifest = {}
    if (path / "manifest.json").exists():
        try:
            manifest = load_json(path / "manifest.json")
        except Exception as exc:
            manifest = {"manifest_error": str(exc)}
    return {
        "record_dir": str(path),
        "frame_count": len(frames),
        "scene": manifest.get("scene"),
        "obstacle_desc": manifest.get("obstacle_desc"),
        "duration": manifest.get("duration"),
        "source": manifest.get("source"),
    }


def normalize_decoupling(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "record_dir": raw.get("record_dir"),
        "delta_r": raw.get("delta_r"),
        "delta_eval": raw.get("delta_eval"),
        "omegas": raw.get("omegas") or raw.get("omega"),
        "metrics": raw["metrics"],
        "recording": recording_stats(raw.get("record_dir")),
    }


def normalize_warning(raw: dict[str, Any], source_path: Path) -> dict[str, Any]:
    return {
        "trials": raw.get("trials"),
        "metrics": raw["metrics"],
        "source_path": str(source_path),
    }


def build_summary() -> dict[str, Any]:
    missing = [str(path) for path in list(DECOUPLING_SOURCES.values()) + list(WARNING_SOURCES.values()) if not path.exists()]
    if missing:
        raise FileNotFoundError("missing source files:\n" + "\n".join(missing))

    decoupling = {
        name: normalize_decoupling(load_json(path))
        for name, path in DECOUPLING_SOURCES.items()
    }
    warning = {
        name: normalize_warning(load_json(path), path)
        for name, path in WARNING_SOURCES.items()
    }
    return {
        "source": "reused Chapter 4 real replay results for Chapter 6.2",
        "decoupling_sources": {k: str(v) for k, v in DECOUPLING_SOURCES.items()},
        "warning_sources": {k: str(v) for k, v in WARNING_SOURCES.items()},
        "decoupling": decoupling,
        "warning": warning,
    }


def table_decoupling(summary: dict[str, Any]) -> str:
    headers = ["scene", "method", "frames", "R_res", "R_false", "R_keep", "R_det", "R_over", "sigma_c", "T_dec(ms)"]
    rows = []
    for scene, data in summary["decoupling"].items():
        frames = data.get("recording", {}).get("frame_count") or "-"
        for method in ("workspace", "ksi_like", "ours"):
            vals = data["metrics"][method]
            rows.append(
                [
                    SCENE_NAMES_42[scene],
                    METHOD_NAMES_42[method],
                    str(frames),
                    fmt(vals.get("R_res")),
                    fmt(vals.get("R_false")),
                    fmt(vals.get("R_keep")),
                    fmt(vals.get("R_det")),
                    fmt(vals.get("R_over")),
                    fmt(vals.get("sigma_c")),
                    fmt(vals.get("T_dec_ms_mean")),
                ]
            )
    return markdown(headers, rows)


def table_warning(summary: dict[str, Any]) -> str:
    headers = ["scene", "method", "trials", "T_lead", "R_miss", "R_false_time", "D_trigger_ref", "T_recover", "N_switch"]
    rows = []
    for scene, data in summary["warning"].items():
        trials = data.get("trials") or "-"
        for method in ("dsa", "ssm", "ours_wo_temporal", "ours"):
            vals = data["metrics"][method]
            rows.append(
                [
                    SCENE_NAMES_43[scene],
                    METHOD_NAMES_43[method],
                    str(trials),
                    fmt(vals.get("T_lead")),
                    fmt(vals.get("R_miss")),
                    fmt(vals.get("R_false_time")),
                    fmt(vals.get("D_trigger_ref")),
                    fmt(vals.get("T_recover")),
                    fmt(vals.get("N_switch")),
                ]
            )
    return markdown(headers, rows)


def table_reuse(summary: dict[str, Any]) -> str:
    headers = ["item", "source", "recording/trials", "reuse decision"]
    rows = []
    for scene, data in summary["decoupling"].items():
        rec = data.get("recording", {})
        rows.append(
            [
                SCENE_NAMES_42[scene],
                summary["decoupling_sources"][scene],
                f"{rec.get('frame_count', '-')} frames",
                "reuse for 6.2 real decoupling",
            ]
        )
    for scene, data in summary["warning"].items():
        rows.append(
            [
                SCENE_NAMES_43[scene],
                summary["warning_sources"][scene],
                f"{data.get('trials', '-')} trials",
                "reuse for 6.2 object-level warning",
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
    parser = argparse.ArgumentParser(description="Build Chapter 6.2 real-data summary from existing results.")
    parser.add_argument("--output", default="data/results/ch6_2_real")
    parser.add_argument("--copy-figures", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    summary = build_summary()
    with (output / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, allow_nan=True)

    tables = {
        "table_6_2_real_decoupling.md": table_decoupling(summary),
        "table_6_2_real_warning.md": table_warning(summary),
        "reuse_audit.md": table_reuse(summary),
    }
    for name, text in tables.items():
        with (output / name).open("w", encoding="utf-8") as handle:
            handle.write(text + "\n")

    if args.copy_figures:
        fig_dir = output / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        for name, src in FIGURE_SOURCES.items():
            if src.exists():
                shutil.copy2(src, fig_dir / name)

    print(tables["table_6_2_real_decoupling.md"])
    print()
    print(tables["table_6_2_real_warning.md"])
    print(f"\n[exp_62_real] saved summary to {output}")


if __name__ == "__main__":
    main()
