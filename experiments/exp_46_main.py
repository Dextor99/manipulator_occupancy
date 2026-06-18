"""Chapter 4.6 ablation and timing aggregation entry point."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.exp_46_ablation import AblationMetrics, table_46_ablation
from experiments.exp_46_timing import TimingAnalyzer, table_e2e, table_timing


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Chapter 4.6 ablation and timing aggregation.")
    parser.add_argument("--results-43", default="data/results/ch4_3")
    parser.add_argument("--results-45", default="data/results/ch4_5")
    parser.add_argument("--timing-logs", default="data/results/ch4_5")
    parser.add_argument("--output", default="data/results/ch4_6")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    ablation = AblationMetrics(args.results_43, args.results_45).build_ablation_table()
    timing_analyzer = TimingAnalyzer(args.timing_logs)
    timing = timing_analyzer.compute_timing_stats()
    e2e = timing_analyzer.compute_e2e_stats()
    quality = build_quality_report(ablation, timing, e2e, args.results_43, args.results_45, args.timing_logs)

    with (output / "ablation.json").open("w", encoding="utf-8") as handle:
        json.dump(ablation, handle, indent=2, ensure_ascii=False)
    with (output / "timing.json").open("w", encoding="utf-8") as handle:
        json.dump({"timing": timing, "e2e": e2e}, handle, indent=2, ensure_ascii=False)
    with (output / "quality_check.json").open("w", encoding="utf-8") as handle:
        json.dump(quality, handle, indent=2, ensure_ascii=False)

    table47 = table_46_ablation(ablation)
    table48 = table_timing(timing)
    table49 = table_e2e(e2e)
    (output / "table_4_7.md").write_text(table47 + "\n", encoding="utf-8")
    (output / "table_4_8.md").write_text(table48 + "\n", encoding="utf-8")
    (output / "table_4_9.md").write_text(table49 + "\n", encoding="utf-8")
    (output / "quality_check.md").write_text(format_quality_report(quality), encoding="utf-8")

    print("表 4-7 消融结果")
    print(table47)
    print("\n表 4-8 模块耗时")
    print(table48)
    print("\n表 4-9 端到端与控制频率")
    print(table49)
    print("\n数据合理性检查")
    print(format_quality_report(quality))
    print(f"\n[exp_46] saved results to {output}")


def build_quality_report(
    ablation: dict,
    timing: dict,
    e2e: dict,
    results_43: str,
    results_45: str,
    timing_logs: str,
) -> dict:
    full = ablation.get("Full", {})
    wo_temporal = ablation.get("w/o Temporal Risk", {})
    wo_rep = ablation.get("w/o Repulsive Vector", {})
    wo_filter = ablation.get("w/o Safety Filter", {})
    nonzero_modules = timing.get("_meta", {}).get("nonzero_modules", [])
    checks = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    add(
        "时空预测消融",
        _num(full.get("T_lead")) > _num(wo_temporal.get("T_lead")),
        f"Full T_lead={_fmt(full.get('T_lead'))}, w/o Temporal={_fmt(wo_temporal.get('T_lead'))}",
    )
    add(
        "排斥向量消融",
        _num(full.get("D_min_ref")) > _num(wo_rep.get("D_min_ref")) and _num(wo_rep.get("T_viol")) > 0,
        f"Full D_min_ref={_fmt(full.get('D_min_ref'))}, w/o Rep D_min_ref={_fmt(wo_rep.get('D_min_ref'))}, "
        f"w/o Rep T_viol={_fmt(wo_rep.get('T_viol'))}",
    )
    add(
        "安全滤波消融",
        _num(full.get("D_min_ref")) >= _num(wo_filter.get("D_min_ref")),
        f"Full D_min_ref={_fmt(full.get('D_min_ref'))}, w/o Filter D_min_ref={_fmt(wo_filter.get('D_min_ref'))}",
    )
    add(
        "试次数量",
        int(full.get("n_trials", 0) or 0) >= 5,
        f"4.5 闭环消融统计 n_trials={full.get('n_trials')}",
    )
    add(
        "实时性日志完整性",
        len(nonzero_modules) > 3,
        "当前 4.5 日志只对控制器/排斥计算段计时；感知预处理、解耦、聚类、跟踪、预测列为 0。",
    )
    add(
        "控制段实时性目标",
        _num(e2e.get("T_ctrl_p95_ms")) < 20.0,
        f"T_ctrl^95={_fmt(e2e.get('T_ctrl_p95_ms'))} ms，目标 < 20 ms。",
    )
    add(
        "端到端实时性目标",
        _num(e2e.get("T_e2e_p95_ms")) < 100.0 and len(nonzero_modules) > 3,
        f"T_frame^95={_fmt(e2e.get('T_e2e_p95_ms'))} ms；但当前缺少完整感知模块计时。",
    )

    return {
        "sources": {
            "results_43": results_43,
            "results_45": results_45,
            "timing_logs": timing_logs,
        },
        "summary": {
            "usable_for_ablation": all(c["passed"] for c in checks[:4]),
            "usable_for_full_e2e_timing": False,
            "usable_for_controller_timing": True,
            "frame_count": e2e.get("frame_count", 0),
            "nonzero_timing_modules": nonzero_modules,
        },
        "checks": checks,
    }


def format_quality_report(report: dict) -> str:
    lines = ["| 检查项 | 结论 | 说明 |", "| --- | --- | --- |"]
    for check in report.get("checks", []):
        result = "通过" if check.get("passed") else "需说明"
        lines.append(f"| {check.get('name')} | {result} | {check.get('detail')} |")
    return "\n".join(lines) + "\n"


def _num(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("-inf")


def _fmt(value) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


if __name__ == "__main__":
    main()
