"""Chapter 4.6 ablation and timing aggregation entry point."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.exp_46_ablation import AblationMetrics, table_46_ablation
from experiments.exp_46_timing import TimingAnalyzer, table_timing


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

    with (output / "ablation.json").open("w", encoding="utf-8") as handle:
        json.dump(ablation, handle, indent=2, ensure_ascii=False)
    with (output / "timing.json").open("w", encoding="utf-8") as handle:
        json.dump({"timing": timing, "e2e": e2e}, handle, indent=2, ensure_ascii=False)

    print("表 4-7 消融结果")
    print(table_46_ablation(ablation))
    print("\n表 4-8 模块耗时")
    print(table_timing(timing))
    print(f"\n[exp_46] saved results to {output}")


if __name__ == "__main__":
    main()
