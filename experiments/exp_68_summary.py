"""Build Chapter 6.8 discussion, evidence-chain, and limitation summary."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


RESULTS = {
    "6.2 sim": Path("data/results/ch6_2_sim/metrics.json"),
    "6.2 real": Path("data/results/ch6_2_real/metrics.json"),
    "6.3 sim": Path("data/results/ch6_3_sim/metrics.json"),
    "6.3 real": Path("data/results/ch6_3_real/metrics.json"),
    "6.4": Path("data/results/ch6_4/metrics.json"),
    "6.5": Path("data/results/ch6_5/metrics.json"),
    "6.6": Path("data/results/ch6_6/metrics.json"),
    "6.7": Path("data/results/ch6_7/metrics.json"),
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_summary() -> dict[str, Any]:
    availability = {
        name: {"path": str(path), "exists": path.exists()}
        for name, path in RESULTS.items()
    }
    data = {name: load_json(path) for name, path in RESULTS.items() if path.exists()}
    p7_complete = bool(data.get("6.6", {}).get("real_robot_complete", False))
    evidence = [
        {
            "claim": "对象级时空占据可支持动态障碍预警",
            "section": "6.2",
            "evidence": "sim/real occupancy and warning tables",
            "table": "data/results/ch6_2_sim/table_6_2_sim.md; data/results/ch6_2_real/table_6_2_real_warning.md",
            "paper_ready": True,
            "boundary": "真实结果为回放/统计预警，不是实机轨迹切换。",
        },
        {
            "claim": "全身风险距离能发现末端方法遗漏的中间连杆风险",
            "section": "6.3",
            "evidence": "body-link sim and real replay counterexamples",
            "table": "data/results/ch6_3_sim/table_6_3_sim.md; data/results/ch6_3_real/table_6_3_real_methods.md",
            "paper_ready": True,
            "boundary": "真实部分为采集序列回放，不发送新控制命令。",
        },
        {
            "claim": "CCRO-NUBS 可在静态风险场景中生成通过 dense 复核的连续轨迹",
            "section": "6.4",
            "evidence": "NUBS internal ablation + MINCO/RRT external baselines",
            "table": "data/results/ch6_4/table_6_4_static_risk.md; data/results/ch6_4/table_6_4_external_baselines.md",
            "paper_ready": True,
            "boundary": "MINCO/RRT 为关节空间复现基线，不宣称官方工程直接部署。",
        },
        {
            "claim": "动态障碍下风险触发重规划可提升最小距离并支持安全接管",
            "section": "6.5",
            "evidence": "virtual closed loop and rolling replanning",
            "table": "data/results/ch6_5/table_6_5_virtual_loop.md; data/results/ch6_5/table_6_5_rolling_replan.md",
            "paper_ready": True,
            "boundary": "主要为虚拟闭环和软件闭环。",
        },
        {
            "claim": "真实系统具备感知回放安全响应和 fail-closed 软件门控",
            "section": "6.6",
            "evidence": "real replay + P7 dry-run",
            "table": "data/results/ch6_6/table_6_6_real_dynamic_response.md; data/results/ch6_6/table_6_6_readiness_gate.md",
            "paper_ready": True,
            "boundary": "真实在线 NUBS 轨迹切换仍 pending。" if not p7_complete else "真实在线切换已完成。",
        },
        {
            "claim": "时序预测、全身风险、排斥向量和安全滤波均有独立贡献",
            "section": "6.7",
            "evidence": "risk and control ablations",
            "table": "data/results/ch6_7/table_6_7_risk_ablation.md; data/results/ch6_7/table_6_7_control_ablation.md",
            "paper_ready": True,
            "boundary": "真实端到端完整分模块计时仍需补非零感知日志。",
        },
    ]
    limitations = [
        "动态障碍预测采用短时匀速模型，复杂非线性人手运动下可能保守或滞后。",
        "Mesh 表面距离和有限差分风险梯度在高采样密度下耗时较高。",
        "在线 NUBS 重规划适合触发式低频执行，不适合每帧高频优化。",
        "真实 RGB-D 噪声、遮挡和外参误差会导致误触发和恢复延迟。",
        "真实在线轨迹切换仍需通过 P7 的 watchdog、通信、时间戳、急停和厂家限位检查。",
    ]
    next_steps = [
        "补真实端到端非零分模块计时。",
        "完成 P7 low_speed_switch 实机闭环并记录 Real-N1/N2/N3。",
        "增加 RRT-Connect 多随机种子统计。",
        "生成 6.2-6.7 的最终论文图。",
        "如需严格官方基线，额外部署 GCOPTER C++/ROS 并说明任务迁移边界。",
    ]
    return {
        "source": "Chapter 6.8 discussion summary",
        "availability": availability,
        "evidence_chain": evidence,
        "limitations": limitations,
        "next_steps": next_steps,
        "paper_ready": all(row["paper_ready"] for row in evidence),
        "real_online_switch_complete": p7_complete,
    }


def markdown(headers: list[str], rows: list[list[str]]) -> str:
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
            *["| " + " | ".join(row) + " |" for row in rows],
        ]
    )


def table_evidence(summary: dict[str, Any]) -> str:
    rows = []
    for row in summary["evidence_chain"]:
        rows.append(
            [
                row["section"],
                row["claim"],
                row["evidence"],
                row["table"],
                "yes" if row["paper_ready"] else "no",
                row["boundary"],
            ]
        )
    return markdown(["section", "claim", "evidence", "result table", "paper-ready", "boundary"], rows)


def table_availability(summary: dict[str, Any]) -> str:
    rows = [
        [name, "yes" if row["exists"] else "no", row["path"]]
        for name, row in summary["availability"].items()
    ]
    return markdown(["result", "exists", "path"], rows)


def lines(title: str, values: list[str]) -> str:
    return title + "\n\n" + "\n".join(f"- {value}" for value in values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Chapter 6.8 discussion summary.")
    parser.add_argument("--output", default="data/results/ch6_8")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    summary = build_summary()
    (output / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tables = {
        "table_6_8_evidence_chain.md": table_evidence(summary),
        "table_6_8_result_availability.md": table_availability(summary),
        "limitations.md": lines("Limitations", summary["limitations"]),
        "next_steps.md": lines("Next steps", summary["next_steps"]),
    }
    for name, text in tables.items():
        (output / name).write_text(text + "\n", encoding="utf-8")
    print(tables["table_6_8_evidence_chain.md"])
    print(f"\n[exp_68] saved summary to {output}")


if __name__ == "__main__":
    main()
