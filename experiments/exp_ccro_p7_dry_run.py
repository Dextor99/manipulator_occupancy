"""P7 fail-closed dry run; never imports or calls a robot SDK."""
from __future__ import annotations
import argparse,json
from dataclasses import asdict
from pathlib import Path
import yaml
from robot.ccro_safety_gate import evaluate_gate
ROOT=Path(__file__).resolve().parents[1]
def run(config_path=ROOT/"config/ccro_p7.yaml"):
 cfg=yaml.safe_load(Path(config_path).read_text(encoding="utf-8"));out=ROOT/cfg["output_dir"];out.mkdir(parents=True,exist_ok=True);r=cfg["readiness"];phrase=cfg["required_operator_phrase"]
 decisions={m:asdict(evaluate_gate(m,r,cfg["allow_real_robot_commands"],None,phrase)) for m in ("dry_run","shadow","safety_layer","low_speed_switch")}
 unsafe_blocked=not decisions["low_speed_switch"]["allowed"] and not decisions["safety_layer"]["allowed"]
 metrics={"stage":"P7","scope":"software readiness only; no robot SDK imported","decisions":decisions,"unsafe_switch_blocked":unsafe_blocked,"dry_run_accepted":bool(decisions["dry_run"]["allowed"] and decisions["shadow"]["allowed"] and unsafe_blocked),"real_robot_complete":False,"accepted":False,"pending":[k for k,v in r.items() if not v]}
 (out/"metrics.json").write_text(json.dumps(metrics,indent=2,ensure_ascii=False)+"\n",encoding="utf-8");(out/"table_p7.md").write_text(f"| check | value |\n|---|---:|\n| dry run | {metrics['dry_run_accepted']} |\n| unsafe switch blocked | {unsafe_blocked} |\n| real robot complete | False |\n\nSoftware dry-run: **{'PASS' if metrics['dry_run_accepted'] else 'FAIL'}**  \nReal robot: **PENDING**\n",encoding="utf-8");return metrics
def main():
 p=argparse.ArgumentParser();p.add_argument("--config",default=str(ROOT/"config/ccro_p7.yaml"));a=p.parse_args();m=run(a.config);print(json.dumps({"dry_run_accepted":m["dry_run_accepted"],"unsafe_switch_blocked":m["unsafe_switch_blocked"],"real_robot_complete":m["real_robot_complete"],"pending":m["pending"]},indent=2));raise SystemExit(0 if m["dry_run_accepted"] else 2)
if __name__=="__main__":main()
