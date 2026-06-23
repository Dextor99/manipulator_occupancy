"""P6 unified ablation tables and time-weight sensitivity."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,yaml
from planning.nubs_trajectory import NUBSTrajectory6D
from planning.optimizer import FixedTimeNUBSOptimizer,JointLimits
from planning.time_optimizer import VariableTimeNUBSOptimizer
ROOT=Path(__file__).resolve().parents[1]

def _read(path): return json.loads((ROOT/path).read_text(encoding="utf-8"))
def run(config_path=ROOT/"config/ccro_p6.yaml"):
    cfg=yaml.safe_load(Path(config_path).read_text(encoding="utf-8")); out=ROOT/cfg["output_dir"];out.mkdir(parents=True,exist_ok=True)
    s1,p1,p2,p4,p5=(_read(cfg[k]) for k in ("source_stage1","source_p1","source_p2","source_p4","source_p5")); c=yaml.safe_load((ROOT/cfg["stage1_config"]).read_text(encoding="utf-8")); tr=c["trajectory"]; head=NUBSTrajectory6D.make_boundary_state(tr["q_start"],tr["qd_start"],tr["qdd_start"]);tail=NUBSTrajectory6D.make_boundary_state(tr["q_goal"],tr["qd_goal"],tr["qdd_goal"]);dur=np.asarray(tr["segment_durations"],float);r=c["robot"];limits=JointLimits.from_arrays(r["q_min"],r["q_max"],r["qd_max"],r["qdd_max"])
    single=NUBSTrajectory6D().generate(np.empty((0,6)),head,tail,np.array([dur.sum()])); fixed=FixedTimeNUBSOptimizer(head,tail,dur,limits,**c["optimizer"]).optimize(); sweep=[]
    for weight in cfg["lambda_time_sweep"]:
        opt=VariableTimeNUBSOptimizer(head,tail,dur,limits,mode="total",lambda_time=weight,max_iterations=200); result=opt.optimize(fixed.p_inner); sweep.append({"lambda_time":float(weight),"success":bool(result.success),"duration":float(result.durations.sum()),"jerk_energy":float(result.final_energy),"objective":float(result.final_cost),"max_limit_violation":float(max(result.max_q_violation,result.max_qd_violation,result.max_qdd_violation))})
    representation=[{"method":"single_quintic","duration":float(dur.sum()),"jerk_energy":single.energy()},{"method":"nubs_fixed","duration":p2["fixed"]["total_duration"],"jerk_energy":p2["fixed"]["jerk_energy"]},{"method":"nubs_variable","duration":p2["total"]["total_duration"],"jerk_energy":p2["total"]["jerk_energy"]}]
    risk={"pooled":p1["pooled"],"body_counterexample_rate":p1["body_counterexample_rate"]}
    closed={k:{"A4_Dmin":v["A4"]["min_distance"],"A5_Dmin":v["A5"]["min_distance"],"A6_Dmin":v["A6"]["min_distance"],"A5_replans":v["A5"]["accepted_count"],"A6_finished":v["A6"]["finished"]} for k,v in p4["scenarios"].items()}
    sources={"stage1":bool(s1["accepted"]),"P1":bool(p1["accepted"]),"P2":bool(p2["accepted"]),"P4":bool(p4["accepted"]),"P5":bool(p5["accepted"])}; sweep_rate=float(np.mean([x["success"] and x["max_limit_violation"]<=1e-8 for x in sweep])); checks={"sources":all(sources.values()),"sweep":sweep_rate>=cfg["acceptance"]["sweep_success_rate_min"]};metrics={"stage":"P6","sources":sources,"representation":representation,"risk_ablation":risk,"closed_loop":closed,"realtime":p5["timing"],"lambda_time_sweep":sweep,"sweep_success_rate":sweep_rate,"checks":checks,"accepted":bool(all(checks.values()))}
    (out/"metrics.json").write_text(json.dumps(metrics,indent=2,ensure_ascii=False)+"\n",encoding="utf-8");(out/"lambda_time_sweep.jsonl").write_text("".join(json.dumps(x)+"\n" for x in sweep),encoding="utf-8")
    lines=["# P6 统一基线与消融","","## 轨迹表示","","| method | duration / s | jerk energy |","|---|---:|---:|"]+[f"| {x['method']} | {x['duration']:.5f} | {x['jerk_energy']:.8f} |" for x in representation]+["","## 时间权重","","| lambda_time | duration / s | jerk energy | objective |","|---:|---:|---:|---:|"]+[f"| {x['lambda_time']:.3f} | {x['duration']:.5f} | {x['jerk_energy']:.8f} | {x['objective']:.8f} |" for x in sweep]+["",f"Overall: **{'PASS' if metrics['accepted'] else 'FAIL'}**"];(out/"table_p6.md").write_text("\n".join(lines)+"\n",encoding="utf-8");return metrics
def main():
 p=argparse.ArgumentParser();p.add_argument("--config",default=str(ROOT/"config/ccro_p6.yaml"));a=p.parse_args();m=run(a.config);print(json.dumps({"accepted":m["accepted"],"sources":m["sources"],"sweep_success_rate":m["sweep_success_rate"]},indent=2));raise SystemExit(0 if m["accepted"] else 2)
if __name__=="__main__":main()
