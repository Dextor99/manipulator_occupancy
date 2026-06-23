"""Unified A4/A5/A6 virtual closed-loop experiment."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np,yaml
from experiments.exp_ccro_stage4 import (_active_loop,_baseline,_limits,_make_components,_make_forecast,_passive,_states)
from planning.robot_surface_model import RobotSurfaceModel
from planning.safety_executor import SafetyExecutor

ROOT=Path(__file__).resolve().parents[1]

def _summary(rows,dt,goal):
    distances=np.asarray([r["distance"] for r in rows],float); return {"sample_count":len(rows),"min_distance":float(np.min(distances)),"time_below_stop":float(sum(r["distance"]<=r["d_stop"] for r in rows)*dt),"goal_error":float(np.linalg.norm(np.asarray(rows[-1]["q"])-goal)),"finished":bool(np.linalg.norm(np.asarray(rows[-1]["q"])-goal)<=.05)}

def _a6(timeline,evaluator,forecast,goal,config):
    cc=config["controller"]; source=config["_source"]; rc=source["risk"]; dt=float(source["virtual_loop"]["dt"])
    ex=SafetyExecutor(rc["d_stop"],rc["d_safe"],cc["tracking_gain"],cc["repulsive_gain"],cc["qd_limit"],cc["state_error_limit"])
    q=np.asarray(timeline[0]["q"],float).copy(); rows=[]; timings=[]
    for row in timeline:
        tau=float(row["time"]); qref=np.asarray(row["q"],float); qdref=np.asarray(row["qd"],float)
        started=time.perf_counter(); risk=evaluator.configuration(q,forecast,tau,density=cc["density"],with_gradient=True); cmd=ex.command(q,qref,qdref,risk.min_distance,risk.gradient_q); timings.append((time.perf_counter()-started)*1000)
        dense=evaluator.configuration(q,forecast,tau,density="dense",with_gradient=False); rows.append({"time":tau,"q":q.tolist(),"distance":dense.min_distance,"d_stop":rc["d_stop"],"state":cmd.state,"state_error":cmd.state_error,"speed_scale":cmd.speed_scale}); q=q+cmd.qd*dt
    # One second of safe goal recovery remains within the 9 s forecast horizon.
    tau=float(timeline[-1]["time"])
    for _ in range(int(1.0/dt)):
        tau+=dt
        if tau>forecast.valid_horizon: break
        risk=evaluator.configuration(q,forecast,tau,density=cc["density"],with_gradient=True); cmd=ex.command(q,goal,np.zeros(6),risk.min_distance,risk.gradient_q); q=q+cmd.qd*dt
        dense=evaluator.configuration(q,forecast,tau,density="dense",with_gradient=False); rows.append({"time":tau,"q":q.tolist(),"distance":dense.min_distance,"d_stop":rc["d_stop"],"state":cmd.state,"state_error":cmd.state_error,"speed_scale":cmd.speed_scale})
    result=_summary(rows,dt,goal); result.update({"control_mean_ms":float(np.mean(timings)),"control_p95_ms":float(np.percentile(timings,95)),"control_max_ms":float(np.max(timings)),"high_holds":sum(r["state"]=="high_hold" for r in rows),"state_mismatch_holds":sum(r["state"]=="state_mismatch_hold" for r in rows)}); return result

def run(config_path=ROOT/"config/ccro_p4.yaml"):
    cfg=yaml.safe_load(Path(config_path).read_text(encoding="utf-8")); source=yaml.safe_load((ROOT/cfg["source_config"]).read_text(encoding="utf-8")); cfg["_source"]=source; out=ROOT/cfg["output_dir"]; out.mkdir(parents=True,exist_ok=True)
    robot=source["robot"]; surf=source["surface"]; model=RobotSurfaceModel(ROOT/robot["urdf_path"],robot["joint_names"],surf["density_totals"],seed=surf["random_seed"],min_points_per_link=surf["min_points_per_link"],cache_dir=surf["cache_dir"],geometry=surf["geometry"])
    head,tail,durations=_states(source); limits=_limits(source); baseline=_baseline(source,head,tail,durations,limits).trajectory; goal=tail[:,0]; dt=float(source["virtual_loop"]["dt"]); results={}; checks=[]
    for name,scenario in source["experiment"]["scenarios"].items():
        forecast,info=_make_forecast(source,scenario,model,baseline); evaluator,verifier,factory=_make_components(source,model,limits,forecast)
        passive=_passive(baseline,evaluator,forecast,dt); a4={"min_distance":float(min(r["actual_min_distance"] for r in passive)),"goal_error":0.,"finished":True}
        active=_active_loop(source,baseline,goal,durations,evaluator,verifier,factory,forecast); timeline=active.pop("timeline"); a5={"min_distance":float(min(r["actual_min_distance"] for r in timeline)),"goal_error":active["goal_error"],"finished":active["finished"],"replan_count":active["replan_count"],"accepted_count":active["accepted_count"],"planning_control_cycles":active["planning_control_cycles"],"safety_events":active["safety_events"]}
        a6=_a6(timeline,evaluator,forecast,goal,cfg)
        expected=scenario["expected"]
        if expected=="accepted_replan": ok=a5["accepted_count"]>=1 and a5["min_distance"]>a4["min_distance"]+.002 and a6["min_distance"]>=source["risk"]["d_stop"]-cfg["acceptance"]["distance_tolerance"] and a6["control_p95_ms"]<cfg["acceptance"]["control_p95_ms"]
        elif expected=="no_trigger": ok=a5["replan_count"]==0 and a6["high_holds"]==0
        else: ok=len(a5["safety_events"])==1 and a6["high_holds"]>=1
        results[name]={"expected":expected,"forecast":info,"A4":a4,"A5":a5,"A6":a6,"accepted":bool(ok)}; checks.append(ok)
    metrics={"stage":"P4","scope":"unified URDF/Mesh virtual closed loop; no robot command","scenarios":results,"accepted":bool(all(checks))}; (out/"metrics.json").write_text(json.dumps(metrics,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    lines=["| scenario | A4 Dmin | A5 Dmin | A6 Dmin | A5 replans | A6 p95 ms | pass |","|---|---:|---:|---:|---:|---:|---:|"]
    for k,v in results.items(): lines.append(f"| {k} | {v['A4']['min_distance']:.5f} | {v['A5']['min_distance']:.5f} | {v['A6']['min_distance']:.5f} | {v['A5']['accepted_count']} | {v['A6']['control_p95_ms']:.2f} | {v['accepted']} |")
    lines+=['',f"Overall: **{'PASS' if metrics['accepted'] else 'FAIL'}**"]; (out/"table_p4.md").write_text("\n".join(lines)+"\n",encoding="utf-8"); return metrics

def main():
    p=argparse.ArgumentParser();p.add_argument("--config",default=str(ROOT/"config/ccro_p4.yaml"));a=p.parse_args();m=run(a.config);print(json.dumps({"accepted":m["accepted"],"scenarios":{k:v["accepted"] for k,v in m["scenarios"].items()}},indent=2));raise SystemExit(0 if m["accepted"] else 2)
if __name__=="__main__":main()
