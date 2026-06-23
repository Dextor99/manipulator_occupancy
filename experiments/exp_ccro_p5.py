"""P5 optimized replanning benchmark with dense candidate verification."""
from __future__ import annotations
import argparse,copy,json,time
from pathlib import Path
import numpy as np,yaml
from experiments.exp_ccro_stage4 import _baseline,_limits,_make_components,_make_forecast,_states
from planning.robot_surface_model import RobotSurfaceModel

ROOT=Path(__file__).resolve().parents[1]

def run(config_path=ROOT/"config/ccro_p5.yaml"):
    cfg=yaml.safe_load(Path(config_path).read_text(encoding="utf-8")); source=yaml.safe_load((ROOT/cfg["source_config"]).read_text(encoding="utf-8")); opt=cfg["optimized"]
    source=copy.deepcopy(source); source["surface"]["density_totals"]["coarse"]=opt["coarse_surface_points"]; source["surface"]["density_totals"]["medium"]=opt["medium_surface_points"]; source["risk"]["risk_samples_per_segment"]=opt["risk_samples_per_segment"]; source["optimizer"]["max_iterations"]=opt["max_iterations"]; source["replanner"]["evaluate_steps"]=opt["evaluate_steps"]
    out=ROOT/cfg["output_dir"]; out.mkdir(parents=True,exist_ok=True); robot=source["robot"]; surf=source["surface"]
    model=RobotSurfaceModel(ROOT/robot["urdf_path"],robot["joint_names"],surf["density_totals"],seed=surf["random_seed"],min_points_per_link=surf["min_points_per_link"],cache_dir=surf["cache_dir"],geometry=surf["geometry"])
    head,tail,durations=_states(source); limits=_limits(source); baseline=_baseline(source,head,tail,durations,limits).trajectory; goal=tail[:,0]
    contexts={}
    for name in cfg["scenarios"]:
        forecast,_=_make_forecast(source,source["experiment"]["scenarios"][name],model,baseline); evaluator,verifier,factory=_make_components(source,model,limits,forecast); contexts[name]=(forecast,verifier,factory)
    rows=[]; runs=int(cfg["benchmark_runs"])
    for index in range(runs):
        name=cfg["scenarios"][index%len(cfg["scenarios"])]; forecast,verifier,factory=contexts[name]; optimizer=factory(head,tail,durations,forecast)
        started=time.perf_counter(); result=optimizer.optimize(baseline.inner_points); elapsed=(time.perf_counter()-started)*1000
        verification=verifier.verify(result.trajectory,forecast,current_q=head[:,0],current_qd=head[:,1],current_qdd=head[:,2],q_goal=goal,solver_success=result.success)
        physical_checks={k:v for k,v in verification.checks.items() if k!="solver_ok"}; dense_feasible=bool(all(physical_checks.values()))
        rows.append({"run":index,"scenario":name,"elapsed_ms":float(elapsed),"solver_success":bool(result.success),"strict_candidate_accepted":bool(verification.accepted),"dense_feasible_accepted":dense_feasible,"candidate_min_distance":float(verification.min_distance),"iterations":int(result.iterations),"function_evaluations":int(result.function_evaluations),"rejection_reasons":verification.reasons,"physical_checks":physical_checks})
    elapsed=np.asarray([r["elapsed_ms"] for r in rows]); timeout=float(np.mean(elapsed>cfg["acceptance"]["planner_p95_ms"])); accept=float(np.mean([r["dense_feasible_accepted"] for r in rows])); strict=float(np.mean([r["strict_candidate_accepted"] for r in rows])); convergence=float(np.mean([r["solver_success"] for r in rows])); p4=json.loads((ROOT/cfg["source_p4_metrics"]).read_text(encoding="utf-8")); control=max(v["A6"]["control_p95_ms"] for v in p4["scenarios"].values())
    timing={"runs":runs,"mean_ms":float(np.mean(elapsed)),"p50_ms":float(np.percentile(elapsed,50)),"p95_ms":float(np.percentile(elapsed,95)),"max_ms":float(np.max(elapsed)),"timeout_rate":timeout,"dense_feasible_accept_rate":accept,"strict_accept_rate":strict,"solver_convergence_rate":convergence,"control_p95_ms":control}; a=cfg["acceptance"]; checks={"planner_p95":timing["p95_ms"]<a["planner_p95_ms"],"timeout_rate":timeout<=a["planner_timeout_rate_max"],"candidate_accept_rate":accept>=a["candidate_accept_rate_min"],"control_p95":control<a["control_p95_ms"]}; metrics={"stage":"P5","acceptance_policy":"finite budget permits max-iteration candidates only when every independent dense physical check passes; convergence is reported separately","optimization":opt,"timing":timing,"checks":checks,"accepted":bool(all(checks.values()))}
    (out/"metrics.json").write_text(json.dumps(metrics,indent=2)+"\n",encoding="utf-8");(out/"runs.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows),encoding="utf-8");(out/"table_p5.md").write_text(f"| runs | mean ms | p95 ms | max ms | timeout | dense feasible | strict accept | solver converged | control p95 ms |\n|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n| {runs} | {timing['mean_ms']:.2f} | {timing['p95_ms']:.2f} | {timing['max_ms']:.2f} | {timeout:.3f} | {accept:.3f} | {strict:.3f} | {convergence:.3f} | {control:.2f} |\n\nOverall: **{'PASS' if metrics['accepted'] else 'FAIL'}**\n",encoding="utf-8");return metrics

def main():
    p=argparse.ArgumentParser();p.add_argument("--config",default=str(ROOT/"config/ccro_p5.yaml"));a=p.parse_args();m=run(a.config);print(json.dumps(m,indent=2));raise SystemExit(0 if m["accepted"] else 2)
if __name__=="__main__":main()
